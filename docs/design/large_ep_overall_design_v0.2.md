# XTuner 大EP：整体设计、四库链路梳理与路线评审（v0.2）

> 状态：v0.2 讨论稿，承接规划 v0.1（W0–W7 方向不变，本文回答"具体怎么设计、怎么拆"），用于与框架组技术负责人对齐。
> 日期：2026-08-26 · 作者：FDE（衔接框架组 × 算子组）
> **2026-09-17 更新**：§2 的契约类型已被 [统一 EP 接口 v1.0](unified_ep_contract_v1.0.md) 取代，对照表见其 §2.7：`Layout` / `SFFormat` → `ExpertRows.alignment` 与 `hidden_scales`；`ExpertSpace` → 三条 docstring 约定；`Balancer` / `BalancePlan` → `LayerEPExecution` 五钩子与 `EPCall.plan`；`ExpertWeights.storage` 与 `ExpertParamAllocator` → 不进契约，归入解耦线（v1.0 §2.8）。§2.10 关于 `efsdp=1` 的一处修正：FSDP2 对 size 1 的 mesh 仍维护 unsharded 拷贝，固定地址要求专家跳过 `fully_shard`。本文其余章节（链路梳理、四库对照、路线）仍有效。
> 一手材料（本次逐文件复核，`file:line` 可点）：XTuner `4d7e23d6`、MoonEP 集成分支 `sh/a3-stable-moonep` @ 2ea7caaf、DeepEP dd758ca（v2.1.0）、MoonEP 0f385f0、UltraEP 94cab09（v1.0.0）、DeepGEMM 559d79f、sonic-moe、GroupedGEMM、MetaMoE 分支的优化记录、DeepEP PR #605、DeepGEMM PR #304/#323/#360。推断处标 **[推断]**。

---

## 0. 一页结论

### 0.1 对三个问题的直接回答

**Q1 · XTuner 现状是"通信库解耦"的吗？**
调用层解耦了：六段式 `GenericDispatcher` 协议（`dispatch_preprocess → dispatch → dispatch_postprocess → [experts] → combine_preprocess → combine → combine_postprocess`，`v1/module/dispatcher/base.py:70-159`）+ `build_dispatcher` 工厂 + experts 只消费 `PostDispatchResult{hidden_states, tokens_per_expert}`。**数据契约有且只有一种，而且是隐式的**：`PostDispatchResult{hidden_states, tokens_per_expert}` 只能表达一种形态——按本地专家连续、无 padding/gap、counts 是唯一分段信息、bf16 单张量、probs 未乘（由 `unpermute` 乘）、物理专家数 = `E/ep`。这六个假设里只有前三个写在类型字段上，其余散在 `GroupedLinear` 构造与各 dispatcher 的 post/pre-process 代码里；后端也无法声明自己产出/消费什么形态。要接的每一个库（DeepEP v2 expand、MoonEP、UltraEP、DeepGEMM psum、SonicMoE）都至少打破两条。同事的 MoonEP 集成分支是现成证据：为绕开这层，+2358 行、fork 了 640 行 kernel、`if is_moonep` 分支散落 6 处、每层 ≥6 次 host sync、丢掉 zero-copy 与 micro-batch overlap（§1.4）。

**Q2 · "多 backend 通信库" 还是 "MoE 阶段一个大类（ExpertBatch）"？**
不是二选一，两者不在一层：前者是**传输轴**（谁搬 token），后者应当是**数据契约**（搬完之后每个 rank 手里的张量长什么样）——而不是一个把 dispatch+GEMM+combine 包起来的"大类"（那会变成 #通信库 × #GEMM × #均衡器 的组合爆炸，moonep 分支的 `MoonEPMetaMoEBlock` 就是这种大类的雏形）。建议形态：**三个正交轴 + 一个显式契约 + 一个调度器 + 一个旁路**：

| 部件 | 内容 | 实现候选 |
|---|---|---|
| 轴 1 `TokenDispatcher` | 搬 token，六段式不变，但每段的类型换成契约 | all2all / all2all_dedup / deepep_v1 / **deepep_v2** / moonep / hier |
| 轴 2 `Balancer` | router 之后、dispatch 之前：改写路由到物理槽位 + 副本权重物化 + 反向梯度归约 | none / **ultraep** / moonep_plan |
| 轴 3 `GroupedGemmBackend` | 消费契约做 GEMM-1/act/GEMM-2 | triton / adaptive_fp8 / **deepgemm** / sonic / cutlass |
| 契约 `ExpertBatch` | layout 家族（R/E0/EA/M）+ `seg_start/seg_count`（GPU int32）+ 对齐 + SF 格式 + probs 状态 + 物理专家数 + 后端私有 handle | 三轴之间**唯一**的数据接口 |
| `MoEScheduler` | 从各后端声明的 `caps` 推导 overlap / micro-batch / recompute（EP-DR）/ CUDA graph | 一套，不再每库手写 |
| 旁路 `FusedMoE` | 吞掉六段式的第七种后端 | MegaMoE / UniEP 形态 |

任何一对 (dispatcher, gemm) 之间只允许三种廉价适配器（派生索引 / SF 转置 / compact），禁止专用胶水。接口代码在 §2，可直接拿去评审。

**Q3 · 工作设想怎么样？**
八条主线方向都对，顺序也对（解耦 → 契约 → v2 → 均衡 → 算子 → 融合）。有几个前提要修正，否则会在 v2 主线上撞墙（详见 §4.2）：

1. **DeepEP v2 的"固定 shape / 无 CPU 参与 / CUDA graph"是有条件的**：`do_cpu_sync=False` 时输出按最坏 `R × S_max × min(K, E_local)` 行预留（EP8、S=4096、K=8、H=7168 单层 `recv_x` 3.8 GB；EP64 30 GB），训练想要静态 shape 必须"均衡器给接收上界 + 给 DeepEP 打一个 allocation-hint 小 patch"；否则只能 `do_cpu_sync=True`（精确 shape，但 host 同步、不可 graph）。
2. **UltraEP 不是传输库**。你写的"当前 UltraEP 使用的是 hybrid EP、NVSHMEM"里，"hybrid EP"是它 Megatron 示例用的 token dispatcher **HybridEP（DeepEP v1 分支）**，NVSHMEM 只用于副本权重/梯度的对称堆。3.2 与 3.3 是**同一条代码路径**（求解器按 NVL 域自动分片），且副本**永不出 NVLink 域**——3.3 应改写为"跨节点 EP 组（DeepEP v2 hybrid）+ 域内实时均衡（UltraEP）+ 低频跨域 master 重排"。
3. **MoonEP "跨平台"只能是语义/契约移植**：规划算法与布局平台中性（`tests/planning_reference.py` 就是纯 torch 实现），但 7 个 CuTe DSL kernel + VMM 对称内存 + NVSwitch 多播 + 内联 PTX 全是 CUDA。
4. **FSDP/EP 解耦已在 MetaMoE 分支实现并有测试**（`DECOUPLE_EP_FSDP=1`，三维 mesh `(replicate, efsdp, ep)`，EP8 单机 −22 GB，8/16/64 rank 单测）。第 1 条的工作是**上游化到 `moe.py` 基类 + 与 #2007 Expert TP 合并 mesh 构建 + 走 L0–L4 验收 + 发版**，不是从零做。
5. **MegaKernel 在 H200 上还没有可用实现**：DeepGEMM MegaMoE 的 H200 移植 #360、H20 移植 #323 均未合入，#323 有寄存器溢出回归（H800 上反而慢于 DeepEP+DeepGEMM）。第 8 条的第一阶段应是"静态 shape + 整层 CUDA graph + shared-expert 掩体"，不是写 kernel。

### 0.2 需要负责人拍板的决策（§5 展开）

① P1 契约（`ExpertBatch` + `GroupGemmProtocol` 扩展）先于一切新 backend 合入；② MetaMoE 分支解耦实现的上游化归属，以及与 #2007 的 mesh 合并方式；③ DeepEP v2 `do_expand + expert_alignment=128 + psum` 与 DeepGEMM 作为默认训练路径（DeepGEMM 版本钉死进依赖）；④ 均衡器先 UltraEP 外挂（8 卡域），MoonEP-planning 作为同一 `Balancer` 接口的第二实现；⑤ 大 EP 默认 `efsdp=1`（专家只 EP、不 FSDP），并让专家权重 allocator 可替换（VMM / NVSHMEM / IPC）；⑥ topk 权重乘法迁入 SwiGLU 核（需数值 A/B）；⑦ 给 DeepEP v2 提一个 `max_recv_rows_hint` 的小 patch。

---

## 1. XTuner 现状诊断（代码级，`Xtuner/xtuner` @ 4d7e23d6，路径相对 `xtuner/v1/`）

### 1.1 现有架构一图

```
MoEDecoderLayer.forward / _micro_batch_forward                 (module/decoder_layer/moe_decoder_layer.py:393-604)
  ├─ _pre_moe_forward: attn + norm + MoEGate → RouterResults{topk_ids[T,K] i64, topk_weights[T,K] f32}
  ├─ dispatcher.dispatch_preprocess → dispatch → dispatch_postprocess     ┐ 六段式协议 (dispatcher/base.py:70-159)
  │        → PostDispatchResult{hidden_states[N,H] bf16, tokens_per_expert[E_local] (GPU)}   (base.py:28-45)
  ├─ MoEBlock: fused_w1w3(GroupedLinear) → moe_act(swiglu) → fused_w2                         (:150-191)
  ├─ dispatcher.combine_preprocess → combine → combine_postprocess          ┘
  └─ shared experts / residual
GroupedLinear.forward → xtuner.v1.ops.group_gemm(x[M,din], w[E_local,dout,din], tokens_per_expert)   (ops/moe/protocol.py:6-12)
```

**三个 dispatcher**（`build_dispatcher`，`dispatcher/__init__.py:30-79`；`ep_size==1` 强制 Naive；工厂注释自认 "does not follow the Liskov Substitution Principle"）：

| dispatcher | 传输 | 到达布局 | permute 在哪 | counts / host sync | probs 乘在哪 | 位置 |
|---|---|---|---|---|---|---|
| `all2all`（默认） | `all_to_all_single_autograd` | 按 (源 rank, 本地专家) | 发送前 `permute`（按 global expert 排序）+ 到达后第二次 `permute` | `histc` → a2a 交换计数 → **`.to("cpu").tolist()`** | 源侧 `combine_postprocess` 的 `unpermute(probs)` | `torch_all2all.py:78-113,328,416-461,566-570` |
| `deepep`（**legacy v1** normal 内核） | `Buffer.dispatch/combine` | 按源 rank 分段 + `recv_topk_idx`（-1 非本地） | 到达后框架 `permute(recv_x, recv_topk_idx)` | `num_recv_tokens_per_expert_list` 是 **host list**（v1 内部 host 等 GPU 计数，`deepep_op.py:187` 注释自认与 CUDA graph 不兼容）→ pinned H2D | 专家侧 `combine_preprocess` 的 `unpermute(probs)` 做部分和 | `deepep.py:349-421`；`ops/comm/deepep_op.py:146-223` |
| `agrs` | all-gather + reduce-scatter | 全量 token | — | — | RS 内求和 | `agrs.py:109-129,182-184`；前提 grouped router `ep_size==n_groups==8 && top_k==8`（`model/moe/moe.py:172-176`） |

**三个 GEMM 后端**（import 时按 device 一次性绑定，`ops/moe/__init__.py:17-38`）：

| 后端 | 契约 | counts | 对齐 | 精度 | 位置 |
|---|---|---|---|---|---|
| Triton TMA varlen（默认） | 按专家连续 + counts | GPU | kernel 内"虚拟 128 对齐"（`cdiv(size,128)` 生成 tile 表，不改 A 内存） | bf16 | `ops/moe/cuda/group_gemm.py:23-37`；`triton_kernels/m_grouped_gemm_TMA_triton3_4.py:274-352` |
| CUTLASS（fanshiqing `grouped_gemm`） | 同上 | **`.cpu()`** | 无 | bf16 | `group_gemm_cutlass.py:74-88` |
| AdaptiveGEMM FP8（InternLM，DeepGEMM 分支，arXiv 2508.16584） | 同上；A 1×128 tile / B 128×128 block，**fp32 row-major SF** | GPU | 前向 varlen；**wgrad 前 `trans_per_*_quant_expand_128x` 把每专家段 pad 到 128 倍**；dgrad 每步物理转置权重 | FP8 e4m3 | `float8/float8_gmm_tile_wise.py:28-36,86-153` |

DeepEP 用法的几个细节：Buffer 以 `low_latency_mode=True, num_qps_per_rank=max(E//ep, num_sms//2)` 创建（`deepep_op.py:89-95`）却只调 normal 内核（全仓无 `low_latency_dispatch`）；`Buffer.set_num_sms(20)` 是 import 时全局静态设置（`:24`）；`fwd/bwd_comm_dtype_fp8` 四处 `NotImplementedError`（`:314-316,390-392,509-511`）——**FP8 训练下通信仍是 bf16，量化在每个接收 rank 各自做一遍**；`deepep_op.py:300-431` 与 `dispatcher/deepep.py:76-219` 是两套重复的 `DeepEPDispatch/Combine`，`*BwdOnly/global_event_dict`（`:434-670`）无人调用。模型级并行精度测试只跑 `all2all`，`deepep` 用例被注释（`tests/model/test_moe.py:77`）。

### 1.2 调度、重计算与并行 mesh

- **层内 2-MB 两段式 overlap**（"domino EP"）写死在 `_micro_batch_forward`（`moe_decoder_layer.py:471-604`）：先对所有 MB 跑 attn+gate+`dispatch_preprocess(async)`；再逐 MB `dispatch(async)`→等 event→GEMM→`combine_preprocess`；所有 MB 的 `combine(async)` 连发，用 shared experts 做掩体；最后 `combine_postprocess`。机制是类级单例 comm stream + event（all2all）或 DeepEP `EventOverlap`（`previous_event` / `allocate_on_comm_stream`）。反向靠 `grad_fn.register_prehook/hook` 回填 event（`torch_all2all.py:255-274`；`deepep.py:222-240`）——脆弱且与 compile 不兼容（EP>1 时整层 forward 被移出 compile，`moe.py:84-101`）。没有跨层 1F1B/DualPipe 式重排，没有专家级 chunk 流水；comm/compute 的 SM 划分静态（DeepEP 20 SM，Triton `XTUNER_SM_MARGIN`）。
- **重计算**：`recompute_ratio` 默认 1.0，整层 `checkpoint_wrapper(REENTRANT)`（`moe.py:1151-1155`）→ 每层每 step **dispatch/combine 通信各执行 3 次**（fwd / recompute-fwd / bwd）。没有 EP-DR（只存 dispatch 输入、反向重放 dispatch），也没有"保留 dispatched 激活只重算 attention"的选择性策略。
- **EP × FSDP**：`_init_device_mesh` 固定 `(world//ep, ep)` 2D mesh（`moe.py:1342-1393`），非专家参数 `distribute_tensor(param, ep_mesh, [Replicate()])`（`moe.py:1137-1138,1415-1427`），HSDP 与 EP 互斥（`config/fsdp.py:49-51`）。`examples/v1/config/sft_glm5p2.py:36` 注释直白："EP=8 leaves FSDP size at 1 and replicates non-expert params"。fp8 reduce mesh 硬编码第三维 `ep_or_tp = world//fsdp`（`float8/float8_handler.py:157-161`）。
- **NPU**：`ops/moe/npu` 只实现 permute 子集（不支持 `num_out_tokens` / `-1` 索引，`permute_unpermute.py:13-20`），DeepEP 类后处理在 NPU 上不可用；没有 HCCL 原生 dispatch/combine 的接入点，没有 NPU 专属 dispatcher。

### 1.3 "通信库解耦"到什么程度：调用可插拔 ✓，数据契约单一且隐式 ✗

更准确的说法不是"没有契约"，而是**只有一种契约，且没有写全**：`PostDispatchResult{hidden_states, tokens_per_expert}`（`base.py:28-45`）能表达的形态只有一种；六个假设分别落在类型字段、`GroupedLinear` 构造和各 dispatcher 的代码路径里（所以换 backend 时没有任何类型检查会提醒你），每一个都被至少一个目标库打破：

| 隐式假设 | 写在哪里 | 谁打破它 |
|---|---|---|
| 按本地专家连续、无 padding / gap（下文记 **E0**） | 类型：`hidden_states` 行数 = `sum(tokens_per_expert)`，无起点/对齐字段（`base.py:28-45`）；GEMM 协议只有 `split_sizes`（`ops/moe/protocol.py:6-12`） | DeepEP v2 expand（段起点按 A 对齐、段间 gap）、MoonEP（128 对齐 + `[E+B]` 全局组）、HybridEP `pad_multiple` |
| counts 是唯一分段信息（无起点、无对齐语义） | 类型：同上 | DeepGEMM psum（要 `align(end, BLOCK_M)` 起点）、SonicMoE（要 `gather_idx` / offset） |
| `hidden_states` 是一张 bf16 | 类型：`hidden_states: Tensor` 单字段；fp8 量化在 `TileWiseFloat8GroupedLinear` 内做（`float8_gmm_tile_wise.py:86-110`） | DeepEP v2 FP8 dispatch `(fp8, sf)`，且 SF 有三种布局（row-major fp32 / MN-major TMA fp32 / UE8M0×4） |
| probs 在 `unpermute` 里乘 | 代码：各 dispatcher 的 `combine_preprocess / combine_postprocess`（`deepep.py:417`；`torch_all2all.py:566-570`） | MoonEP 与 v2 expand 给的是按行对齐的一维 `probs`，且三家 combine 都不乘 |
| 物理专家数 = `num_routed_experts // ep` | 代码：`GroupedLinear.__init__`（`moe_group_linear.py:24-29`）+ `len(tokens_per_expert)` | UltraEP（`E_local + R` 副本槽）、MoonEP（`E + B`） |
| permute 由框架做（fanshiqing `backend.permute`） | 代码：各 dispatcher 的 `dispatch_postprocess`（`deepep.py:367`；`torch_all2all.py:416-461`） | v2 expand / MoonEP 已在 dispatch 写终位，再 permute 是纯浪费 |

一句话：**调用层是可插拔的（六段式 + 工厂），数据层只有一个写死的形态（E0 + counts + bf16 + probs 未乘 + P_local = E/ep）**——新后端要么迁就它（多付 permute、拷贝、host sync、接收侧重新量化），要么绕开它（moonep 分支的做法）。此外三个"框架级"耦合：**host sync**（`.tolist()` / host list / `.cpu()`）；**全局单例**（`_buffer` 三个 getter、`set_num_sms`、类级 `_comm_stream`——同进程多 EP group / 多 hidden（VL 文本+视觉）不可能）；**async 双实现**（每个 dispatcher 手写 sync/async 两套 + 六个 TypedDict 各扩 event 字段，新后端约 500 行样板）。

### 1.4 反面教材：moonep 分支的集成方式（`sh/a3-stable-moonep`，两个 commit +2358/−65，无测试无配置样例）

| 现象 | 位置 | 后果 |
|---|---|---|
| `dispatch()/combine()` 新增**必填** kwarg `weights`，违反基类签名 | `dispatcher/moonep.py:413,554` | 调用侧无法多态 → `if is_moonep` 分支 `moe_decoder_layer.py` 3 处（`:414-426,438-452,468-484`）、`meta_moe_decoder_layer.py` 3 处 |
| 权重侧通信（prefetch / grad-reduce / B scratch）塞进 dispatcher，`build_dispatcher` 多 5 个只为 MoonEP 服务的参数 | `ops/comm/moonep_op.py:955-1007,1134-1168`；`dispatcher/__init__.py:30-40` | dispatcher 不再 token-only |
| 绕过 MoonEP 公开 `prefetch_weight/reduce_grad`，fork ≈640 行 CuTe DSL grad-reduce kernel（bf16 梯度、E/B 拆分布局） | `moonep_op.py:185-829` | MoonEP 升级即漂移；bf16 梯度累加、专家无 fp32 master |
| GEMM 走 fanshiqing CUTLASS，E 段/B 段两次 GMM + `cat`，`tokens_per_expert.cpu()`、`.item()` | `grouped_linear/moonep.py:23-44,158-198` | 每层前向 ≥6 次 host sync；MoonEP 的三个卖点（静态 shape / 无 host sync / zero-copy）只剩第一个 |
| `zero_copy=False`；`_micro_batch_forward` 未适配（会 `TypeError`）；不进 compile；`extra_ignored_params` 把专家剔出 FSDP → **DP=1** | `moonep_op.py:982-989`；`moe_decoder_layer.py:565-579`；`meta_moe.py:1582-1592` | 无 overlap、无 DP、无 checkpoint 验证 |
| 进程级全局状态按 `data_ptr()` 反查权重空间 | `moonep_op.py:133-134`；`grouped_linear/moonep.py:147-151` | optimizer 非原地更新 / offload / compile functionalization 都会炸 |

这不是同事的问题——是抽象层放错了缝：**契约不显式，任何新库都只能靠"特化签名 + 分支"接进去**。§2 的设计就是把这层缝补上；`support_multilayer` 那个 commit 做的"B 池全进程共享、E 权重按层"其实正是 MoonEP README 要求的形态，可以直接迁到 §2.5 的 `ExpertWeights`。

### 1.5 MetaMoE 分支已经做了的（要上游化，不要重做）

MetaMoE 分支的优化记录（Mobius-v0-30B-48L SFT，8×H200，naive EP8 2,580 → 6,286 tok/GPU/s）记录了四项与大EP直接相关、**不在 XTuner 主线也不在 moonep 分支**的实现：

| 项 | 开关 | 收益 | 实现 |
|---|---|---|---|
| **FSDP/EP 正式解耦** `root = (replicate, efsdp = dp_shard/ep, ep)`；dense FSDP over `flatten(efsdp, ep)`，expert `Shard(ep) + FSDP(efsdp)` | `DECOUPLE_EP_FSDP=1` | EP8 单机 dense 复制消失：峰值 96.30 → 73.95 GB（**−22 GB**），吞吐 +1.4%（噪声内） | `meta_moe.py::_init_decoupled_device_mesh`、`_scale_and_reduce_grad_decoupled`（expert `div_(ep)`，gate 显式 EP all-reduce，coalescing）、`_fsdp_mesh_for`（HF load/save 按参数类别选 mesh）、`_all_gather_chunked`（保存时 36 GiB → 4 GiB 分块）；测试 `tests/model/test_metamoe_decoupled_ep_fsdp_mesh.py` 覆盖 8/16/64 rank、EP1/2/4/8、HSDP placements；legacy 开关默认关、bit-exact |
| rank-dedup all2all v2 | `DISPATCHER=all2all_dedup` | top-8/EP8 平均访问 5.25 个 rank，双向 payload −34%；16K pack +2.8%（4K −5.7%，长 pack 专用） | `dispatcher/torch_all2all_dedup.py` |
| 共享专家池 dW 原位累加 | `XTUNER_MOE_DW_ACC=1` | +22.2%（48 层共享池下 dW 物化占 27% wall、1 TB/步 HBM 读写） | `_GroupedGemmAccDw` + `k_grouped_gemm_acc`（跳空专家，DTensor 直接进 Function） |
| 流水化 SwapAdamW / z-loss recompute | `SWAP_OPTIMIZER=1` / 层内 checkpoint | +24.5% / 稳态 −27 GB | `optim/swap_adamw.py`（3 组 ring slot 三流流水）；`meta_moe_decoder_layer.py` |

含义：**你计划的第 1 条不是"做解耦"，而是"把 `meta_moe.py` 里的解耦泛化到 `moe.py` 基类 + HSDP/DCP/fp8 mesh 补齐 + 与 #2007（Expert TP）合并 mesh 构建入口 + 走 proposal 的 L0–L4 验收 + 发版"**。这比从零做省一半以上，也是第一个能"先拿到收益"的 PR。`all2all_dedup` 与 dW acc 也应一起上游化（前者是 §2 里 `all2all` 后端的 dedup 变体，后者是 `GroupedGemmBackend.wgrad(acc=)` 的动机）。

---

## 2. 整体设计建议：三轴、一契约、一调度器

### 2.1 为什么不是二选一

"多 backend 通信库"回答的是**谁搬 token**；"ExpertBatch 大类"如果做成把 dispatch+GEMM+combine 都包进去的类，就会变成 (#通信库 × #GEMM × #均衡器) 的组合爆炸——moonep 分支的 `MoonEPMetaMoEBlock` + `MoonEPDispatcher` + `MoonEPGroupedLinear` 三件套正是这种"大类"的雏形。真正缺的是把今天隐式的 `tokens_per_expert` 升格为**显式数据契约**，让三个轴各自独立演进：

```
                          ┌──────────────── Balancer（轴 2，可选）────────────────┐
Router ──Routing──▶       │ plan(): 逻辑 expert → 物理 slot（GPU，无 host sync）     │──Routing'──▶
                          │ materialize(): 副本权重物化（weight_sync / prefetch）   │
                          │ reduce_grads(): 副本梯度归约回 master                   │
                          └───────────────────────────────────────────────────────┘
          ┌────────────── TokenDispatcher（轴 1）──────────────┐
          │ dispatch_preprocess → dispatch → dispatch_postprocess │──▶ ExpertBatch ──▶ GroupedExperts（轴 3）──▶ ExpertBatch'
          │ combine_postprocess ← combine ← combine_preprocess   │◀── ExpertBatch' ◀──   gemm1 → act(×probs, quant) → gemm2
          └───────────────────────────────────────────────────────┘
                ▲ 六段全部由 MoEScheduler 驱动：micro-batch 交错 / stream / recompute / graph，由各后端 caps 推导
          ┌────────────── FusedMoE（旁路）──────────────┐
          │ forward(hidden, Routing, ExpertWeights) → out │   MegaMoE / UniEP 形态，吞掉六段式
          └───────────────────────────────────────────────┘
```

五条设计原则（本文补代码与后端映射）：

1. **layout 是 dispatcher 与 experts 之间的显式契约，不是 dispatcher 的实现细节。**
2. **后端只声明产出 / 消费的 layout 家族，不互相感知**；中间只允许三种廉价适配器（派生索引 / SF 转置 / compact），禁止"为某对组合写专用胶水"。
3. **均衡器是 dispatcher 之前的插件**，对 experts 侧的唯一影响是"物理专家数 `P_local ≥ E_local`"，对 dispatcher 的唯一影响是"`num_experts` 变成物理 id 空间"。
4. **topk 权重乘法、SwiGLU、FP8 重量化三合一**放在 GEMM-1 与 GEMM-2 之间（MegaMoE `mega_moe.cuh` / DeepGEMM 生态的 `swiglu_apply_weight_to_fp8` / SonicMoE `gemm_act` 的共识），所有 combine 退化为纯求和——三家库（DeepEP v2、MoonEP、UltraEP）的 combine 本来就都不乘权重。
5. **分离形态永远保留**（调试、对拍、非静态 shape 回退），fused 整层后端作为第七种 backend 挂进同一 registry。

### 2.2 四个 layout 家族

| 家族 | 定义 | 产出者 | 消费者 |
|---|---|---|---|
| **R**（rank-grouped） | `[N_recv, H]` 按源 rank 分组，一 token 一行 + `topk_local[N, K]`（-1 无效）或 `gather_idx` | DeepEP v1 normal、DeepEP v2 `do_expand=False`、torch all2all（第一次 a2a 之后） | 框架 permute → E0/EA；**SonicMoE 型 gather-GEMM 可直接吃**（`A_idx` 就是 permute 索引） |
| **E0**（expert-contig） | `[N, H]` 按本地物理专家连续，无 gap，`alignment=1` | XTuner permute、NCCL EP HT、HybridEP `pad_multiple=1` | Triton varlen、AdaptiveGEMM、CUTLASS、SonicMoE（idx=identity）；**DeepGEMM 不行**（组起点未对齐） |
| **EA**（expert-aligned） | `[N_pad, H]`，`seg_start[i] = align(seg_start[i-1]+seg_count[i-1], A)`，gap 行零填或未定义 | **DeepEP v2 expand + `expert_alignment=A`**、**MoonEP**（A=128，零填，组数 `E+B`）、HybridEP `pad_multiple=A` | DeepGEMM psum / `m_indices`（A=BLOCK_M）；以及所有 E0 消费者——传 `aligned_counts()` 当 counts 即可（每专家 ≤A−1 行零计算） |
| **M**（masked 3D） | `[P_local, M_max, H]` + `masked_m[P_local]` | DeepEP v1 LL、NCCL EP LL | DeepGEMM masked；decode / rollout 专用，训练主线不用 |

EA ⊃ E0（A=1 即 E0）。**只要 dispatcher 统一产 EA（并暴露 start/count 两个 GPU 数组），XTuner 现有三个 GEMM 后端和 DeepGEMM 可以同时挂上，适配代价只剩 SF 格式与派生 `m_indices`。**

### 2.3 `ExpertBatch`：唯一的数据契约（替换 `PostDispatchResult`）

```python
# xtuner/v1/module/moe/contract.py —— 契约层：不 import 任何通信库 / GEMM 库
from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, Optional, Protocol
import torch
from torch import Tensor


class Layout(str, Enum):
    R  = "rank_grouped"    # 按源 rank 分组，一 token 一行；配 topk_local[N,K] 或 gather_idx
    E0 = "expert_contig"   # 按物理专家连续，无 gap（alignment=1）
    EA = "expert_aligned"  # 按物理专家分段，段起点 = align(prev_end, alignment)；gap 行零填/未定义
    M  = "masked"          # [P_local, M_max, H] + masked_m（decode / LL 专用）


class SFFormat(str, Enum):
    NONE                = "none"
    FP32_ROW_1x128      = "fp32_rowmajor_1x128"    # XTuner AdaptiveGEMM / DeepEP v1 / v2 默认
    FP32_MN_TMA_1x128   = "fp32_mn_tma_1x128"      # DeepGEMM SM90 SFA：stride(-2)==1, stride(-1)==align(M,4)
    UE8M0X4_MN_TMA_1x32 = "ue8m0x4_mn_tma_1x32"    # DeepGEMM SM100 packed UE8M0（v2 use_tma_aligned_col_major_sf）


@dataclass
class Routing:
    """router → (Balancer) → TokenDispatcher。id 空间可以是逻辑专家也可以是物理槽位，由 num_experts 说明。"""
    topk_ids: Tensor                              # [T, K] int32|int64；-1 = 屏蔽（DeepEP 语义）
    topk_weights: Tensor                          # [T, K] fp32
    num_experts: int                              # 该 id 空间的专家总数；物理空间时 = ep_size * P_local
    tokens_per_expert: Optional[Tensor] = None    # [num_experts] int32 GPU，本 rank 局部计数（MoonEP / UltraEP 需要）


@dataclass
class ExpertBatch:
    """dispatch_postprocess 产出 → GroupedExperts 消费 → combine_preprocess 回收。
    所有索引张量 int32、GPU 驻留；任何字段都不允许隐含 host sync。"""
    hidden: Tensor                        # [N_pad, H] bf16 | fp8_e4m3（M 布局为 [P_local, M_max, H]）
    layout: Layout
    num_phys_experts: int                 # P_local：E_local | E_local+R（UltraEP）| E+B（MoonEP，全局组）
    seg_count: Tensor                     # int32 [P_local] 真实计数（不含 pad）
    seg_start: Optional[Tensor] = None    # int32 [P_local]；E0 时可由 seg_count 派生
    alignment: int = 1                    # E0=1；EA=A（H 系 DeepGEMM 128）
    padding_zeroed: bool = False          # EA：gap 行是否已清零（DeepGEMM ensure_zero_padding=True 需要）
    num_valid_rows: Optional[Tensor] = None  # int32 标量 GPU：真实有效行数（= psum[-1]）；喂 act/量化核与 GEMM heuristics
    # --- R 布局附加 ---
    topk_local: Optional[Tensor] = None   # [N, K] 本地专家 id，-1 无效
    gather_idx: Optional[Tensor] = None   # [N_sorted] expert-sorted 行 → hidden 行号（SonicMoE 型 gather-GEMM 消费者）
    # --- M 布局附加 ---
    masked_m: Optional[Tensor] = None
    # --- 低精度 ---
    sf: Optional[Tensor] = None
    sf_format: SFFormat = SFFormat.NONE
    # --- 路由权重 ---
    probs: Optional[Tensor] = None        # fp32 [N_pad]（E0/EA，按行对齐）或 [N, K]（R）
    probs_applied: bool = False           # False → experts 侧须在 act 核里乘；True → combine 纯求和
    # --- 语义 ---
    deterministic_order: bool = False     # 段内序是否确定（W7 bitwise 需求）
    handle: Any = None                    # 后端私有：EPHandle / MoonEPCommPlan / (input_splits, output_splits, row_id_map)

    # ---- 派生视图：O(P_local) 或 O(N) 的 int32 小 kernel / view，带缓存，绝不 host sync ----
    def starts(self) -> Tensor:
        if self.seg_start is None:                       # E0
            self.seg_start = torch.cumsum(self.seg_count, 0) - self.seg_count
        return self.seg_start

    def psum(self) -> Tensor:                            # DeepGEMM use_psum_layout=True 的 grouped_layout
        return self.starts() + self.seg_count            # == DeepEP v2 handle.psum_num_recv_tokens_per_expert

    def aligned_counts(self) -> Tensor:                  # 喂 counts-only 消费者（Triton varlen / AdaptiveGEMM）
        s = self.starts()
        end_pad = torch.cat([s[1:], _align_up(self.psum()[-1:], self.alignment)])
        return end_pad - s                               # 每专家多算 ≤A-1 行零

    def m_indices(self) -> Tensor:                       # DeepGEMM m_indices 型 [N_pad]，gap 行 = -1
        return _seg_to_m_indices(self.starts(), self.seg_count, self.hidden.shape[0])   # 复用 triton m_indices_pad

    def cu_seqlens(self) -> Tensor:                      # [P_local+1]；仅 alignment==1 时对 SonicMoE 合法
        return torch.cat([torch.zeros_like(self.seg_count[:1]), self.psum()])

    def with_hidden(self, hidden: Tensor, **changes) -> "ExpertBatch":
        return replace(self, hidden=hidden, **changes)  # experts 产出同布局的新 batch（零拷贝元数据）

    def to_layout(self, layout: Layout, alignment: int = 1) -> "ExpertBatch":
        """唯一允许的重排：R→E0/EA（permute，可指定对齐）、EA→E0（compact）。O(N·H) 拷贝，只在退路使用。"""
        ...
```

说明：
- `seg_start/seg_count` 与 DeepEP v2 的 `psum_num_recv_tokens_per_expert` / `num_unaligned_recv_tokens_per_expert` 一一对应（§3.2.2），与 MoonEP 的 `cu_seqlens` + `zero_fill_ranges` 一一对应（§3.3），与 XTuner 现有 `tokens_per_expert` 在 `alignment=1` 时完全等价——**现有三个 dispatcher × 三个 GEMM 在 E0 下 bit-exact 回归**是 P1 的验收标准。
- `num_valid_rows` 是 GPU 标量而不是 host int：无 host sync 模式下 hidden 的行数是上界，真实行数只在 GPU 上（DeepEP v2 `psum[-1]`），act / 量化核用它裁剪（DeepGEMM 生态 `avail_tokens` 语义），GEMM 用 `expected_m_for_psum_layout` 做 heuristics。
- `probs` 的形状由 layout 决定：EA/E0 是按行对齐的一维（v2 expand 的 `recv_topk_weights[N]`、MoonEP 的 `route_weights_nvs[NvS]`），R 是 `[N, K]`（v1、v2 非 expand）。
- 现有 permute kernel（fanshiqing `permute.cu`）只需**多导出 `sorted_row_id`**（排序输出 ÷ K）并**加一个 `alignment` 参数**，就能同时产 EA 与 SonicMoE 需要的 `gather_idx`——这是 P1 里唯一的 kernel 改动。

### 2.4 `TokenDispatcher`：六段式不变，加 capabilities，反向统一

```python
@dataclass(frozen=True)
class DispatcherCaps:
    """后端声明自己能做什么；调度器据此推导方案，配置校验据此提前报错（而不是 NotImplementedError 到运行时）。"""
    produces: tuple[Layout, ...]                  # 可产出的布局家族，首个为默认
    alignments: tuple[int, ...] = (1,)            # 可选段对齐
    fp8_dispatch: bool = False                    # 接受 (fp8, sf) payload
    sf_formats: tuple[SFFormat, ...] = (SFFormat.NONE,)
    static_shape: bool = False                    # 输出 shape 与路由无关（最坏预留 / 完美均衡 / 接收上界）
    host_sync_free: bool = False                  # 前向无 .item() / list / .cpu()
    cuda_graph: bool = False
    replayable_dispatch: bool = False             # dispatch(handle=) 可重放（EP-DR 的前提）
    cross_node: bool = False
    phys_expert_space: bool = False               # 接受 num_experts = ep_size * P_local 的物理 id 空间（均衡器）
    deterministic: bool = False
    async_stream: bool = True                     # comm stream + event
    combine_applies_probs: bool = False           # 三家库都 False；v1 LL 才 True
    max_topk: int = 32
    hidden_multiple: int = 1                      # DeepEP v2 combine 要求 256；MoonEP 要求 128
    fixed_tokens_per_rank: bool = False           # MoonEP：每 rank 每次恰好 S 个 token（框架 pad）
    inter_rank_barrier: Literal["none", "per_call"] = "none"   # MoonEP inter_rank_sync 是正确性屏障


@dataclass
class MoESpec:            # setup 时一次性给足，后端据此建 buffer / JIT；进程内允许多个 spec（去掉全局单例）
    hidden: int
    topk: int
    num_experts: int                  # 物理 id 空间总数（无均衡器时 = 逻辑专家数）
    num_local_experts: int            # P_local
    max_tokens_per_rank: int          # 发送侧上界（DeepEP v2 num_max_tokens_per_rank / MoonEP S）
    dtype: torch.dtype
    alignment: int                    # 由 GEMM 后端 required_alignment 决定
    sf_format: SFFormat               # 由 GEMM 后端 sf_formats[0] 决定 → dispatcher 按需产出，不做转置适配
    ep_group: torch.distributed.ProcessGroup
    max_recv_rows_hint: Optional[int] = None     # 均衡器给的接收上界；有则静态 shape 不必最坏预留（§4.2-1）


@dataclass
class StageCtx:
    async_op: bool = False
    prev_event: Any = None            # 上游 event（跨 stream 依赖）
    replay: bool = False              # 反向重放 dispatch（EP-DR）
    layer: int = 0
    micro_batch: int = 0


class TokenDispatcher(Protocol):
    caps: DispatcherCaps
    def setup(self, spec: MoESpec) -> None: ...
    # ---- 六段式保持不变（调度器需要它做交错）；变的是每段的类型与语义 ----
    def dispatch_preprocess(self, hidden: Tensor, route: Routing, *, ctx: StageCtx) -> Any: ...   # 量化 / 排序 / 计数
    def dispatch(self, pre: Any, *, ctx: StageCtx) -> Any: ...                                     # 只搬 token
    def dispatch_postprocess(self, dispatched: Any, *, ctx: StageCtx) -> ExpertBatch: ...          # 落到契约（尽量 zero-copy）
    def combine_preprocess(self, out: ExpertBatch, *, ctx: StageCtx) -> Any: ...                   # probs 已乘：多为 identity
    def combine(self, pre: Any, *, ctx: StageCtx) -> Any: ...
    def combine_postprocess(self, combined: Any, *, ctx: StageCtx) -> Tensor: ...                   # [T, H]
```

**反向由基类统一实现，不再每个后端手写两套 `autograd.Function`。** 三家库的反向都是"转置"：`dispatch-bwd = combine(grad, handle, topk_weights=grad_probs)`、`combine-bwd = dispatch(grad, handle=handle)`（DeepEP v2 `elastic.py:937-948`；MoonEP README；v1 同）。基类 `TransposeBackwardDispatcher` 提供两个通用 Function，子类只实现 `_dispatch_impl/_combine_impl(handle)`；`all2all` 因 `all_to_all_single_autograd` 自带反向，标 `backward="autograd"`。`grad_probs`（一维、按行）通过 combine 的 `topk_weights` 通道 gather 回 `[T, K]`（v2 `combined_topk_weights`；MoonEP `route_weights_nvs` gather；all2all `unpermute_bwd`）。流同步用 `ctx.prev_event` 显式传递，删除 `grad_fn.register_prehook` 那套。

### 2.5 `Balancer`：dispatcher 之前的插件

```python
@dataclass
class BalancePlan:
    routing: Routing                  # 已重映射到物理 id 空间（num_experts = ep_size * P_local）
    num_phys_experts_local: int
    handle: Any = None                # UltraEP: (vid, placement)；MoonEP: MoonEPCommPlan
    max_recv_rows: Optional[int] = None   # 该 plan 承诺的接收上界（UltraEP τ·mean / MoonEP S·K）


class Balancer(Protocol):
    num_phys_experts_local: int                                        # E_local + R（UltraEP）| E + B（MoonEP）
    def setup(self, spec: MoESpec, weights: "ExpertWeights") -> None: ...   # 注册 master 指针 / 建对称槽
    def plan(self, route: Routing, *, ctx: StageCtx) -> BalancePlan: ...    # 全 GPU：本地直方图 → all-gather → 求解 → reroute
    def materialize(self, plan: BalancePlan, *, ctx: StageCtx) -> "ExpertWeights": ...   # weight_sync / prefetch（comm stream，返回含 event 的视图）
    def restore_for_backward(self, plan: BalancePlan) -> None: ...          # 副本槽跨层复用被覆写 → 反向再物化（UltraEP RestoreReplicaWeights / MoonEP 再 prefetch）
    def reduce_grads(self, plan: BalancePlan, *, ctx: StageCtx) -> Any: ... # grad_reduce / reduce_grad（async，返回 token）
    def join(self, token: Any) -> None: ...                                 # 下一层 wgrad 覆盖共享槽之前 join
```

两种落法：

| | `ultraep`（方案 A） | `moonep_plan`（方案 B 的规划层） |
|---|---|---|
| `plan()` | `topk_local_sum` → `all_gather_into_tensor`（NCCL 即可，不必 NVSHMEM fcollect）→ `solve_placement_for_test(..., num_nvl_ranks=8, rank_quota_source_rank=my_rank)`（`ultra_ep.hpp:427-533`，**免 `init_runtime`**）→ `reroute_sparse` 原地改 `topk_ids` 为物理 id（`reroute.cu:492-515`，需补一个 pybind + 确定性版本） | `launch_planning`（`planning.py:1294`，或先用 `tests/planning_reference.py` 的纯 torch 版）→ 把 `dst` 的 (dest rank, slot) 翻译成 `ep_size×(E_local+B)` 空间的物理 id |
| `materialize()` | `weight_sync(layer, async)`：NVL 域 ≤8 走 `thread_copy_kernel` direct（`manager.py:114-122`）；副本落 `[R, fc1+fc2]` 对称堆 strided 视图（`manager.py:180-245`） | `prefetch_weight(plan, full_*)`：warp 特化 2D TMA 从 home rank 对称 VA 读到 `[E, E+B)` 槽（`prefetch.py:115-126`） |
| `reduce_grads()` | `grad_reduce(layer, async)`：42 SM，TMA 拉远端副本梯度 → 累加 master 并清零远端（`grad_reduce.cu:252-257`），**fp32 强制** | `reduce_grad(plan, full_*_grad, *_reduce_buffer)`：远端 pull `[R,B]` 槽 + 本地累加 + 本地清零（`grad_reduce.py:326-389`），**无入口屏障，框架须先同步** |
| 副本存储 | NVSHMEM 对称堆（可换 `IpcManager` 的 cudaIpc / fabric handle，kernel 一行不改，`ipc_manager.cu:76-137`） | VMM `cuMemCreate` + fd 交换（`nvl_shared_buffer.cuh:108-241`） |
| 硬限制 | 副本不出 NVL 域；`MAX_EXPERTS_PER_NVL=512`（`placement.cu:22`：**E=2560/EP8 → 2560 > 512，需改常数重编**）；权重/梯度 `data_ptr` 稳定；一进程一个 EP group | `B = E/R`、`R ≤ 128`、`E % R == 0`、`K ≤ 32`、`H,H' % 128 == 0`、每 rank 固定 S |

对 dispatcher 的要求只有一条：`caps.phys_expert_space=True`——DeepEP v2 的 expert→rank 映射是纯整除 `expert_id // (num_experts / num_ranks)`（`dispatch.cuh:104,320`），`num_experts` 只是一个整数，天然满足；all2all 同理。

### 2.6 `ExpertWeights`、`GroupedExperts`、`GroupedGemmBackend`、`moe_act`

```python
@dataclass
class ExpertWeights:
    """experts 侧看到的权重：若干"段"，每段 [P_seg, N, K] K-major 连续；段与段可来自不同分配。"""
    w1w3: tuple[Tensor, ...]          # (master[E_local,2I,H], replica[R,2I,H]) | (full[E+B,2I,H],) | (w[E_local,2I,H],)
    w2:   tuple[Tensor, ...]
    seg_sizes: tuple[int, ...]        # (E_local, R) | (E+B,) | (E_local,)
    storage: Literal["dtensor_fsdp", "dtensor_ep_only", "symmetric_vmm", "symmetric_nvshmem", "ipc"]
    grad_sinks: Optional[tuple[Tensor, ...]] = None   # 副本梯度落点（fp32），供 Balancer.reduce_grads
    ready: Any = None                                 # 物化完成 event


class GroupedGemmBackend(Protocol):
    name: str
    accepts: tuple[Layout, ...]
    dtypes: tuple[torch.dtype, ...]
    sf_formats: tuple[SFFormat, ...]              # 首个 = 希望 dispatcher / act 核直接产出的格式
    required_alignment: int = 1                   # DeepGEMM = BLOCK_M（SM90 128；SM100 32…224）
    needs_host_counts: bool = False               # fanshiqing CUTLASS（cuBLAS 路径）True → 只做退路
    supports_wgrad_acc: bool = False              # DeepGEMM k-grouped 必传 c；Triton k_grouped_gemm_acc
    def fwd(self, x: ExpertBatch, w: Tensor, *, out_dtype=torch.bfloat16) -> Tensor: ...       # [N_pad, N_out]
    def dgrad(self, g: ExpertBatch, w: Tensor) -> Tensor: ...
    def wgrad(self, g: ExpertBatch, x: ExpertBatch, *, acc: Optional[Tensor] = None) -> Tensor: ...


class ActKernel(Protocol):
    """SwiGLU × probs × (FP8 重量化) 三合一；反向同一 epilogue 出 dH、d_probs（SonicMoE dS = <dA', A> 公式）。"""
    def __call__(self, h: ExpertBatch, *, apply_probs: bool, quant_to: SFFormat) -> ExpertBatch: ...


class GroupedExperts(torch.nn.Module):
    def __init__(self, gemm: GroupedGemmBackend, act: ActKernel): ...
    def forward(self, x: ExpertBatch, w: ExpertWeights) -> ExpertBatch:
        assert x.layout in self.gemm.accepts and x.alignment % self.gemm.required_alignment == 0
        h = _gemm_over_segments(self.gemm.fwd, x, w.w1w3, w.seg_sizes)       # 多段 → 多次 launch（EA 下段天然连续，零拷贝）或 ptr-array 一次
        a = self.act(x.with_hidden(h), apply_probs=not x.probs_applied, quant_to=self.gemm.sf_formats[0])
        y = _gemm_over_segments(self.gemm.fwd, a, w.w2, w.seg_sizes)
        return x.with_hidden(y, sf=None, sf_format=SFFormat.NONE, probs_applied=True)
```

说明：
- `_gemm_over_segments`：UltraEP 的物理编号是"本 rank 的 M 个 master 在前、R 个 replica 在后"（`placement.cu:1493-1501`），EA 布局下两段在 `hidden` 里天然连续，行段 `[0, seg_start[E_local])` × master、`[seg_start[E_local], N_pad)` × replica 各一次 launch，**零拷贝**；优化项是 CUTLASS ptr-array grouped GEMM 合成一次。MoonEP 是单段 `[E+B]`（绝大多数组空，DeepGEMM/Triton 都跳空组）。
- `wgrad(acc=)`：承接 MetaMoE 的 dW 原位累加（共享池 48 次 backward 只物化一次 dW）；DeepGEMM k-grouped 本来就要求传 `c`。
- `sf_format` 由消费者决定、生产者按需产出：DeepGEMM 要 MN-major TMA（v2 `use_tma_aligned_col_major_sf=True` 直接给），AdaptiveGEMM 要 row-major fp32（`=False`）——**不需要转置适配器**，只需在 `setup` 时把 GEMM 的 `sf_formats[0]` 写进 `MoESpec`。
- act 核里乘 probs 在数学上与在 GEMM-2 之后乘等价（逐行标量与线性映射可交换），但改变了 bf16/fp8 舍入位置，P3 必须做 loss 级 A/B。

### 2.7 `MoEScheduler`：由 caps 推导，不再每库一套

```python
@dataclass
class SchedulePlan:
    micro_batches: int                                        # 1 | 2（层内交错）
    overlap: Literal["none", "intra_layer_2mb", "shared_expert_cover", "both"]
    recompute: Literal["full_layer", "ep_dr", "none"]         # ep_dr：只存 dispatch 输入，反向重放 dispatch（K3 EP-DR）
    graph: bool                                               # 整层 CUDA graph 捕获
    comm_sms: int


def derive_plan(disp: DispatcherCaps, bal: Optional[Balancer], cfg: "MoEParallelConfig") -> SchedulePlan:
    static = disp.static_shape or (bal is not None and cfg.max_recv_rows_hint is not None)
    return SchedulePlan(
        micro_batches = 2 if (disp.async_stream and cfg.intra_layer_micro_batch > 1) else 1,
        overlap       = "both" if disp.async_stream else "shared_expert_cover",
        recompute     = ("ep_dr" if (cfg.recompute and disp.replayable_dispatch)
                         else "full_layer" if cfg.recompute else "none"),
        graph         = static and disp.host_sync_free and disp.cuda_graph and cfg.allow_cuda_graph,
        comm_sms      = cfg.comm_sms or disp_default_sms(disp, cfg),
    )


class MoEScheduler:
    """把六段式 + Balancer 钩子 + experts 按 SchedulePlan 编排。现有 _micro_batch_forward 的两段式就是 plan=intra_layer_2mb 的特例。"""
    def __init__(self, dispatcher: TokenDispatcher, experts: GroupedExperts, balancer: Optional[Balancer], plan: SchedulePlan): ...

    def forward(self, hidden_list: list[Tensor], route_list: list[Routing], weights: ExpertWeights) -> list[Tensor]:
        d, b, plan = self.dispatcher, self.balancer, self.plan
        # 阶段 1：所有 MB 的 balancer.plan + dispatch_preprocess（可与前一 MB 的 attention 重叠）
        bplans = [b.plan(r, ctx=StageCtx(async_op=True, micro_batch=i)) if b else None for i, r in enumerate(route_list)]
        pres   = [d.dispatch_preprocess(h, bp.routing if bp else r, ctx=StageCtx(async_op=True, micro_batch=i))
                  for i, (h, r, bp) in enumerate(zip(hidden_list, route_list, bplans))]
        # 阶段 2：MB(i+1) 的 dispatch 在 comm stream 上与 MB(i) 的 experts 重叠
        outs = []
        for i, pre in enumerate(pres):
            w_i = b.materialize(bplans[i], ctx=...) if b else weights          # weight_sync ∥ dispatch
            disp = d.dispatch(pre, ctx=StageCtx(async_op=plan.micro_batches > 1, micro_batch=i))
            batch = d.dispatch_postprocess(disp, ctx=...)                        # 等 comm event；零拷贝落 ExpertBatch
            outs.append(d.combine_preprocess(self.experts(batch, w_i), ctx=...))
        # 阶段 3：所有 MB 的 combine 连发，shared experts 做掩体
        combined = [d.combine(o, ctx=StageCtx(async_op=True, micro_batch=i)) for i, o in enumerate(outs)]
        shared   = [self.shared_experts(h) for h in hidden_list] if plan.overlap != "none" else None
        return [d.combine_postprocess(c, ctx=...) + s for c, s in zip(combined, shared or [0] * len(combined))]
```

三种 overlap 模板都是**同一套六段的不同编排**，后端一行不改：`intra_layer_2mb`（现状）、`shared_expert_cover`（K3 的 SE1/SE2：dispatch 藏在 SE1 后、combine 藏在 SE2 后，单 MB 也能用）、`ep_dr`（反向只重放 dispatch，不重放整层——把重计算段的通信从 2 次降到 1 次，前提 `caps.replayable_dispatch`：v2 cached handle、MoonEP `dispatch(plan=)`、all2all 保存 splits 都满足）。`graph=True` 的前提是三个 caps 同时成立——这就是 W3 均衡 → W4 图化的依赖在代码里的样子。

#### 2.7.1 两个时序例子：同一个调度器，两个后端

现有 `_micro_batch_forward` 的两段式（`moe_decoder_layer.py:471-604`）在新设计里就是 `plan.overlap="intra_layer_2mb"`，逐段对应如下（`D` = dispatch，`C` = combine，`X` = experts，`SE` = shared experts，`P` = balancer.plan，`W` = balancer.materialize）：

```
deepep_v2 + ultraep（2 个 micro-batch，前向）
compute : attn+gate(mb0) attn+gate(mb1) │ X(mb0)            │ X(mb1)            │ SE(mb0) SE(mb1) │ +res
comm    :                P(mb0) P(mb1)  │ W(mb0)∥D(mb0)     │ W(mb1)∥D(mb1)     │ C(mb0)  C(mb1)  │
依赖    : D(mb1) 只等阶段 1 的 event → 与 X(mb0) 并行；C(*) 连发后用 SE 掩体；W 必须先于 D 完成（UltraEP 要求 dispatch 前 weight_sync 结束）

moonep（同一调度器；差别全部来自 caps）
compute : attn+gate(mb0) attn+gate(mb1) │ X(mb0)            │ X(mb1)            │ SE(mb0) SE(mb1) │ +res
comm    :                               │ D(mb0)[含 planning]→W(mb0) │ D(mb1)→W(mb1) │ C(mb0)  C(mb1)  │
差别    : ① planning 融合在 dispatch 内 → caps.plan_inside_dispatch=True，调度器把 W 排在 D 之后（prefetch 需要 plan）
          ② caps.fixed_tokens_per_rank=True → 每个 mb 都 pad 到 S
          ③ caps.inter_rank_barrier="per_call" → 所有 rank 的 D/C 调用顺序必须一致（本来就一致），不能跨 rank 乱序
          ④ zero_copy=True → X 必须原位写回同一 [NvS,H]（ExpertBatch.with_hidden 保证）

反向（两者相同的骨架）
C-bwd = D(handle)   → 若 recompute="ep_dr"：只重放这一段 dispatch，不重放整层
X-bwd（dgrad ∥ wgrad(acc)）
D-bwd = C(handle, topk_weights=d_probs) → 拿回 grad_x 与 grad_topk_weights
balancer.restore_for_backward 在 X-bwd 之前（UltraEP 阻塞 weight_sync / MoonEP 再 prefetch）
balancer.reduce_grads 在 X-bwd 之后异步，下一层 X-bwd 之前 join（UltraEP 42 SM；MoonEP reduce_grad 前先 inter_rank_sync）
```

`StageCtx(async_op, prev_event, micro_batch)` 是唯一在六段之间传递流依赖的载体：后端在 comm stream 上执行并把 event 放进返回值，下一段用 `ctx.prev_event` 等待；调度器只决定"谁先谁后、谁跟谁并行"，不知道也不需要知道后端是 NCCL Gin 还是 VMM push。今天每个 dispatcher 各写一套 sync/async 双路径 + `grad_fn.register_prehook` 回填 event 的做法（§1.2）被这一个对象替掉。

### 2.8 `FusedMoE` 旁路

```python
class FusedMoEBackend(Protocol):
    caps: DispatcherCaps                   # 复用同一 caps 描述（static_shape / fp8 / cross_node…）
    def setup(self, spec: MoESpec) -> None: ...
    def forward(self, hidden: Tensor, route: Routing, weights: ExpertWeights) -> Tensor: ...   # dispatch→GEMM→act→GEMM→combine 一个 kernel
```

MegaMoE（`deep_gemm.fp8_fp4_mega_moe`，SM100、仅前向、≤72 rank、token 对齐 384）就是这个形态；UniEP 形态的训练侧原型也挂这里。`MoEDecoderLayer` 按 `config.dispatcher == "fused"` 选它，分离形态永远保留作对拍。

### 2.9 各后端映射到契约

| backend | 六段怎么落 | 产出 | 物理专家 | FP8 | 静态 shape | host sync | 备注 |
|---|---|---|---|---|---|---|---|
| `all2all` / `all2all_dedup` | pre: permute(按 global expert, alignment=A) + histc + a2a 交换 counts；dispatch: `all_to_all_single_autograd`；post: 第二次 permute → **EA**（或 E0）；combine_pre: unpermute；post: 纯求和 | EA/E0 | E_local（或物理空间） | ✗ | ✗ | `.tolist()`（可用 `num_worst_tokens` 式预留消掉） | 保底与对拍；dedup 变体上游化 |
| `deepep_v1`（现 `deepep`，改名保留） | 同现状：R → permute → E0/EA | R→EA | E_local | ✗ | ✗ | host list | 回退路径 |
| **`deepep_v2`** | pre: 预量化 `(fp8, sf)` + `topk_ids` int64；dispatch: `ElasticBuffer.dispatch(do_expand=True, expert_alignment=A, do_zero_padding=True, use_tma_aligned_col_major_sf=(GEMM 要 MN-major), do_cpu_sync=not static)`；post: **零拷贝**——`seg_start = align(psum[:-1], A)`、`seg_count = num_unaligned`、`probs = recv_topk_weights[N]`、`handle = EPHandle`；combine_pre: identity；combine: `combine(x_bf16, handle)`；post: identity | **EA**（或 R） | E_local 或 E_local+R | ✓ 预量化 | `do_cpu_sync=False` 最坏预留 / hint | 可免 | 主力；训练必须 `allow_multiple_reduction=True`（否则 dispatch 反向不能带 `topk_weights`，`combine.cuh:67-69`）；`hidden % 256 == 0`；`deterministic=True` 可选 |
| **`moonep`** | pre: pad 到 S + `bincount` + int32；dispatch: `Buffer.dispatch(hidden_sh, w_sk, topk_sk, tpe, zero_copy=True)`（planning 融合在内）；post: **零拷贝**——`P = E+B`、`seg_start/seg_count` 由 `cu_seqlens` 与 `zero_fill_ranges` 派生、`probs = route_weights_nvs`、`padding_zeroed=True`、`alignment=128`；combine: `Buffer.combine(plan, out_nvsh)` | **EA**（A=128） | E+B（全局组） | ✗ bf16 | ✓ `NvS` | 无（`.item()` 仅 `destroy`） | dispatcher 与 balancer 合一：`Balancer` 接口包 `prefetch_weight/reduce_grad`；权重须 `[E+B]` 对称 VA（§2.10） |
| `hier`（W2-b） | 域间 `deepep_v2` R 模式（到达同轨 GPU，`recv_topk_idx` 已是池内 id）→ 域内 `moonep` → EA | EA | E+B | 域间 ✓ / 域内 ✗ | 域内 ✓ | — | 6.2b 变体：v2 EA 全 a2a 到 home + moonep K=1 再均衡；后续再决定 |
| `fused` | 旁路 | — | — | ✓ | ✓ | 无 | MegaMoE / UniEP |

**MoonEP 在新设计里的三种接法**（回答"到时候是调用它的 CuTe DSL，还是 DeepEP v2 + MoonEP 的 balancer"）：

| 接法 | 谁搬 token | 谁均衡 / 搬权重 | 用到 MoonEP 的什么 | 适用 | 判断 |
|---|---|---|---|---|---|
| **A · `moonep` 整包**（dispatcher + balancer 一起选，配置校验强制成对） | MoonEP `Buffer.dispatch/combine`（CuTe DSL push/pull，VMM 对称 VA） | MoonEP planning（融合在 dispatch 内）+ `prefetch_weight` + `reduce_grad` | **只用公开 Python API**（`Buffer.__init__/dispatch/combine/prefetch_weight/reduce_grad` + `create_nvl_dist_tensor` 分配器）；kernel 由它内部 JIT，XTuner 不写、不 fork CuTe DSL | 单 NVLink 域（HGX 8 卡、NVL72） | **域内首选**——静态 `S·K`、zero-copy、无 host sync 三个收益只有整包才拿全（MoonEP 自己的 benchmark 就是 vs v2 的对照） |
| **B · `deepep_v2` 搬 token + `moonep_plan` 均衡** | DeepEP v2 expand/psum（可 FP8） | MoonEP planning（`launch_planning` 或 `planning_reference` 纯 torch）→ `dst` 翻译成 `ep_size×(E_local+B)` 物理 id；权重侧仍用 MoonEP 的 `prefetch_weight/reduce_grad`（它们只依赖 `[E+B]` 对称 VA，不依赖 hidden_buf） | planning + prefetch + reduce_grad 三个 kernel；不用 dispatch/combine | 想要"FP8 dispatch + 精确 `S·K` 均衡"的域内场景；或作为 planning 算法的独立测试床 | 域内 bf16 dispatch 在 NVLink 上不是瓶颈，B 比 A 多一次 v2 epilogue 拷贝、少 zero-copy；**低优先级**，按需 |
| **C · `hier`：`deepep_v2` 域间 + `moonep` 域内** | 域间 v2（R 模式到同轨 GPU 停住），域内 MoonEP dispatch/combine | MoonEP planning 在池内 | 整包（域内） | 多节点 EP 组且需要域内精确均衡 + 静态 shape（W2-b） | 6.2b 变体先测；是否立项看 W0 profiler 的节点级残余 |

三种接法对 XTuner 侧是**同一份代码**：`dispatcher/moonep.py` 实现六段式（preprocess = pad 到 S + `bincount` + int32；dispatch = `Buffer.dispatch(zero_copy=True)`；postprocess = 从 `cu_seqlens`/`zero_fill_ranges` 派生 `ExpertBatch(EA, A=128, P=E+B)`，零拷贝；combine_preprocess = identity；combine = `Buffer.combine(plan, out)`；反向 = 基类转置 Function 用 `plan`），`balancer/moonep_plan.py` 实现 `Balancer`（`materialize` = `prefetch_weight`，`reduce_grads` = `inter_rank_sync` + `reduce_grad`，`restore_for_backward` = 再 `prefetch_weight`），`ExpertWeights(storage="symmetric_vmm")` 由可替换 allocator 产出 `[E+B]` 权重与 `[E,E+B)` 行 alias 到 reduce buffer 的 fp32 梯度。同事分支里 `_SharedBProjectionSpace`（B 池全进程共享、E 权重按层）的组织方式可以直接搬进 `ExpertWeights`，其余（fork 的 grad-reduce kernel、两次 GMM+`cat`、`if is_moonep`、`.item()`）全部不要。要向上游提两个小需求：把 `launch_planning` 提升为公开的 `plan()`（B/C 接法需要），以及给 `reduce_grad`/`prefetch` 加可选入口屏障。

**GEMM 消费侧**：

| GEMM backend | 吃的 layout | 需要的字段 | 精度 / SF | 架构 | 适配器 |
|---|---|---|---|---|---|
| Triton TMA varlen（现默认） | E0 / **EA**（传 `aligned_counts()`） | counts | bf16 | Hopper | 无 |
| AdaptiveGEMM（现 FP8） | E0 / EA | counts + `FP32_ROW_1x128` | FP8 | SM90 | 无（dispatcher 按 spec 产 row-major SF）；EA A=128 使 wgrad 的 `expand_128x` **消失** |
| CUTLASS（fanshiqing） | E0 | counts（host） | bf16 | SM80 | `to_layout(E0)` + `.cpu()`；仅退路 |
| **DeepGEMM** | **EA**（A=BLOCK_M） | `psum()` 或 `m_indices()`；SF `FP32_MN_TMA`（SM90）/ `UE8M0X4`（SM100） | bf16 / FP8 / FP8×FP4 | SM90/SM100 | 无（零拷贝）；**SM90 wgrad 的 k-grouped 不支持 psum、要 `ks_cpu`**（`gemm.hpp:598-599`）→ H200 上 wgrad 继续用 Triton/AdaptiveGEMM 的 k-grouped（GPU counts），SM100 才用 DeepGEMM psum wgrad |
| **SonicMoE** | **R**（`gather_idx`）或 E0（idx=identity） | `expert_frequency_offset`、`x_gather_idx`（从 `seg_start/seg_count/gather_idx` 直接拼，跳过其三个 Triton 排序 kernel） | bf16 | SM90/100/120 | K=1 快捷路径要加；EA 的 gap 无法表达 → 走 R/compact 视图 |

**矩阵里没有任何一对需要专用胶水。** 三种适配器的成本：派生 `psum/m_indices/aligned_counts` 是 O(P_local) 或 O(N) 的 int32 kernel（几 µs）；SF 转置只在"dispatcher 无法按需产出"时发生（v2 两种都能产，实际不发生）；compact 是唯一 O(N·H) 的，只在退路出现。

### 2.10 与 W1 解耦、FSDP、权重存储的关系

三种权重存储形态与并行 mesh 的关系，是这套设计里**最容易被低估的耦合点**：

| 形态 | 谁需要 | 与 FSDP 的关系 |
|---|---|---|
| `dtensor_fsdp`：`Shard(ep) + FSDP(efsdp)` | 现状 / W1 解耦后默认 | FSDP2 在 forward 前 all-gather 出**临时** unsharded 参数，`data_ptr` 每步变；`reshard_after_forward` 后释放 |
| `dtensor_ep_only`：`Shard(ep)`，`efsdp=1` | **UltraEP**（`construct_local_master_ptr_pool` 按层注册 `data_ptr`，要求稳定，`manager.py:359-428`） | 与 FSDP 无冲突（专家不做 FSDP）；fp32 master / optimizer 仍走 XTuner 现有路径 |
| `symmetric_vmm` / `symmetric_nvshmem` / `ipc`：跨 rank 可寻址的连续分配 | **MoonEP**（`[E+B,H,H']` 对称 VA，"Contiguity is a hard requirement"，README:45-51；grad `[E,E+B)` 行 alias 到 reduce buffer）、UltraEP 副本槽 | FSDP flat-param 存储不可导出；需要**可替换的专家参数 allocator**（`cuMemCreate` pinned device + fd 导出，`nvl_shared_buffer.cuh:108-148`），并让 optimizer/grad-reduce 只看 `[0,E)` 中本 rank 那段 |

由此的建议：**大 EP + 均衡器的默认配置是 `efsdp=1`（专家只 EP、不 FSDP；dense 走 `dp_shard` 全量 FSDP/HSDP）**。理由：① 两个均衡器都要求 master 权重指针稳定 / 可导出；② EP ≥ 32 时每 rank 专家分片已经很小，FSDP 再切收益有限；③ W1 解耦恰好把 `efsdp` 做成了独立维度，`efsdp=1` 是它的一个合法取值，dense 侧显存收益不受影响。需要 `efsdp>1` 省显存时（Meta-MoE 单节点 B=320 那种场景），Balancer 的指针注册必须挂在 FSDP unshard 之后（`fsdp_pre_all_gather` 钩子）——标为风险项，首期不做。

MetaMoE 的"多层共享一个专家池"在这套接口里的落点：`ExpertWeights` 由模型级 `meta_experts[block]` 提供、被多层的 `GroupedExperts` 共享；`wgrad(acc=)` 承接 dW 原位累加；Balancer 的副本槽跨层共享本来就是 UltraEP / MoonEP 的设计（`support_multilayer` commit 做的正是这件事）。

### 2.11 PR 拆分与验收

| # | 改造 | 触及 | 验收 |
|---|---|---|---|
| **P0** | W1 解耦上游化：`meta_moe.py` → `moe.py` 基类；HSDP / DCP / HF / fp8 reduce mesh；与 #2007 合并 root mesh 构建；`all2all_dedup`、dW acc 一并上游 | `model/moe/moe.py`、`config/fsdp.py`、`float8_handler.py`、`dispatcher/` | proposal L0–L4；旧路径 bit-exact；EP8 dense 显存 ≈1/8 断言 |
| **P1** | `ExpertBatch` / `Routing` / `DispatcherCaps` / `GroupGemmBackend` 契约；现有三 dispatcher × 三 GEMM 迁到契约；permute kernel 加 `alignment` + 导出 `sorted_row_id`；删重复的 `DeepEPDispatch`、死代码；去全局单例 | `dispatcher/base.py`、`ops/moe/protocol.py`、`moe_group_linear.py`、`permute.cu` | E0 下 bit-exact 回归；同一 `ExpertBatch` 三 GEMM 对拍 |
| **P2** | `deepep_v2` backend（ElasticBuffer；expand/A=128/zero_pad/按 spec 产 SF；cached handle 反向；`deepep`→`deepep_v1`） | 新 `ops/comm/deepep_v2_op.py`、`dispatcher/deepep_v2.py` | 与 v1 loss 容差对齐；nsys 无 `cudaStreamSynchronize`（`do_cpu_sync=False` 模式） |
| **P3** | probs 乘法迁入 act 核 + dS 改写 + fp8 重量化融合；`unpermute` 去掉 probs | `moe_decoder_layer.py`、act kernel（triton/CuTeDSL） | 逐元素对拍（bf16 容差）+ loss 级 A/B |
| **P4** | DeepGEMM 后端（psum/m_indices；SM90 SF MN-major；wgrad 策略按架构） | `ops/moe/cuda/deep_gemm.py` | 三 GEMM 同一 EA 输入可切换 + 单测（v0.1 §7a 验收） |
| **P5** | FP8 dispatch（预量化 + `(fp8, sf)` 贯通 + `sf_format` 协商） | P2 + `float8/` | W7 一致性基线 |
| **P6** | `Balancer` 接口 + UltraEP 适配（`solve_placement_for_test` + `reroute_sparse` pybind + 确定性 reroute + `E_local+R` 两段 GEMM + 三个反向钩子 + `efsdp=1` 存储） | 新 `module/moe/balancer/` | `tests/test_solving.py` 式模拟 + 8 卡 demo ≥92% ideal；`MAX_EXPERTS_PER_NVL` 重编 |
| **P7** | `MoEScheduler`：把 `_micro_batch_forward` 抽成 plan 驱动；`shared_expert_cover`、`ep_dr`、整层 CUDA graph（依赖 P2 static + P6 hint） | `moe_decoder_layer.py` | 与现状两段式吞吐持平或更好；graph 模式下 step 时间平坦 |
| **P8** | DeepEP v2 `max_recv_rows_hint` allocation patch（上游 PR 或本地 patch，溢出置 flag 兜底） | DeepEP `buffer.hpp:1065-1071` | EP8 单层 `recv_x` 3.8 GB → ~0.5 GB |
| **P9** | SonicMoE 后端（R/E0 直吃，K=1 快捷路径；激活显存评估） | `ops/moe/cuda/sonic.py` | `a[N,I]` 不存；recompute 可关 |
| **P10** | `moonep` backend（公开 API：`dispatch/combine/prefetch_weight/reduce_grad`，不 fork kernel）+ `[E+B]` 对称 VA allocator + `hier` 拼接 | W1/P1 之后 | 共享梯度验收；替换 `sh/a3-stable-moonep` 分支 |

依赖：P1 → P2 → (P3, P4, P5) 并行 → P6/P8 → P7；P0 独立且最先；P9/P10 在 P1 之后随时可开。

### 2.12 算子组的 Grouped GEMM 工作清单与"统一 GroupGEMM"

**现状**：`ops/moe/__init__.py` 在 import 时按 device 一次性绑定 `group_gemm`（cuda 下由环境变量 `XTUNER_USE_CUTLASS_GROUP_GEMM` 在 Triton / CUTLASS 之间二选一，`:17-38`），FP8 走另一条 `TileWiseFloat8GroupedLinear`（`moe_group_linear.py:52-70`）——没有按 layout / dtype / shape / op 选择的能力，也没有性能表。

**"统一 GroupGEMM + 自动选后端"是对的，但要满足四个约束才不会返工**：

1. **选择发生在 setup（每层一次），不在每次调用。** 静态 shape 模式下真实 M 只在 GPU 上（`num_valid_rows`），任何"看 M 选 kernel"的逻辑都会引入 host sync；选择只能依据 `GemmSpec`（arch / op / dtype / sf_format / layout / alignment / N / K / `expected_m`），每次调用只做 assert。
2. **先可行性过滤，再查性能表。** 过滤按 `GroupedGemmBackend` 的 caps（accepts layout、dtypes、sf_formats、required_alignment、needs_host_counts、supports_wgrad_acc）；性能表 key = `(arch, op, dtype, layout, N, K, bucket(expected_m))`，无记录时**离线** autotune 写表（`XTUNER_GEMM_AUTOTUNE=offline`），线上只查表，避免首步抖动与非确定性。
3. **三个 op 可以选不同后端。** H200 上典型组合：fwd = DeepGEMM psum（FP8）/ Triton varlen（bf16）；wgrad = Triton / AdaptiveGEMM k-grouped（GPU counts；DeepGEMM SM90 不支持 psum wgrad）；dgrad = DeepGEMM `nn`（SM90 FP8 需缓存转置权重）或 Triton。
4. **SF 格式由 selector 输出、上游按需产出。** selector 把选中后端的 `sf_formats[0]` 写进 `MoESpec`，dispatcher（v2 `use_tma_aligned_col_major_sf`）与 act 核按此产出，不做事后转置。

不要把 permute / act 融合塞进"统一 GroupGEMM"：act 核（G1）是独立算子，GEMM 库是可换的；SonicMoE 那种 gather 融合通过 `accepts=(Layout.R,)` 的后端声明进入同一 registry 即可。

```python
@dataclass(frozen=True)
class GemmSpec:
    arch: str                                   # "sm90" | "sm100"
    op: Literal["fwd", "dgrad", "wgrad"]
    dtype: torch.dtype; sf_format: SFFormat
    layout: Layout; alignment: int
    n: int; k: int; expected_m: int             # expected_m 来自 S·K/E_local 或 profiler，不是运行期 M
    needs_acc: bool; deterministic: bool


class GroupedGemmRegistry:
    def __init__(self): self.backends: dict[str, GroupedGemmBackend] = {}; self.table = load_autotune_table()
    def register(self, b: GroupedGemmBackend): self.backends[b.name] = b
    def select(self, spec: GemmSpec) -> GroupedGemmBackend:
        cands = [b for b in self.backends.values() if b.supports(spec)]          # caps 过滤
        if not cands: raise ConfigError(f"no grouped-gemm backend for {spec}")   # 配置期报错，不是运行期 NotImplementedError
        key = (spec.arch, spec.op, spec.dtype, spec.layout, spec.n, spec.k, bucket(spec.expected_m))
        name = self.table.get(key)
        if name is None:                                                          # 离线 autotune；线上禁止
            name = autotune(cands, spec); self.table[key] = name; save_autotune_table(self.table)
        return self.backends[name]


class GroupedLinear(torch.nn.Module):
    def setup(self, registry, spec_fwd, spec_dgrad, spec_wgrad):
        self.fwd_be, self.dgrad_be, self.wgrad_be = map(registry.select, (spec_fwd, spec_dgrad, spec_wgrad))
        self.sf_format = self.fwd_be.sf_formats[0]                                 # 协商给 dispatcher / act 核
```

**算子组工作清单（按优先级）**：

| # | 工作 | 对应 PR | 依赖 | 备注 |
|---|---|---|---|---|
| G1 | **act 融合核**：SwiGLU × probs × FP8 重量化（fwd）；dact 同一 epilogue 出 dH 与 d_probs（SonicMoE `dS = <dA', A>`）；bf16 / fp8 两版；按 `num_valid_rows` 裁剪；Triton 起步，热点再 CuTeDSL | P3 | P1 | 三家库都把乘法留给框架，这一个核让所有 combine 变纯求和 |
| G2 | **DeepGEMM 后端**：psum / `m_indices` 两种输入；SM90 SF MN-major；dgrad `nn`（FP8 权重转置缓存）；wgrad 策略按架构；版本钉死 | P4 | P1 | 与 v2 expand 零拷贝对接 |
| G3 | **统一 GroupGEMM registry + selector + 离线 autotune 表**；Triton / AdaptiveGEMM / DeepGEMM 三后端接入；CUTLASS 降为退路 | P4 | P1 | 上面的代码骨架 |
| G4 | **permute/unpermute**：`alignment` 参数 + `sorted_row_id` 导出（fanshiqing `permute.cu`）；`unpermute` 去 probs（纯求和） | P1 | — | 长期目标：dispatcher 写终位后不再 permute |
| G5 | **wgrad `acc` 原位累加**（fp32 / bf16 可选）+ 跳空专家（上游 MetaMoE `k_grouped_gemm_acc`）；EA 下删除 `trans_per_*_quant_expand_128x` 的 pad | P0 / P1 | — | 共享专家池 +22% 的来源 |
| G6 | **FP8 量化核统一**：`per_token_cast_to_fp8` 产出协商好的 SF 格式（row-major fp32 / MN-major TMA / UE8M0×4）；转置 + 量化融合保留 | P5 | P2 | 通信字节减半的前提 |
| G7 | **SonicMoE 评估与后端**：H200 bf16 吞吐 + 激活显存（免 recompute）；K=1 快捷路径；EP 接入 `moe_general_routing_inputs` | P9 | P1 | Blackwell 数字不能平移到 H200 |
| G8 | **对拍与 benchmark harness**：同一 `ExpertBatch` → 全部后端，正确性 + 性能矩阵（arch × dtype × layout × shape），进 CI；也是 G3 性能表的生产工具 | W0 | P1 | 验收口径：全链路零 `.cpu()/.tolist()` |
| G9 | **workload-aware 调度**（K3）：rank 内 per-expert 偏斜下的 tile 调度 / block size 选择，由 `expected_m` 分布驱动 autotune | 后续 | G3 | 完美均衡只保证 rank 级均衡 |
| G10 | 后续：masked GEMM（rollout / decode）、FP4（SM100）、MegaKernel 构件（persistent tile scheduler、epilogue 融合） | 后续 | — | W4 的算子侧地基 |

---

## 3. 四条链路逐段梳理：permute / unpermute / FP8 / layout / GroupGEMM

### 3.0 统一坐标系

把一层 MoE 拆成九段，四条链路在同一坐标系下对照（§3.5 总表）：

```
① router 输出 → ② pre-dispatch（排序 / 量化 / 计数）→ ③ dispatch 传输（dedup？）→ ④ 到达布局
→ ⑤ post-dispatch（permute？counts 从哪来？host sync？）→ ⑥ GEMM-1 → act(+probs?+quant?) → GEMM-2
→ ⑦ pre-combine（unpermute？部分和？）→ ⑧ combine（乘权重？精度？）→ ⑨ post-combine → 反向
```

### 3.1 XTuner 现状：`all2all` 与 `deepep`（v1）

**all2all**（`dispatcher/torch_all2all.py`）：
① `topk_ids` i64 / `topk_weights` f32 → ② `permute(hidden, topk_ids.int32)` 按 global expert id 排序得 `[T·K, H]`（因此按目的 rank 连续）+ `row_id_map`（`:328`）；`histc` → `all_to_all_single` 交换计数 → `input/output_splits` **`.to("cpu").tolist()`**（`:100-103`）→ ③ `all_to_all_single_autograd`，双向各 `K·T·H` 字节（无 dedup；`all2all_dedup` 变体每 (token, rank) 只发一份，−34%）→ ④ 按 (源 rank, 本地专家) 排列 → ⑤ `repeat_interleave(expert_ids_per_ep_rank, token_counts)` 得每行本地专家 id → **第二次 `permute`** → E0，`tokens_per_expert = tokens_per_expert_group.sum(0)`（GPU）（`:416-461`）→ ⑥ Triton varlen（bf16）或 AdaptiveGEMM（fp8：**在接收侧 `per_tile_quant`**，`float8_gmm_tile_wise.py:86-110`）；`moe_act` 独立 kernel，不乘 probs → ⑦ `unpermute(experts_out, row_id_map)` **不带 probs** 回到 (源 rank, 专家) 序（`:474-477`）→ ⑧ 反向 a2a（splits 互换），纯搬运 → ⑨ 源侧 `unpermute(..., probs=topk_weights)` 做 top-k 加权求和 `[T, H]`（`:566-570`）。反向：`all_to_all_single_autograd` 自带；异步时 `_AsyncDispatch/_AsyncCombine` 在 comm stream 上反向 a2a。

**deepep（v1 normal）**（`dispatcher/deepep.py`、`ops/comm/deepep_op.py`）：
① 同上，`topk_ids.to(int64)` → ② 无 permute；`get_dispatch_layout` 算 `num_tokens_per_rank/rdma_rank/expert`、`is_token_in_rank`（`deepep_op.py:179`）→ ③ `Buffer.dispatch(x, topk_idx, topk_weights, ..., async_finish=True, allocate_on_comm_stream=True)`，按 rank dedup（同 token 多专家落同 rank 只发一份）→ ④ **R**：`recv_x[N, H]` 按源 rank 分段 + `recv_topk_idx[N, K]`（非本地 -1）+ `recv_topk_weights[N, K]` + `num_recv_tokens_per_expert_list`（**host list**，v1 内部 host 自旋等 GPU 计数）+ `handle` → ⑤ `permute(recv_x, recv_topk_idx.int(), num_out_tokens=sum(list), num_negative_one_in_indices)` → E0（`deepep.py:367`）；`tokens_per_expert = torch.tensor(list, pin_memory=True).to(device, non_blocking=True)`（`:385-395`）→ ⑥ 同上 → ⑦ `unpermute(experts_out, row_ids_map, probs=recv_topk_weights)` **在专家侧**做本 rank 命中的部分加权和 `[N, H]`（`:417-421`）→ ⑧ `Buffer.combine(x, handle, topk_weights=None)` 跨 rank 求和（`deepep_op.py:264-270`）→ ⑨ `view_as` + 等 event。反向：`DeepEPDispatch.backward = combine(grad, handle, topk_weights=grad_recv_topk_weights)`；`DeepEPCombine.backward = dispatch(grad, handle=handle)`（`:226-297`），`handle` 三次复用。

共同点：**FP8 通信未打通**（v1 normal 其实支持预量化 `[T, H/128]` fp32 SF，但 `fwd_comm_dtype_fp8` 抛 `NotImplementedError`）；整层 REENTRANT checkpoint 让 ③⑧ 各执行 3 次；permute 由框架做，每次多一次 `[N, H]` 读写。

### 3.2 DeepEP v2（`ElasticBuffer`，`deep_ep/buffers/elastic.py`）

#### 3.2.1 它是什么

- **一个类、一对 kernel、一个后端。** `ElasticBuffer` 统一 v1 的 normal 与 low-latency 两套接口（v1 整体搬到 `legacy`，两套同时导出，`deep_ep/__init__.py:88-89`）；README:9-31 明说 v2 **不再支持 0-SM RDMA low-latency**、buffer 比 v1 大。同一块 symmetric window 还承载 Engram / PP / AGRS（实验性）。
- **后端 = NCCL Gin（GPU-Initiated Networking）设备 API**：`ncclDevCommCreate`（`csrc/kernels/backend/nccl.cu:108`，NCCL ≥ 2.31 / PyTorch ≥ 2.10）、`ncclCommWindowRegister`、`ncclGetLsaDevicePointer`；**直接复用 PyTorch 的 NCCL communicator**（`deep_ep/utils/comm.py:61-64`）。设备侧：目标在 NVLink（LSA team）内 → `ncclGetLsaPointer` + TMA store；否则 `gin.put`（RDMA）。NVSHMEM 只为 legacy 保留。
- **LSA 与 GIN 的关系**（常被问到）：两者都是 NCCL 2.28+ 设备端 API（`nccl_device.h`）的组成部分，**不是从属关系**。LSA（Load/Store Accessible）= NVLink/P2P 可直接访存的 rank 集合（`ncclTeamLsa`）+ 对称窗口的裸指针翻译（`ncclGetLsaPointer`），之后就是普通 ld/st/TMA 过 NVLink；GIN（GPU-Initiated Networking）= 设备侧发起的 RDMA 动词（`ncclGin.put/signal`），走 NIC。DeepEP v2 的设备句柄虽然叫 `NCCLGin`，里面同时持有 `gin` 与三个 team（`team_world / team_lsa / team_rail`，`common/handle.cuh:21-22`）和 `lsa_base_ptr`（`:34`）；每个访问点先 `is_nvlink_accessible` → 取 LSA 指针，拿不到才 `gin.put`（`:64-92`）；`handle.cuh:181` 的注释明说 `gin.put` 即使目标是 NVLink 邻居也照走 NIC。一次 `ncclCommWindowRegister`（`nccl.cu:140`）让同一块内存同时被 LSA 域内 peer map 进各自 VA、又注册进 NIC 的 MR——"一块内存、一个 offset 空间、两条硬件通路"。所以单机 EP8 的流量**全走 LSA，GIN 数据面零参与**（无 NIC 机器 `EP_DISABLE_GIN=1`）；跨机 hybrid 才是 Rail team 上的 GIN put + LSA 转发。
- **direct vs hybrid**：`allow_hybrid_mode=True` 且多机 → hybrid（scaleout = RDMA rank 按 rail，`NCCL_GIN_CONNECTION_RAIL`；跨机沿 rail 发，落地后 NVLink 转发，`impls/hybrid_dispatch.cuh`）；否则 direct（team World 全连接）。单机纯 NVLink 时全走 TMA；单机无 NIC 需 `EP_DISABLE_GIN=1`（`nccl.cu:86-90`）。**注意与 HybridEP（Megatron flex backend、DeepEP v1 分支、未合入 main）同名不同物。**
- **两段式 kernel + PDL**：主通信 kernel 只用 `num_sms` 个 SM 写 symmetric buffer，`cudaTriggerProgrammaticLaunchCompletion()`（`dispatch.cuh:403`）后由 PDL 启动的 **copy epilogue**（`dispatch_copy_epilogue_impl`，全部 SM）把 token 从 buffer 搬到最终张量并完成布局；combine 同构。这就是它"zero-copy"与"SM 少"的来源。
- **SM 数解析式**：`get_theoretical_num_sms(num_experts, num_topk)`（`elastic.py:728-834`），按 NVLink/RDMA 带宽与每 SM 读写带宽反推，`≥4、偶数`；README：V3 式训练 24 → 4–6 SM，EP8×2 12 SM，NVLink EP8 24–64 SM。注释明说假设均衡 gate。
- **硬上限**：`kNumMaxRanks=1024, kNumMaxExperts=2048, kNumMaxExpertsPerRank=256`（`common/layout.cuh:19-21`；README 的 "EP2048" 与代码 1024 不一致）；`num_topk ≤ 32`；combine 要求 **`hidden % 256 == 0`**（`combine.cuh:109`）。**Meta-MoE E=2560 / EP8 E_local=320 同时超过两个上限**，需改常数验证 [推断：是否只是常数]。

#### 3.2.2 `use_psum_layout` 到底是什么

先纠正一个名字：**DeepEP 里没有 `use_psum_layout` 这个参数**（全仓只命中 `psum_num_recv_tokens_*`）。`use_psum_layout` 是 **DeepGEMM** 的 GEMM 参数（`DeepGEMM/csrc/apis/gemm.hpp:175,467,679,741`）。DeepEP 侧的开关是 `dispatch(do_expand=True, expert_alignment=A, do_zero_padding=..., use_tma_aligned_col_major_sf=...)`，它产出的 `handle.psum_num_recv_tokens_per_expert` 就是 DeepGEMM 的 `grouped_layout`。两边的定义逐字对应：

**expand 模式的内存排列**（`do_expand=True`）：`recv_x` 形状 `[num_expanded_tokens, H]`，**一行 = 一个 (token, 本地 expert) 对**，按本地 expert `e ∈ [0, E_local)` 分段，段间按 `A` 补洞：

```
start_0     = 0
start_{e+1} = align_up(start_e + n_e, A)          n_e = 真实计数（不含 pad）
psum[e]     = start_e + n_e                       ← handle.psum_num_recv_tokens_per_expert[e]："真实结束行"
num_unaligned_recv_tokens_per_expert[e] = n_e
```

来源：notify warps 统计每 expert 计数并对齐（`dispatch.cuh:205-215`），写 exclusive prefix sum（`:254-257`）；copy epilogue 对每个落在本 rank 的 top-k 槽 `dst = atomicAdd(psum + e, 1)`（`dispatch_copy_epilogue.cuh:120-121`）——跑完后 psum 自然变成"起点 + 真实数"。段内行序 = atomic 到达序（非确定；`deterministic=True` 时 Python 侧按源 token 全局号稳定排序，`elastic.py:100-192`）。洞行默认**未初始化**，`do_zero_padding=True` 时 epilogue 尾部把洞行的 `recv_x / recv_sf / recv_topk_weights` 置零（`:231-322`）。

**DeepGEMM 侧的契约**（`deep_gemm/include/deep_gemm/scheduler/gemm.cuh:100-103,217-237,318-319`）：`grouped_layout[g]` = 累计真实结束行，组 g 起点 = `align(grouped_layout[g-1], BLOCK_M)`，块内有效行 `< psum[g]`。因此 **`expert_alignment` 必须 == DeepGEMM 的 `get_mk_alignment_for_contiguous_layout()`**：SM90 固定 128；SM100 `get_theoretical_mk_alignment_for_contiguous_layout(expected_m)` 从 224 按 32 递减、最小 32（`heuristics/runtime.hpp:10,47-57`）。`ensure_zero_padding=True`（默认）整块计算并写零 → 要求洞行 A 为确定值（配 `do_zero_padding=True`）；`False` 只算 `align(valid,16)` 行。FP8 时 SF 打包 kernel 按 psum 跳过洞行（`impls/smxx_layout.hpp:188-196`）。

**一个例子**（E_local=3，A=128，n=[200, 0, 130]）：

```
expert 0: rows [0, 200)     psum[0]=200    洞 [200,256)
expert 1: rows [256, 256)   psum[1]=256    空组占 0 行
expert 2: rows [256, 386)   psum[2]=386    洞 [386,512)
num_expanded_tokens（无 CPU 同步时是上界）≥ 512
等价 m_indices = [0]*200 + [-1]*56 + [2]*130 + [-1]*126   ← DeepGEMM 另一种输入，O(M)
```

**为什么要它**（相对 `m_indices[M]` 与 `masked[G, M_max]`）：① 元数据只有 G 个 int32，不随 M 增长，GPU 驻留，天然无 host sync、CUDA-graph 友好；② epilogue 用 `atomicAdd` 就能定行号，不需要排序；③ 与 masked 3D 相比不浪费 `M_max` 预留、GEMM 不用逐组 `ceil(masked_m/BLOCK_M)`；④ `psum[-1]` 就是真实有效行数（GPU 标量），直接喂 act/量化核（`swiglu_apply_weight_to_fp8(..., avail_tokens=psum[-1])`，`DeepGEMM/tests/test_mega_moe.py:296-302`）与 GEMM heuristics（`expected_m_for_psum_layout`）。

**一个坑**：`do_expand=False` 时 `psum_num_recv_tokens_per_expert` 语义**不同**——是对齐计数的 inclusive psum（`buffer.hpp:1118-1121`），仅供框架自己 permute 用；不能直接喂 DeepGEMM。

在 `ExpertBatch` 里：`seg_start = align_up(cat([0], psum[:-1]), A)`、`seg_count = num_unaligned`、`psum() == handle.psum_num_recv_tokens_per_expert`——**零拷贝、零派生**。

#### 3.2.3 FP8 / SF 布局

v2 **不融合量化**（v1 LL 才在 kernel 内量化；v2 砍掉了）：`x = (fp8_e4m3[T, H], sf[T, num_sf_packs])`，sf 元素必须 4 字节（`buffer.hpp:767`）——fp32（SM90，recipe (1,1,128)，`[T, H/128]`）或 UE8M0×4 packed int32（SM100，(1,1,32)，`[T, ceil(H/128)]`），输入 stride 任意；kernel 只搬字节。`use_tma_aligned_col_major_sf=True` 时输出 SF 为 MN-major、行 stride `align(N, 4)`（`buffer.hpp:1090-1099`），恰是 DeepGEMM SFA 的 TMA 对齐要求（`DeepGEMM/csrc/utils/layout.hpp:113-117`）→ 零拷贝；`False` 则 row-major，恰是 AdaptiveGEMM 要的。**combine 只收 bf16**（`buffer.hpp:1203`），无 FP8 combine、无 FP4。量化位置因此从"接收侧各做一遍"提前到"发送前做一次"，通信字节减半。

#### 3.2.4 combine 与反向

- `combine(x_bf16[N, H], handle, topk_weights=None, bias=None)` → `(combined_x[T, H] bf16, combined_topk_weights[T, K] | None)`。**不乘 topk 权重**：`topk_weights` 只是被原样运回（`combine.cuh:215-225`；`test_ep.py:507` 断言相等）。expand 模式两级归约：`allow_multiple_reduction=True`（默认）发送侧先把同 token 的多个本地 expert 行求和（`combine.cuh:144-176`），接收端 epilogue 再对 `min(R, K)` 或 `K` 个槽求和（≤2 路无 bias 用 bf16 hadd，否则 fp32，`combine_utils.cuh:67-73,111-168`）；`False` 时每个展开行单独发、只在接收端 fp32 归约一次（精度最好，MegaMoE 测试用它），**但此时不能传 `topk_weights`**（`combine.cuh:67-69`）→ 训练（需要 `grad_topk_weights`）必须 `True`。
- 反向（README:207-252 范式，`elastic.py:937-948`）：dispatch-bwd = `combine(grad_recv_x, handle, topk_weights=grad_recv_topk_weights[N])` → `(grad_x, grad_topk_weights[T,K])`；combine-bwd = `dispatch(grad_combined_x, handle=handle)`：cached 模式，无 notify warps、无 CPU 同步，复用 `dst_buffer_slot_idx`，expand 时按 `recv_src_metadata[:, 2:]` 放行，梯度直接落在与前向相同的 psum 布局里；cached dispatch 也接受 `(fp8, sf)`。`handle.recv_src_metadata[N, 2+K]` 就是 unpermute 映射（列 2.. = 该 top-k 槽在 `recv_x` 的行号）。
- wgrad：SM100 用 DeepGEMM `k_grouped_*_tn_contiguous(..., use_psum_layout=True)`（`ks_cpu` 可省）；SM90 不支持 psum、要 `ks_cpu` → H200 上继续用 Triton / AdaptiveGEMM 的 k-grouped（GPU counts），且 EA A=128 使 AdaptiveGEMM 的 `expand_128x` pad 不再需要。

#### 3.2.5 buffer 与"固定 shape"的真实含义

- `do_cpu_sync=True`（默认）：notify warps 把计数写进 mapped pinned host workspace，host 自旋（`buffer.hpp:1017-1064`）→ 精确 `num_recv_tokens`，**有 host 同步、不可 graph**。
- `do_cpu_sync=False`（cached handle 时强制）：**按最坏情况分配**（`:1065-1071`）：`num_recv_tokens = num_max_tokens_per_rank × R`，`num_expanded_tokens = align(R × S_max × min(K, E_local) + (A−1)·E_local, A)`；真实计数只在 GPU 的 `psum[-1]`。EP8、S=4096、K=8、H=7168 bf16：`recv_x` = 262144 × 7168 × 2 B ≈ **3.8 GB / 层**，GEMM-1 输出再 `[N, 2I]`；EP64 ≈ 30 GB。这就是 PR #605 "CUDA graph compatible, **or** faster CPU sync in saving memory mode" 那个 "or"。
- 通信 buffer 本身（`get_buffer_size_hint`）：hybrid EP64=8×8、S=4096、H=7168、K=8、FP8 dispatch 约 **4.8 GB/GPU**（`buffer.hpp:586-650` 公式估算），随 `R × S_max × H` 线性。
- CUDA graph：**官方口径只在 PR #605 的描述里**（"CUDA graph compatible, or faster CPU sync in saving memory mode"）；本仓库 README 与 docs 对 v2 一字未提（`docs/legacy.md:263` 只说 v1 LL 兼容），也没有 graph 测试（grep `cudaGraph` 为空）。从代码看，可捕获需要三个条件同时成立：① `do_cpu_sync=False`（唯一 host 阻塞点，`buffer.hpp:1017-1064`）；② `EP_AVOID_RECORD_STREAM=1`（`buffer.hpp:566-573`；`event.py:28` 注释明说 `record_stream` 与 graph 不兼容）；③ handle / 输出按最坏尺寸预分配并复用（cached handle）。所以"兼容"= 满足这三条时**理论上**可捕获；要自己做一次捕获实验（direct 模式先）。
- **结论**：v2 的"固定 shape"是**悲观预留**（MoonEP 是**乐观构造**：完美均衡使每 rank 恒收 `S·K`，`NvS = S·K + 254·E/R` 行，同配置 ≈ 0.47 GB）。大 EP 训练想同时拿到静态 shape 与可承受的显存，需要 **均衡器给出接收上界 + DeepEP 接受一个 `max_recv_rows_hint`**（v1 normal 曾有 `num_worst_tokens`；v2 没有；HybridEP 用 `num_permuted_tokens` + 溢出置 flag）——P8 的由来。

#### 3.2.6 均衡钩子

DeepEP 内没有任何复制 / 均衡逻辑（README:35-36 路线图的 "EP replay" 未实现；`cumulative_local_expert_recv_stats` 仅监控）。可用的插入点只有一个但足够：expert→rank 映射是纯整除，`num_experts` 只是整数（`% num_ranks == 0`），所以 balancer 在 dispatch 前把 `topk_idx` 从逻辑 expert 重映射到物理槽位 id（`num_experts = R × P_local`，`P_local ≤ 256`、总数 ≤ 2048），handle 保存重映射后的 `topk_idx`，combine 不用改，psum 布局按物理 expert 分段，GEMM 直接对物理槽位权重 `[P_local, N, K]` 做。约束：同一 token 不能两次落同一物理 expert；`num_experts/hidden/num_topk/expert_alignment` 是 JIT 模板参数。

### 3.3 MoonEP（`EP/MoonEP` @ 0f385f0）

**它只做 5 类通信/规划算子，不含 GEMM、不含 autograd、不含 FP8。** `moonep/__init__.py` 只导出 `Buffer` 与 `MoonEPCommPlan`。

```python
Buffer(S, H, K, E, num_ep_ranks, num_sms=None, token_padding=128, B=None, group=None,
       comm_stream_priority=-1, enable_pdl=True, explicitly_destroy=False)                       # api.py:439-453
dispatch(hidden_sh[S,H] bf16, route_weights_sk[S,K] f32, topk_experts_sk[S,K] i32, tokens_per_expert[E] i32,
         plan=None, async_finish=False, *, inter_rank_sync=True, zero_copy=False)
  -> (hidden_nvsh[NvS,H] bf16, route_weights_nvs[NvS] f32, cu_seqlens[E+B] i32 | None, plan[, event])   # api.py:685-696
prefetch_weight(plan, async_finish=False, *, full_gate_weight, full_up_weight, full_down_weight)   # 三个 [E+B,H,H'] bf16 连续
combine(plan, hidden_nvsh, route_weights_nvs=None, async_finish=False, inter_rank_sync=True, *, zero_copy=False)
  -> (hidden_sh[S,H] bf16, route_weights_sk[S,K] f32 | None, event | None)                       # api.py:881-890
reduce_grad(plan, async_finish=False, full_gate_grad, full_up_grad, full_down_grad,
            gate_reduce_buffer, up_reduce_buffer, down_reduce_buffer)   # grad fp32 [E+B,H,H']；reduce buffer fp32 [R,B,H,H'] 全 rank 映射
MoonEPCommPlan(dst[N], experts_to_copy[R,B], zero_fill_ranges[E+B,2], remote_stats[2], N,R,E,B,NvS,K, dup_groups, dup_loffs, dup_counts)
```

九段：① `topk` int32、`route_weights` fp32、**每 rank 固定 S 个 token**（变长 micro-batch 由框架 pad，`api.py:741`）、`tokens_per_expert[E]` = 本 rank 局部 `bincount` → ② 无 permute、无量化；planning 融合在 `dispatch(plan=None)` 内：单次 cooperative launch，各 rank 把 `tpe` 推到 rank 0，rank 0 单 warp 跑 Theorem 1 的贪心填平（`planning.py:671-701`：`balance[r] = group_tokens[r] − S·K`，反复取最大 surplus 与最小 deficit 一次填满）→ 配额分配 `alloc[rank, expert]`（`:717-798`）→ 每个 dest rank 选 top-B 远端专家 `experts_to_copy[R,B]`（`:834-885`）→ `multimem.st` 多播整个 plan（`:955-960`）→ 各 rank 本地稳定排序 + 二分定 `dst[N]`（`:1048-1073`）→ ③ **push**：warp 解码 `dst = drank·NvS + loff`，`cp.async.bulk` S2G 直写远端 rank 的分组位置（`dispatch.py:406-427`）——**permute 融进 dispatch**；同 token 多专家落同 rank 只发一份，目的侧本地展开（`dispatch_epilogue.py:1-17`）；zero warp 把 padding 行写零（`dispatch.py:451-497`，注释 "the segment-padding rows DeepGEMM will read"）→ ④ **EA**：单个固定 `[NvS, H]` bf16，`NvS = S·K + (128−1)·2·(E/R)`（`api.py:278-279`），逻辑上 `E+B` 个 VM group：`[0, E)` 全局 expert id（训练时只有本 rank 的 E/R 个非空）、`[E, E+B)` 预取槽；`cu_seqlens[g]` = **含 padding 的累计结束偏移**（不是带前导 0 的 `[E+1]`），`zero_fill_ranges[g] = (pad_start, n_pad)`，真实数 = 段长 − n_pad；`[cu_seqlens[-1], NvS)` 尾部未定义 → ⑤ **无 permute、无 host sync**（`moonep/` 中 `.item()/.cpu()/synchronize` 仅在 `destroy()`）；`prefetch_weight` 把 `experts_to_copy[rank]` 指定的远端专家权重经 NVLink TMA 读到本地 `[E, E+B)` 槽（`prefetch.py:115-126`）→ ⑥ 仓库**无 GEMM**；契约是"每个投影一个 `[E+B, H, H']` 连续对称 VA + `cu_seqlens[E+B]` 按行索引选组"（README:45-51 "Contiguity is a hard requirement"）；零填充行照算；**框架须在 `[NvS]` 域自己乘 `route_weights_nvs`**（`test_combine.py:201-208` 参考值 = `hidden·K`）→ ⑦ FFN 输出**按原位写回同一 `[NvS, H]`**（zero_copy 时断言 `data_ptr` 相同，`api.py:936-946`）→ ⑧ **pull**：源 rank 按 `plan.dst` 远端读，4 个 ACC warp fp32 累加 K 行写 `[S, H]`（`combine.py:353-475`）；**不乘权重**（`combine.py:403-407` 纯 `acc += f32(x)`）；`route_weights_nvs` 原样 gather 回 `[S, K]`；重复行 prologue 归并多一次 bf16 舍入（1 ulp，`test_combine.py:246-249`）→ ⑨ 无。

反向（README 映射，**无 autograd，框架自写 Function**）：combine-bwd = `dispatch(grad_sh, plan=plan)`（跳过 planning/dedup builder，`dispatch.py:861-864`）；dispatch-bwd = `combine(plan, grad_nvsh, route_weights_nvs=d_w)`（K 路求和 + 把 `d_w[NvS]` gather 回 `[S, K]`）+ `reduce_grad`（远端 pull `[R,B]` 槽 fp32 累加 → barrier → 本地清零，`grad_reduce.py:326-389`）。**benchmark 里反向再 prefetch 一次**（槽池跨层共享被覆写，`bench_vs_deepep.py:246-250`）。

框架必须知道的三条正确性约束：`inter_rank_sync=True` 是"peer 的 combine(L) 读完我的 shard 之后别人的 dispatch(L+1) 才覆写"的**唯一屏障**（`api.py:601-602,646-647`），不是性能开关；`reduce_grad` **无入口屏障**（`grad_reduce.py:468-469`）；`prefetch` 无任何屏障，optimizer step 后到下一次预取之间要跨 rank 同步。硬限制：`B = E/R`（训练）、`R ≤ 128`（`dst` 去重的 2×i64 位集，`planning.py:1095-1111,1282`）、`E % R == 0`、`K ≤ 32`、`H, H' % 128 == 0`、`num_sms=32`。

传输：CUDA VMM 对称内存（`cuMemCreate` PINNED/DEVICE → POSIX fd 经 unix socket `SCM_RIGHTS` 交换 → `cuMemMap` 进一段连续 VA，`nvl_shared_buffer.cuh:108-241`，**强制同主机**）+ NVSwitch 多播（仅广播 plan）+ 7 个 CuTe DSL kernel（cooperative launch、warp specialization、TMA/mbarrier、`red.release.sys`、`multimem.st`、PDL）。无 NVSHMEM、无 NCCL 数据面、无 RDMA。

**跨平台评估**（回答设想第 5 条）：

| 层 | 平台中性 | CUDA-bound |
|---|---|---|
| 契约层 | `api.py` 签名与张量语义；`[NvS,H]` / `cu_seqlens` / `[E+B,H,H']` / `[R,B,H,H']` 布局；`dst` 与 dedup 编码；comm stream + event 编排 | `torch.cuda.Stream/Event`、`cuda.bindings.driver` |
| 规划层 | 算法本身（`tests/planning_reference.py:25-312` 用 `dist.all_gather` + CPU 循环给出等价实现） | 单 warp 寄存器求解、`match.any/redux/shfl`、多播广播、`meta_buf` 偏移表 |
| 数据面 | 无 | 7 个 kernel、TMA/mbarrier、proxy fence、cooperative launch、PDL |
| 内存层 | 无 | VMM + 多播、fd 交换 |

所以"MoonEP 作为跨平台库"的可行形态是：**把契约层 + 规划层抽成平台无关的 `moonep-core`（就是 §2 的 `ExpertBatch(EA, A=128, P=E+B)` + `ExpertWeights(symmetric, [E+B])` + `Balancer(prefetch/reduce_grad)`），底下四个接口按平台实现**：(a) 对称 buffer 分配器 `alloc(shape, dtype) → (full_view[R×chunk], local_view, multicast_view?)`；(b) 7 个算子的 launcher，张量契约不变；(c) 跨 rank barrier 原语（现在是 `red.release.sys` + `ld.acquire.sys` 自旋）；(d) stream/event。可退化项：多播（改 unicast 或全 rank 同构规划——UltraEP 的做法）、PDL、zero_copy、dedup builder 的 warp 原语（改排序 + 分段扫描）。

### 3.4 UltraEP（`EP/UltraEP` @ 94cab09）：不是 dispatcher

**仓库里没有 dispatch/combine。** 它只做三件事：冗余专家槽位管理、均衡 plan 求解、专家**权重/梯度**的 NVLink 复制与归约（`README.md:11`；`csrc/kernels/` 仅 `placement/reroute/weight_sync/grad_reduce/load_compute` 五个 `.cu`）。token 数据面从不经过它。

"hybrid EP" 的出处：参考集成脚本用 Megatron flex dispatcher + **HybridEP** 后端（`--moe-flex-dispatcher-backend hybridep`，`examples/train_qwen3_235b.sh:179-183`；`examples/README.md:31-40` 明说 "HybridEP, an optimized branch of DeepEP-V1. DeepEP-V2 support is under testing"）。HybridEP 被选中的直接原因：它吃 dense `routing_map[T, P] bool`，与 UltraEP `reroute` 输出零转换；DeepEP v2 只吃 `topk_idx[T, K]`。NVSHMEM 只用主机侧 API 做对称堆与 `nvshmem_ptr`（`csrc/ultra_ep.cpp:311,334,367,378`；`csrc/kernels/` 里**零 NVSHMEM 设备 API**），跨节点流量只有 planning 前的负载 all-gather（KB 级）。

九段（框架侧）：① `topk_ids[T,K] i64` + `topk_weights f32` → ② **`Balancer.plan`**：`topk_local_sum` → NVSHMEM `fcollect`（或 NCCL all-gather）得 `[R, L]` → `solve_placement`：单 CTA/NVL 域、128 线程（`placement.cu:788-1482`），阈值 τ 二分（`lo = ceil(mean × 1.0)`、`hi = max_rank_load`；先试 `τ = lo × 1.01` 快速 oracle），可行性 oracle 按 excess 降序把过载 rank 的 master 负载导出到"slack 最大且有空槽且无该专家副本"的 rank（`q ≥ min_tokens_per_replica=1024`），物化 `quota/l2p/p2l`（全 rank 逐比特一致，无广播）与 `rank_quota_prefix`（每 rank 不同，locality-aware 先填本 rank 实例）；EP64 实测 0.067–0.078 ms → **reroute**：dense `[T,L]→[T,P]`（两遍 kernel，确定性，可 interleave）或 sparse `topk_ids` **原地**改成物理 id（`reroute.cu:492-515`，`atomicAdd` 序号、非确定、**无 autograd**）；`P = (M+R)·R_ep`，rank r 拥有 `[r(M+R), (r+1)(M+R))`，前 M 个 master、后 R 个 replica slot → **`materialize`**：`weight_sync(layer, async)`（NVL 域 ≤8 走 `thread_copy_kernel` direct，>8 走 TMA 双缓冲 + chunk relay 树；副本落 `[R, fc1+fc2]` 对称堆 strided 视图；FP8 权重时连 scale 一起 4 个 shard）→ ③④⑤ **由所用 dispatcher 决定**（HybridEP `dispatch_with_permute(..., pad_multiple)` → EA；接 DeepEP v2 就是 §3.2 的 expand/psum，`num_experts = P`、`num_local_experts = M+R`）→ ⑥ grouped GEMM 对 `M+R` 个本地物理专家：master 权重是框架参数、replica 权重是 UltraEP 暴露的 buffer（**两个来源**，§2.6 两次 launch）→ ⑦⑧⑨ 由 dispatcher 决定 → 反向：`RestoreReplicaWeights`（**阻塞**重跑 `weight_sync(vid)`，因为槽跨层复用被覆写）→ expert bwd（副本 wgrad 必须落在 `local_replica_*_grad_buffer`，**fp32 强制**）→ `StartGradReduce`（async，42 SM，TMA 拉远端副本梯度累加 master 并清零远端，可 order-preserving 确定）→ 下一层 wgrad 覆盖共享槽前 `JoinGradReduce`。

解耦评估（回答设想 3.2/3.3）：

| 层 | 代码 | 依赖 |
|---|---|---|
| 负载统计 / 求解 / reroute / 权重复制 / 梯度归约 kernel | `load_compute.cu` / `placement.cu` / `reroute.cu` / `weight_sync.cu` / `grad_reduce.cu` | **纯 CUDA**，只要远端裸指针 |
| 负载 all-gather | `ultra_ep.cpp:711-722` | NVSHMEM fcollect → 可换 `torch.distributed.all_gather_into_tensor` |
| 分配 / 指针 / 屏障 | `ultra_ep.cpp`、`runtime.cpp`、`utils/nvshmem.cuh` | NVSHMEM → 可换自带 `IpcManager` 的 cudaIpc / fabric handle（`ipc_manager.cu:76-137`，kernel 一行不改） |

已有**免传输入口**（不需 `init_runtime`）：`_C.solve_placement_for_test(expert_loads, expert_loads_per_rank, num_ranks, M, R, num_nvl_ranks, ..., rank_quota_source_rank)`（`ultra_ep.hpp:427-533`）、`_C.dense_reroute_for_test(...)`（`:535-612`）；sparse reroute kernel `run_sparse_reroute_quota`（`reroute.cu:547-577`）加一个 pybind 即可导出。

**8 卡 NVL 域 + DeepEP v2 的接法**（3.2）：router → 本地直方图 → NCCL all-gather → `solve_placement_for_test(num_nvl_ranks=8)` → `reroute_sparse` 物理 id → `ElasticBuffer.dispatch(x, topk_idx=phys, num_experts=P, ...)`（契约就是 `[T,K]` int64、`-1` 空、`num_experts` 整数）→ GEMM（master | replica 两段）→ combine 不改 → 权重侧保留 `weight_sync/grad_reduce`（NVSHMEM 与 DeepEP v2 的 NCCL window 无冲突，但要付 ≥512 MB 对称堆粒度；或换 IpcManager **彻底去掉 NVSHMEM**）。会失去的：dense 路径的 interleave 与确定性（sparse 用 atomic 序号；若 v2 开 `deterministic=True`，建议重写一个 segmented-prefix 版 sparse reroute，约百行）。

**跨 IB + NVL**（3.3）：token 侧无障碍（v2 hybrid 把 token 搬到任意物理 id）。但 UltraEP **按构造只在域内复制**：求解按域分 CTA、副本不出域（`placement.cu:819,838`；`tests/test_solving.py:71-75` 断言），残余不均衡下限 = `max_domain_mean / global_mean`（`tests/utils.py:587-595`）。要跨域复制需把 `num_nvl_ranks` 设成全 EP 大小并**新写一条 RDMA 权重广播传输**（UltraEP 没有）；[推断] 以单专家 ≈37.7 MB（Qwen3-235B）计，400 Gb/s 下每份跨节点副本 ≈0.75 ms，落在每层每 micro-batch 关键路径上不可接受——所以 3.3 的正确形态是"跨节点 EP 组 + 域内实时均衡 + **低频**跨域 master 重排（EPLB 式，10²–10³ 步，优化器状态随迁）"，即 v0.1 §5.2 的三时间尺度分频。

硬限制：SM90/SM100、NVSHMEM 3.4.5 + `-rdc=true`、PyTorch ≥ 2.10；NVL 域 ≤72；`MAX_EXPERTS_PER_NVL=512`（`M × nvl_size ≤ 512`，**Meta-MoE E=2560 需改常数 + 核 shared memory**）；梯度 fp32；每专家权重/梯度须独立连续、`data_ptr` 稳定（→ §2.10 `efsdp=1`）；一进程一个 EP group；profiler 不兼容 CUDA graph（`docs/blog_v1.md:138`）；relay 模式的 host 侧 `_weight_sync_epoch` 进 kernel 参数，graph 回放存疑 [推断]（direct 模式无此问题）。

### 3.5 四库对照总表

| 维度 | XTuner all2all | XTuner deepep (v1) | **DeepEP v2 (expand)** | **MoonEP** | **UltraEP**（+任意 dispatcher） |
|---|---|---|---|---|---|
| 角色 | 搬运 | 搬运 | 搬运 | 搬运 + 均衡（合一） | **均衡器**（不搬 token） |
| ① 路由输入 | `[T,K]` i64/f32 | `[T,K]` i64/f32 | `[T,K]` i64（可编译成 i32）/f32，-1 屏蔽 | `[S,K]` **i32**/f32，每 rank 固定 S，+ 本地 `tpe[E]` | `[T,L]` dense bool + f32（dense）或 `[T,K]` i64（sparse） |
| ② pre-dispatch permute | 框架 permute（按 global expert） | 无 | 无（kernel 按原序读） | 无 | — |
| ② 量化位置 | 接收侧（AdaptiveGEMM `per_tile_quant`） | 接收侧 | **发送前预量化** `(fp8, sf)`，kernel 只搬 | 无（bf16-only，全仓无 fp8） | 不涉及（FP8 只有权重+scale） |
| ③ 传输 / dedup | NCCL a2a，无 dedup（dedup 变体 −34%） | NVSHMEM，按 rank dedup | NCCL Gin（NVLink TMA / RDMA rail），按 rank dedup，PDL 两段式 | VMM 对称 VA **push** 直写终位，按 rank dedup + 目的侧展开 | 权重/梯度 NVSHMEM（可换 IPC）；token 由 dispatcher |
| ④ 到达布局 | (源 rank, 专家) | **R** + `recv_topk_idx` | **EA**（psum，段起点 align A）或 R | **EA**（A=128，零填，`E+B` 全局组，`cu_seqlens` 含 pad 末偏移） | 由 dispatcher；只把专家空间 `E → P=(M+R)·R_ep` |
| ⑤ post-dispatch | 第二次 permute → E0 | permute → E0；counts **host list** → H2D | **零拷贝**；counts 在 GPU（`num_unaligned`）；`do_cpu_sync` 可选 | **零拷贝**；无 host sync | — |
| ⑥ GEMM 契约 | counts（E0） | counts（E0） | **DeepGEMM psum**（`use_psum_layout=True`）/ `m_indices` | `[E+B,H,H']` 单段连续对称 VA + `cu_seqlens`（DeepGEMM 风格，零填行照算） | master 逐专家指针 + replica `[R, numel]` strided 视图 → **两个来源** |
| ⑥ probs 乘在哪 | GEMM-2 后 `unpermute(probs)`（源侧） | GEMM-2 后 `unpermute(probs)`（专家侧部分和） | **库不乘**（`recv_topk_weights[N]` 按行给框架） | **库不乘**（`route_weights_nvs[NvS]` 按行给框架） | 改写 probs 但不乘 |
| ⑦ pre-combine | unpermute（无 probs） | unpermute + 部分和 | identity（原位） | identity（**必须原位写回**同一 `[NvS,H]`） | — |
| ⑧ combine | a2a 纯搬运 | 跨 rank 求和，bf16 | 纯求和；两级归约（发送侧同 token 多专家先合，接收端 ≤2 路 bf16 hadd / 否则 fp32）；**只收 bf16** | **pull**，4 ACC warp fp32 累加 → bf16；重复行多一次 bf16 舍入 | — |
| ⑨ post-combine | `unpermute(probs)` 加权求和 | view | view | 无 | — |
| 反向 | autograd a2a + unpermute_bwd | combine(handle) / dispatch(handle) | combine(handle, `topk_weights=grad`) / cached dispatch(handle)（无 notify 无 CPU） | dispatch(plan) / combine(plan, `d_w`) + `reduce_grad`；**无 autograd** | `RestoreReplicaWeights`（阻塞 weight_sync）/ `StartGradReduce` / `JoinGradReduce` |
| 静态 shape | ✗ | ✗ | 最坏 `R·S·K'` 预留 或 CPU sync | ✓ 恒 `S·K`（完美均衡） | 给出上界 τ·mean（1.01–1.04） |
| CUDA graph | ✗ | ✗ | 原理可（未验证） | ✓（benchmark 用 graph 计时） | 声称可（profiler / relay epoch 存疑） |
| 确定性 | permute 稳定 | — | expand 段内 atomic **非确定**；`deterministic=True` 稳定排序 | 稳定计数排序，确定 | dense reroute 确定；sparse 非确定；grad_reduce 可 order-preserving |
| 域 / 规模 | 任意 | NVLink + RDMA | direct / hybrid，≤1024 rank，≤2048 expert，≤256/rank | **单 NVLink 域**，`R ≤ 128` | 副本限 NVL 域（≤72）；EP 组可跨节点 |
| 显存 | — | — | buffer ≈ 4.8 GB/GPU（EP64 例）；输出最坏预留 | `NvS` 恒定 + `B=E/R` 冗余槽 + fp32 grad/reduce buffer | 副本槽 2–4 个跨层复用（108 MB 级） |

### 3.6 Grouped GEMM 三库契约（`GroupGEMM/`）

三库对"专家批"的描述方式互不兼容：**DeepGEMM** 用"物理展开 + 按 BLOCK_M 对齐的分段"（`m_indices` 或 `psum`）；**SonicMoE** 用"紧凑原始 token 表 + gather 索引 + 无对齐 `cu_seqlens`"；**fanshiqing grouped_gemm / XTuner** 用"物理展开 + 无对齐 + `tokens_per_expert`"。§2.2 的 EA（`psum(group_ends, alignment)`，alignment=1 退化为 cu_seqlens）正是三者的最小公共上界，且是 DeepEP v2 的原生输出。

**DeepGEMM（`__version__=2.6.1`，含 2026.04 Mega MoE / FP8×FP4 / PDL）**，只 group M 轴，D 仅 bf16（m-grouped）：

| 布局 | 张量 | 语义 | 出处 |
|---|---|---|---|
| contiguous `m_indices` | `grouped_layout i32 [M]`，pad 行 −1 | 调度器只读每个 M block 首行的 group id（`scheduler/gemm.cuh:314-315`）→ **每组必须按 BLOCK_M 对齐**；SM90 128、SM100 = BLOCK_M（32…224，`heuristics/sm100.hpp:31-43`，m-grouped 恒定 swap-AB） | `tests/generators.py:352-353` |
| **psum** | `grouped_layout i32 [G]` = 累计真实结束行；起点 `align(prev, BLOCK_M)`；gap 可未初始化；`ensure_zero_padding` 见 §3.2.2 | 无 host 同步、graph 友好；`expected_m_for_psum_layout` 给 heuristics | `gemm.cuh:100-103,228-231` |
| masked | A `[G, M_max, K]` + `masked_m[G]` + host `expected_m` | LL / decode；masked→psum 需拷贝 | `gemm.hpp:250-297` |
| k-grouped（wgrad） | A `[ΣK, M]`、B `[ΣK, N]` MN-major，D `[G,M,N]` fp32/bf16 **必传 `c` 累加**；`ks_cpu` host list（128 倍数） | **SM100 psum 免 `ks_cpu`；SM90 不支持 psum** | `gemm.hpp:48-69,299-400,566-608` |

SF：SM90 fp32 `(1,128)+(128,128)`，A/B 都要 K-major（dgrad `nn` 需物理转置权重）；SM100 packed UE8M0 int32、MN-major TMA 对齐，fp32 会自动转（多一个 kernel）。FP4 仅 SM100。**GEMM 内无任何融合**（README:72 "must be handled separately"），SwiGLU × topk × FP8 量化靠第三方 tilelang op。Mega MoE：`fp8_fp4_mega_moe(...)` 仅前向、仅 SM100、≤72 rank、token 对齐 384、需 torch symmetric memory——它定义了 W4 的目标布局，而它的"非融合基线"就是 psum 链路。

**SonicMoE（Dao-AILab，QuACK/CuTeDSL + Triton）**：入口 `moe_general_routing_inputs(x, router_scores[TK], token_indices[TK], expert_indices[TK], w1, b1, w2, b2, E, ...)` 接受任意三元组、varlen-K（EP 接入点）。布局 = 紧凑 `x[T,d]` + `expert_frequency_offset[E+1]`（无对齐）+ `x_gather_idx[TK]`（= `argsort(topk.flatten()) // K`）+ `s_reverse_scatter_idx`，全 device 生成、零 host sync、X 从不物理展开。GEMM-1 `gemm_act(x, w1ᵀ, cu_seqlens_m, A_idx=x_gather_idx, preact_out=h, postact_out=a)`（gather 融合：cp.async 或 TMA gather4；SwiGLU 融合），GEMM-2 连续写 `y[TK,H]` 不 scatter，单独 `token_gather_and_sum` 乘 score；dH 用 `gemm_dact(..., colvec_scale=s, colvec_reduce=True)` 一次出 dH、dS、A'；wgrad varlen-K + 沿 K 的 gather。激活只存 `x[T,d]` 与 `h[TK,2n]`——每层激活与专家粒度无关、免 GEMM 重算（W5-b 的落点）。**bf16 only**（MXFP8/FP4 计划中）、SM90/100/120、EP=1 设计、无 fp32 wgrad 累加、无对齐概念（EA 的 gap 无法表达 → 走 R/compact 视图）。B300 上 vs DeepGEMM fwd +54% / bwd +35%，主要来自 gather 融合（GEMM 本身 ~10%）——**在 H200 上没有对应数字**，需自测。

**fanshiqing grouped_gemm（XTuner 现用 CUTLASS 后端 + permute 来源）**：`gmm(a, b, batch_sizes i64, trans_b)` 默认 cuBLAS 逐专家循环（需 `.cpu()`），`GROUPED_GEMM_USE_CUTLASS=1` 走 SM80 模板并接受 device batch_sizes（`kMaxExperts=512`）；wgrad bf16、无累加、无融合、无 FP8。`permute(act, indices[T,K] i32, num_out_tokens, ...)` cub 稳定排序 + `row_id_map[K][T]`（支持 -1 与 token drop，dtype 含 e4m3/e5m2）；`unpermute(..., probs)` 加权求和。**只需额外导出 `sorted_row_id` 并加 `alignment` 参数**，即可同时产 EA 与 SonicMoE 的 `gather_idx`。

**XTuner 三后端的 delta**（对照 §1.1 表）：Triton 与 AdaptiveGEMM 已经是"展开 + counts"契约，把 `tokens_per_expert` 改为由 `ExpertBatch.aligned_counts()` 派生即可零改动；AdaptiveGEMM wgrad 的 `expand_128x` 在 EA A=128 下消失；CUTLASS 后端只留作退路。

**汇总矩阵**：

| | DeepGEMM 2.6.1 | SonicMoE | grouped_gemm | XTuner triton | XTuner AdaptiveGEMM |
|---|---|---|---|---|---|
| M 维布局 | `m_indices` / **psum** / masked | 紧凑 + `gather_idx` + `cu_seqlens` | 展开 + counts | 展开 + counts（内核虚拟 128 对齐） | 展开 + counts（varlen） |
| 分段对齐 | = BLOCK_M | 无 | 无 | 无 | fwd 无；wgrad K 扩 128 |
| wgrad | k-grouped，fp32 累加，SM90 需 `ks_cpu` | varlen-K + gather，无累加 | bf16 | device cumsum，bf16 | 128 扩展，bf16 |
| dtype / SF | bf16 / fp8 / fp8×fp4；SM90 fp32 SF K-major，SM100 ue8m0 MN-major | bf16 | bf16 | bf16 | fp8，fp32 row-major SF |
| 架构 | SM90/100 | SM90/100/120 | SM80 模板 | Hopper TMA | SM90 |
| 融合 | 无（外部 tilelang） | gather / SwiGLU / topk 缩放 / dS | permute 带 probs | 无 | 量化+转置+扩展 |
| host sync | 无（k-grouped 非 psum 需 `ks_cpu`） | 无 | cuBLAS 路径需 | 无 | 无 |

**对 H200（SM90）的落地建议**：GEMM-1/2 用 DeepGEMM psum（FP8）或 Triton varlen（bf16，传 `aligned_counts`），wgrad 用 AdaptiveGEMM / Triton k-grouped（GPU counts），DeepGEMM k-grouped psum 留给 SM100；SonicMoE 先做激活显存与 bf16 吞吐评估，K=1 快捷路径与 EP 接入是它的集成成本。

---

## 4. 对"技术路线与规划设想"的评审

### 4.1 逐条评审

| # | 设想 | 判断 | 修正 / 建议 |
|---|---|---|---|
| 1 | FSDP 与 EP 解耦先行，先拿到收益，集成发版 | ✅ 同意，且**已有实现** | 工作 = 上游化（`meta_moe.py` → `moe.py` 基类）+ HSDP/DCP/HF/fp8 mesh 补齐 + 与 #2007 合并 root mesh 构建入口 + L0–L4 验收 + 发版（P0）。EP8 单机收益是 −22 GB 显存、吞吐持平；EP 越大收益越大。顺带上游 `all2all_dedup`、dW acc |
| 2 | 整体设计，多 backend 通信库兼容 | ✅ 但要重定义 | 不是"多 backend 通信库"一层，是三轴 + 契约 + 调度器（§2）。**P1 契约先于一切新 backend**：先让现有三 dispatcher × 三 GEMM 在契约上 bit-exact，再接 v2 |
| 3 | DeepEP v2 为主（固定 shape、无 CPU、CUDA graph、SM 少） | ✅ 主线正确，**三个前提有条件** | "SM 少"成立（解析式 4–64）；"无 CPU"= `do_cpu_sync=False`，代价是最坏预留（§3.2.5）；"CUDA graph"仓库未验证；"固定 shape"= 悲观预留，**要靠均衡器上界 + P8 hint 才可承受**。另：训练必须 `allow_multiple_reduction=True`；`hidden % 256`；`deterministic=True` 有 Python 排序开销；E≤2048 / E_local≤256 上限对 Meta-MoE 是硬伤 |
| 3.1 | DeepEP v2 + 配套 GroupGEMM 集成到 XTuner | ✅ | = P2 + P4 + P3 + P5；配套 = DeepGEMM psum（A=128）；SM90 wgrad 走 Triton/AdaptiveGEMM k-grouped；FP8 预量化贯通是 v2 收益的一半（通信字节减半） |
| 3.2 | v2 基础上，8 卡 NVL 域用 UltraEP 外挂做专家均衡（"UltraEP 用 hybrid EP、NVSHMEM"） | ✅ 可行，**术语修正** | UltraEP 自身不用 HybridEP；接法 = 虚拟专家 id 空间（§3.4）；`solve_placement_for_test` + `reroute_sparse` pybind；可用 IpcManager 彻底去 NVSHMEM；GEMM 两段 launch；fp32 副本梯度；`data_ptr` 稳定 → `efsdp=1`；`MAX_EXPERTS_PER_NVL=512` 对 E=2560 要改；产出 = P6 |
| 3.3 | v2 基础上，IB 域 + NVL 域用 UltraEP 做专家均衡 | ⚠️ **重定义** | 与 3.2 是同一代码路径（按 NVL 域自动分片，`placement.cu:819`）；UltraEP **不能跨域复制**，也不该做（每副本 ≈0.75 ms 上关键路径）。改为"跨节点 EP 组（v2 hybrid）+ 域内实时均衡（3.2 原样）+ **低频**跨域 master 重排（10²–10³ 步，optimizer 状态随迁）"；跨域残余用 W0 profiler 量化后再决定要不要上 W2-b 分层 a2a |
| 4 | 已有库的规范化重构（torch.alltoall、MoonEP、UltraEP） | ✅ | "重构" = 迁到契约（P1）；MoonEP 分支**重写**而非修补（用公开 API、不 fork kernel，P10）；`deepep` 改名 `deepep_v1` 保留；删两套重复的 `DeepEPDispatch` 与死代码；去全局单例 |
| 6 | 每个库接入时考虑其调度、overlap、多 micro-batch | ✅ | 不要每库一套：`MoEScheduler` 由 `caps` 推导（§2.7）；三种 overlap 模板；EP-DR；库特有的正确性约束（MoonEP `inter_rank_sync` / `reduce_grad` 无屏障、UltraEP 反向 restore/join 时序）进 `caps` 与 `Balancer` 钩子，而不是散在 decoder layer |
| 7 | GroupedGEMM 分析与集成 | ✅ | DeepGEMM psum 首选（与 v2 零拷贝）；SonicMoE 的主卖点是激活显存（免 recompute），H200 吞吐需自测，bf16-only、EP 接入要 K=1 快捷路径；Triton varlen 保底；**P3 的 SwiGLU×probs×FP8 融合核是算子组最该先做的一个**（三家库都把乘法留给了框架） |
| 8 | MegaKernel（3+4 熟悉后再开始） | ✅ | 第一阶段不写 kernel：静态 shape（P6+P8）+ 整层 CUDA graph（P7）+ shared-expert 掩体；H200 上 MegaMoE 移植 #360 未合入、#323 有回归；训练侧对标 UniEP 形态；判据 6144 FLOPs/Byte 测算表（W0） |

### 4.2 需要修正的前提（展开）

**(1) "固定 shape" 在 v2 里是悲观预留，不是免费的。** `do_cpu_sync=False` 下 `recv_x` 按 `R·S_max·min(K,E_local)` 行分配（§3.2.5），EP8 单层 3.8 GB、EP64 30 GB——训练不可承受；`do_cpu_sync=True` 精确但 host 同步。出路是把 W3 均衡器的上界喂给通信库：UltraEP 承诺 τ ≈ 1.01–1.04 × mean、MoonEP 恒 `S·K`，对应 `recv_x` ≈ 0.5 GB。DeepEP v2 目前没有接收上界参数（v1 normal 有 `num_worst_tokens`，HybridEP 有 `num_permuted_tokens` + 溢出 flag），所以 **P8 是把 W3 与 W4 接起来的那颗螺丝**——一个小 patch，值得直接提上游。四级溢出兜底（v0.1 §5.2）也挂在这个 flag 上。

**(2) UltraEP 的定位。** 它是均衡器，输入 `[T,K]`/`[T,L]` 路由、输出物理 id 空间的路由；权重/梯度走 NVSHMEM（可换 IPC）；token 一律由 dispatcher 搬。与 DeepEP v2 的对接缺口只有两条：dense→idx 转换（或直接用 sparse 路径）与 `num_experts=P`。副本不出 NVL 域是构造性的，3.3 的"IB+NVL 用 UltraEP 均衡"在 8 卡域集群上只能得到域内版（`README.md:12` 原话 imbalance floor "slightly above the ideal 1.0"）。


**(3) 解耦已实现。** §1.5；P0 的风险在"泛化到非 MetaMoE 模型 + HSDP + fp8 reduce mesh + 与 #2007 合并"，不在算法。

**(4) MegaKernel 现状。** DeepGEMM MegaMoE main 只有 SM100（`mega.hpp:260-280`）；#360（H200）open、单算子 2.04–6.29× 但 L2 命中 ~40%；#323（H20）open、社区反馈寄存器溢出导致慢于非融合基线。训练侧 UniEP（Triton-distributed）是更贴近的蓝本。**在 H200 上，W4 前两个阶段（图化、训练侧原型）没有可拿来即用的 kernel，第三阶段（rollout 侧 MegaMoE）要等 SM100 机器或 #360 合入。**

### 4.3 几条横向建议

- **专家数 E 的取值要同时看三个库的硬上限**：DeepEP v2 `E ≤ 2048, E_local ≤ 256`（`layout.cuh:19-21`）、UltraEP `M × nvl_size ≤ 512`（`placement.cu:22`）、MoonEP `E % R == 0, R ≤ 128`。Meta-MoE 的 2560 三个都不满足或勉强；v0.1 建议的 2304/3072 仍超 DeepEP 2048。**E=2048 是唯一不改任何库常数就能同时过 DeepEP v2（EP8 时 E_local=256 恰好到顶）与 MoonEP（被 8/16/32/64/128 整除）的取值**；UltraEP 仍需改 512 常数。384 倍数的诉求只在 EP=384/768 时才成立（不在计划内）。建议把这条带给算法组，比"先定 2560 再逐库打 patch"省很多事。
- **W7 的确定性**：v2 expand 段内序是 atomic 到达序，`deterministic=True` 用 Python 稳定排序补救（有开销）；UltraEP sparse reroute 非确定；MoonEP 确定。bitwise 需求（RL 训推一致）下要么付排序开销、要么 MoonEP，要么自写 segmented-prefix 版 reroute + 让 v2 epilogue 走确定序（需 kernel 改动）。
- **全层 REENTRANT checkpoint 与大 EP 冲突**：通信执行 3 次。EP-DR（K3）+ SonicMoE 式激活策略（不存 gathered X / a / y）应作为 P7 的一部分，而不是等 W4。
- **NCCL EP**（`contrib/nccl_ep`，2603.13606）：API 形态（group/handle 两级资源模型、`send_only` 分阶段）与本文 `MoESpec`/`handle`/六段式一致，作为 `caps` 设计的对照；HT（训练）模式成熟后是 `deepep_v2` 之外的第二个跨节点 backend，追踪即可。
- **上游贡献清单**（小而高价值）：DeepEP v2 `max_recv_rows_hint`（P8）；UltraEP `reroute_sparse` pybind + 确定性版本 + `MAX_EXPERTS_PER_NVL` 参数化；fanshiqing permute 的 `alignment`/`sorted_row_id`；MoonEP `reduce_grad`/`prefetch` 入口屏障选项。

### 4.4 版本与环境门槛（各库 vs XTuner 当前镜像）

**XTuner 当前镜像**（`image_build.sh`、`.github/workflows/docker-inner.yml`）：基础镜像 NGC `pytorch:25.03-py3`（CUDA 12.8.1、Python 3.12、自带 torch 2.7.0a0 但被覆盖）→ `pip install torch==${TORCH_VERSION}`，默认 **`TORCH_VERSION=2.9.1`**（cu128 wheel；CI 同）；DeepEP 钉在 **v1.2.1**（`9af0e0d`）+ `nvidia-nvshmem-cu12==3.4.5`；DeepGEMM **v2.1.1.post3**（2025-10-15）；AdaptiveGEMM `10411e0`；`pyproject` 只要求 `torch>=2.6`。torch wheel 自带的 NCCL 是 **2.27.x**（torch 2.8 → 2.27.3，2.9 / 2.10 → 2.27.5，见 pytorch `.ci/docker/ci_commit_pins/nccl-cu12.txt`）——所以"升到 torch 2.10"本身并不带来 DeepEP v2 需要的 NCCL。

| 库 | 官方声明的要求 | 代码里真正的耦合点 | 对 XTuner 镜像的含义 |
|---|---|---|---|
| **DeepEP v2**（2.1.0） | Hopper SM90+；Python ≥ 3.8；CUDA ≥ 12.3；**PyTorch ≥ 2.10**；**NCCL ≥ 2.30.4**；NVLink / RDMA（`README.md:63-72`，由 PR #605 改写，此前是 PyTorch ≥ 2.1） | torch 侧只用 `ProcessGroupNCCL._comm_ptr()` 复用通信器（`deep_ep/utils/comm.py:60-63`），该绑定 **torch 2.8 / 2.9 / 2.10 都有**（`torch/csrc/distributed/c10d/init.cpp` 三个 release 分支核实）；没有时回退自建 comm（`create_nccl_comm`，`comm.py:66-74`）。NCCL 侧：Gin 设备 API 需 ≥ 2.30.4，且**进程里加载的 `libnccl.so.2` 必须就是编译时那个**——2.31 之前编译/运行版本必须相等（`nccl.cu:104`，`EP_SUPPRESS_NCCL_CHECK` 可关），2.31+ 走 `useRuntimeVersion`（#688）。legacy v1 路径仍链接 NVSHMEM（`setup.py:93` "TODO: make NVSHMEM and legacy optional"）。 | "PyTorch ≥ 2.10" 是官方支持基线，不是 import 门槛；**真正要动的是 NCCL**：在 2.9.1 镜像上 `pip install "nvidia-nccl-cu12>=2.30.4"`（覆盖 torch 钉的 2.27.5；pip 会警告，NCCL ABI 向后兼容，PyPI 现有 2.30.4 / 2.30.7 / 2.31.2）后再编 DeepEP，**需实测**。两个坑：① EP 子组懒初始化时 `_comm_ptr()` 返回 0 会段错误（issue #726，2026-08，本 checkout dd758ca 未修）→ 建 `ElasticBuffer` 前在 EP group 上先做一次 collective，或 `EP_REUSE_NCCL_COMM=0`；XTuner 的 `init_process_group(backend=backend)` 未传 `device_id`（`train/trainer.py:1543`）；② 单机/无 NIC 需 `EP_DISABLE_GIN=1`；GIN 的 RDMA 路径是 GDAKI（DOCA GPUNetIO），NIC 需 ConnectX-6 Dx 以上（README 性能表用 CX7）。同一 `deep_ep` 包装 v2 后 legacy `Buffer` API 签名不变 → `deepep_v1` 后端零改动。 |
| **DeepGEMM**（main 2.6.1） | SM90 / SM100；Python ≥ 3.8；C++20 编译器；CUDA ≥ 12.3（**强烈推荐 ≥ 12.9**）；PyTorch ≥ 2.1；Mega MoE 需 PyTorch ≥ 2.9（`README.md:120`，symmetric memory）；CUTLASS 4.0 | psum 布局在 #280（2026-01-16）引入；**XTuner 钉的 v2.1.1.post3 没有**（`git merge-base` 核实） | 升到含 #280 的版本；JIT 用 NVRTC，镜像 CUDA 12.8.1 可用但性能口径是 12.9；与 AdaptiveGEMM 包名不同，可共存 |
| **MoonEP**（0f385f0） | NVIDIA GPU（NVSwitch 多播）；`nvidia-cutlass-dsl==4.4.2`（`setup.py:73`）；`cuda.bindings.driver`；`-lcuda` | CuTe DSL 4.4.x：Python 3.10–3.13、**CUDA Toolkit 12.9 或 13.1**、驱动"与对应 Toolkit 一致"（12.9 → R575，575.51.03+）；VMM + `cuMulticast*` 需 NVSwitch 机型（`buffer.py:221` 断言多播支持） | **镜像的 CUDA 12.8.1 不够；集群驱动若是 R570 也不够**——MoonEP（以及任何 CuTe DSL 内核）要先升 toolkit/驱动；与 SonicMoE 的 DSL 版本互斥（见下） |
| **UltraEP**（1.0.0） | SM90 / SM100；Python ≥ 3.10；**PyTorch ≥ 2.10**；C++17；CUDA ≥ 12.3（SM90）/ ≥ 12.9（SM100）；`nvidia-nvshmem-cu12/13==3.4.5`；`-rdc=true`（`README.md:32-46`） | Python 侧没有 2.10 专属 API（grep 无 `torch.__version__` / graph / symmetric 依赖）[推断：2.10 是其测试基线] | NVSHMEM 3.4.5 与 XTuner 镜像一致；torch 2.9.1 大概率可编译运行，**需实测**；走 `IpcManager` 去 NVSHMEM 后依赖更少 |
| **SonicMoE** | SM90 / 100 / 120；CUDA ≥ 12.9（B300 13.0+）；**Python ≥ 3.12；PyTorch ≥ 2.11**（推荐 2.12）；`quack-kernels>=0.6.4` → `nvidia-cutlass-dsl==4.7.0` + `torch-c-dlpack-ext`（`pyproject.toml:9-16`；quack `pyproject.toml:8-13`） | 版本要求来自 QuACK / CuTe DSL 4.7 | **与当前镜像差三档**（torch 2.9.1 / CUDA 12.8 / DSL 4.4.2 vs 2.11 / 12.9 / 4.7.0）；且 `cutlass-dsl==4.7.0` 与 MoonEP 的 `==4.4.2` **同一环境不可共存**——G7 评估要单独环境 |
| AdaptiveGEMM（现用 FP8） | DeepGEMM `3b3783d` 分支 | 与新 DeepGEMM 包名不同 | 可共存 |

**结论与建议**

1. "DeepEP v2 要 torch 2.10"不是 import 级门槛，真正的门槛是 **NCCL ≥ 2.30.4 且与 torch 加载的一致**。建议先做一个两小时的可行性实验：2.9.1 镜像 → `pip install nvidia-nccl-cu12==2.31.2`（2.31 走 `useRuntimeVersion`，对版本漂移最宽容）→ 编 DeepEP main → 单机 8 卡跑 `tests/elastic/test_ep.py`（无 NIC 加 `EP_DISABLE_GIN=1`）→ 再跑一次 XTuner 现有 `deepep`（legacy）回归。通过即可把 P2 的环境风险关掉。
2. 长期镜像路线：与其逐库打补丁，不如一次把 torch / NCCL / CUDA / 驱动升到 CuTe DSL 4.7 与 SonicMoE 都满足的水位（NGC 26.05+ 自带 NCCL 2.30.4、CUDA 13.2、torch 2.12 [按 NVIDIA support matrix，需复核]）。**驱动升级（R570 → R575+）是集群运维事项，要最早提**，它同时卡住 MoonEP 与 SonicMoE。
3. 依赖隔离：MoonEP（DSL 4.4.2）与 SonicMoE（DSL 4.7.0）互斥，评估期各用独立 venv；正式集成时向 MoonEP 提 PR 放宽 pin。
4. XTuner 侧两处小改：`init_process_group` 传 `device_id`（消除 #726 的懒初始化坑），`deepep_v2` backend 的 `setup()` 对 EP group 做一次 warm-up collective 再建 `ElasticBuffer`。

### 4.5 风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| DeepEP v2 `E ≤ 2048 / E_local ≤ 256` 对目标模型不够 | Meta-MoE 无法用 v2 | 改常数验证是否只是常数（JIT 模板参数）；或 E=2048 |
| `do_cpu_sync=False` 最坏预留显存 | 静态 shape 不可承受 | P8 hint + 均衡器上界；过渡期用 `do_cpu_sync=True` |
| UltraEP 依赖 `data_ptr` 稳定 / fp32 副本梯度 | 与 FSDP over experts 冲突 | `efsdp=1` 默认；`efsdp>1` 时挂 unshard 钩子（后续） |
| MoonEP 权重须对称 VA 连续分配 | 与 FSDP flat-param 冲突 | 可替换 allocator（P10，W1 之后） |
| v2 expand 非确定序 | W7 bitwise | `deterministic=True` 或 MoonEP |
| CUDA graph 在 v2 / UltraEP relay 模式未验证 | 图化收益延后 | 先做 graph 捕获实验（direct 模式 + cached handle） |
| probs 乘法迁入 act 核改变舍入位置 | 数值漂移 | P3 loss 级 A/B，保留旧路径开关 |
| MegaMoE H200 移植未合入 | W4 第三阶段无载体 | 等 #360 / SM100 机器；训练侧走 UniEP 形态 |
| 镜像 / 驱动版本：DeepEP v2 需 NCCL ≥ 2.30.4（torch wheel 自带 2.27.x）；CuTe DSL 内核（MoonEP / SonicMoE）需 CUDA 12.9 + R575 驱动；SonicMoE 需 torch ≥ 2.11 | P2 / P10 / G7 环境准备期被低估 | §4.4 的可行性实验先行；驱动升级早提运维；评估期独立 venv |
| moonep 分支已有投入 | 重写阻力 | 复用其 `_SharedBProjectionSpace` 与 autograd 映射，替换为公开 API + 契约 |

---

## 5. 与技术负责人讨论的开放问题

1. **P1 契约的合入方式**：作为一个"无行为变化"的重构 PR 先进主线（三 dispatcher × 三 GEMM bit-exact），还是随 `deepep_v2` 一起进？建议前者——它同时是 #2007 Expert TP 与 MoonEP 分支的公共地基。
2. **P0 解耦上游化的归属**：MetaMoE 分支的作者主导泛化，还是框架组接手？与 #2007 是各自 PR 还是合成一次 mesh 重构？root mesh 维度命名（`replicate / efsdp / ep / etp?`）与构建入口只能有一套。
3. **默认训练路径**：是否同意 `deepep_v2(expand, A=128) + DeepGEMM psum (FP8) / Triton (bf16)` 为大 EP 默认，`deepep_v1`/`all2all` 降为回退？DeepGEMM 进主线依赖的版本与 JIT 缓存策略。
4. **均衡器形态**：先 UltraEP 外挂（8 卡域）还是直接做 MoonEP-planning（B=E/R 显存、静态 shape 更强）？建议 UltraEP 先，因为它不改 token 面、与 v2 零冲突、且 `solve_placement_for_test` 已可独立调用；MoonEP-planning 作为同一 `Balancer` 接口的第二实现。
5. **专家权重存储策略**：是否接受"大 EP + 均衡器默认 `efsdp=1`"？专家参数 allocator 可替换（VMM / IPC）是否在 W1 解耦 PR 里预留接口？
6. **probs 乘法迁入 act 核**（P3）：改变舍入位置，需要算法组认可数值 A/B 口径。
7. **DeepEP v2 patch 策略**：`max_recv_rows_hint`（P8）与 `E ≤ 2048` 常数——本地 patch 维护还是提上游等合入？
8. **E 的取值**：是否把"E=2048"作为建议带给算法组（§4.3）？
9. **调度器抽象的范围**（P7）：只做层内（两段式 + shared-expert 掩体 + EP-DR），还是把跨层 1F1B/DualPipe 式重排也纳入？建议首期只做层内。
10. **验收口径**：全链路（dispatch→GEMM→act→GEMM→combine）**零 `.cpu()/.tolist()`** 是否作为 `deepep_v2` 路径的硬验收；一致性测试基线（不同 EP / 均衡 / 后端下 loss 逐位或容差对齐分级）由谁维护。

---

## 附录 A：本文与 v0.1 的关系

- 本文替换 v0.1 §4.2(a)（dispatcher 多 backend）与 §7(a)（GEMM 后端可切换）的设计部分，并把 `ExpertBatch` 描述符落成接口代码。
- v0.1 的 W1（解耦）状态从"proposal 已备"改为"MetaMoE 分支已实现，待上游化"（§1.5）。
- v0.1 §7(a) "DeepEP v2 MoE layout" 应改为 "DeepGEMM psum layout（`use_psum_layout`）与 DeepEP v2 `dispatch(do_expand, expert_alignment, use_tma_aligned_col_major_sf)` 零拷贝对接"；"dynamic A/B swap" 注明仅 SM100。
- v0.1 §5.2 UltraEP 行："外挂形态"精确为"不碰 token 面，只改路由 + 搬权重；依赖 dispatcher 接受物理 id 空间"。
- v0.1 §6.1 MegaMoE 行补充：H200/H20 移植 #360/#323 均未合入。
- v0.1 §9 W7 补充：v2 expand 段内序非确定（atomic），需 `deterministic=True` 或 MoonEP。

## 附录 B：引用位置索引

| 主题 | 文件:行 |
|---|---|
| XTuner 六段式协议 / `PostDispatchResult` | `xtuner/v1/module/dispatcher/base.py:28-45,70-159`；`dispatcher/__init__.py:28-79` |
| XTuner all2all 路径 | `dispatcher/torch_all2all.py:78-113,100-103,255-274,315-316,328,416-461,474-477,516-521,566-570` |
| XTuner DeepEP v1 路径 | `dispatcher/deepep.py:222-240,277-305,349-421,461-473,497-514`；`ops/comm/deepep_op.py:24,61-105,142-223,226-297,314-316,390-392,509-511,434-670` |
| XTuner GEMM 三后端 | `ops/moe/protocol.py:6-12`；`ops/moe/cuda/group_gemm.py:8-37`；`group_gemm_cutlass.py:74-88`；`triton_kernels/m_grouped_gemm_TMA_triton3_4.py:274-352`；`float8/float8_gmm_tile_wise.py:28-36,42-83,86-153` |
| XTuner overlap / recompute / mesh | `module/decoder_layer/moe_decoder_layer.py:393-445,471-604,697-723`；`model/moe/moe.py:84-101,172-176,486-619,1137-1138,1151-1155,1302-1340,1342-1393,1415-1427`；`config/fsdp.py:18,49-51`；`float8/float8_handler.py:157-161` |
| XTuner NPU | `ops/moe/npu/permute_unpermute.py:6-34`；`ops/moe/npu/group_gemm.py:5-10`；`utils/device.py:10-33` |
| moonep 分支集成 | `sh/a3-stable-moonep` 分支 `dispatcher/moonep.py:138-302,305-314,331-332,358-544`；`grouped_linear/moonep.py:23-77,109-151,158-198,258`；`ops/comm/moonep_op.py:17-93,133-134,185-829,898-1007,1010-1102,1115-1187`；`moe_decoder_layer.py:274-296,414-484,512-645`；`meta_moe.py:1411-1421,1582-1592` |
| MetaMoE 分支已实现项 | MetaMoE 分支优化记录 §2、§4.2、§4.3、§5、§6 |
| DeepEP v2 API / handle / 上限 | `deep_ep/buffers/elastic.py:42-74,100-192,228-246,381-385,728-834,855-876,937-948,1046-1056`；`deep_ep/include/deep_ep/common/layout.cuh:19-21,194-207`；`csrc/elastic/buffer.hpp:566-573,586-686,755-771,805,1017-1071,1090-1121,1203,1219,1237-1247` |
| DeepEP v2 expand / psum / epilogue / combine | `impls/dispatch.cuh:79-107,104,205-215,254-257,280,320,332,344-345,371-393,403`；`impls/dispatch_copy_epilogue.cuh:40-42,63-64,66-81,104-121,192-206,231-322`；`impls/combine.cuh:45-46,67-69,88-92,109,115-118,144-176,215-225`；`impls/combine_utils.cuh:67-73,111-168` |
| DeepEP v2 后端 / 模式 | `csrc/kernels/backend/nccl.cu:56-60,86-90,108,118-125,140-141`；`deep_ep/utils/comm.py:61-64`；`common/handle.cuh:11-228`；`impls/hybrid_dispatch.cuh` |
| DeepGEMM psum / 对齐 / API | `deep_gemm/include/deep_gemm/scheduler/gemm.cuh:100-103,189-195,217-237,302-319`；`csrc/jit_kernels/heuristics/{runtime.hpp:10,47-57, sm90.hpp:31-33, sm100.hpp:31-43}`；`csrc/apis/gemm.hpp:48-69,166-232,250-297,299-400,464-517,566-608,673-681,738-743`；`csrc/utils/layout.hpp:32-34,90-129`；`csrc/jit_kernels/impls/smxx_layout.hpp:181-216`；`tests/test_mega_moe.py:225-306`；`tests/generators.py:345-379,405-408` |
| DeepGEMM Mega MoE | `csrc/apis/mega.hpp:19-21,98-115,153-203,260-280`；`deep_gemm/mega/__init__.py:97-149` |
| SonicMoE | `sonicmoe/functional/__init__.py:104-171,227-362,260-275,287-296,339-348,365-479,389,410-418,465-473,482-557,560-566,571-652`；`functional/backward.py:187-194,228-240`；`functional/triton_kernels/__init__.py:78-148`；`reduction_over_k_gather.py`；blog `assets/2026-04-22-sonicmoe-blackwell.md` |
| fanshiqing grouped_gemm | `grouped_gemm/ops.py:12-42`；`backend.py:29-35`；`csrc/grouped_gemm.cu:40-81,283-304,398-437,511-514`；`csrc/permute.cu:41-99,565-568,680-705`；`csrc/permute.h:15-34` |
| MoonEP API / 布局 / 限制 | `moonep/api.py:158-173,196-210,244-279,298-347,352-355,439-453,523-524,588-630,685-696,704-707,719-723,741,748-766,782-824,881-890,929-946,1009-1019`；`planning.py:31-88,365-393,398-517,535,601-701,717-798,834-960,1048-1113,1202-1250,1282,1294`；`dispatch.py:301,406-497,569-576,631-639,861-864,883`；`combine.py:336-348,353-475,485-511,604-606`；`dispatch_epilogue.py:1-17`；`combine_prologue.py:1-19,456-478`；`prefetch.py:115-126,336-339,353`；`grad_reduce.py:133-138,237,310-389,468-469` |
| MoonEP 传输 / 工具链 | `csrc/nvl_shared_buffer.cuh:108-241,298-418`；`moonep/buffer.py:23-27,55-114,145,221,264`；`moonep/_common.py:40-460,249-346`；`setup.py:63`；`benchmarks/bench_vs_deepep.py:3-8,16,148-199,246-262,279-292,369`；`tests/planning_reference.py:25-312` |
| UltraEP 求解 / reroute / 权重 | `ultra_ep/manager.py:19-36,66-67,84-86,114-122,180-245,309-330,359-428,430,464,538,552-558`；`csrc/ultra_ep.hpp:21-59,125-126,207,267-270,427-612`；`csrc/ultra_ep.cpp:123-158,263-277,303-395,403-422,503-508,649-666,668-746,748-832,834-892,909,968,989,1129`；`csrc/kernels/placement.cu:22-24,76-84,337-667,788-1482,1493-1501,1593-1595,1630-1631`；`csrc/kernels/reroute.cu:21-51,125-132,139-264,270-297,492-515,547-577`；`csrc/kernels/weight_sync.cu:37-39,205-522,674,699,714-775,782-986,1001-1018`；`csrc/kernels/grad_reduce.cu:51-53,236-257,289-394`；`csrc/kernels/load_compute.cu:51-73,111-129`；`csrc/kernels/config.cuh:10-12` |
| UltraEP 传输 / 依赖 / 示例 | `csrc/utils/nvshmem.cuh:35-39,100-102`；`csrc/utils/ipc_manager.cu:25-50,76-137,150-208`；`ultra_ep/runtime.py:14-15,22,24,34-51`；`csrc/runtime.cpp:35-45`；`setup.py:83-125`；`examples/train_qwen3_235b.sh:179-183,201-206`；`examples/README.md:31-40`；`tests/test_e2e.py:539-588`；`tests/test_solving.py:71-75`；`tests/utils.py:587-595`；`docs/blog_v1.md:61,97,138,186` |
| 线上 PR | DeepEP #605（v2）；DeepGEMM #304（Mega MoE, merged）、#323（H20, open, 寄存器溢出）、#360（H200, open） |

*v0.2 · 2026-08-26 · 讨论后迭代：决策清单落地、P0–P10 归属、E 取值与算法组对齐。*
