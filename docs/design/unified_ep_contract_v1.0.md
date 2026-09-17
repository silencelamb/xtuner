# XTuner 统一 EP 接口 v1.0：最终态设计、参考实现与三后端接入（2026-09-17）

面向框架组与算子组同事的讲解稿。本文只讲最终态：代码在 [silencelamb/xtuner 分支 `feat/deepepv2`](https://github.com/silencelamb/xtuner/tree/feat/deepepv2)，基于 upstream `7d377424`，两个原地提交；本文与 [统一 Grouped GEMM 算子库 v1.0](unified_grouped_gemm_v1.0.md) 一起随第 1 个提交进入 `docs/design/`。讲解前的问答整理进 §2.6–2.7、§5.1–5.2、§6.1–6.2（2026-09-17）。算子侧配套：[统一 Grouped GEMM 算子库 v1.0](unified_grouped_gemm_v1.0.md)（§2.4 的 GEMM 入口在算子侧怎么扩成注册表与多后端，含开发指南）。

## 概要

| 内容 | 指向 |
|---|---|
| 核心在 `dispatcher/base.py` | §2 整节。数据面看 §2.1：`ExpertRows` 和 `ExpertWeights`，都挂在 `PostDispatchResult` 上，`Layout` 是 `ExpertRows` 按 `alignment` 派生的布局家族（E0 / EA，R / M 预留）；执行面看 §2.3：`LayerEPExecution` 五个钩子、`EPExecutionRuntime` 四个边界 |
| 与 #2056 的异同 | §3 的逐接缝对照表，再加 §2.6 讲每个类型沿用还是新设计、落地到哪一步 |
| MoonEP 接入 | §5 逐项对照，§5.1 改动量，§5.2 对私有 API v3 的依赖 |
| UltraEP 接入 | §6 逐项对照，§6.1 通信底座，§6.2 两个补丁 |
| 对现有代码的影响 | §2.5，框架组最关心这个：legacy 逐位不变、同速、回归怎么做 |

## 0. 一页结论

| 提交 | 性质 | 改动面 | 证据 |
|---|---|---|---|
| 第 1 个提交 `[Refactor]` 契约 v0 原地落地 | **无行为变化**的接缝重构 | 10 个文件 +619/−206：`dispatcher/base.py`、四个 legacy dispatcher 各 +8 行、`MoEBlock.forward(batch)`、两个 `GroupedLinear` 吃 `rows`、decoder 单批/多批合成一个交错核心、`MoE` 持有 runtime | 跨树逐位对比 66/66（EP=2：all2all / DeepEP v1 × BF16 / FP8 × mb1 / mb2；EP=8：AGRS × BF16 / FP8 × mb1 / mb2，加 all2all / DeepEP v1 各一条 BF16；EP=1：naive × BF16 / FP8；每条比全部 rank 的输出、dX、全部参数梯度）与上游原样**逐位相等**；原有 dispatcher / grouped-linear 测试 14 通过 1 跳过（该跳过是上游自带的 `skipIf(True)`）；GLM-5.2 / Qwen3 上 legacy 路径同速 |
| 第 2 个提交 `[Feature]` DeepEP V2 + DeepGEMM psum | 新后端 | 5 个新文件（V2 dispatcher、DeepEP V2 op、DeepGEMM GEMM op、两个 Triton kernel）+ 现有文件的接入点 | 2 卡六种模式对照 all2all；静态段 CUDA graph 捕获重放；专家块 fullgraph 编译；单卡 8 条 kernel 路径对 fp32 参考 |

第 1 个提交的回归全部通过且逐位对齐：跨树对比 66 条全部 BIT-EXACT（EP=1 / 2 / 8，含 AGRS），现有 dispatcher / grouped-linear 单测通过，详见 §2.5。loss 曲线全部与 legacy 一致（第 20 步逐位或 1e-4 内）。契约重构本身零代价：新树的 legacy 行与上游原样树同速。

三个后端的接入方式：DeepEP V2 = 纯 dispatcher（不用执行钩子）；MoonEP = dispatcher + 模型级 runtime + 1 段 call-local 权重；UltraEP = DeepEP V2 dispatcher + 模型级 runtime + 2 段权重 + 5 个钩子。详见 §5、§6。

## 1. 设计约束（专项当初的结论怎么落的）

1. **新后端走新文件，legacy 的 dispatcher 与 kernel 一行不改**：做到了。四个 legacy dispatcher 只在 `dispatch_postprocess` 末尾多填一个 `rows`，`dispatch_preprocess` 多收两个它们不读的参数。
2. **调用点只能有一份**：三个后端要接进同一个 decoder，decoder 就必须说一种语言。这份语言就是契约；它落在 `moe_decoder_layer.py` 和 `dispatcher/base.py`，这是唯一不可避免的"公共文件改动"。§8 专门讨论这个取舍。
3. **契约里没有 host sync**：行布局是 GPU 张量；host 镜像是可选字段，只给声明需要的消费者。
4. **对 `torch.compile` / CUDA graph 友好**：进专家块的数据对象键固定；控制状态（handle、事件、plan）不进编译边界。
5. **GEMM 后端 setup 期选择，按布局能力匹配**，不在热路径分支。

## 2. 契约 v0 最终形态（`xtuner/v1/module/dispatcher/base.py`）

### 2.1 数据对象

```python
class ExpertRows(NamedTuple):            # 一次 dispatch 后本 rank 上专家批的行布局，全部 GPU 张量
    starts: Tensor                       # [P] 每个本地专家段的起始行
    compute_counts: Tensor               # [P] GEMM 可触碰的行数（alignment 的倍数）
    valid_counts: Tensor | None          # [P] 真实 token 行数（<= compute_counts）
    alignment: int                       # 1 = 未对齐（legacy），128 = 段对齐（DeepGEMM psum 布局）
    padding_zeroed: bool                 # 对齐 padding 行是否已被生产者清零
    host_counts: Tensor | None = None    # 可选 CPU 镜像

rows_from_counts(counts)                                   # legacy tokens_per_expert -> 未对齐布局
rows_from_psum(psum, valid, alignment=, padding_zeroed=)   # DeepEP V2 / DeepGEMM 的前缀和 -> 对齐布局
rows_psum(rows)                                            # DeepGEMM use_psum_layout 要的 grouped_layout

class PostDispatchResult(TypedDict):     # dispatcher 交给 MoEBlock.forward 的唯一对象，键固定
    hidden_states: Tensor                # [capacity, hidden]，布局由 rows 描述；FP8 时可为 bf16 类型的 Float8Tensor 包装
    hidden_scales: Tensor | None         # 生产者直接给 fp8 + per-tile scale 时用
    tokens_per_expert: Tensor            # == rows.compute_counts，给只认计数的 kernel
    rows: ExpertRows
    expert_weights: ExpertWeights        # segments=None 表示用模块自身参数

class ExpertWeights(NamedTuple):         # call-local 权重段（UltraEP replica / MoonEP alias 的归宿）
    segments: tuple[ExpertWeightSegment, ...] | None = None
# ExpertWeightSegment(first_slot, w1w3: ProjectionWeight, w2: ProjectionWeight)
# ProjectionWeight(value, backward_read="saved"|"restore_before_dgrad", grad=GradBinding(path, buffer, write_op))
```

语义要点：`capacity` 可以大于 `starts[-1] + compute_counts[-1]`（静态模式的最坏预留），尾部行任何 kernel 都不读不规约；对齐 padding 行必须是有限零（wgrad 的计数 kernel 会规约到它们）。这两条是 DeepEP V2 静态模式与 DeepGEMM 能接进来的前提。

布局家族（`Layout`，与 `ExpertRows` 同在 `dispatcher/base.py`）：

| 家族 | 定义 | 产出者 | 消费者 |
|---|---|---|---|
| **R**（rank-grouped） | `[N_recv, H]` 按源 rank 分组，一 token 一行 + `topk_local[N_recv, K]`（-1 无效）或 `gather_idx` | DeepEP v1 normal、DeepEP v2 `do_expand=False`、torch all2all（第一次 a2a 之后） | 框架 permute → E0/EA；SonicMoE 型 gather-GEMM 可直接吃（`A_idx` 就是 permute 索引） |
| **E0**（expert-contig） | `[N, H]` 按本地物理专家连续，无 gap，`alignment=1` | XTuner permute、NCCL EP HT、CANN `MoeTokenPermuteWithEP`、HybridEP `pad_multiple=1` | Triton varlen、AdaptiveGEMM、CUTLASS、SonicMoE（idx=identity）、aclnnGroupedMatmulV5；DeepGEMM 不行（组起点未对齐） |
| **EA**（expert-aligned） | `[N_pad, H]`，`seg_start[i] = align(seg_start[i-1]+seg_count[i-1], A)`，gap 行零填或未定义 | DeepEP v2 expand + `expert_alignment=A`、MoonEP（A=128，零填，组数 `E+B`）、HybridEP `pad_multiple=A` | DeepGEMM psum / `m_indices`（A=BLOCK_M）；以及所有 E0 消费者，传 `aligned_counts()`（v1.0 里就是 `rows.compute_counts`）当 counts 即可（每专家 ≤A−1 行零计算） |
| **M**（masked 3D） | `[P_local, M_max, H]` + `masked_m[P_local]` | DeepEP v1 LL、NCCL EP LL | DeepGEMM masked；decode / rollout 专用，训练主线不用 |

`ExpertRows` 只表达 E0 与 EA，`rows.layout` 按 `alignment` 派生（1 是 E0，大于 1 是 EA），不是新字段；R 与 M 在 `Layout` 里占位，等第一个需要它们的后端出现再加 `gather_idx` / `masked_m` 字段。

```python
class Layout(str, Enum):
    E0 = "expert_contig"
    EA = "expert_aligned"
    R = "rank_grouped"       # reserved
    M = "masked"             # reserved

class ExpertRows(NamedTuple):
    ...
    @property
    def layout(self) -> Layout:
        return Layout.EA if self.alignment > 1 else Layout.E0
```

做成派生属性而不是字段，`PostDispatchResult` 的键不变，编译边界与逐位验证都不受影响；GEMM 库用它做后端能力声明与配置期过滤（GEMM 文档 §3.1、§4.1）。

### 2.2 六段式：签名基本不动

```python
dispatch_preprocess(hidden_states, topk_ids, topk_weights, async_op, tokens_per_expert=None, layer_state=None)
dispatch(pre_dispatched, topk_weights, async_op)
dispatch_postprocess(pre_dispatched, dispatched, async_op) -> PostDispatchResult
combine_preprocess(hidden_states, pre_dispatched, dispatched, post_dispatched, async_op)
combine(pre_dispatched, dispatched, post_dispatched, pre_combined, async_op)
combine_postprocess(pre_dispatched, dispatched, post_dispatched, pre_combined, combined, async_op)
```

只有第一段多了两个可选参数：`tokens_per_expert`（Router 的逻辑计数，MoonEP planning 用）与 `layer_state`（本次调用的 `EPCall`，MoonEP 把它的 invocation token 放这里）。后面五段通过 `pre_dispatched` 拿到它们，不再逐段传。每个 dispatcher 声明 `caps: DispatcherCaps`（对齐、是否 host 计数、是否静态形状、是否 FP8 dispatch 等），配置期校验与调度决策只看它。

### 2.3 执行层：5 个层钩子 + 4 个模型级边界

```python
class LayerEPExecution(Protocol):            # 每层一个，默认 NoOp（identity，不建 autograd 节点）
    prepare_layer_inputs(inputs) -> (inputs, [EPCall])   # 层入口：放 Join / 生成 EPCall
    prepare_dispatch(call, hidden, topk_ids, topk_weights) -> (hidden, topk_ids, topk_weights)  # 重路由、附着
    prepare_experts(call, batch) -> batch                 # 填 expert_weights 段 / 恢复 replica
    attach_after_experts(call, out) -> out                # 梯度桥 / 归约触发
    attach_after_combine(call, out) -> out                # 层出口

class EPExecutionRuntime(Protocol):          # 每模型一个，默认 NoOp
    bind_layer(layer_fqn, layer_idx, projections) -> LayerEPExecution
    validate_before_fsdp(fsdp_config); install_after_fsdp(fsdp_root, execution_order); close()
```

decoder 的调用点（`MoEDecoderLayer._moe_forward`，单批与多批同一份）：

```
inputs, calls = ep_exec.prepare_layer_inputs(inputs)
for each mb:   attention+router -> ep_exec.prepare_dispatch -> dispatcher.dispatch_preprocess(..., layer_state=call)
for each mb:   dispatch -> dispatch_postprocess -> ep_exec.prepare_experts -> [mark_dynamic] -> experts(batch)
               -> ep_exec.attach_after_experts -> combine_preprocess
for each mb:   combine
shared experts (通信在飞时算)
for each mb:   combine_postprocess -> ep_exec.attach_after_combine -> residual
```

交错顺序与 legacy 多微批完全一致（attention 全排完再 dispatch；experts A 排在 dispatch B 前；combine 全排完再 postprocess）。`MoE` 在构造末尾对每个 MoE decoder 层（含 MTP）调 `runtime.bind_layer`，在 `fully_shard` 前后调 `validate_before_fsdp` / `install_after_fsdp`。

### 2.4 专家 GEMM：`GroupedLinear.forward(x, rows, *, x_scale=None, weight=None)`

| 布局 | BF16 | FP8 |
|---|---|---|
| `alignment == 1`（legacy 与 DeepEP V2 未对齐） | 现有 Triton `m_grouped_gemm` / cutlass（不改） | 现有 AdaptiveGEMM per-tile（不改；新增接受预量化输入） |
| `alignment == 128`（DeepEP V2 对齐） | DeepGEMM psum `m_grouped_bf16_gemm_{nt,nn}`；wgrad Triton `k_grouped_gemm` | DeepGEMM psum `m_grouped_fp8_gemm_nt` 前向与 dgrad；wgrad 用 AdaptiveGEMM 的 k-grouped dW（bf16 直出、device 计数），无 AdaptiveGEMM 时退回 DeepGEMM k-grouped（fp32 累加，慢 2 倍） |

选择在 `XTUNER_EXPERT_GEMM_BACKEND=auto|legacy|deepgemm`（`auto`：对齐布局且 DeepGEMM 可导入就用它），按 `rows.alignment` 这个 Python 常量分支，编译期折叠。`MoEBlock.forward(batch)` 是编译入口，DeepGEMM / AdaptiveGEMM 的调用都是注册过的 custom op，`fullgraph=True` 下前反向都能 trace。

### 2.5 对 legacy 的影响面与回归方法

- 行为面：legacy 路径的数值逐位不变（跨树对比：在上游原样树上 dump `MoEDecoderLayer` 的输出 / dX / 全部参数梯度，在新树上比）；速度不变（GLM-5.2 6347 vs 6261，Qwen3 mb1 1.798 s vs 1.788 s，噪声内）。最近一次完整回归（2026-09-19）：三级逐位对比 66 条全部 BIT-EXACT，其中 AGRS 用分组路由器（16 专家、top-k 8）在 8 卡上跑；现有单测 14 通过 1 跳过；新增单测（契约、8 条 kernel 路径、DeepEP V2 六模式与 CUDA graph / compile）25 通过，单机无 GIN 网卡时要带 `EP_DISABLE_GIN=1`。
- 接口面：`MoEBlock.forward` / `GroupedLinear.forward` 的签名变了（内部 API，仓内调用点只有 decoder）；`PostDispatchResult` 加键（纯新增）；decoder 单批路径在 EP>1 时改为异步通信（与多批一致，#2056 也这么做，逐位验证过）。
- 建议框架组的回归：对 commit 1 单独跑全量回归（它不引入新依赖），parity 工具可以扩到任意模型族（换 `build_layer` 里的配置即可）。

### 2.6 三个类型的来历与实现程度（讲解问答整理）

讲解时最常被问的是"哪些是沿用的、哪些是新设计的、新设计的落地到什么程度"。按类型列：

| 类型 | 来历 | #2056 里有没有 | #2050 里有没有 | 最终态实现程度 |
|---|---|---|---|---|
| `ExpertRows` | 新设计（第 1 个提交落地） | 没有。其 `PostDispatchResult` 是 `hidden_states + tokens_per_expert + expert_weight_layout`，行布局仍是 legacy 的隐含假设 | 没有。加的是 `tokens_per_expert_cpu`，被吸收为 `rows.host_counts` | 全部实现：`rows_from_counts / rows_from_psum / rows_psum / check_rows`，`GroupedLinear` 按 `alignment` 选后端 |
| `ExpertWeights` | 从 #2056 的 `ExpertWeightLayout(trainable_weights)` 扩展 | 有前身 | 没有；对应物是 `_ultra_ep_replica_*` 模块态 + `select_ultra_ep_slot` | **只实现了 #2056 的语义**：`_call_local_weights`（`moe_decoder_layer.py:249`）只接受 `None` 或恰好 1 段，多段抛 `NotImplementedError`；`first_slot / backward_read / grad` 三个字段没有任何消费者；#2050 的 dual-base kernel 未移植 |
| `EPExecutionRuntime` | 沿用 #2056（`dispatcher/__init__.py:33-91`），四个方法名与 `MoE` 上的调用位置原样 | 有 | 形状相似的 `UltraEPManagerProvider`（`install_after_fsdp(fsdp_root, targets)`），但 HEAD 从未接线 | 实现；唯一实质改动是 `bind_layer` 返回 `LayerEPExecution` 而不是 dispatcher（#2056 返回 `MoonEPDispatcher`，NoOp 返回 `None` 让工厂回退），并多收 `layer_idx` |
| `LayerEPExecution` | 新设计 | 只有一个钩子的前身：dispatcher 基类上的 `prepare_layer_inputs`（`base.py:122`） | 四个 `autograd.Function`（`moe_decoder_layer.py:83-139`）加 19 处 `if ultraep` 分支 | 实现；默认 `NoOpLayerEPExecution` 是 identity，不建 autograd 节点 |
| `EPCall` | 新设计 | 前身是 `object \| None` 的不透明 token | 前身是 `virtual_layer_id` 整数 | 实现 |
| `Layout` | 新设计（2026-09-20 加入第 1 个提交） | 没有。两个 PR 的 GEMM 都只认 `tokens_per_expert`，即隐含的 E0 | 没有 | 实现：枚举 E0 / EA / R / M 加 `ExpertRows.layout` 派生属性，`test_expert_rows.py` 三条断言；原因见下 |

**`ExpertRows` 为什么必须有。** legacy 只传一个 `tokens_per_expert`，它隐含三个假设：段紧密连续无 padding；buffer 总行数等于计数之和；第 i 段起始行等于前 i 个计数的累加。框架的 permute / unpermute 就是把任意通信库的输出整理成满足这三个假设的形状。DeepEP V2 的 `do_expand` 输出三个都不满足：段按 128 对齐有 padding 行；静态模式下 buffer 是最坏容量，尾部有不属于任何专家的行；DeepGEMM psum 布局要的是每段真实结束行的前缀和而不是计数。只有两条路：框架再 permute 加 pad 揉回 legacy 形状（这正是 v1 路径的开销，去掉后 Qwen3 +8.2%），或者把布局显式描述出来让 GEMM 直接吃。一个细节：`PostDispatchResult.tokens_per_expert == rows.compute_counts` 而不是 `valid_counts`，老 kernel 在对齐布局上会把 padding 行也算一遍，靠 `padding_zeroed` 保证无害，换取老 kernel 一行不改。

**为什么显式定义 `Layout`。** E0 与 EA 已经能由 `alignment` 隐含表达，单看契约不需要枚举。显式定义它有四个理由。一是算子侧要用它做能力声明与配置期过滤（GEMM 文档的 `GemmCaps.layouts` 与 `GemmSpec.layout`），没有一个可枚举的名字，判断就会散成各处的 `if rows.alignment == 128`。二是 R 与 M 是真实存在、`ExpertRows` 又表达不了的两个家族：DeepEP v1 与 all2all 的原始接收布局是 R，decode 的 masked GEMM 是 M。先在枚举里占位，等第一个需要它们的后端出现再加 `gather_idx` / `masked_m` 字段，届时不用再改公共类型的名字。三是做成派生属性而不是新字段：`PostDispatchResult` 的键不变，`torch.compile` 的固定键约束与已有的逐位验证都不受影响。四是词汇统一：契约、GEMM 文档、各库对照表从此用同一套名字，不再各叫各的。

**`ExpertWeights` 与 `ExpertWeightLayout` 的异同。** 相同：都是 `PostDispatchResult` 里的 call-local 权重槽；默认值表示用模块自身参数；`GroupedLinear` 有就用没有就取参数，kernel 不改；dW 走 autograd 回 alias 张量而不是 `nn.Parameter`（MoonEP 的 `_MoonEPExpertGradBridge` 原样沿用）；MoonEP 的 `(w13, w2)` 对映射到 1 段是零拷贝。不同：一对张量变成带 `first_slot` 的多段；每个投影多了 `backward_read`（autograd 存的那份到 dgrad 时还有效吗）和 `grad`（dW 写 autograd 还是外部 buffer、overwrite 还是 add）；由执行层钩子 `prepare_experts` 填而不是 dispatcher 内部填，这样 UltraEP runtime 才能给 DeepEP V2 dispatcher 的批附加权重；改名是因为它描述的是权重归属不是布局。#2056 的 docstring 明确排除的两项（two-segment storage、direct-output WGrad）恰好是 UltraEP 需要的两项，最终态把它们从"不支持"改成"类型能表达、首期不实现"。现在就加字段的理由：`PostDispatchResult` 键要固定才能过 `torch.compile`，签名一次定好才能止住三个 PR 各改一次 `MoEBlock` 的漂移；等 UltraEP 接入时再改公共文件要再过一轮全量回归。

**`LayerEPExecution` 的五个位置怎么来的。** 取两个 PR 需要碰 decoder 的位置的并集：

| 钩子 | #2056 MoonEP 的原型 | #2050 UltraEP 的原型 |
|---|---|---|
| `prepare_layer_inputs` | dispatcher 基类同名方法，MoonEP 用它放 `_MoonEPLayerGradJoin` 并生成 token | `_UltraEPGradReduceJoin.apply` 放层入口 |
| `prepare_dispatch` | 无 | `update_placement → sync_weights → reroute → _UltraEPGradReduceStart.apply` |
| `prepare_experts` | dispatcher 内部 `prepare_experts`：等 `weights_ready`、取 `[2B]` 视图、过 Bridge、填权重 | `bind_virtual_layer_slot` 加 replica restore |
| `attach_after_experts` | 无 | `_UltraEPWeightRestoreJoin.apply` |
| `attach_after_combine` | dW 发布 | 层出口归约触发 |

把 `prepare_layer_inputs` 从 dispatcher 挪到执行层是分层的关键一步：它在 #2056 里放 dispatcher 上，是因为 MoonEP 的 dispatcher 就是它的执行层；两层分开后，"层入口放 Join"显然属于执行层。默认钩子不建 autograd 节点，是跨树逐位相等能通过的前提；若默认实现是空的 `autograd.Function`，反向图就变了。

### 2.7 FSDP bf16 权重与固定地址：两个 PR 的解法与 allocator 的位置

**现状。** 上游 `7d377424` 还没有解耦：`moe.py:1302` 仍调用 `_replicate_other_params` 把 dense 参数按 EP 倍数复制，专家与 dense 共用一个 `fsdp_mesh`。`efsdp` 维度只存在于 MetaMoE 分支的 `DECOUPLE_EP_FSDP=1` 实现里。所以 `efsdp=1` 加 `dtensor_ep_only`（专家只 EP、不 FSDP）在当前树上不可用。

**两个 PR 都绕开了它**，思路相同：FSDP 仍是参数唯一 owner，在 unshard 边界做适配。解法不同，因为两家对地址的要求不同：

| | MoonEP #2056 | UltraEP #2050 |
|---|---|---|
| 对地址的硬要求 | 远端 rank 用 TMA 直读本 rank 的 home 权重，bf16 工作副本必须落在**对称 VA** 上且地址跨步固定 | 只需要 master 权重在本次层调用期间有一个有效的**本地**指针；副本住在 UltraEP 自己的 NVSHMEM 堆里 |
| 机制 | **direct landing**（`fsdp_vmm_landing.py`）：用 `types.MethodType` 覆盖 `FSDPParam` 的 `init_all_gather_outputs / alloc_all_gather_outputs / free_unsharded_param`，让 FSDP 的 all-gather 输出直接写进 MoonEP 用 `nvl_dist_alloc` 分配的固定 VMM landing，`free_unsharded_param` 改成空操作。两代 landing 按奇偶层交替，因为 FSDP 在本层计算时预取下一层的 all-gather | **read-only binding**（`fsdp_expert_binding.py`）：不改分配，只记录 `FSDPParam` 身份；每次 `sync_weights` 前读当前 unsharded 参数的 `data_ptr`，变了就刷新原生库的 master 指针池 |
| 代价 | 钉死 `torch == 2.12.1+cu132`（`_TARGET_TORCH_VERSION`，不符直接报错）；不支持 FSDP padding；拒绝 FP8 子类；两代 landing 常驻，reshard 不释放专家权重 | 每层一次指针刷新；同样 import 私有 FSDP2 类型，但对 2.9 的命名做了回退，本环境 torch 2.9.1 能跑 |
| 与契约的关系 | 全部在 `install_after_fsdp` 里，契约不感知 | 同 |

所以矛盾能解，两家分别靠 direct landing 和 binding 解了，且都装在 `install_after_fsdp` 边界内。这正是 v1.0 的 `ExpertWeights` 没有 `storage` 字段的原因：存储形态是各 runtime 私有的事，契约只描述"这次调用用哪块权重、反向怎么读、dW 写哪"。两家也不能共用一个方案：UltraEP 的 binding 对 MoonEP 不够（要对称 VA），MoonEP 的 landing 对 UltraEP 过重（master 不需要对称）。

**`ExpertParamAllocator` 的位置。** 它不是契约 v0 的内容，而是"解耦 FSDP 与 EP"那条线的后半段。要说明一点：即使做到 `efsdp=1`，FSDP2 在 world size 为 1 时仍走 `_fsdp_param_group.py:357` 的路径，把 sharded 参数拷贝进一份 unsharded 输出，地址仍归 FSDP 管。要真正得到"分配一次、终身固定"的地址，专家模块必须**不进 `fully_shard`**，用 `DTensor.from_local(Shard(ep))` 保住 checkpoint 语义；这意味着专家的 fp32 master、optimizer state、梯度归约都要走单独路径（MetaMoE 分支的 `_scale_and_reduce_grad_decoupled`）。前置是三步：解耦 mesh 上游化；`efsdp=1` 成为合法配置且专家跳过 `fully_shard`；然后才有 allocator 挂接点。它值得保留，但属于解耦提案的后期目标，收益写明：替换掉 MoonEP 那个钉 torch 版本的 FSDP 私有方法覆盖。`efsdp=1` 的显存代价（专家 fp32 master 与 Adam 不再按 efsdp 切分）对大 EP 可忽略，对单节点大 E 不行。

## 3. 从两个 PR 沿用了什么、改了什么

| 接缝 | #2056 MoonEP 的做法 | #2050 UltraEP 的做法 | 最终态 |
|---|---|---|---|
| experts 的输入 | `PostDispatchResult + expert_weight_layout` | `+ tokens_per_expert_cpu` | `PostDispatchResult{rows, hidden_scales, expert_weights}`；host 计数进 `rows.host_counts` |
| `MoEBlock.forward` | `(x, tpe, *, weight_layout)` | `(x, tpe, decoding, tpe_cpu)` | `(batch)`，一个对象止住签名漂移 |
| `GroupedLinear.forward` | `trainable_weight=` alias 走 autograd | 模块态 `_ultra_ep_replica_*` | `(x, rows, x_scale, weight)`；权重段从 `batch["expert_weights"]` 传入，不放模块态 |
| GEMM 协议 | `grad_weight_out` kw | `tokens_per_expert_cpu, replica_weight, replica_grad` 位置参数 | 统一吃 `rows`；dual-base kernel 是"2 段权重"时的实现细节（后置） |
| GEMM 选择 | env 二选一 | `XTUNER_GROUP_GEMM` + 可注入 | `XTUNER_EXPERT_GEMM_BACKEND` + 按 `rows.alignment` 能力匹配（沿用 #2050 的"显式名字"） |
| 模型级 runtime | `EPExecutionRuntime{bind_layer, validate_before_fsdp, install_after_fsdp, close}` | `UltraEPManagerProvider` | **沿用 #2056 的四个方法名**；`bind_layer` 改为返回层钩子而不是 dispatcher（dispatcher 由工厂建，runtime 与 dispatcher 解耦，UltraEP 才接得上） |
| 层钩子 | `dispatcher.prepare_layer_inputs`（Join 放层入口） | 4 个 `Function` + decoder 里 19 处分支 | `LayerEPExecution` 5 钩子，decoder 无条件调用；`prepare_layer_inputs` 从 dispatcher 挪到执行层 |
| 路由进 dispatcher | `+ tokens_per_expert, layer_state` | 传重路由后的物理 `topk_ids` | 沿用 #2056 的两个参数；重路由放 `prepare_dispatch` 钩子 |
| 路由权重乘法 | HEAD `8f3d0aab` 仍在 `Buffer.combine(hidden_scales_nvs=)` 内乘（`moonep.py:634-641`）；作者自评的方向是挪到 `combine_preprocess` | — | 统一在 `combine_preprocess` 乘（DeepEP V2 用融合 Triton kernel）；MoonEP 照此改后顺带去掉一项私有 API 依赖（§5.2） |
| 配置 | 平铺进 `MoEConfig` | `ultraep_cfg` 子对象 | 子对象：`deepep_v2_cfg: DeepEPV2Config`（同理 `moonep_cfg` / `ultraep_cfg`） |
| 生命周期关闭 | `TrainEngine.close()` → `runtime.close()` | 无调用者 | 协议里有 `close()`，engine 调用点待加（#2056 那两行） |
| `decoding` 参数 | 删除 | 保留 | GEMM 层删除（六段式保留） |

## 4. DeepEP V2 接入（`dispatcher/deepep_v2.py`）

### 4.1 输出布局：两种模式都已经是"按专家分段"的，框架不做 permute

DeepEP V2 的 `do_expand=True`：接收缓冲经库内的 copy epilogue 直接写成"每个 (token, 本地专家) 一行、按专家分段、段按 `expert_alignment` 对齐"的布局。`dispatch_postprocess` 零拷贝：收到的张量就是 `hidden_states`，`rows` 由 handle 里的 `psum_num_recv_tokens_per_expert` / `num_unaligned_recv_tokens_per_expert` 算出。`cpu_sync` 只决定 host 是否得到精确计数：

| | `cpu_sync=True` | `cpu_sync=False` |
|---|---|---|
| 输出布局 | 按专家分段（已"permute"） | 同左 |
| 输出行数 | 精确（host 等一次计数，与 v1 的 `num_recv_tokens_per_expert_list` 同级） | 最坏容量 = tokens/rank × EP × topk（EP4 下是期望的 4 倍） |
| 形状 | 动态 | 静态（可 CUDA graph） |
| 相对 v1 少了什么 | 框架的 permute / unpermute 两次全量 gather、`row_ids_map`、FP8 的 128 扩展 pad | 再少一次 host 停顿 |

所以 `cpu_sync=True` 就已经有收益：Qwen3 上仅去掉 permute（`alignment=1`，legacy kernel 不变）就 +8.2%，再换 DeepGEMM +11.0%。

### 4.2 `alignment` 与 GEMM 的配对

- `alignment=1`：BF16 走 Triton varlen（device 计数，容量尾部不读，静态模式可用）；FP8 走 AdaptiveGEMM（它从激活行数推 M、要求 ΣK == M，**不能吃静态容量**，配置期拒绝 `cpu_sync=False` + FP8 + `alignment=1`）。
- `alignment=128`：BF16 与 FP8 都走 DeepGEMM psum m-grouped（DeepGEMM 2.6 有 BF16 的 `m_grouped_bf16_gemm_{nt,nn}_contiguous`，psum 布局，不需要 permute 也不需要 pad kernel）；wgrad 见 §2.4。

### 4.3 FP8 dispatch

dispatch 前 1×128 per-tile 量化，NVLink 上传 e4m3 + scale（通信量约减半），接收后直接进 DeepGEMM FP8 psum GEMM。激活跨 autograd 时用 bf16 类型的 `Float8Tensor` 包装（autograd 会把梯度 cast 成输入 dtype，裸 fp8 边会把 dX 静默量化）。combine 与两条反向原语仍是 BF16。

### 4.4 反向

dispatch 的反向 = `combine(grad, handle, topk_weights=grad_probs)`；combine 的反向 = cached `dispatch(grad, handle)`。路由权重在 `combine_preprocess` 乘（库的 combine 是纯求和），融合 Triton kernel，梯度按行回传。

### 4.5 静态模式（`cpu_sync=0`）：根因、容量因子与现状

- **为什么第一轮跑不起来**：profile 显示静态模式一步里 DeepEP 的 combine 内核 10.4 s（动态 0.24 s），但 2 卡隔离微基准里静态只比动态慢 30%。host 侧一步有 268 次 `cudaEventSynchronize` 共 7.5 s：PyTorch 缓存分配器在缓存不命中时等待 `record_stream` 事件，这些事件排在自旋等待对端的通信内核后面；最坏容量（tokens/rank × EP × topk，EP4 下是期望的 4 倍）让每层 2 GB 的接收张量频繁不命中，一个 rank 在 host 上卡住，其余 rank 的通信内核跟着自旋。`EP_AVOID_RECORD_STREAM=1` 下 Qwen3 最坏容量也稳定（1.91 s，比动态慢 12%，峰值 +11 GB），GLM 容量 2.0 倍从 12 s/步回到 2.57 s（与动态持平）；但该开关要求调用方自己保证跨流张量的生命周期，反向里的临时梯度张量还没做持有，是正式启用前的必做项。
- **容量因子**：DeepEP V2 原生没有"容量小于最坏值"的接口，本地给 DeepEP V2 打了 46 行补丁（`dispatch(..., num_max_expanded_tokens, overflow_flag)`，copy epilogue 越界置 flag 丢行），框架侧 `DeepEPV2Config.capacity_factor`，溢出经 pinned 镜像 + event 延迟一次调用检测，不引入 host 停顿；溢出时报错要求调大因子（应用层的 drop / 重试策略以后再加）。这是提给 DeepEP 上游或算子组的需求原型。
- **Qwen3 mb1 扫描**（10 步）：1.5 倍第 4 步溢出；**2.0 倍 1.69 至 1.73 s，峰值 72 GB，无溢出**；2.5 倍开始抖；3.0 倍回到病态。动态模式 1.62 至 1.66 s。20 步的静态行见 §9 表。
- **CUDA graph**：契约层已验证静态段可捕获重放；接进训练步的收益上限按 GLM profile 估约 3.6%，排在容量因子和 `EP_AVOID_RECORD_STREAM` 的生命周期管理之后。

## 5. MoonEP 接入方案（对照 #2056 逐项）

| #2056 的部件 | 落到契约的位置 | 改动量 |
|---|---|---|
| `MoonEPModelRuntime`（bind_layer / validate_before_fsdp / install_after_fsdp / close） | 实现 `EPExecutionRuntime`；`bind_layer` 返回 `MoonEPLayerExecution`（不再返回 dispatcher） | 方法名已一致，改返回值 |
| `dispatcher.prepare_layer_inputs`（`_MoonEPLayerGradJoin` 放层入口） | `LayerEPExecution.prepare_layer_inputs` | 搬家 |
| invocation token 进 dispatch 阶段 1 | `dispatch_preprocess(layer_state=call)`，token 放 `call.plan`，后续阶段从 `pre_dispatched` 读 | 无需改六段签名 |
| `ExpertWeightLayout(trainable_weights)`（`[2B]` alias：前 B 行是 FSDP landing，后 B 行 duplicate chunk） | `prepare_experts` 填 `ExpertWeights(segments=(1 段 [2B]，backward_read="restore_before_dgrad"，grad=autograd),)`；`GroupedLinear(weight=)` 直接用 | `GroupedLinear` 已支持 1 段 |
| `_MoonEPExpertGradBridge` → `reduce_grad_bf16` → 共享 H → Join 一次发布 | alias 权重是 MoonEP 控制的 autograd 叶子，dW 天然进 Bridge；发布放 `attach_after_combine` 或 Join | 不变 |
| `rows` | `rows_from_counts([2B] 计数)`，`valid == compute`，`alignment=1` | 一行 |
| 路由权重乘法 | 挪到 `combine_preprocess`（与 DeepEP V2 同位置；HEAD 仍在 combine 内乘，见 §3） | 框架侧一次 `[NvS, H]` 逐元素乘，去掉 `hidden_scales_nvs` 依赖 |
| `comm_stream` / `enqueue` | dispatcher 内部 | 不变 |
| VMM landing / FSDP `FSDPParam` 三个方法覆盖 | `install_after_fsdp` | 不变 |
| `check_config` / `check_fsdp_policy` | `MoE.__init__` 校验 + `validate_before_fsdp` | 搬家 |
| `TrainEngine.close()` 调 `close_ep_runtime` | `runtime.close()`；engine 调用点补上 | 2 行 |
| `MoEConfig` 平铺的 `moonep_*` | `moonep_cfg` 子对象 | 配置整理 |

需要 MoonEP 方确认：`comm_stream` 实为当前流的语义；MoonEP API v3（`xtuner-integration` 分支 d4494473，私有）与公开版的差异（§5.2 列了逐项对照）。

### 5.1 改动量的真实构成

#2056 对上游的 diff 是 31 个文件、+2831/−137。上表的"搬家、不变、一行"只对接缝成立，按三类看：

- **后端本体约 1900 行，原样搬过来**：`moonep.py` 1045、`moonep_workspace.py` 539、`fsdp_vmm_landing.py` 239、`moonep_capability.py` 106。契约不碰这些。
- **公共文件改动约 500 行，契约吸收一部分**：

| #2056 改的公共文件 | 行数 | 契约下的归宿 |
|---|---|---|
| `base.py`、`moe_group_linear.py` | 60 | 已被契约提交覆盖 |
| `ops/moe/cuda/route_weight.py` | 70 | 最终态已有 `row_scale`，二选一 |
| `group_gemm.py`、`group_gemm_cutlass.py`、两个 `k_grouped_gemm_TMA*` 的 `grad_weight_out` | 约 180 | 对应 `GradBinding(path="external")`，最终态未实现，见下 |
| `router/greedy.py`、`noaux_router.py`、`protocol.py` | 39 | 修 `topkens_per_expert` 拼写、histc 留在 GPU；契约的 `dispatch_preprocess(tokens_per_expert=)` 要它，最终态没改 router，**仍需要**，建议单独成小 PR |
| `utils/fsdp.py`、`dtensor.py`、`interleaved_shard.py`、`train/trainer.py` | 37 | VMM landing 对 FSDP 的钩子，不变 |

- **一项被"搬家"掩盖的工作**：`grad_weight_out` 让 wgrad kernel 直接把 dW 写进 `[2B]` alias 的梯度区，省一次拷贝。契约里这是 `GradBinding(path="external", buffer, write_op)`，但最终态的 `_call_local_weights` 只读 `.value`，external 路径没实现。首期二选一：用 autograd 路径，dW 多拷贝一次进 H；或把 external 路径实现掉，UltraEP 的 replica 段同样要用，建议在两者之前做一次。

### 5.2 对 MoonEP 私有 API v3 的依赖

契约改的是 XTuner 侧调用点，不改 #2056 对 `moonep` 包的调用，所以**按 §5 接入依然依赖 v3**。`moonep.py:37-59` 要求 `moonep.XTUNER_INTEGRATION_API_VERSION == 3`；`xtuner-integration` 分支 `d4494473` 在公开仓库不存在。按调用点与公开 `master`（`EP/MoonEP` @ `0f385f0`）对照：

| 依赖项 | 公开 master 有没有 | 契约下的出路 |
|---|---|---|
| `combine(hidden_scales_nvs=)` 在 combine 内乘路由权重（`moonep.py:634-641`） | 没有（`api.py:881` 无此参数） | **契约直接消掉**：乘法放 `combine_preprocess` 用 `row_scale`，多一次 `[NvS, H]` 逐元素遍历，V2 已在付 |
| `reduce_grad_bf16(plan, local_grads=[2B], distributed_duplicate_grads=[R,B])`（`moonep_workspace.py:499`） | 没有，只有 fp32 的 `reduce_grad(full_*_grad, *_reduce_buffer)`（`api.py:1009`） | **设计上可绕**，与 UltraEP 补丁二同类：dW 走 fp32 reduce buffer，归约后 cast 成 bf16 给 FSDP。但公开接口吃 `[E+B]` 全表布局而非 `[2B]` 本地 alias，workspace 的梯度视图要重排，是最大的一块工程量 |
| `dispatch` 返回四元组 `(hidden_nvsh, route_weights_nvs, cu_seqlens, plan)` | 输入签名同形（`api.py:685`），返回形态**未核实** | 要拿到 `d4494473` 的 diff |
| `Buffer.__init__` 多 `explicitly_destroy`、`num_sms`，测试传 `B=`、`token_padding=16` | 未核实 | 同上 |
| `prefetch_weight(plan, projections=)` | 公开是 `full_gate_weight / full_up_weight / full_down_weight`（`api.py:816`） | 签名级差异；VMM landing 已把权重放进对称分配，改传全表视图 |
| `moonep._C` 的 `get_vmm_granularity / nvl_dist_alloc / nvl_release_mem_handle / nvl_dist_map`、`buffer._exchange_ipc_fds`（`moonep_workspace.py:225,285,323`） | **有**（`buffer.py:55`） | 不是 v3 专属，是下划线私有符号；风险是接口稳定性不是可得性 |
| `XTUNER_INTEGRATION_API_VERSION == 3` 门禁 | 没有 | 一个常量，随上面几项处理 |

两条路：等 MoonEP 方发布 v3 并钉版本，或按公开 master 移植（工程量最大，加上 H200 上 FA3 + micro2 没有吞吐收益，所以 MoonEP 排在三个后端最后）。不论哪条路，先做两件与 v3 无关的事：router 的 39 行单独合入；路由权重乘法挪到 `combine_preprocess`。"改动量小"和"依赖私有 API"是两个正交的维度：契约压缩的是 XTuner 侧接缝，通信库的 API 面在契约之外；判断一个依赖能否被契约消掉，看它是不是"框架本可以自己做的事"，路由权重乘法是，梯度归约不是。

## 6. UltraEP 接入方案（对照 #2050 逐项）

| #2050 的部件 | 落到契约的位置 |
|---|---|
| `UltraEPManagerProvider` | 实现 `EPExecutionRuntime`；`MoE._bind_ep_execution` 无条件绑定，修掉了 HEAD 里 provider 从未接线的问题 |
| decoder 里 4 个 `Function` 与 19 处分支 | 收进 `UltraEPLayerExecution` 的 5 个钩子：`GradReduceJoin` → `prepare_layer_inputs` / `attach_after_combine`；replica restore → `prepare_experts`；重路由后的物理 `topk_ids` → `prepare_dispatch` |
| `_ultra_ep_replica_weight / _grad` 模块态 | `ExpertWeights` 两段：master `[M]`（saved / autograd）、replica `[R]`（跨步 view，restore_before_dgrad / external overwrite → grad_reduce） |
| `num_experts = E + ep×R` | 直接给 DeepEP V2 dispatcher（虚拟专家，dispatcher 不感知均衡器） |
| `GroupGemmProtocol` 三个新位置参数 | `rows`（含 `host_counts`）+ 权重段；dual-base Triton kernel 在"2 段"时启用 |
| `XTUNER_GROUP_GEMM ∈ {te, triton, triton_dual, cutlass}` | 并入 `XTUNER_EXPERT_GEMM_BACKEND` |
| `ultraep_cfg: UltraEPConfig` | 保留子对象形式 |
| 首期拒绝清单（recompute、MTP、bias、ExpertTP、FP8） | 放 `validate_before_fsdp` |

UltraEP 与 DeepEP V2 是正交的：前者是执行层（均衡器 + 权重段），后者是通信层。

### 6.1 通信底座：#2050 在 DeepEP v1 上，接 V2 后 UltraEP 自己的流量仍走 NVSHMEM

分两层看。**token 通信层**：`moe.py:206-207` 硬性校验，开了 `ultraep_cfg` 就必须 `dispatcher="deepep"`，即 legacy dispatcher 加环境里的 DeepEP 1.2.1（NVSHMEM + IBGDA）。**均衡层**：`ultraep/runtime.py:44,100` 直接构造 `ultra_ep.Manager`，从它取 replica 权重 / 梯度 buffer，调 `update_placement_sparse / reroute_sparse / weight_sync / grad_reduce`。原生库用 NVSHMEM 主机侧 API 建对称堆、`nvshmem_ptr` 取 peer 指针，kernel 内 TMA 直写（`csrc/kernels/` 只有 `load_compute.cu`、`api.cuh` 两个文件碰 NVSHMEM 符号）；planning 前的负载 all-gather 默认走 NVSHMEM 集合通信（`ultra_ep/runtime.py:24` `NVSHMEM_DISABLE_NCCL=1`）。"UltraEP 用 hybrid EP"的说法来自其 Megatron 参考脚本用了 HybridEP（DeepEP v1 分支），XTuner PR 里没有。`ultra_ep` 包在本环境未安装，#2050 在这里跑不起来。

按 §6 叠到 DeepEP V2 后，四条流量各归各：

| 流量 | 归谁 | #2050 现状 | 叠到 DeepEP V2 后 |
|---|---|---|---|
| token dispatch / combine | dispatcher | DeepEP v1：NVSHMEM + IBGDA | DeepEP V2：NCCL 设备 API，域内 LSA 对称指针，跨节点 GIN，NCCL ≥ 2.30.4 |
| 副本权重分发 `weight_sync` | UltraEP 原生库 | NVSHMEM 对称堆 | **不变** |
| 副本梯度归约 `grad_reduce` | UltraEP 原生库 | 同上 | **不变** |
| planning 前负载 all-gather | UltraEP 原生库 | NVSHMEM，可切 NCCL | **不变**，KB 级 |

换 dispatcher 只换第一行。UltraEP README:43 说 NVSHMEM 是唯一依赖、未来切 NCCL GIN，那是上游路线图，不是 XTuner 接入时做的事。并存要写清三件事：

- 运行时共存：UltraEP 默认 `NVSHMEM_DISABLE_NCCL=1`，NVSHMEM 自 bootstrap，不依赖 NCCL，与 NCCL 2.31 同进程不冲突；但 NVSHMEM 3.4.5 wheel 与镜像的 CUDA / 驱动要对齐，是又一个压在镜像升级上的依赖。
- 显存双份预留：V2 的 ElasticBuffer 是 VMM 对称窗口，UltraEP 的对称堆是 NVSHMEM 另一段，各按最坏容量预留。
- SM 争抢：V2 dispatch 占 4 至 6 个 SM，UltraEP `weight_sync` 在关键路径上拉满 SM，`grad_reduce` 默认 42 个 SM 与反向重叠；三者同飞的顺序进 §7 的 `MoESchedule` 讨论。

彻底去 NVSHMEM 有一条比等上游快的路：副本按构造永不出 NVLink 域（`nvshmem_ptr` 对域外 PE 返回空），对称堆可换成域内 IPC（`cuMemCreate` + fd 导出）或直接复用 NCCL 的 LSA 窗口；跨节点只剩 KB 级 all-gather，走 NCCL 的开关已有。主要改动在 `csrc/ultra_ep.cpp` 的堆分配与 peer 指针获取，是算子组体量的工作。

### 6.2 两个 native 补丁：契约消不掉，各有绕法

`patch/ultraep/` 的两个补丁改的是 `ultra_ep` 原生库内部（replica buffer 布局、grad_reduce kernel 元素类型），在契约之下：契约不管权重段的 `value` 从哪块内存切出、`GradBinding.buffer` 是什么 dtype。契约合入不依赖它们，只在 UltraEP 接入那一期出现。

- **`ultraep-multimicrobatch-slots.patch`**（8 文件，+158/−54）：上游 replica 权重 / 梯度 buffer 全层共用一套 `[R, numel]`，假设同一层两次调用串行。§2.3 的交错第一个循环对每个微批做 `prepare_dispatch`，UltraEP 的 `sync_weights` 落在这里，于是 mb0、mb1 的 `sync_weights` 都在 mb0 的 experts 之前发出，共用槽位时 mb1 覆盖 mb0；反向两个微批的 replica wgrad 也都落同一 buffer。补丁扩成 `[max_mb, R, numel]` 按 slot 取偏移。契约把槽位选择从模块态变成每次调用的第二段 `value`（按 slot 切的视图），但存储仍靠补丁。**绕法**：首期在 `validate_before_fsdp` 的拒绝清单加 `intra_layer_micro_batch > 1`，mb1 下退化成上游布局。把 `sync_weights` 挪到 `prepare_experts` 并加流事件也能绕，但 sync 进了关键路径，且违背 UltraEP "dispatch 前完成 weight_sync 以避免 NVLink 争抢"的建议，不推荐。
- **`ultraep-native-bf16-grad-reduce.patch`**（7 文件）：上游 `manager.py:65-67` 硬拒 `grad_dtype != fp32`（面向 Megatron 的 fp32 main_grad）；XTuner FSDP2 专家梯度是 bf16，#2050 让原生库在 bf16 staging 上做 `m += r` 再零拷贝绑成 `.grad`，并刻意不留 fp32 回退。**绕法**：走上游原生的 fp32 路径，replica 梯度 buffer 与 master staging 用 fp32，归约后一次 cast 写 `.grad`。需配合两处：dual-base wgrad kernel 对 replica 段 store fp32（Triton 里改一个 dtype）；master staging 每 rank 多一份 fp32 显存。数值不会更差，补丁的确定性路径本来就是 fp32 累加再一次舍入。`GradBinding.buffer` 不限定 dtype，两种路径都能表达。

建议：补丁二首期不打，走 fp32 加 cast，把改原生库的风险面减半；补丁一用拒绝 mb>1 绕开，多微批要不要支持看 V2 上 mb2 的收益（Qwen3 +4% 至 5%，GLM 主要靠 offload）；两个补丁的上游化仍要有人推，XTuner 不应长期维护 fork 的 UltraEP。MoonEP 和 UltraEP 在 bf16 归约上撞到同一堵墙，解法同构：两家上游都只支持 fp32 归约，两个 PR 都靠私有改动绕开。

## 7. 调度：现状与后续

现状：三类后端共用 legacy 的层内多微批交错（§2.3），dispatcher 只暴露事件；没有新的调度器。后续按需求分三层开口：

1. **微批交错顺序可插拔**：把 `_moe_forward` 里的顺序抽成 `MoESchedule` 对象，默认实现就是 legacy 顺序。工作量小，先做。
2. **段内流水**：MoonEP 或 SonicMoE 式的按 chunk 交错通信与 GEMM，dispatcher 需要自己执行 experts。契约上加能力位 `owns_segment`，这类后端提供 `run_segment(calls, hidden, routing, experts)` 替换 dispatch→experts→combine 三段，其余钩子不变。等第一个需要它的后端出现再开。
3. **跨层重叠**（combine 压到下一层 attention 之后）：走 `EPExecutionRuntime` 加层边界钩子。
4. **静态模式 + CUDA graph**：§4.5。
5. **容量因子 / 溢出**：需要通信库配合（DeepEP 需求或算子组 kernel）。

## 8. 关于"新路径"还是"原地"

专项当初的结论是"新接口走新的分支路径，完全不影响以前的"。最终态在两个层面上分别处理：

- **通信库与 kernel 层：完全做到了。** DeepEP V2 是新文件，legacy dispatcher / kernel 一行不改；MoonEP、UltraEP 将来也是新文件。
- **调用点层（decoder / `MoEBlock` / `GroupedLinear` 签名）：不可能既有统一接口又不改。** 不改调用点，三个后端就要各维护一份 decoder（第一版并行路径 `MoEDecoderLayerV2` 就是这样，等于把 `moe_decoder_layer.py` 复制一份），而 #2056、#2050 本身也都改了 `moe_decoder_layer.py`、`base.py`、`moe_group_linear.py`。

所以取舍是：把"公共文件改动"压缩成一个**无行为变化、可逐位验证**的提交（commit 1），配一个跨树对比工具作为回归证据，请框架组对它单独跑全量回归；新后端从 commit 2 起只加文件。并行路径的实现保留在分支 `archive/deepepv2-parallel-path`，如果框架组仍要求先以并行路径合入、契约后补，代码是现成的，但两份 decoder 的维护成本会一直在。

## 9. 基准现状与待办


- Qwen3 BF16：去 permute +8.2%，再加 DeepGEMM 共 +11.0%（mb1）；mb2 +4% 至 +5%。
- 静态模式（容量 2.0 倍期望）：Qwen3 mb1 1.70 s、mb2 1.72 s，峰值显存 72 GB（动态 69 GB），无溢出，loss 一致；GLM 在 100 GB 显存水位下仍是 12 s/步，加 `EP_AVOID_RECORD_STREAM=1` 后 2.57 s 与动态持平（峰值 107.6 GB），正式启用前要补反向临时张量的生命周期持有（§4.5）。
- 峰值显存（rank 0 `max_memory`，GB）：

| 配置 | legacy deepep(v1) | V2 legacy kernel | V2 DeepGEMM 动态 | V2 DeepGEMM 静态（容量 2.0 倍） |
|---|---:|---:|---:|---:|
| Qwen3 mb1 / mb2 | 68.8 / 68.7 | 69.0 / 68.7 | 69.0 / 68.8 | 72.0 / 71.6 |
| GLM mb1 | 99.8 | 100.1 | 100.0 | 104.6（+`EP_AVOID_RECORD_STREAM` 107.6） |
| GLM mb2 + offload | 103.6 | — | 103.7 | 105.9 |

  动态模式 V2 比 legacy 多不到 0.2 GB（第一轮并行路径实现曾多 9 GB，是 fp32 临时张量，已消掉）；静态模式多 3 至 5 GB，是容量预留。
- 两个新 Triton kernel 的适用范围：`row_scale` / `row_scale_bwd`（`ops/moe/cuda/triton_kernels/row_scale.py`，路由权重逐行乘）是 DeepEP V2 所有模式都用的，包括 BF16，它替代 legacy unpermute 里融合的那次乘法；`trans_quant_kgrouped`（`float8/triton_kernels/`）只在 FP8 且没装 AdaptiveGEMM 时的 DeepGEMM k-grouped wgrad 回退路径用。
- V2 DeepGEMM 不带 FP8 dispatch 的 GLM mb1 行两次重跑都带周期性毛刺（均值 2.65 s，中位数 2.56 s，20 步里 2 步 3.0 至 3.5 s）；按中位数与其它 V2 行同速，summary 表因此加了中位数列。
- GLM-5.2 FP8：mb1 +1%（MoE 段占比小，FP8 通信在单机 NVLink 下不额外提速）；mb2 + offload 从 4589 到 6264（legacy 在显存上限附近抖动，V2 少了 permute 的临时张量后不再抖动，待 profile 证实）。
- 待办：`EP_AVOID_RECORD_STREAM=1` 的张量生命周期持有（让 GLM 静态模式与最坏容量都可用）；把容量因子 / 溢出信号提给 DeepEP 上游；静态模式接 CUDA graph；`cudnn_dsa` 后端的 GLM 一份；`MoESchedule` 抽象；engine 的 `runtime.close()` 调用点；`GradBinding(path="external")` 的 wgrad 直写路径（MoonEP / UltraEP 共用，§5.1）；#2056 的 router 39 行单独 PR；向 MoonEP 方索取 `d4494473` 的 diff 核对 `dispatch` 返回形态与 `Buffer` 构造参数（§5.2）；UltraEP 首期拒绝 mb>1、走 fp32 grad 路径（§6.2）；`DispatcherCaps` 加 `accepts_physical_ids` 位并在 `MoE.__init__` 校验；`ExpertParamAllocator` 与 `efsdp=1` 跳过 `fully_shard` 归入解耦提案的后期范围（§2.7）。

## 10. 源码锚点与验证命令

| 文件 | 内容 |
|---|---|
| [xtuner/v1/module/dispatcher/base.py](../../xtuner/v1/module/dispatcher/base.py) | 契约类型（含 `Layout`）、六段式基类、NoOp 执行层 |
| [xtuner/v1/module/decoder_layer/moe_decoder_layer.py](../../xtuner/v1/module/decoder_layer/moe_decoder_layer.py) | MoEBlock.forward(batch)、_moe_forward、bind_ep_execution |
| [xtuner/v1/module/grouped_linear/moe_group_linear.py](../../xtuner/v1/module/grouped_linear/moe_group_linear.py) | GroupedLinear(rows)、use_deepgemm |
| [xtuner/v1/float8/float8_gmm_tile_wise.py](../../xtuner/v1/float8/float8_gmm_tile_wise.py) | FP8 GroupedLinear(rows)、预量化输入 |
| [xtuner/v1/module/dispatcher/deepep_v2.py](../../xtuner/v1/module/dispatcher/deepep_v2.py) | DeepEPV2Config、DeepEPV2Dispatcher |
| [xtuner/v1/ops/comm/deepep_v2_op.py](../../xtuner/v1/ops/comm/deepep_v2_op.py) | ElasticBuffer 注册表与四个原语 |
| [xtuner/v1/ops/moe/cuda/group_gemm_deepgemm.py](../../xtuner/v1/ops/moe/cuda/group_gemm_deepgemm.py) | DeepGEMM psum BF16/FP8、wgrad |
| [xtuner/v1/model/moe/moe.py](../../xtuner/v1/model/moe/moe.py) | deepep_v2_cfg、ep_runtime 绑定、FSDP 边界 |
| [tests/module/dispatcher/test_expert_rows.py](../../tests/module/dispatcher/test_expert_rows.py) | 契约单测（CPU） |
| [tests/module/dispatcher/test_deepep_v2.py](../../tests/module/dispatcher/test_deepep_v2.py) | 六模式对照、CUDA graph、compile（2 卡） |
| [tests/module/test_moe_block_rows.py](../../tests/module/test_moe_block_rows.py) | 8 条 kernel 路径对 fp32 参考（单卡） |
