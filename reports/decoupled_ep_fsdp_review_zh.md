# FSDP / EP 解耦代码审查结论

> 状态更新（2026-09-03）：C1 已由 commit `5923dc1d`（`[Fix] Gather RL layer-wise weights with the submodule that owns them`）修复，并附 `tests/rl/test_weight_iterator.py::TestLayerBatchesGatherWithParamOwner` 回归测试；N4 已在上游 main 修复。W1、W2、N1–N3 与 L1–L3 gate 仍待处理。以下正文保持审查时原样。

## 1. 审查范围与总评

审查范围是 `b934f462` 之后、截至 `56021e3e127362252a5dd0e36188583eca35f3c0` 的五个提交：

1. `2098db9e`：固定 legacy MoE EP/FSDP mesh 与 placement 基线；
2. `3055e4ce`：实现 dp2ep mesh、两级 FSDP、梯度归约和 L1；
3. `6fee4dd3`：补 efsdp/HSDP 的 L2 验证；
4. `2d667775`：补 FP8、HF/DCP、RL weight sync 和 L3；
5. `56021e3e`：在 torch 2.9 上复验，并补 GLM-5.2-30B 报告。

当前 HEAD 在目标提交之后还有两个仅修改 `reports/GLM52.md` 的文档提交；产品代码与目标提交相同。本文不把那两个提交的 allocator-counter、GC/autotune 结论倒算进本次审查证据。

**总评：核心 SFT 训练实现方向正确，GLM-5.2 的目标配方达标；整个改动目前应是 `REQUEST_CHANGES`，不建议按“所有外围路径均已支持”直接合入。** 阻塞项是 compose MoE 在 IPC/Turbomind RL 分层权重同步时选错 gather owner/mesh，会发送不完整权重。除此之外，还有一个自定义 `fp32_keys_pattern` 命中 routed expert 时的梯度正确性边界，需要修复或显式拒绝。

可以分别下结论：

- **核心 BF16/AdamW SFT：有条件通过。** mesh、placement、两级 FSDP、dense 去复制和单机数值语义证据较强。
- **目标 GLM-5.2 配方：通过。** 8×H200 上 EP4/EP8 的显存目标均达到；EP4 耗时持平，EP8 因避开 allocator ceiling 显著变快。
- **HF 权重与 DCP：已验证的同步 BF16/tiny 范围通过。** step 0 权重导出 bit-exact；DCP 同布局续训和跨布局 reshard 数值连续。不能扩展为 config、异步保存、GLM/MTP、FP8 checkpoint 均已验证。
- **FP8：训练 smoke 通过，严格等价未证明。** padding/reduce-mesh 设计合理，medium 和真实 GLM 均跑通；FP8+HSDP、FP8+checkpoint 未测，报告的“完全处于同配置重复噪声内”不成立。
- **RL：plain MoE 的 mesh 选择逻辑成立，但 compose MoE 的 Turbomind IPC 路径有确定性错误。** ProduceBatchResult、RoutedExperts 对象语义和 Ray concurrency 不受本改动影响。
- **HSDP+EP：单机缩比拓扑与数学路径通过，真正双节点 collective 尚未验证。**
- **legacy：开关默认关闭、旧分支隔离和 placement 回归成立；没有做 `b934f462` 与新代码 `decouple=False` 的端到端逐张量 bitwise 对照，因此只能说“高可信兼容”，不能说“已实验证明 bit-exact”。**

实现导读见 [decoupled_ep_fsdp_implementation_zh.md](decoupled_ep_fsdp_implementation_zh.md)。

## 2. Issues

### Critical

#### C1. compose MoE + IPC/Turbomind 的 RL 分层权重同步会发送不完整权重

位置：

- `xtuner/v1/rl/weight_update/weight_iterator.py:36-41,139-187`
- `xtuner/v1/model/compose/base.py:105-127`
- `xtuner/v1/model/base.py:1907-1937`
- `xtuner/v1/utils/load_spec.py:880-913`

触发条件：compose 模型的 language tower 是 MoE，rollout 使用 IPC + Turbomind，并开启 decoupled；`efsdp > 1` 时 expert 必现，HSDP 下 dense 也会命中。

故障机制：

1. `iter_layer_batches()` 从 `language_model` 取参数和 `LoadSpec`；
2. 内部 `get_params()` 却调用外层 compose `model._fsdp_foreach_allgather()`；
3. 外层 compose 的 `fsdp_mesh` 是 world mesh，`expert_fsdp_mesh` 仍为 `None`，不是 language tower 的 `dp_shard/efsdp` mesh；
4. `LoadSpec.plan_hf_save(gather_process_group=...)` 会保留所有不属于传入 group 的 shard，因此 child expert 的 efsdp shard没有被 gather；HSDP child dense 的 `dp_shard` 也可能被保留；
5. Turbomind 最终收到 rank-local 碎片，可能直接 shape mismatch，也可能静默更新成错误权重。

无 HSDP 的 dense 因 `dp_shard == world`、以及 `efsdp == 1` 的 expert 会偶然正确，所以普通 smoke 很容易漏掉这个问题。现有 `tests/rl/test_weight_iterator.py:17-153` 只覆盖 `iter_hf_batches()`，没有覆盖该调用链。

建议修复：让 `get_params()` 显式接收参数所属 owner；language 参数调用 `language_model._fsdp_foreach_allgather()`，vision/projector 调各自 owner。更稳妥的长期方案是让 gather 直接由 `LoadSpec` 的 shard metadata 决定，不依赖 receiver 的 mesh 属性。至少新增：

- compose + decouple + `efsdp=2` + IPC/Turbomind；
- compose + HSDP + IPC/Turbomind；
- 对每批 tensor 的 HF shape、key 和完整值做断言。

### Warning

#### W1. `fp32_keys_pattern` 命中 routed expert 时会漏掉 efsdp/HSDP replica 梯度归约

位置：

- `xtuner/v1/model/base.py:675-693`
- `xtuner/v1/model/moe/moe.py:1583-1628`，关键分支在 `1599-1601`

expert 在 EP 初始化后已经是 `DTensor(Shard(0) on ep)`。若 HF FP32 pattern 命中它，`BaseModel._fully_shard()` 只把它放入 `ignored_params`，不会再沿 `expert_fsdp_mesh` 分片；梯度路径看到名字含 `.experts` 后只做 `/ep` 并立即 `continue`，没有跨 efsdp 或 HSDP replicate 做 all-reduce。`efsdp > 1` 或 `replicate > 1` 时，各副本会发生 optimizer 更新分叉。

当前内置 Qwen3.5 pattern 只匹配 linear-attention 的 dense 参数，GLM-5.2 没有该 expert pattern，所以现有报告未触发。它仍是公开配置能力上的正确性问题。

建议短期显式拒绝 decoupled 下 FP32 pattern 命中 `MoEBlock`；完整方案应把 ignored expert 表达为 `Replicate(replicate), Replicate(efsdp), Shard(ep)`，对 `replicate × efsdp` 求均值，再做 `/ep`。需补 efsdp、HSDP 和一步 optimizer 的定向测试。

#### W2. expert 分类使用参数名，和 FSDP 包装使用模块类型，不是同一份事实来源

位置：

- `xtuner/v1/model/moe/moe.py:1599`
- `xtuner/v1/model/moe/moe.py:1631-1649`

FSDP 通过 `isinstance(MoEBlock)` 找 routed expert，梯度缩放却依赖 FQN 中包含 `.experts`。未来模型若把 `MoEBlock` 挂在其他名字下，会被 expert mesh 分片却不做 `/ep`，梯度放大 `ep` 倍；反方向也可能误匹配。建议在包装时记录 expert parameter identity/FQN，梯度、FP8、save/RL 均复用这一分类。

#### W3. FP8 数值结论写得比数据更强

位置：

- `reports/L3.md:73-89,102`
- `reports/L3.md:91-93`

报告中 C vs B 的 20-step 最大相对 loss 差是 `6.59e-4`，而三组“同配置重复”的最大值最高为 A/A2 的 `4.84e-4`，B/B2 和 C/C2 只有 `1.46e-4`、`1.04e-4`。所以当前数据支持“20 步稳定、差异较小、量级接近 FP8 拓扑/运行波动”，不支持“decoupled-vs-legacy 已落在同配置重复噪声内”的严格表述。

此外 medium FP8 的 warm step time 是 B `242 ms`、C `251-258 ms`，约有 4%–7% 回退；不能把 GLM EP8 的巨大提速泛化为 decoupling 总会加速。GLM 的提速主要来自释放 dense 复制、跨过显存/allocator ceiling，而非通信本身天然快 7 倍。

#### W4. HF step-10 的 “allclose / ≤1 bf16 ULP”没有由脚本证明

位置：

- `tests/model/run_decoupled_ep_fsdp_ckpt.py:72-103`
- `reports/L3.md:42-57,100`

`compare_hf()` 只统计 `max_abs/max_rel/mismatched`，没有执行 `assert_close`，也没有计算 ULP 距离；报告省略了脚本已经计算的 max-rel。step 10 相对 ep1 约 45M/108M 元素不相等，`max_abs=1.099e-3~1.221e-3` 本身不能推出“至多一个 bf16 ULP”。正确表述应是：

- step 0：807 keys、108,139,520 elements **bit-exact**，这是很强的 layout/HF mapping 证据；
- step 10：训练轨迹在不同 reduction order 下**数值接近但非 bit-exact**，需要定义 atol/rtol 或真实 ULP gate 后才能称 allclose。

#### W5. L1/L2/L3 是人工实验脚本，不是自动验收 gate

位置：

- `tests/model/run_decoupled_ep_fsdp_numerics.py:239-267`
- `tests/model/summarize_decoupled_ep_fsdp_numerics.py:37-90`
- `tests/model/run_decoupled_ep_fsdp_ckpt.py:72-103,143-216`

这些脚本主要写 JSON、打印统计，没有对 loss、grad、key、shape、HF diff、resume/cross-load 阈值做失败断言；即使数值严重回归，进程仍可能 exit 0。应把报告中的 acceptance 条件落成 GPU pytest/assert gate。

目标提交也未包含 L1/L2/L3 原始 JSON；`reports/GLM52.md@56021e3e:8-9` 声称保存的 `reports/glm52_pt29_logs/` 不在该 commit tree，且 `.gitignore:60` 忽略 `*.log`。因此 Markdown 数字可读，但仅凭提交不能独立审计或重算。

#### W6. HSDP、compile 和若干组合能力的证据边界需要收窄

位置：

- `reports/L2.md:5-9,20-38`
- `tests/model/test_decoupled_ep_fsdp_mesh.py:66-68`
- `reports/GLM52.md@56021e3e:11-34,156-179`
- `examples/v1/config/sft_glm5p2.py:52,107-114`
- `xtuner/v1/train/trainer.py:2165-2174`

具体边界：

- HSDP+EP 的训练只在单机 8 卡缩比拓扑实跑；16/64 rank 是 fake-PG placement，不执行跨节点 NCCL collective。可说“单机数学路径通过”，不可说“双节点生产验证完成”。
- 设计要求 L0 无 GPU 可进 CI，但整个测试文件在 CUDA 不可用时 skip，CPU CI 实际没有 gate mesh/config/placement。
- GLM 报告只记录 `MODEL_COMPILE=1`；示例配置的 `TORCH_COMPILE` 默认是 0，而 Trainer 会在它为 0 时强制关闭 `model_cfg.compile_cfg`。外部脚本可能同时设置二者，但目标提交没有精确命令或 config dump，单凭 commit 无法闭环证明 compile 确实开启。
- FP8+HSDP、Muon、CPU offload、FP8 checkpoint、GLM/MTP save/export、async HF/DCP 均未覆盖。
- empty-expert smoke 仅用短序列概率性制造空 expert，没有记录 route count 或断言 `M_e=0`。

### Nit / compatibility risk

#### N1. 新拓扑约束使用 `assert`，在 `python -O` 下会消失

位置：

- `xtuner/v1/config/fsdp.py:57-65`
- `xtuner/v1/model/moe/moe.py:1672-1677`

例如 `ep=8, hsdp_sharding_size=4, decouple=True` 在优化模式下可能绕过校验，进入非法 mesh shape。建议改成 Pydantic validator + `ValueError`，运行时也使用显式异常，并校验正整数。

#### N2. L0 对 PyTorch 版本覆盖小于项目声明范围

位置：

- `pyproject.toml:40`
- `xtuner/v1/model/moe/moe.py:1212-1244,1708-1715`
- `reports/decisions.md:131-139`

项目声明 `torch>=2.6.0`，两次 mixed-mesh FSDP 和私有 `_flatten(name)` 只在 2.8/2.9 报告中验证。建议补 2.6/2.7 CI，或为该 feature 明确更高的最低版本并 fail fast。

#### N3. checkpoint 测试的 `release()` 没有释放调用方引用

位置：`tests/model/run_decoupled_ep_fsdp_ckpt.py:106-109,169-176`。

函数里的 `del engine` 只删除形参，caller 的 `engine`/`resumed` 仍存活。tiny 数值结论不因此失效，但更大模型可能同时保留两套 engine，额外吃显存甚至 OOM。应在调用方 `del engine` 后再 GC/empty-cache，或让作用域真正结束。

#### N4. FP8 + plain MoE + Turbomind 的既有问题不属于本五个提交，但限制联合能力声明

位置：

- `xtuner/v1/rl/weight_update/weight_iterator.py:139-145`
- `xtuner/v1/model/base.py:1099-1118`

`_to_float8()` 的第三个参数要求 `list[bool]`，调用方传入的却是 tensor tuple，多元素 tensor 会触发 ambiguous-bool。该行在基线 `b934f462` 已存在，所以不是本改动回归；但在修复前不能宣称“FP8 + Turbomind RL 联合路径可用”。

## 3. 对齐判定

| 能力 | 审查判定 | 准确表述 |
|---|---|---|
| mesh / placement / 两级 FSDP | 通过 | 目标环境 L0 29/29；dense 与 expert placement 符合设计 |
| BF16 loss / grad | 较充分 | 单机 tiny/medium/GLM 训练语义对齐，不是逐 bit 对齐 |
| dense 显存去复制 | 通过 | local dense storage 符合 `1/ep` 预期，真实 GLM 峰值明显下降 |
| `efsdp > 1` | 单机通过 | C4/H22 和 GLM EP4 覆盖；无跨节点证明 |
| HSDP + EP | 部分通过 | 单机缩比训练通过；双节点 collective 未测 |
| FP8 | smoke 通过 | medium/GLM 能训；严格噪声结论、HSDP、checkpoint 未证 |
| HF 权重 | 指定范围通过 | tiny BF16 同步导出 step 0 bit-exact；config/async/GLM/FP8 未审验 |
| DCP | 指定范围通过 | model+AdamW 同布局 resume、C↔A/C4→H41 跨布局续训在既定噪声内 |
| RL weight sync | 不通过 | plain owner 逻辑可用；compose+Turbomind IPC 有 C1 |
| legacy bit-exact | 未完整证明 | 默认关闭与旧 placement 回归成立；缺 base-vs-branch 端到端 bitwise 对照 |
| compile / DeepEP / MTP / SP | smoke/报告支持 | 真实 GLM 跑通记录存在；compile 开关缺提交内完整配置快照 |
| ExpertTP | 不支持 | 代码显式拒绝，见 `xtuner/v1/model/moe/moe.py:1665-1666` |

## 4. 已复验项目

- `PYTHONPATH=. pytest -q tests/model/test_decoupled_ep_fsdp_mesh.py`：`29 passed`。
- `PYTHONPATH=. pytest -q tests/rl/test_weight_iterator.py tests/utils/test_load_spec.py tests/model/test_model_config.py`：`38 passed`。
- `git diff --check b934f462 56021e3e`：通过。

这些测试不会否定 C1：现有 RL 测试没有进入 compose/Turbomind 的 `iter_layer_batches()` 分支。

## 5. Verdict

**REQUEST_CHANGES**。

合入前最低要求：

1. 修复 C1，并添加 compose + efsdp/HSDP + IPC/Turbomind 权重完整性测试；
2. 修复或显式拒绝 W1 的 FP32 routed-expert 情况；
3. 把关键 L1/L2/L3 acceptance 条件变成会失败的测试断言；
4. 收窄 FP8、HF ULP、multi-node HSDP、legacy bit-exact 的文档表述，或补足对应证据。

若当前交付范围明确限定为“GLM-5.2 单机 SFT、现有内置 FP32 pattern、AdamW、非 HSDP”，核心实现已经达到可试用水平；不能据此放行 compose RL 或宣称完整生态已闭环。
