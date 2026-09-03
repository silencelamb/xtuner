# FSDP / EP 解耦实现说明

## 1. 一句话理解

旧实现把 EP 当成 FSDP mesh 的一条正交维度，结果 dense 参数在每个 EP rank 上复制；新实现把 data-parallel shard 拆成 `efsdp × ep`，dense 在两维 flatten 后的完整 `dp_shard` 上分片，routed expert 则保持 EP 分专家、再只沿 `efsdp` 做 FSDP。

核心公式：

```text
dp_shard = hsdp_sharding_size or world_size
replicate = world_size / dp_shard
efsdp = dp_shard / ep

root mesh = (replicate, efsdp, ep)
dense     = FSDP over flatten(efsdp, ep) = dp_shard
expert    = EP Shard(0) over ep + FSDP over efsdp
HSDP      = replicate 上复制并归约，dp_shard/efsdp 上分片
```

它解决的不是“把 expert 分得更多”这一件事，而是让 dense 的 FSDP 分片度不再被 `world / ep` 锁死。EP 越大时，旧实现越容易复制大量 dense 参数；新实现让 EP 和 FSDP 可以独立表达。

## 2. 三个具体拓扑例子

| 场景 | root mesh | dense | routed expert |
|---|---|---|---|
| 8 卡，EP=8，无 HSDP | `(1, efsdp=1, ep=8)` | 在 8 卡 `dp_shard` 上 FSDP，不再复制 8 份 | 纯 EP=8，efsdp=1 |
| 8 卡，EP=4，无 HSDP | `(1, efsdp=2, ep=4)` | 在 8 卡 `dp_shard` 上 FSDP | EP=4 后，再跨间隔为 4 的两卡做 expert FSDP |
| 16 卡，HSDP shard=8，EP=8 | `(replicate=2, efsdp=1, ep=8)` | 两个 HSDP replica，每个 replica 内 shard 8 | 每个 replica 内 EP=8；replica 间由 HSDP 归约 |

因此“EP=8 且 FSDP=8”不再是矛盾配置：对 dense 来说 FSDP shard size 是 8；对 expert 来说这一例的 efsdp 是 1，因为 expert 已经由 EP 切成 8 份。

## 3. 关键代码按执行顺序

### 3.1 配置入口与兼容开关

位置：

- `xtuner/v1/config/fsdp.py:48-65`
- `examples/v1/config/sft_glm5p2.py:107-115`

`FSDPConfig` 新增 `decouple_ep_fsdp=False`，默认值确保 legacy 路径不自动改变。旧的 “HSDP 时 EP 必须等于 1” 约束仅保留在 legacy；新路径改为要求 `hsdp_sharding_size % ep_size == 0`，因为必须存在整数 `efsdp`。

GLM 示例通过 `DECOUPLE_EP_FSDP` 和 `HSDP_SHARDING_SIZE` 环境变量打开新布局。

### 3.2 建立统一 root mesh

位置：`xtuner/v1/model/moe/moe.py:1651-1715`。

新分支建立 `(replicate, efsdp, ep)` root，保持 `ep` 为最内层连续 rank 维，以兼容原有 dispatcher/DeepEP 的 EP group 假设。随后从同一 root 派生：

- `ep_mesh = root[ep]`：token dispatch 和 expert ownership；
- `fsdp_mesh = flatten(efsdp, ep)`：dense 的完整 `dp_shard`；
- `hsdp_mesh = root[replicate, dp_shard]`：有 HSDP 时的 dense mesh；
- `expert_fsdp_mesh = root[efsdp]` 或 `root[replicate, efsdp]`：expert 的 FSDP/HSDP mesh。

所有子 mesh 来自同一 root 很重要：expert 参数最终需要同时保留 EP shard 和叠加 efsdp shard，LoadSpec/HF 保存也依赖这份可组合的 placement 描述。

ExpertTP 在新路径被显式拒绝：`xtuner/v1/model/moe/moe.py:1665-1666`。

### 3.3 两级 `fully_shard`

位置：

- `xtuner/v1/model/moe/moe.py:1193-1244`
- `xtuner/v1/model/moe/moe.py:1327-1360`
- `xtuner/v1/model/moe/moe.py:1631-1649`
- `xtuner/v1/module/decoder_layer/moe_decoder_layer.py:150-200,285`

旧路径先把非 expert 参数沿 EP 复制，再用 `world/ep` 的 FSDP；新路径跳过该复制。

每个 decoder layer 的处理顺序变为：

1. 找出 `MoEBlock`；该模块只包含 routed-expert grouped linears、activation 等 expert 计算；
2. 先在 `expert_fsdp_mesh` 上对 `MoEBlock` 做一次 `fully_shard`；
3. 再在 dense `fsdp_mesh/hsdp_mesh` 上包装整个 decoder layer；
4. FSDP2 会保留内层 expert wrapper，从而形成 expert 的 EP shard + efsdp FSDP shard，而其余 dense 参数只走完整 dp_shard。

MTP layer 使用同样的两级包装，并为内外两套 FSDP wrapper 都设置 forward prefetch，避免额外 all-gather 串行化。

### 3.4 梯度为什么要单独处理

位置：`xtuner/v1/model/moe/moe.py:1392-1395,1583-1628`。

最终目标是每类参数的 gradient 都等价于全 data-parallel world 上的均值：

| 参数类 | 谁完成归约 | 新路径额外动作 |
|---|---|---|
| dense FSDP 参数 | FSDP 在 `dp_shard` 上 reduce-scatter；HSDP 再跨 replicate all-reduce | 无 |
| routed expert | FSDP 在 `efsdp` 上归约；每个 expert 已看过整个 EP group 路由来的 token | 手工 `/ep` |
| FSDP ignored、全 Replicate 的 FP32 dense 参数 | 没有 FSDP 自动归约 | 按所有 Replicate mesh 维 flatten 后 coalesced all-reduce 并求均值 |

这也是新路径不能继续复用 legacy 梯度逻辑的原因：legacy dense 是 EP replicate，需要手工跨 EP 归约；新 dense 已在完整 dp_shard 上由 FSDP 处理，再做一次会重复归约。

当前实现有两个需修正的分类边界：`.experts` 字符串与 `MoEBlock` 类型不是同一事实来源；FP32 pattern 命中 expert 时会被 ignored，却仍提前走 `/ep` 分支。详见 review 的 W1/W2。

### 3.5 FP8 适配

位置：

- `xtuner/v1/model/moe/moe.py:1162-1176`
- `xtuner/v1/model/base.py:1002-1008`
- `xtuner/v1/float8/float8_handler.py:121-238,311-338`
- `xtuner/v1/float8/fsdp_utils.py:119-179`

tile-wise FP8 需要知道一个本地权重会被多少个 FSDP rank 切，以及 scale 的 reduce-max 应跨哪些 rank。解耦后 dense 和 expert 不再共享同一个答案：

- dense linear padding 按 `dp_shard` chunk 数计算，reduce mesh 的 rank stride 是 1；
- grouped-expert linear padding 按 `efsdp` chunk 数计算，reduce mesh 的 rank stride 是 `ep_size`；
- scale 预计算按 dense module type 和 expert grouped-linear type 分两次执行，分别用自己的 mesh mapping。

这部分实现与新 placement 是一致的。现有证据覆盖无 HSDP 的 medium FP8 和真实 GLM FP8；未覆盖 FP8+HSDP、FP8 checkpoint。

### 3.6 HF / DCP 为什么大部分不需要模型专用补丁

位置：

- `xtuner/v1/model/base.py:1035-1095,1358-1425`
- `xtuner/v1/utils/load_spec.py:721-770,880-913`
- `tests/model/run_decoupled_ep_fsdp_ckpt.py:111-216`

FSDP/EP 的两维 shard 会记录进 `LoadSpec`。同步 HF save 根据 shard 描述逆向 unshard，再调用模型已有的 HF key/fused-expert 映射；DCP 本身按 DTensor placement 保存并在载入时 reshard。因此本组提交没有为每个模型重写 `state_dict_adapter`，主要工作是让新 mesh 产生正确的 LoadSpec。

已证明的范围：

- tiny Qwen3-MoE BF16 在 A/C/C4/H41 四布局刚 `from_hf` 后导出，807 keys、108,139,520 elements 全部 bit-exact；
- DCP 包含 model + AdamW state，同布局 step-5 resume 后 loss 连续；
- C↔A、C4→H41 跨布局载入后继续训练，loss 在既定运行噪声内。

没有重新验证的范围：`config.json`/Transformers round-trip、vLLM/SGLang 字段合同、async HF/DCP、GLM MTP export、FP8 checkpoint、world-size 变化。step 10 的导出是数值接近，不是 bit-exact。

### 3.7 RL weight sync 改了什么，以及哪里出了问题

位置：

- `xtuner/v1/model/base.py:1907-1940`
- `xtuner/v1/rl/weight_update/weight_iterator.py:36-54,124-210`

Turbomind 的 layer-wise 更新只想 gather FSDP shard，同时保留 EP-local expert slice。新 helper 会查看每个 `LoadSpec`：如果它包含 expert efsdp group，就用 expert group gather；否则用 dense fsdp group。这对参数 owner 就是当前 MoE model 的 plain 模型成立。

compose 模型却从 child language tower 取 LoadSpec、用 outer compose 的 mesh 做选择，导致 review C1。修复前，RL 结论必须限定为 plain MoE；compose MoE + IPC/Turbomind 不可用。

按 RL 专项语义检查：

- ProduceBatchResult：没有改 rollout result/status/reward/timing/counter 聚合，不受影响；
- RoutedExperts：没有改 RL RoutedExperts ownership 或 shared-store 引用，不受影响；
- Ray concurrency：没有改 actor method、concurrency group 或控制 RPC，不受影响。

这里的错误是 collective owner/mesh 选择，不是 Ray 并发问题。

## 4. 实际效果与“是否对齐”

### 4.1 BF16 和单机拓扑

依据 `reports/L1.md:24-50,75-100`、`reports/L2.md:20-38,147-154`：

- EP8 tiny：C vs legacy B 的 50-step loss 最大相对差 `1.9e-5`；step-0 expert grad-norm 最大相对差约 `1.6e-5`；
- medium 3.4B：dense local storage `836 -> 105 MiB`，peak allocated `11861 -> 8203 MiB`，steady step `170 -> 158 ms`；
- C4、H41、H22 分别覆盖 `efsdp>1`、HSDP replicate、两者同时存在，单机 loss/grad 对齐；
- 这说明训练语义对齐，但不是逐参数、逐 step 的 bit-exact。不同 reduction order 会使轨迹逐步分叉，报告后期个别 per-param grad-norm 相对差可到百分比级，而总体 loss 仍在预设噪声量级。

### 4.2 GLM-5.2-30B 目标配方

依据目标提交 `reports/GLM52.md@56021e3e:11-34,38-113`：

| EP | legacy | decoupled | 结论 |
|---|---|---|---|
| 4 | `1.688 s / 99.50 GiB` | `1.658 s / 83.49 GiB` | 峰值少 16.0 GiB，耗时持平 |
| 8 | `11.835 s / 120.03 GiB` | `1.636 s / 76.32 GiB` | 峰值少 43.7 GiB；40-step 仍保持约 1.62 s |

EP8 的 7 倍级加速不能理解为新 collective 本身快 7 倍。目标提交中的合理解释是 legacy 已把 reserved memory 推到约 132.5 GiB ceiling，allocator 频繁 release/retry；decoupling 释放 43.7 GiB 后离开该慢区。目标提交当时把它标成“与所有运行一致但尚未用 counters 证实的推断”；后续两个文档提交才增加 counter/GC-autotune 证据，不属于本次五提交。

同一目标 profile 还实际组合了 DeepEP、tile-wise FP8、MTP 和模型 compile 配置，并记录 loss/grad-norm 贴近；但提交内缺精确命令/config dump，compile 是否同时满足 `MODEL_COMPILE=1` 与 `TORCH_COMPILE=1` 无法独立复核。

### 4.3 对齐结论的正确措辞

| 项目 | 是否对齐 | 限定 |
|---|---|---|
| mesh/placement | 是 | L0 29 passed，含 16/64 fake-PG shape |
| BF16 训练 | 是，数值语义对齐 | 非 base-vs-branch bit-exact；单机证据 |
| GLM 目标训练 | 是 | 已测 8×H200 配方；性能不泛化到所有模型/shape |
| HF step 0 权重 | 是，bit-exact | tiny Qwen3-MoE BF16 同步导出 |
| DCP resume/reshard | 是，数值连续 | tiny + AdamW；未覆盖 async/FP8/GLM/world-size change |
| FP8 | smoke 对齐 | “严格处于重复噪声内”未证明；HSDP/ckpt 未测 |
| HSDP+EP | 部分 | 单机缩比通过，双节点 collective 未测 |
| RL | 否，不能整体放行 | plain owner 逻辑成立；compose/Turbomind 有确定性 bug |
| legacy bit-exact | 未证明 | 默认开关隔离和 placement 回归通过 |

## 5. 五个提交各自做了什么

| commit | 作用 | 关键文件 |
|---|---|---|
| `2098db9e` | 在动核心代码前固定 legacy mesh/placement，作为 L0 回归基线 | `tests/model/test_decoupled_ep_fsdp_mesh.py`、`reports/baseline.md`、`reports/decisions.md` |
| `3055e4ce` | 加开关、root mesh、dense/expert 两级 FSDP、prefetch、梯度归约；完成 L1 | `xtuner/v1/config/fsdp.py`、`xtuner/v1/model/moe/moe.py`、`reports/L1.md` |
| `6fee4dd3` | 不再加主执行代码，补单机 efsdp/HSDP/DeepEP/empty-expert 的 L2 记录 | `reports/L2.md` |
| `2d667775` | dense/expert 分离的 FP8 padding/reduce mesh；RL FSDP-only gather；HF/DCP/FP8 L3 | `xtuner/v1/float8/*`、`xtuner/v1/model/base.py`、`tests/model/run_decoupled_ep_fsdp_ckpt.py`、`reports/L3.md` |
| `56021e3e` | torch 2.9 复验、GLM 示例开关和 8×H200 GLM-5.2 实验文档 | `examples/v1/config/sft_glm5p2.py`、`reports/GLM52.md`、其余报告更新 |

## 6. 推荐阅读顺序

只想理解主实现：

1. `DESIGN.md:45-65`：目标语义；
2. `xtuner/v1/model/moe/moe.py:1651-1715`：mesh；
3. `xtuner/v1/model/moe/moe.py:1193-1244`：两级 FSDP；
4. `xtuner/v1/model/moe/moe.py:1583-1628`：梯度数学；
5. `xtuner/v1/float8/float8_handler.py:121-238`：FP8；
6. `xtuner/v1/model/base.py:1907-1940`：RL gather；
7. [decoupled_ep_fsdp_review_zh.md](decoupled_ep_fsdp_review_zh.md)：风险与放行条件。
