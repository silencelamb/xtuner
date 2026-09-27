# 解耦布局下专家组的 reshard 时机与反向预取

本文说明 PR #2093 解耦布局（`FSDPConfig.decouple_ep_fsdp=True`）开发过程中发现的一个问题和对应的两个修复：为什么要改、改了什么、测到了什么。
文中「修复前」指 PR 的早期 head `212c05ee`（专家组用 FSDP2 默认的逐次 reshard），不是 main；main 上只有旧布局，
PR 最终代码相对 main 的效果看「旧布局」与「解耦，修复后」两列。
对应提交：

- `[Fix] Reshard decoupled expert FSDP groups once per layer instead of after every call`
- `[Fix] Point decoupled expert groups' backward prefetch at the layer below`
- `[Test] Count decoupled expert all-gathers under intra-layer micro-batches and recompute`（回归测试，见 §4）

两个修复都没有新增配置项，解耦路径下默认生效，旧布局不受影响。

## 0. 结论

- **问题**：解耦布局把专家组的 FSDP 单元放在了整层的 reentrant 重算**里面**。按 FSDP2 默认的逐次 reshard，专家参数每层每步要
  all-gather 多次（mb4、全重算时 13 次，旧布局的整层单元 2 次）；重算还打乱了默认的反向预取，每步 46 次 all-gather 发出后
  从未被使用，缓冲留到反向结束，正好压在显存峰值上。
- **修复**：专家组每层前向后、反向后各 reshard 一次；专家组的反向预取显式指向下一层。不加配置项，只影响解耦路径，
  主干层和 MTP 层都覆盖。
- **效果**（单机 8×H200、同一批交错、DeepEP v1、PR 树）：
  - Qwen3-30B-A3B，EP4，mb4，64K token/卡/步：解耦原来比旧布局**慢 5.3%、峰值高 3.2 GB**，修复后**快 0.6%、峰值低 8.3 GB**
    （对修复前 +6.2%）。专家 all-gather 614 → 96 次/步，从未被使用的 46 → 1，暴露的 all-gather + reduce-scatter 297 → 42 ms/步。
  - 同上，EP8（efsdp=1）：对旧布局 −1.6% → +1.0%（对修复前 +2.7%）。
  - GLM-5.2-30B（含 1 层 MTP），EP4，mb2：对修复前 +4.4%，峰值 87.0 → 83.2 GB；MTP 层的专家组同样每步最多 2 次 all-gather。
  - 末步 loss 与旧布局一致，差异在同一布局重复运行的波动之内。
- **两机 16 卡**（2×8 H200，专家组的 all-gather 跨机走 IB，§3.3）：修复后对修复前 Qwen3 EP8（efsdp=2）**+8.0%**、
  EP4（efsdp=4）**+11.2%**、GLM EP4 **+4.7%**，都比单机大；修复后与旧布局持平（Qwen3 +0.5% / +0.2%），显存少 8.6 / 3.8 GB。
- **未覆盖**：两机以上、efsdp > 4、HSDP + EP 的吞吐。

## 1. 问题

### 1.1 专家组落在 reentrant 重算里面，每层的 all-gather 次数随微批和重算成倍增加

解耦布局对每个 decoder layer 做两级包装（设计文档 §3.3）：先把 `MoEBlock` 在 `expert_fsdp_mesh` 上 `fully_shard`，
再对整层套 activation checkpoint，最后在 dense mesh 上 `fully_shard` 整层。运行时的结构是：

```
旧布局： FSDP(整层)  [ 重算 [ attention ... 专家 ... ] ]
解耦：   FSDP(dense) [ 重算 [ attention ... FSDP(专家) ... ] ]
```

XTuner 的重算是 reentrant 实现（`xtuner/v1/model/utils/checkpointing.py`）：原始前向在 `no_grad` 下执行，反向到这一层时
先用保存的输入把前向重跑一遍，再在反向内部对重跑出的小图调用一次 `torch.autograd.backward`。`checkpointing.py` 的注释写明
「FSDP must wrap this checkpoint boundary」：外层的 FSDP 单元不会被重跑，所以旧布局下整层单元每层每步只 all-gather 两次
（前向一次，反向一次）。解耦布局里的专家单元在边界之内，它的 FSDP hook 会在重跑时再次触发。

FSDP2 的默认行为（torch 2.9.1，`_fsdp_param_group.py`）是 `reshard_after_forward=True`、`reshard_after_backward=True`：
每次前向调用结束、每次反向调用结束都释放参数，下次用时重新 all-gather。层内微批让专家模块每层被调用 N 次，于是 mb=4、
全重算时，每层专家单元每步的 all-gather 是：

| 来源 | 次数 |
|---|---:|
| 前向，4 个微批各一次 | 4 |
| 反向开始时 dense 单元的默认预取（给重算用） | 1 |
| 重算，第 2~4 个微批各一次 | 3 |
| 反向，4 个微批各一次 | 4 |
| 打错目标的预取（见 1.2） | 1 |
| **合计** | **13**（旧布局 2） |

mb=1 时同样存在：每个 pack 每层要拼 3 次（前向、重算、反向），外加 1 次浪费。

### 1.2 重算打乱了 FSDP2 的默认反向预取

FSDP2 在每个单元开始反向时，会按「前向完成顺序」预取上一个单元（`post_forward_order`）。专家单元在重算时重新执行
`post_forward`，会在**反向期间**往这张表末尾追加记录。于是第 i 层专家最后一次开始反向时，查到的「上一个」是第 i+1 层
专家在重算时留下的记录，它会去预取一个反向已经做完的单元：

- 这次 all-gather 的结果从未被使用（没有对应的 `all_gather_copy_out`），而 FSDP2 要到反向结束的 `finalize_backward` 才回收它。
  Qwen3-30B-A3B、EP4 下每个 0.3 GB，每步 46 个，反向后段最多累积约 14 GB，而且恰好落在峰值上。
- 同时没有任何单元去预取第 i−1 层的 dense 单元。它的 all-gather 只能在用到时才发，此时第 i 层 dense 单元的 reduce-scatter
  刚在同一个通信组上发出，两者在 GPU 上串行，计算流只能等着。

### 1.3 为什么 PR 原有的测试没有发现

L0~L3 都是正确性检查：多拼几次、多发一次没人用的 all-gather，参数值完全一样，loss 和梯度一个 bit 也不变；
显存断言检查的是每卡**参数**字节数（静态分片大小），不是峰值；而且测试没有开层内微批。唯一带步时和峰值的数据是 EP8
（efsdp=1），那时专家不需要跨卡拼，打错目标的预取也不会分配新缓冲，恰好是问题最不明显的配置。

## 2. 修复

### 2.1 专家组每层每个阶段只 reshard 一次

`_fully_shard_expert_blocks` 把专家组的 `reshard_after_forward` 和 `reshard_after_backward` 都设为 `False`，
`_reshard_expert_blocks_per_layer` 在外层（dense）FSDP 单元上挂两个 hook，把 reshard 放回到「层」的粒度：

- **前向**：外层单元的 forward post-hook。重算执行的是被包在里面的模块，不会调用它，所以只在原始前向里触发。
  最后一层（无 MTP 时）沿用它自身的 `reshard_after_forward=False`，不在前向后释放。
- **反向**：给该层输入张量注册的梯度 hook。该层的重算、所有微批的反向以及专家组 reduce-scatter 的发出都完成之后，
  输入的梯度才会产生，此时 reshard。
- **兜底**：模型的 forward pre-hook 在每次前向开始前 reshard 所有专家组。任何一层的反向 hook 没有触发时（比如某层没有
  需要梯度的输入），它保证下一步不会拿优化器更新前的旧权重计算；正常情况下是空操作。

结果是专家组每层每步前向拼一次、反向拼一次，和旧布局的整层单元节奏一致；同一时刻最多只有一层专家处于拼好的状态。

### 2.2 专家组的反向预取显式指向下一层

`_set_expert_backward_prefetch` 给第 i 层（i ≥ 1）的每个专家组设置 `set_modules_to_backward_prefetch([第 i−1 层])`。
FSDP2 中显式列表会替换默认目标，所以同时解决了 1.2 的两个问题：不再预取已经做完的单元；第 i−1 层 dense 的 all-gather
在第 i 层专家第一次开始反向时就发出，排在第 i 层 dense 的 reduce-scatter 前面。每个栈的第一层保留默认目标
（主干第 0 层还会有一次浪费的预取，约 1 ms、0.3 GB，影响可忽略）。

### 2.3 MTP

MTP layer 使用同样的两级包装，同样落在重算里面，2.1 和 2.2 一并应用（MTP 的第一层保留默认预取）。

### 2.4 为什么不加配置项

两个修复都只改变 reshard 与预取的时机：同一时刻最多一层专家处于拼好的状态，这和修复前相同，所以不引入新的常驻显存。
按层 reshard 让一层专家在该层的几个微批之间不再释放，峰值最多多出一层专家的拼好缓冲（Qwen3 EP4 上 0.3 GB；§3.4 的 V2 数据里
71.7 → 72.0 GB），比 2.2 去掉的未使用缓冲（约 14 GB）小得多。测量里吞吐和峰值都没有变差，没有需要用户权衡的地方。曾评估过的另一种做法是让专家组 `reshard_after_forward=False` 常驻到反向：它同样减少 all-gather，
但所有层的专家同时常驻（Qwen3 EP4 上限 48 × 0.302 GB = 14.5 GB），只能做成开关，已放弃。

## 3. 测量

### 3.1 trace 计数（Qwen3-30B-A3B，EP4，mb4，64K token/卡/步，第 8 步，rank 0）

修复前 = PR head `212c05ee`，修复后 = 本文三个提交，旧布局 = 修复后的树、`decouple_ep_fsdp=False`；DeepEP v1，同一台机器。
「暴露」是该集合通信 kernel 运行期间，任何 stream 上都没有计算 kernel 的时长。解耦两列是 09-24 同一批；旧布局一列 09-27 补抓，
次数与批次无关，时长类数字跨批比较有 0.3%~3.7% 的漂移（§6），吞吐以 §3.2 的同批数据为准。

| 每步 | 旧布局 | 解耦，修复前 | 解耦，修复后 |
|---|---:|---:|---:|
| 单步 span | 5741 ms | 6065 ms | 5715 ms |
| 专家组 all-gather，前向 | —（专家在整层单元里） | 189 | 48 |
| 专家组 all-gather，重算 + 反向 | — | 425 | 48 |
| 其中发出后从未被使用 | 0 | 46 | 1（第 0 层） |
| 每层外层单元 all-gather，前向 / 反向 | 48 / 47（整层，含专家） | 48 / 47（dense） | 48 / 47（dense） |
| FSDP all-gather 合计 | 100 | 715 | **196** |
| all-gather 总时长 / 暴露 | 127.4 / 6.6 ms | 709.0 / 197.9 ms | 216.8 / 24.7 ms |
| reduce-scatter 次数 / 暴露 | 51 / 4.2 ms | 99 / 99.5 ms | 99 / 17.0 ms |

**次数只能在两列解耦之间比，不能拿来比布局。** 解耦把一层拆成 dense 与专家两个 FSDP 单元，每个单元每步各拼 2 次，
所以次数是旧布局的两倍（all-gather 196 对 100，reduce-scatter 99 对 51）；但每次拼的量相应变小，dense 与专家两次拼出的参数
加起来和旧布局整层单元一次相同。比较两种布局要看吞吐、显存（§3.2）和暴露时间；次数只用来验证机制：修复后专家组每层每步
前向 1 次、反向 1 次，和旧布局整层单元的节奏一致。

修复前后：reduce-scatter 的次数不变，暴露时长从 99.5 降到 17.0 ms。修复前第 i−1 层 dense 的 all-gather 排在第 i 层 dense 的
reduce-scatter 后面，计算流在等它（1.2），这段等待落在 reduce-scatter 运行期间。

修复后和旧布局只差很小的一点暴露：all-gather 24.7 对 6.6 ms，reduce-scatter 17.0 对 4.2 ms，合计每步多约 31 ms（单步的 0.5%）。
all-gather 里最大的一块是 dense 的反向 all-gather（13.1 ms）：2.2 让第 i−1 层 dense 的 all-gather 在第 i 层专家组第一次开始反向时
才发出，还有一段没被计算盖住；其余是 embed / lm_head / norm 在步首步尾的几次（4.0 ms）、专家组的 all-gather（5.4 ms）和
dense 的前向 all-gather（2.3 ms）。reduce-scatter 的 17.0 ms 里 dense 占 11.7 ms，没有再逐条归因。把所有单元的反向预取都改成
显式链能把 dense 这部分再压下去（上限约 0.5%），但要额外维护一条跨单元的预取链，为保持改动简单没有做（§5）。

GLM-5.2-30B（EP4、mb2、16K token/卡/步；3 个专家组 = 2 层 MoE + 1 层 MTP）：

| 每步 | 旧布局 | 解耦，修复前 | 解耦，修复后 |
|---|---:|---:|---:|
| 单步 span | —（见下） | 2753 ms | 2635 ms |
| 专家组 all-gather，前向 / 重算 + 反向 | —（专家在整层单元里） | 5 / 11 | 3 / 2 |
| 其中发出后从未被使用 | 0 | 2 | 0 |
| 每层外层单元 all-gather，前向 / 反向 | 6 / 5（整层，含专家） | 6 / 5（dense） | 6 / 5（dense） |
| FSDP all-gather 合计 | 20 | 37 | 26 |
| 暴露的 all-gather / reduce-scatter | —（见下） | 45.1 / 29.5 ms | 35.3 / 18.9 ms |

修复后反向只有 2 次：MTP 层是最后一层，沿用 `reshard_after_forward=False`，专家组从前向一直保持到反向。
旧布局这一列只给次数：GLM 的旧布局贴着显存上限（峰值 103.4 GB，reserved 131.6 GB），抓 trace 那次每步 6~11 s
（同批吞吐里平均 2.9 s），单次 dense reduce-scatter 长到 386 ms，是分配器抖动，时长没有可比性。

### 3.2 同批吞吐与显存（DeepEP v1，PR 树）

| 工作点 | 处理 | n | tgs（最小~最大） | 对旧布局 | 对修复前 | 峰值显存 | 末步 loss |
|---|---|---:|---|---:|---:|---:|---|
| Qwen3-30B-A3B，EP4（efsdp=2），mb4 × 16K | 旧布局 | 3 | 11510（11492~11525） | — | | 79.4 GB | 1.5768~1.5771 |
| | 解耦，修复前 | 3 | 10900（10894~10904） | −5.3% | — | 82.6 GB | 1.5768~1.5769 |
| | **解耦，修复后** | 3 | **11575（11570~11581）** | **+0.6%** | **+6.2%** | **71.1 GB** | 1.5766~1.5770 |
| Qwen3-30B-A3B，EP8（efsdp=1），mb4 × 16K | 旧布局 | 2 | 10088（10086~10090） | — | | 91.3 GB | 1.5770~1.5771 |
| | 解耦，修复前 | 2 | 9922（9920~9923） | −1.6% | — | 73.6 GB | 1.5768~1.5769 |
| | **解耦，修复后** | 2 | **10190（10179~10201）** | **+1.0%** | **+2.7%** | **73.1 GB** | 1.5769~1.5773 |
| GLM-5.2-30B（含 MTP），EP4（efsdp=2），mb2 × 8K，FP8 | 旧布局 | 2 | 5624（5527~5722） | — | | 103.4 GB | 8.1975 |
| | 解耦，修复前 | 2 | 6163（6157~6170） | +9.6% | — | 87.0 GB | 8.1975~8.1976 |
| | **解耦，修复后** | 2 | **6435（6434~6435）** | **+14.4%** | **+4.4%** | **83.2 GB** | 8.1973~8.1974 |

所有「修复后」对「修复前」、对旧布局的差异，最小~最大区间都不重叠。

- **EP4 显存**：解耦按闭式应比旧布局少 `dense × (ep − 1) / world × 16 B` ≈ 9.2 GB。修复前反而多 3.2 GB，多出来的就是 1.2 那些
  从未被使用的 all-gather 缓冲；修复后少 8.3 GB，接近闭式。
- **EP8（efsdp=1）**：专家组只有一张卡，all-gather 不走网络，但仍有 2.7% 的收益。按代码推断（EP8 行没有抓 trace，未 profile 验证）：
  修复前每次调用后 reshard、下次调用重新拷贝出完整参数的开销仍在；重算打乱默认预取与组大小无关，dense 的反向 all-gather
  同样排在 reduce-scatter 后面。
- **GLM 的旧布局**离显存上限近，步时波动大（std 0.17 s，其他行 ≤ 0.03 s），它的均值只作参考；修复前后两行的比较不受影响。

### 3.3 两机 16 卡（2999 + 2296）

2 台 8×H200，RoCE 8 轨；2026-09-27 10:22~11:28 同一批、按轮次交错，每格 2 次；代码、配方与 §3.2 相同，
每卡每步 token 数不变（Qwen3 GBS 64，GLM GBS 32）。ep 维在最内层，EP 组都在机内（DeepEP 只走 NVLink），
跨机的是 FSDP 组：

- EP8：旧布局 fsdp=2（dense 在 ep 维复制 8 份）；解耦 dense 切 16 份，专家 efsdp=2，组是 {r, r+8}，整组跨机。
- EP4：旧布局 fsdp=4；解耦 dense 切 16 份，专家 efsdp=4（2 张本机 + 2 张对机）。

| 工作点 | 处理 | tgs（最小~最大） | 对旧布局 | 对修复前 | 峰值显存 | 末步 loss |
|---|---|---|---:|---:|---:|---|
| Qwen3-30B-A3B，EP8（efsdp=2），mb4 × 16K | 旧布局 | 9853（9833~9873） | — | | 58.4 GB | 1.5846~1.5849 |
| | 解耦，修复前 | 9168（9109~9228） | −6.9% | — | 51.7 GB | 1.5845~1.5848 |
| | **解耦，修复后** | **9898（9858~9938）** | **+0.5%** | **+8.0%** | **49.8 GB** | 1.5847~1.5848 |
| Qwen3-30B-A3B，EP4（efsdp=4），mb4 × 16K | 旧布局 | 11277（11239~11314） | — | | 51.5 GB | 1.5846~1.5848 |
| | 解耦，修复前 | 10166（10160~10173） | −9.8% | — | 54.8 GB | 1.5846~1.5847 |
| | **解耦，修复后** | **11303（11257~11349）** | **+0.2%** | **+11.2%** | **47.7 GB** | 1.5846~1.5850 |
| GLM-5.2-30B（含 MTP），EP4（efsdp=4），mb2 × 8K，FP8 | 旧布局 | 5956（5927~5985） | — | | 66.8 GB | 8.1812~8.1817 |
| | 解耦，修复前 | 6042（6017~6067） | +1.4% | — | 59.2 GB | 8.1813~8.1817 |
| | **解耦，修复后** | **6328（6302~6355）** | **+6.2%** | **+4.7%** | **55.1 GB** | 8.1813~8.1814 |

- 「修复后」对「修复前」的区间都不重叠；Qwen3 两个点上「修复后」与旧布局的区间重叠，即持平。两机的步时 std 是 0.11~0.22 s
  （单机 ≤ 0.03 s），所以只报持平，不报 +0.2% / +0.5% 这样的小差。
- 修复的收益比单机大（EP4：单机 +6.2%、两机 +11.2%；EP8：+2.7%、+8.0%）：每一次多余的专家 all-gather 现在都要跨机。
  修复前 EP4 还比旧布局多用 3.3 GB 显存，同样是 1.2 那些从未被使用的缓冲。
- 显存：解耦比旧布局少 3.8 GB（EP4）、8.6 GB（EP8），闭式 `dense × (ep − 1) / world × 16 B` 给出 4.6 / 10.8 GB。

trace（EP8，第 8 步，rank 0）：

| 每步 | 旧布局 | 解耦，修复前 | 解耦，修复后 |
|---|---:|---:|---:|
| 单步 span | 6619 ms | 7137 ms | 6734 ms |
| 计算 kernel 忙时 | 5196 ms | 5310 ms | 5192 ms |
| FSDP all-gather 次数 / 暴露 | 100 / 30 ms | 715 / 473 ms | 196 / 283 ms |
| 其中专家组，前向 / 重算 + 反向 | —（与 dense 同一单元） | 189 / 425 | 48 / 48 |
| 发出后从未被使用 | 0 | 46 | 1 |
| reduce-scatter 次数 / 暴露 | 51 / 25 ms | 99 / 307 ms | 99 / 273 ms |

all-gather 次数与单机（§3.1）逐项相同，读法也同 §3.1：旧布局的 100 次不能直接和解耦的 196 次比。修复后这一步比旧布局多约 115 ms，计算时间相同，多出来的是解耦布局本身的代价：
dense 切 16 份，它的 reduce-scatter（暴露 215 ms）和反向 all-gather（89 ms）都跨机；旧布局的 dense 只在 2 张卡之间切，
其余靠机内复制。这 115 ms 小于两机的步时 std（0.11~0.22 s），单步 span 的差不代表吞吐，吞吐以上表 20 步均值为准（持平）。
这部分是布局本身的代价，不属于本文修的问题；要在多机上继续压，方向是 HSDP（dense 只在机内切），本文没有测。

### 3.4 补充：DeepEP V2 上的数据

DeepEP V2 不在上游，下列数据来自 fork 的 `feat/deepepv2` 合并 PR #2093 后的评测树，同一台机器、同一批、每格 3 次交错，
修复以环境变量开关的形式实现（逻辑与本 PR 相同）。Qwen3-30B-A3B，EP4，mb4，64K token/卡/步：

| 处理 | tgs（最小~最大） | 对旧布局 | 峰值显存 |
|---|---|---:|---:|
| 旧布局 | 12243（12222~12256） | — | 80.3 GB |
| 解耦，修复前 | 11608（11604~11613） | −5.2% | 83.3 GB |
| 解耦，只加 2.2 | 11749（11747~11751） | −4.0% | 71.7 GB |
| **解耦，2.1 + 2.2** | **12324（12309~12335）** | **+0.7%** | **72.0 GB** |

只加 2.2 就把峰值从 83.3 GB 降到 71.7 GB：那约 14 GB 未使用的 all-gather 缓冲正是 1.2 所说的峰值来源。
解耦布局在 EP4 上按闭式应比旧布局少 `dense × (ep − 1) / world × 16 B` ≈ 9.2 GB，修复后实测少 8.3 GB。

### 3.5 口径与复现

- **机器与批次**：单机 8×H200（2999），2026-09-24 00:00~01:10 同一批，按轮次交错（每轮依次跑 旧布局 → 解耦修复前 →
  解耦修复后），Qwen3 EP4 每格 3 次，其余每格 2 次。
- **代码**：修复前 = PR #2093 head `212c05ee`；修复后 = `212c05ee` + 本文三个提交。旧布局的行用修复后的树、
  `decouple_ep_fsdp=False` 跑（修复只在解耦路径生效）。
- **配方**：DeepEP v1 dispatcher，`torch.compile` 开，默认全重算（`recompute_ratio=1.0`），层内微批 `intra_layer_micro_batch`。
  - Qwen3-30B-A3B：BF16，pack 16384 × mb4 = 64K token/卡/步，GBS 32；EP4（efsdp=2）与 EP8（efsdp=1）。
  - GLM-5.2-30B（缩层版：3 层 dense + 2 层 MoE + 1 层 MTP，256 专家 top-8）：tile-wise FP8，pack 8192 × mb2 = 16K token/卡/步，
    GBS 16，EP4。用它覆盖 MTP 层。
- **指标**：每次 20 步，tgs 取第 5 步之后的均值，是**单卡**每秒 token；峰值是 rank 0 日志的 `max_memory`
  （`max_memory_allocated`，按 1024³ 计）在所有步上的最大值。
- **trace**：`TOTAL_STEP=10`，torch profiler 抓第 8 步 rank 0，按 FSDP 的 `record_function` 标签统计每个单元的 all-gather、
  `all_gather_copy_out`（被使用）与 reduce-scatter。
- 启动脚本、统计脚本和原始日志在规划仓库 `bench/moe_v2/`（单机 `queue_pr2093_final.sh`、两机 `queue_pr2093_2node.sh`、
  `aggregate_reps.py`、`fsdp_trace_audit.py`）。两机的 work_dir 在共享 NFS 上，PR 树没有 NFS 上的 work_dir 探测修复，
  所以每行起跑前预先放好 `.xtuner` 与数据缓存；这只是跑法，与被测代码无关。

## 4. 测试

`tests/engine/test_decoupled_ep_fsdp_train_engine.py::TestDecoupledEpFsdpNumerics::test_expert_groups_gather_once_per_layer_with_micro_batches_and_recompute`
（8 卡）：tiny Qwen3-MoE（4 层），解耦 EP4（efsdp=2），2 个层内微批，默认全重算，训练 3 步，在第 3 步统计每个 FSDP 单元
发出的 all-gather 与被使用的 all-gather（`xtuner._testing.decoupled_ep_fsdp.count_fsdp_all_gathers`，替换
`FSDPParamGroup.unshard` / `wait_for_unshard` 计数，不改变行为）。断言：

- 每层一个专家组，共 4 个；
- 每个专家组每步被使用的 all-gather ≤ 2（前向一次、反向一次）；
- 发出后未被使用的 all-gather 合计 ≤ 1（只允许第 0 层保留默认预取的那一次）。

这个用例只断言通信次数，不比较数值：开层内微批时，`all2all` dispatcher 的异步路径偶发非有限梯度范数（XTuner 会跳过那一步的
优化器更新），旧布局和解耦布局都有，与本修复无关（见 §5）。loss / 梯度与旧布局的一致性由同文件里 L1~L2 的用例覆盖。

结果（8×H200，torch 2.9.1）：

- 修复前的代码（`212c05ee`）上失败，停在第二条断言：`layers.0` 的专家组每步被使用 6 次 all-gather
  （前向 2 + 反向开始时的默认预取 1 + 重算第 2 个微批 1 + 反向 2）。
- 修复后通过；在最终代码上单独重复 5 次，均通过（每次约 19 s）。
- 与 PR 原有用例一起：`tests/model/test_decoupled_ep_fsdp_mesh.py`、`tests/rl/test_weight_iterator.py`、
  `tests/engine/test_decoupled_ep_fsdp_train_engine.py` 在最终代码上合跑 51 passed。
- `ruff check` / `ruff format --check` 在改动文件上通过；`mypy xtuner/v1` 修复前后都是 328 个既有报错，改动的行没有新增。

## 5. 边界与未做

- **只测到两机**：单机 8×H200 与两机 16×H200（§3.3）。更多节点时专家组的 all-gather 更多跨机，方向应与两机一致，没有实测。
- **两机上解耦布局本身的 dense 通信代价**：dense 切满 16 卡、跨机做 reduce-scatter / all-gather，trace 里比旧布局多约 115 ms/步
  暴露（§3.3），20 步均值上与旧布局持平。这是布局的选择，不是本修复的范围。
- **层内微批下 `all2all` dispatcher 的数值问题（既有，与本修复无关）**：`intra_layer_micro_batch ≥ 2` 时 `all2all` 的异步路径
  偶发非有限梯度范数，tiny 模型上旧布局和解耦布局都会出现（修复前解耦 14/18 次运行、修复后 5/18、旧布局约 1/28）；
  `CUDA_LAUNCH_BLOCKING=1`、改用 DeepEP、mb=1 均为 0/6。像是异步 all2all 与计算流之间缺同步，另行跟进。
- **efsdp > 4 未测**：测到 efsdp = 1、2（单机）和 2、4（两机）。
- **HSDP + EP 未单独测吞吐**：L2 的数值门禁覆盖了这些拓扑（代码路径相同）。
- 主干第 0 层保留默认预取，每步 1 次浪费的 all-gather。
- 2.2 在第 i 层专家第一次开始反向时才发第 i−1 层 dense 的 all-gather，Qwen3 EP4 上这次 all-gather 还有约 13 ms/步暴露
  （DeepEP V2 树上约 31 ms）；把所有单元的预取都改成显式链可以再省一点（上限约 0.5%），为保持改动简单没有做。

## 6. 方法说明

同一配置在不同机器、甚至同一机器的不同时间段之间会差 0.3%~3.7%，和这里要测的效果同一量级。所以本文所有对比都只在
**同一批、按轮次交错**的运行之间做，每格至少 2 次，给出均值和最小~最大。
