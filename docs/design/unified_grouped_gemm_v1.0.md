# XTuner 统一 Grouped GEMM 算子库：设计 v1.0（给算子组，对齐统一 EP 接口 v1.0）

状态：2026-09-19 设计稿，给算子组。目的：把框架侧（大 EP 契约）对 Grouped GEMM 的需求一次性说清，标出 `feat/deepepv2` 分支上已经落地的部分（契约的一半、一个 DeepGEMM 适配器、一套单卡 harness），并给出可以直接开工的接口、分工与参考代码（§12）。行布局、权重段、梯度落点直接复用 [统一 EP 接口 v1.0](unified_ep_contract_v1.0.md) 的 `ExpertRows` / `ExpertWeights` / `GradBinding`，本文不另设类型；签名为建议，命名可改。本文与 EP 接口 v1.0 一起放在 `docs/design/`。

---

## 0. 一页结论

1. **要做的不是"再写一个 GEMM kernel"，而是三层东西**：① 一份 kernel 无关的**输入契约**（行布局 / 权重段 / 量化格式 / 输出累加），② 一个**按 setup 期静态信息选后端**的注册表 + 选择器 + 离线 autotune 表，③ 把现有 GEMM 路径（Triton varlen、CUTLASS、AdaptiveGEMM FP8、DeepGEMM psum、TE、Triton dual-base、NPU gmm）和待评估的 SonicMoE 收成同一组 **`fwd / dgrad / wgrad` 三个无 autograd 的算子**，autograd 在框架层组装。
2. **契约的一半已经在分支上**（EP 接口 v1.0 §2.1）：行布局 `ExpertRows` 与 `Layout`、权重段 `ExpertWeights` / `ExpertWeightSegment` / `ProjectionWeight`、梯度落点 `GradBinding` 都在 `xtuner/v1/module/dispatcher/base.py`，`GroupedLinear.forward(x, rows, *, x_scale, weight)` 已经吃它们。本库要补的是算子侧的另一半：`QuantSpec`（SF 格式协商）、`OutSpec`（wgrad 直写 / 累加 / dtype，是 `GradBinding` 在 op 层的形态）、`GemmCaps` / `GemmSpec`（能力声明与静态规格）。
3. **分支上的 DeepGEMM 适配器就是目标形态的样板**（`xtuner/v1/ops/moe/cuda/group_gemm_deepgemm.py`）：每个 kernel 调用是 `torch.library.custom_op` 加 `register_fake`，autograd 在外层 `Function` 组装，三个 op 已经来自不同后端（FP8 的 fwd / dgrad 走 DeepGEMM psum，wgrad 走 AdaptiveGEMM 的 dW kernel），`MoEBlock` 在 `fullgraph=True` 下前反向可 trace。§12.3 的模板就是从它抽出来的。
4. **选择只能发生在 setup 期，不能在每次调用时看 M**。静态模式（`cpu_sync=False`）下真实 token 数只在 GPU 上，任何"看 M 选 kernel"都会引入 host sync。分支现在只有一个两路开关 `XTUNER_EXPERT_GEMM_BACKEND=auto|legacy|deepgemm`，按 `rows.alignment` 这个 Python 常量分支；注册表把它推广成按 `(arch, op, dtype / 量化格式, 布局 / 对齐, N, K, expected_m 桶, 是否累加, 是否确定性, 段数)` 选，运行期只做 assert。
5. **对齐与 SF 格式是"消费者决定、生产者按需产出"**：后端声明 `required_alignment` 与 `preferred_sf_format`，框架把对齐传给 dispatcher（DeepEP V2 的 `expert_alignment`）、把 SF 格式传给量化 / act 核。分支上的适配器目前在 op 内做了一次 `row_major_scales` 转换来吃 DeepEP V2 的 TMA 对齐 SF，这正是 `QuantSpec` 协商要消掉的东西。
6. **profile 说明优先级**（GLM-5.2-30B 8 卡 EP4 FP8）：FP8 grouped GEMM 只占步时 7.8%，FP8 量化加 cast 占 8.7%，等于 GEMM 时间的 42%。所以第一收益是让量化 / act 融合核与任意后端配合并把 counts 全留在 GPU；kernel 升级排第二。分支上的实测与此一致：Qwen3 去 permute +8.2%，再换 DeepGEMM 共 +11.0%（EP 接口 v1.0 §0）。
7. 分四步，S1 已完成一半：**S1** 契约（行布局与权重段已合入；补 QuantSpec / OutSpec / caps）+ 注册表 + 现有后端登记（无行为变化，bit-exact）→ **S2** wgrad 累加 / 双段 / 确定性 / op 级对拍与性能 harness → **S3** DeepGEMM 后端补完（SF 协商、dgrad 转置缓存、与 act 融合核对接）→ **S4** SonicMoE 评估与 R 布局后端。

---

## 1. 现状与问题

### 1.1 分支 `feat/deepepv2` tip 上的 Grouped GEMM 调用面

| 路径 | 入口 | 计数形式 | 布局 / 对齐 | dtype | wgrad | 状态 |
|---|---|---|---|---|---|---|
| Triton varlen（默认 BF16） | `ops/moe/cuda/group_gemm.py` → `m_grouped_gemm(A, B, size_per_group, trans_b)`、`k_grouped_gemm(A, B, size_per_group)` | GPU int，吃 `rows.compute_counts` | E0；kernel 内按 BLOCK_M=128 虚拟对齐，EA 布局下 padding 行参与计算 | bf16 | bf16 新分配 | 分支：不改 kernel，只换入口签名 |
| CUTLASS（`XTUNER_USE_CUTLASS_GROUP_GEMM=1`） | `group_gemm_cutlass.py` → `grouped_gemm.backend.gmm` | **host**（`.cpu()`） | E0 | bf16 | bf16，无累加 | 分支：不改 |
| AdaptiveGEMM（FP8） | `float8/float8_gmm_tile_wise.py` → `m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous((x, sf), (w, sf), counts)`、`k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous(dy_t, dy_t_sf, x_t, x_t_sf, dw, counts_expand)` | GPU int，吃 `rows.compute_counts`；从激活行数推 M，**不能吃容量尾部** | E0 / EA-128 | act per-tile 1×128、weight per-block 128×128，SF row-major fp32 | bf16 `dw` 直写 | 分支：新增接受预量化输入（`x_scale`） |
| **DeepGEMM psum**（`XTUNER_EXPERT_GEMM_BACKEND=auto|deepgemm`） | `ops/moe/cuda/group_gemm_deepgemm.py` → `m_grouped_{bf16,fp8}_gemm_nt_contiguous(a, b, d, psum, use_psum_layout=True)`、`m_grouped_bf16_gemm_nn_contiguous`（BF16 dgrad） | GPU int，吃 `rows_psum(rows)` 与 `rows.compute_counts` | EA-128；容量尾部不读 | bf16 / FP8 | BF16：Triton `k_grouped_gemm`；FP8：AdaptiveGEMM dW（bf16 直出、device 计数），无 AdaptiveGEMM 时 DeepGEMM k-grouped（fp32 累加，慢 2 倍） | 分支：**新增**，是本库的样板 |
| TE / Triton dual-base（PR #2050） | `group_gemm_te.py`；`m_grouped_gemm_dual_weight` | host list / GPU int | E0，组序 `[masters..., replicas...]` | bf16 | 写 `out` list / 复用 k-grouped 再切片 | 未进分支；UltraEP 两段权重时吸收 |
| NPU | `ops/moe/npu/group_gemm.py` → `Ops.gmm` | device groupList | E0 | bf16 | — | 不改 |

选择逻辑：`ops/moe/__init__.py` 仍在 import 期按设备 / 环境变量绑定 legacy 的 `group_gemm`；`moe_group_linear.py::use_deepgemm(rows)` 在 `alignment == 128` 且 DeepGEMM 可导入时切到 psum 路径。FP8 仍是另一个类 `TileWiseFloat8GroupedLinear`，签名已与 BF16 类相同。

### 1.2 问题

1. **输入契约只有一半**：行布局与权重段已有契约（`ExpertRows` / `ExpertWeights`），仍缺 SF 格式、wgrad 落点与累加、双段能力的声明。
2. **没有能力声明**：谁需要 host 计数、谁能累加 wgrad、谁能吃两段权重、谁要什么 SF 格式、谁能吃容量尾部，散在各文件的 assert 里。分支上的例子：AdaptiveGEMM 从激活行数推 M，静态容量下会算错，只能在配置期拒绝 `cpu_sync=False` + FP8 + `alignment=1` 的组合（EP 接口 v1.0 §4.2）。
3. **签名漂移已止住，新字段还没有消费者**：`GroupedLinear.forward(x, rows, *, x_scale, weight)` 止住了三个 PR 各改一次的漂移；#2050 的三个位置参数、#2056 的 `grad_weight_out` 分别对应 `rows.host_counts` + `ExpertWeights` 两段、`GradBinding(path="external")`。后者在分支上**还没有消费者**，是 S2 的第一项。
4. **FP8 量化开销大于 GEMM 本身**，SF 约定仍是隐式的。分支为了吃 DeepEP V2 的 TMA 对齐 SF，在适配器里加了 `row_major_scales` 转换。
5. **harness 只有一半**：分支有 `tests/module/test_moe_block_rows.py`（8 条 kernel 路径对 fp32 参考，含 padding 行必须为零、容量尾部不参与规约）和跨树逐位对比（2026-09-19 在 EP=1 / 2 / 8 上 66 条全部 BIT-EXACT）；缺 op 级 harness（空组、NaN 注入、双段、累加两次）与性能矩阵。

### 1.3 框架侧已经明确的需求（EP 接口 v1.0 §2 的算子侧投影）

| # | 需求 | 来源 | 分支现状 |
|---|---|---|---|
| R1 | 输入是 `hidden[capacity, K]` + `rows: ExpertRows{starts, compute_counts, valid_counts, alignment, padding_zeroed, host_counts}`，GPU int；`capacity` 可以大于 `starts[-1] + compute_counts[-1]` | DeepEP V2 expand / 静态 shape | 已落地 |
| R2 | host 计数是可选镜像 `rows.host_counts`；库内禁止 `.cpu() / .tolist() / .item()`；需要 host 计数的后端在配置期被过滤 | 零 host sync 验收 | 类型已有，无消费者 |
| R3 | 权重是 1 段或 2 段（`ExpertWeightSegment(first_slot, w1w3, w2)`），组序 `[masters..., replicas...]`，replica 专家间可跨步；不允许拼接 | UltraEP | 类型已有；`GroupedLinear` 只支持 `None` 或 1 段 |
| R4 | wgrad 落点：`GradBinding(path="autograd")` 回 autograd；`path="external", write_op="overwrite"` 直写 buffer；`write_op="add"` 累加。输出 dtype bf16 或 fp32；空专家在累加模式下不能清零已有梯度 | MoonEP / UltraEP / MetaMoE 共享池 | 类型已有，无消费者 |
| R5 | 对齐 `A` 由 GEMM 后端声明，框架传给 dispatcher；padding 行内容由 `padding_zeroed` 说明，kernel 不得假设已清零 | EA 布局 | DeepEP V2 `zero_padding` 配置 → `padding_zeroed` |
| R6 | FP8：激活 per-tile (1×128) e4m3、权重 per-block (128×128)；激活 SF 走 `PostDispatchResult.hidden_scales` 或 bf16 类型的 `Float8Tensor` 包装；SF 格式由后端声明 | AdaptiveGEMM + DeepGEMM | 格式未协商，适配器内转换 |
| R7 | 可选确定性（`XTUNER_DETERMINISTIC`）：wgrad 累加序固定 | RL 训推一致 | Triton k-grouped 有确定性配置 |
| R8 | `torch.compile` 可用：每个 op 是 `custom_op` 带 fake impl，op 内无数据相关分支 | `MoEBlock` fullgraph | DeepGEMM 适配器已做，有测试 |
| R9 | 空输入与空专家都要能跑，不靠 host 判断 | 现有 `x.shape[0]==0` 特判要收进 op | 部分：特判仍在 Python |
| R10 | 三个 op 可独立选后端；选择可被配置固定、被环境变量覆盖、被日志记录 | 复现与 A/B | 只有一个两路开关 |

---

## 2. 目标与非目标

**目标**：一个设备无关的 Python 契约 + 注册表，N 个后端适配器，1 套 op 级对拍 / 性能 harness，1 张离线 autotune 表。框架层通过 `build_grouped_linear` 一次性拿到三个 op 的实现。

**非目标**：permute / unpermute（dispatcher 侧）；SwiGLU × 路由权重 × 量化的融合核（独立算子，§7）；MegaKernel；attention / dense GEMM。

---

## 3. 契约层：沿用 v1.0 的三个类型，新增三个

契约类型分两处：行布局 / 权重段 / 梯度落点已在 `xtuner/v1/module/dispatcher/base.py`（框架与算子组共同 owner，不依赖任何 kernel 包）；本节新增的 `QuantSpec` / `OutSpec` / `GemmCaps` / `GemmSpec` 建议放 `xtuner/v1/ops/moe/grouped_gemm/contracts.py`。

### 3.1 行布局：直接用 `ExpertRows`

```python
class ExpertRows(NamedTuple):            # dispatcher/base.py，已落地
    starts: Tensor                       # GPU int32 [G]，组 g 的起始行
    compute_counts: Tensor               # GPU int32 [G]，kernel 可访问的行数（alignment 的倍数）
    valid_counts: Tensor | None          # GPU int32 [G]，真实行数（<= compute_counts）
    alignment: int                       # 1 = E0；128 = EA(128)
    padding_zeroed: bool                 # 对齐 padding 行是否已被生产者清零
    host_counts: Tensor | None = None    # CPU 镜像，只给 caps.needs_host_counts 的后端

rows_from_counts(counts)                                   # legacy 计数 -> E0
rows_from_psum(psum, valid, alignment=, padding_zeroed=)   # DeepEP V2 / DeepGEMM 前缀和 -> EA
rows_psum(rows)                                            # starts + valid_counts，DeepGEMM 的 grouped_layout
check_rows(rows, capacity)                                 # 调试用，会 host sync，禁止上热路径
```

不变量：`starts[0] == 0`，`starts[g+1] == starts[g] + compute_counts[g]`，`valid_counts <= compute_counts`，`starts[-1] + compute_counts[-1] <= capacity`。两条语义（EP 接口 v1.0 §2.1）：容量尾部任何 kernel 都不读不规约；对齐 padding 行必须是有限零，因为计数型 wgrad kernel 会规约到它们。`PostDispatchResult.tokens_per_expert` 恒等于 `rows.compute_counts`，只认计数的老 kernel 在 EA 布局上会把 padding 行也算一遍，靠 `padding_zeroed` 保证无害。

派生量都是 O(G) 的 GPU 小算子：`psum`、`cu_seqlens = cat([0], psum)`（仅 `alignment == 1` 时对 SonicMoE 合法）、`m_indices[capacity]`（gap 行 −1）。四个布局家族（E0 / EA / R / M）的定义、谁产出谁消费、以及 `Layout` 枚举与 `ExpertRows.layout` 派生属性，见 [EP 接口 v1.0 §2.1](unified_ep_contract_v1.0.md)，代码在 `dispatcher/base.py`。契约只描述 E0 与 EA；R 与 M 占枚举位，今天没有后端产出或消费，等第一个需要它们的后端出现再加 `gather_idx` / `masked_m` 字段。本文用 `Layout` 做后端能力声明（`GemmCaps.layouts`）与静态规格（`GemmSpec.layout`）；R 用附加的 `gather_idx`（§6）。

### 3.2 权重段：直接用 `ExpertWeights`

```python
class GradBinding(NamedTuple):           # dispatcher/base.py，已落地
    path: Literal["autograd", "external"]
    buffer: Tensor | None = None         # external 时的落点
    write_op: Literal["overwrite", "add"] | None = None

class ProjectionWeight(NamedTuple):
    value: Tensor                        # [G_seg, N, K]，K-major；允许 stride(0) != N*K（UltraEP replica）
    backward_read: Literal["saved", "restore_before_dgrad"] = "saved"
    grad: GradBinding = GradBinding("autograd")

class ExpertWeightSegment(NamedTuple):
    first_slot: int                      # 该段覆盖的第一个组号（静态）
    w1w3: ProjectionWeight
    w2: ProjectionWeight

class ExpertWeights(NamedTuple):
    segments: tuple[ExpertWeightSegment, ...] | None = None   # None = 用模块自身参数
```

GEMM 库一次只看**一个投影**：op 的权重输入是 `tuple[Tensor, ...]`（各段的 `value`）加各段的 `first_slot`，单段且 `stride(0) == N*K` 时退化为今天的 `w.view(-1, N, K)`。FP8 权重是 `Float8Tensor(_data[G,N,K] e4m3, _scale[G,N/128,K/128] fp32)`。dgrad 需要 `[G, K, N]`：BF16 后端用 `nn` kernel 或 `trans_b`；FP8 后端（SM90）需要 K-major 的 `W^T` 与转置后的 SF，分支和 legacy 一样在每次 backward 里 `transpose(1, 2).contiguous()`（`group_gemm_deepgemm.py` 的 `_DeepGemmFP8.backward`），改成每步一次的 `WeightCache` 是 S3 的一项。

### 3.3 低精度描述 `QuantSpec`（新增）

```python
class SFFormat(str, Enum):
    NONE = "none"
    FP32_ROW_1x128 = "fp32_rowmajor_1x128"        # AdaptiveGEMM / DeepEP v1 默认；[M, K/128] fp32 行主
    FP32_MN_TMA_1x128 = "fp32_mn_tma_1x128"       # DeepGEMM SM90 / DeepEP V2 FP8 dispatch：TMA 对齐 MN-major
    UE8M0X4_MN_TMA_1x32 = "ue8m0x4_mn_tma_1x32"   # DeepGEMM SM100 packed

class QuantSpec(NamedTuple):
    act_dtype: torch.dtype                # bf16 | float8_e4m3fn
    weight_dtype: torch.dtype             # bf16 | float8_e4m3fn
    act_recipe: tuple[int, int]           # (1, 128) per-tile
    weight_recipe: tuple[int, int]        # (128, 128) per-block
    sf_format: SFFormat                   # 激活 SF 的物理格式，由所选后端的 preferred_sf_format 决定
    weight_sf_transposed_cache: bool      # dgrad 是否需要预转置权重与 SF
```

激活的量化不在 GEMM 库内做（属于上游 act / 量化核，或 DeepEP V2 的 FP8 dispatch）。它进 GEMM 的两条路都已落地：`PostDispatchResult.hidden_scales`，或跨 autograd 时用 bf16 类型的 `Float8Tensor` 包装（autograd 会把梯度 cast 成输入 dtype，裸 fp8 边会把 dX 静默量化）。库要提供 `quantize_reference()` 纯 torch 实现供对拍，并**声明**每个后端接受的 `SFFormat`。

### 3.4 输出与累加 `OutSpec`（新增，是 `GradBinding` 在 op 层的形态）

```python
class OutSpec(NamedTuple):
    dtype: torch.dtype                    # fwd / dgrad bf16；wgrad bf16 | fp32
    out: Tensor | None                    # 预分配输出（直写）；None = 库内分配
    beta: Literal[0, 1] = 0               # 1 = 累加到 out（要求 out 非 None）
```

| `ProjectionWeight.grad` | wgrad 的 `OutSpec` | autograd 返回 |
|---|---|---|
| `GradBinding("autograd")` | `OutSpec(dtype=bf16, out=None)` | dW 回该段 `value` |
| `GradBinding("external", buffer, "overwrite")` | `OutSpec(dtype=buffer.dtype, out=buffer, beta=0)` | 该段返回 `None` |
| `GradBinding("external", buffer, "add")` | `OutSpec(dtype=buffer.dtype, out=buffer, beta=1)` | 该段返回 `None` |

`beta=1` 的语义：`out[g] += dy_g^T @ x_g`，0 行的组不改写 `out[g]`。MoonEP 单段 alias 的 dW（#2056 的 `grad_weight_out`）是 `overwrite`，UltraEP 的 replica 段是 `overwrite` 到 `replica_grad`，MetaMoE 共享池是 `add`。

---

## 4. 算子接口：三个无 autograd 的 op + 能力声明

### 4.1 后端协议

```python
@dataclass(frozen=True)
class GemmCaps:
    name: str
    archs: tuple[str, ...]                      # ("sm90",) / ("sm90", "sm100") / ("ascend",)
    ops: tuple[Literal["fwd", "dgrad", "wgrad"], ...]
    act_dtypes: tuple[torch.dtype, ...]
    weight_dtypes: tuple[torch.dtype, ...]
    sf_formats: tuple[SFFormat, ...]            # 首个 = preferred，写进 QuantSpec 让上游产出
    layouts: tuple[Layout, ...]                 # §3.1 的枚举
    required_alignment: int                     # 1 / 128；EA 时 rows.alignment 必须是它的倍数
    accepts_padding_rows: bool                  # 能否对 compute_counts > valid_counts 的洞行安全计算
    accepts_capacity_tail: bool                 # 能否吃 capacity > starts[-1] + compute_counts[-1]（AdaptiveGEMM 不能）
    needs_host_counts: bool                     # True → 只在 rows.host_counts 非 None 时可选
    max_segments: int                           # 1 / 2
    wgrad_out_dtypes: tuple[torch.dtype, ...]   # (bf16,) / (bf16, fp32)
    wgrad_accumulate: bool                      # 支持 beta=1
    deterministic: bool                         # 有确定性变体
    max_groups: int | None                      # CUTLASS kMaxExperts=512 之类


class GroupedGemmBackend(Protocol):
    caps: GemmCaps
    def supports(self, spec: "GemmSpec") -> bool: ...
    def fwd(self, x: Tensor, x_sf: Tensor | None, w: tuple[Tensor, ...], w_sf: tuple[Tensor | None, ...],
            rows: ExpertRows, *, out: OutSpec) -> Tensor: ...                 # [capacity, N]
    def dgrad(self, dy: Tensor, dy_sf: Tensor | None, w: tuple[Tensor, ...], w_sf: tuple[Tensor | None, ...],
              rows: ExpertRows, *, out: OutSpec) -> Tensor: ...               # [capacity, K]
    def wgrad(self, dy: Tensor, dy_sf: Tensor | None, x: Tensor, x_sf: Tensor | None,
              rows: ExpertRows, *, out: tuple[OutSpec, ...]) -> tuple[Tensor, ...]: ...   # 每段一个 [G_seg, N, K]
```

三条硬规则：

1. 每个 op 是 `torch.library.custom_op`（`mutates_args` 正确标注 `out`），带 `register_fake`；不在 op 内按 tensor 内容分支。分支上的 `moe::deepgemm_m_grouped_bf16_nt` 等五个 op 就是这样注册的。
2. op 内**不做**：转置权重、compact / permute 行、SF 转置、`.cpu()`。做不到就 `supports()` 返回 False，让选择器换后端；不做静默退化。（分支适配器里的 `row_major_scales` 与 backward 里的权重转置是待迁出的两处。）
3. 洞行与尾部：`accepts_padding_rows=True` 的后端要么按 `valid_counts` mask，要么要求 `padding_zeroed=True` 并在 `supports()` 里检查；输出的 padding 行必须是有限零（分支适配器用零初始化的输出 buffer 保证）；`accepts_capacity_tail=False` 的后端在静态容量配置下被过滤。反向的 `dy` 同样适用。

### 4.2 autograd 在框架层组装

分支已经按这个分工写：`group_gemm_deepgemm.py` 里 `_DeepGemmBF16` / `_DeepGemmFP8` 两个 `autograd.Function` 只做 save / 调 op / 返回，FP8 的 wgrad 在 backward 里按 `_adaptive_gemm_dw_available()` 走 AdaptiveGEMM 或 DeepGEMM k-grouped。目标形态把"选哪个"提前到 setup 期：

```python
class GroupedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_sf, w_values, w_sfs, rows, plan: "GemmPlan"):
        ctx.plan = plan; ctx.save_for_backward(x, x_sf, *w_values, *rows_tensors(rows))
        return plan.fwd.fwd(x, x_sf, w_values, w_sfs, rows, out=plan.fwd_out)
    @staticmethod
    def backward(ctx, dy):
        dy, dy_sf = plan.quantize_dy(dy)                         # dgrad / wgrad 后端是 FP8 时
        dx = plan.dgrad.dgrad(dy, dy_sf, plan.w_T(), plan.w_sf_T(), rows, out=...)
        dws = plan.wgrad.wgrad(dy_T, dy_T_sf, x_T, x_T_sf, rows, out=plan.wgrad_outs)   # 每段一个
        return (dx, None, *[dw if seg.grad.path == "autograd" else None for dw, seg in zip(dws, plan.segments)], ...)
```

`GemmPlan` 是 setup 期选择的结果：三个后端 + 每段的 `OutSpec` + 是否需要转置缓存。

---

## 5. 选择器与注册表

### 5.1 静态规格 `GemmSpec`

```python
@dataclass(frozen=True)
class GemmSpec:
    arch: str                      # "sm90" | "sm100" | "ascend"，运行时探测
    op: Literal["fwd", "dgrad", "wgrad"]
    quant: QuantSpec
    layout: Layout                 # rows.layout，或 R（gather_idx 后端）
    alignment: int                 # rows.alignment
    static_capacity: bool          # dispatcher 是否给最坏容量（cpu_sync=False）
    n: int; k: int
    expected_m: int                # 每组期望行数：max_tokens_per_rank × topk / P_local；不是运行期 M
    num_groups: int
    segments: int                  # 1 / 2
    out_dtype: torch.dtype
    accumulate: bool
    deterministic: bool
    has_host_counts: bool
```

### 5.2 选择流程

```
候选 = [b for b in registry if b.supports(spec)]     # caps 过滤，含 accepts_capacity_tail 对 static_capacity
若为空 → ConfigError（配置期，列出每个后端被过滤的原因）
key = (arch, op, act_dtype, weight_dtype, layout, n, k, bucket(expected_m), segments, accumulate)
name = perf_table.get(key)                            # 离线 autotune 表（JSON）
若无记录：XTUNER_GEMM_AUTOTUNE=offline 时跑 benchmark 写表；线上默认按 §5.3 静态优先级并 warning
用户覆盖：MoEConfig.grouped_gemm = {"fwd": "deepgemm", "dgrad": "deepgemm", "wgrad": "adaptive"}；
         XTUNER_EXPERT_GEMM_BACKEND=<name>（三 op 同名，向后兼容今天的 auto|legacy|deepgemm）
记录：选择结果写进 trainer 的环境快照
```

`bucket(expected_m)`：`{≤64, 128, 256, 512, 1024, 2048, ≥4096}`，与 H200 BF16 的 roofline 拐点（≈256 行）对齐。

### 5.3 无表时的静态优先级（H200 / SM90，可被表覆盖；分支实测为准）

| op | BF16 | FP8 |
|---|---|---|
| fwd | EA-128：DeepGEMM psum → Triton varlen；E0：Triton varlen → CUTLASS（有 host counts 时） | EA-128：DeepGEMM psum；E0：AdaptiveGEMM（非静态容量） |
| dgrad | 同 fwd（DeepGEMM `nn`） | DeepGEMM `nt` + 转置缓存 → AdaptiveGEMM |
| wgrad | Triton k-grouped（GPU counts；确定性配置存在）→ TE（host counts） | AdaptiveGEMM dW（bf16 直出、device 计数）→ DeepGEMM k-grouped（fp32 累加，慢 2 倍，只作回退） |

两段权重（UltraEP）：fwd / dgrad 只有 Triton dual-base 与 TE 满足 `max_segments=2`；wgrad 用单段 kernel 算 `[M+R]` 再按各段 `OutSpec` 分发是过渡实现，目标是 ptr-array。

---

## 6. 各后端的适配说明与工作量

| 后端 | 契约映射 | 分支已有 / 缺失 | 工作量 |
|---|---|---|---|
| **Triton varlen** | E0 / EA：传 `compute_counts`；padding 行参与计算 → 要求 `padding_zeroed=True` 或加 mask；`k_grouped_gemm_out`（#2056）→ `OutSpec.out` | 已吃 `rows`；缺 `beta=1` 与 fp32 输出 | 小 |
| **CUTLASS** | E0；`needs_host_counts=True`；`ElementC` bf16 硬编码 | 退路与对拍基线 | 小 |
| **AdaptiveGEMM**（FP8） | E0 / EA-128；SF `FP32_ROW_1x128`；`accepts_capacity_tail=False`；EA-128 下 `expand_128x` 变成恒等 | 已吃 `rows` 与预量化输入；缺 `beta=1`、dgrad 转置缓存 | 中 |
| **DeepGEMM 2.6**（psum） | EA-128；`grouped_layout = rows_psum(rows)`；SF `FP32_MN_TMA_1x128`（SM90）；k-grouped SM90 不支持 psum、要 `ks_cpu` | **适配器已有**：BF16 nt / nn、FP8 nt、fp32 k-grouped 回退（`trans_quant_kgrouped`）、`register_fake`、fullgraph 测试。缺：SF 协商（去掉 `row_major_scales`）、dgrad `W^T` 缓存、`OutSpec`、登记进注册表 | 中，S3 主体 |
| **TE**（#2050） | E0；`needs_host_counts=True`；list-of-tensors 天然支持跨步 replica 与两段 | 改为吃 `rows.host_counts` | 小 |
| **Triton dual-base**（#2050） | E0；`max_segments=2` | 改为吃两段 `value`；wgrad 双 `OutSpec` | 小 |
| **SonicMoE** | R 布局：紧凑 `x[T, K]` + `gather_idx` + `cu_seqlens`；gemm-1 融合 gather + SwiGLU；EA 的洞无法表达 → 只接 R 或 E0 | 需独立环境（DSL 4.7 与 MoonEP 的 4.4.2 互斥）；H200 无公开数字 | 大；S4 先评估 |
| **NPU** | E0（device groupList） | 保持协议设备无关 | 小 |

SonicMoE 的接法是声明 `layouts=("R",)` 且 `fused_act=True`，由框架在 `MoEBlock` 层面整体切换，不拆成三个裸 op。

---

## 7. 与量化 / act 融合核的边界

- 融合核（SwiGLU × 路由权重 × FP8 量化，正向产 GEMM-2 的 A 与 wgrad 的 A^T，反向同一 epilogue 出 dH / d_probs）是**独立算子**，PR #2089 的 `swiglu_per_tile_quant_with_trans_per_block` 是雏形。
- 本库与它的接口只有两点：① 它按 `QuantSpec.sf_format` 产出 SF；② 它按 `ExpertRows` 理解分组与洞行（EA-128 下 `expand_128x` 的补零变成对 `valid_counts` 的 mask）。
- 路由权重乘法目前在 `combine_preprocess` 用 `row_scale` Triton kernel 做（EP 接口 v1.0 §4.4），前移到 act 核会改变舍入位置，需要算法侧认可的数值 A/B；库内保留"不乘"的路径。

---

## 8. 数值、性能与验收

### 8.1 对拍 harness

已有两层，作为 S2 的起点：

- **单卡 kernel 路径对 fp32 参考**：`tests/module/test_moe_block_rows.py`，8 条路径（legacy BF16 / FP8 × E0 / EA-128 × bf16 / 预量化输入，DeepGEMM BF16 / FP8），断言有效行容差内、padding 行为精确零、容量尾部不参与规约、反向梯度有限，另有 fullgraph 编译测试。
- **跨树逐位对比**：上游原样树对新树 dump 层输出 / dX / 全部参数梯度，`torch.equal`。2026-09-19 在 EP=1 / 2 / 8（含 AGRS）66 条全部 BIT-EXACT。

S2 要补的 op 级 harness（`tests/ops/grouped_gemm/`）：生成器给定 `(G, N, K, dtype, layout, alignment, expected_m 分布)` 产同一份 `x / w / rows`，含空专家、单专家吃满、容量尾部、洞行注入 NaN、两段跨步权重；参考实现纯 torch fp32 逐组 matmul，FP8 用 `quantize_reference()`。断言分级：**L0 bit-exact**（同一后端契约迁移前后）；**L1 容差**（bf16 后端间 `rtol 2e-2 / atol 2e-2` 起，逐后端标定；FP8 6e-2）；**L2 loss 级**。反向：dX、每段 dW、`beta=1` 累加两次等于单次 ×2、空组不清零；确定性后端两次运行 bitwise 相等。CPU 上跑契约 / 选择器 / fake impl；GPU 标记 `@pytest.mark.gpu`。

### 8.2 性能矩阵

`(arch, op, dtype, layout, N, K, bucket(expected_m), segments)` × 全部候选后端 → TFLOPS 与时间；输出即离线 autotune 表。参考形状：Qwen3-30B-A3B（`K=2048, N=1536` / `K=768, N=2048`，E=128，top-8）、GLM-5.2-30B。每专家行数桶：128 行访存 bound、256 拐点、512 计算 bound。端到端数字以 8 卡训练吞吐为准。

### 8.3 验收口径

| 阶段 | 验收 |
|---|---|
| S1 | 现有后端登记进注册表后，三 dispatcher × Triton / CUTLASS / AdaptiveGEMM / NPU 在 E0 下 L0 bit-exact（沿用跨树对比）；`use_deepgemm()` 两路开关与 `TileWiseFloat8GroupedLinear` 独立路径删除；配置期报错替代运行期 NotImplementedError |
| S2 | op 级 harness 覆盖 §8.1 全部用例；`GradBinding(path="external")` 三种落点有消费者；nsys：GEMM 路径零 `cudaStreamSynchronize` |
| S3 | DeepGEMM 与 AdaptiveGEMM 同一 EA-128 输入 L1 对齐；SF 按 `preferred_sf_format` 直接产出、适配器内无转换；dgrad 权重转置每步一次；FP8 GEMM + 量化总时间较现状下降（量化占比从 42% 降到 <20%） |
| S4 | SonicMoE 在 H200 的 bf16 吞吐与激活显存报告；R 布局后端在 harness 内 L1 通过 |

---

## 9. 接入 XTuner

分支上的入口已经是目标签名（EP 接口 v1.0 §2.4）：

```python
# xtuner/v1/module/grouped_linear/moe_group_linear.py（已落地）
class GroupedLinear(nn.Module):
    def forward(self, x, rows: ExpertRows, *, x_scale=None, weight=None) -> Tensor:
        weight = weight if weight is not None else self._own_weight_view()
        if use_deepgemm(rows):                      # 今天：两路开关，按 rows.alignment 折叠
            return deepgemm_group_gemm(x, weight, rows)
        return group_gemm(x, weight, rows.compute_counts)
```

本库接入只改这一个分支点：`build_grouped_linear` 从 `MoEConfig.grouped_gemm`、`Float8Config`、设备探测拼出三个 `GemmSpec`，`registry.plan()` 一次拿到 `GemmPlan`，`forward` 调 `GroupedLinearFn.apply(x, x_scale, w_values, w_sfs, rows, plan)`；`TileWiseFloat8GroupedLinear` 退化为 `QuantSpec.act_dtype=fp8` 的同一个类。对齐回传：`MoEDecoderLayer` 把 `plan.required_alignment` 写进 `DeepEPV2Config.alignment`，把 `plan.quant.sf_format` 交给量化 / act 核。`expected_m` 来源：`max_tokens_per_rank × topk / P_local`，与 DeepEP V2 的 `num_max_tokens_per_rank` 同源。

---

## 10. 工作分解、分工与里程碑

| # | 工作 | 阶段 | 归属 | 状态 / 依赖 |
|---|---|---|---|---|
| K1 | 契约：`ExpertRows` / `Layout` / `ExpertWeights` / `GradBinding`（已合入）+ `QuantSpec` / `OutSpec` / `GemmCaps` / `GemmSpec` + 纯 torch 参考实现 + 注册表 / 选择器 + 配置期报错 | S1 | 算子组（接口与框架组共同评审） | 一半已落地 |
| K2 | Triton varlen / CUTLASS / AdaptiveGEMM / NPU 登记进注册表（无行为变化，bit-exact） | S1 | 算子组 | Triton / AdaptiveGEMM 已吃 `rows`；K1 |
| K3 | 删除 `ops/moe/__init__.py` import 期选择、`use_deepgemm()` 两路开关、`TileWiseFloat8GroupedLinear` 独立路径；`build_grouped_linear.setup` | S1 | 框架组 + 算子组 | K1, K2 |
| K4 | wgrad `OutSpec`：直写 / `beta=1` / fp32 输出；Triton k-grouped 加 `acc`；空组不清零；`GradBinding(path="external")` 的消费者（MoonEP 单段、UltraEP replica 段共用） | S2 | 算子组 | K2 |
| K5 | 双段权重：dual-base 适配、TE 适配、wgrad 双目标；ptr-array 版为后续 | S2 | 算子组（吸收 #2050 kernel） | K1 |
| K6 | op 级对拍 + 性能 harness + 离线 autotune 表格式与 CLI | S2 | 算子组 | K1；起点是 `test_moe_block_rows.py` |
| K7 | DeepGEMM 后端补完：SF 协商（去 `row_major_scales`）、dgrad `W^T` 缓存、`OutSpec`、登记；wgrad 策略按架构 | S3 | 算子组 | 适配器已有；K1, K6 |
| K8 | 量化 / act 融合核按 `QuantSpec.sf_format` 与 `ExpertRows` 产出（#2089 的核迁移；去掉 `expand_128x`） | S3 | 算子组 | K1, K7 |
| K9 | SonicMoE 评估（独立 venv：torch ≥ 2.11、DSL 4.7）：H200 吞吐 + 激活显存；R 布局后端原型 | S4 | 算子组 | K6 |
| K10 | 确定性变体清单与开关（Triton k-grouped 已有；DeepGEMM / AdaptiveGEMM 待查） | S2–S3 | 算子组 | K4 |

节奏：S1 剩余部分 1–2 周；S2 2–3 周；S3 视 DeepGEMM 补完 2–3 周；S4 并行评估。

---

## 11. 风险与开放问题

1. **DeepGEMM SM90 wgrad**：k-grouped 不支持 psum、要 `ks_cpu`；分支用 `static_ks_host` 走静态 K 绕开 host sync，但 fp32 累加慢 2 倍，H200 上 wgrad 留在 AdaptiveGEMM dW。SM100 才完整。
2. **AdaptiveGEMM 吃不了静态容量**：它从激活行数推 M。静态模式 + FP8 + E0 的组合已在配置期拒绝；EA-128 + DeepGEMM 是静态模式下 FP8 的唯一路径。
3. **依赖互斥**：SonicMoE（`nvidia-cutlass-dsl==4.7.0`）与 MoonEP（`==4.4.2`）不能同环境；DeepGEMM 需 CUDA ≥ 12.3；DeepEP V2 需 NCCL ≥ 2.30.4。镜像升级是共同前置。
4. **AdaptiveGEMM 是 DeepGEMM 旧分支的 fork**：两份 FP8 kernel 并存的维护成本；S3 后评估是否让 AdaptiveGEMM 只保留 SM90 wgrad。
5. **洞行语义**：`accepts_padding_rows` 与 `padding_zeroed` 的组合要在 harness 里用 NaN 注入验证；MoonEP 当前把 padding 折进末组且未显式清零，是第一个会踩到的用例。
6. **`expected_m` 估计偏差**：负载极不均衡时同一层内既有几万行组也有几十行组，单一 kernel 配置不最优；workload-aware tile 调度是后续项。
7. **`torch.compile` 与 custom_op**：TE 的 `.tolist()` 与 cuBLAS 路径的 `.cpu()` 在编译区内是 graph break；`needs_host_counts` 后端与 fullgraph 编译互斥，配置期拒绝。
8. **归属**：契约类型首期放树内（行布局与权重段已在 `dispatcher/base.py`；算子侧的放 `ops/moe/grouped_gemm/`，算子组 owner），S3 后按发布节奏再决定是否拆包；独立成库的条件与规则见 §13。
9. **确定性与性能的取舍口径**：RL 训推一致需要哪些 op 确定性，与 RL 侧对齐。

---

## 12. 开发指南与参考代码

### 12.1 先把现有三条路径跑起来（半天）

在 `feat/deepepv2` 分支的 checkout 里：

```bash
export PYTHONPATH=/opt/ep-v2-site:$PWD        # DeepEP V2 + DeepGEMM 2.6 的站点放前面（路径按机器改）
export EP_DISABLE_GIN=1                        # 单机、无 GIN 网卡
export XTUNER_DETERMINISTIC=true TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# 8 条 kernel 路径对 fp32 参考（单卡）；换 backend 看同一批输入走不同 kernel
XTUNER_EXPERT_GEMM_BACKEND=auto     python -m pytest -q tests/module/test_moe_block_rows.py
XTUNER_EXPERT_GEMM_BACKEND=legacy   python -m pytest -q tests/module/test_moe_block_rows.py
XTUNER_EXPERT_GEMM_BACKEND=deepgemm python -m pytest -q tests/module/test_moe_block_rows.py -k deepgemm

# 契约单测（CPU 即可）与 DeepEP V2 六模式（2 卡）
python -m pytest -q tests/module/dispatcher/test_expert_rows.py tests/module/dispatcher/test_deepep_v2.py
```

读代码的顺序：`dispatcher/base.py` 的 `ExpertRows` 与 `rows_*` 四个函数 → `moe_group_linear.py::GroupedLinear.forward` 与 `use_deepgemm` → `ops/moe/cuda/group_gemm_deepgemm.py`（样板）→ `float8/float8_gmm_tile_wise.py`（legacy FP8，看 `expand_128x` 与每次 backward 的权重转置）→ `tests/module/test_moe_block_rows.py`（harness 怎么造 EA 布局与容量尾部）。

### 12.2 写 kernel 前必读的数值约定

1. **只读 `[starts[g], starts[g] + compute_counts[g])`**，容量尾部不读不写不规约；输出 buffer 零初始化，让 padding 行与尾部保持有限（`group_gemm_deepgemm.py::_out_buffer`）。
2. **padding 行的输出必须是精确零**：计数型 wgrad kernel（Triton `k_grouped_gemm`、AdaptiveGEMM dW）按 `compute_counts` 规约，padding 行两个操作数都是零才不污染 dW。`padding_zeroed=False` 时 kernel 自己 mask。
3. **`tokens_per_expert == rows.compute_counts`**，不是 `valid_counts`。只认计数的 kernel 会把 padding 行算一遍，这是刻意的。
4. **FP8 激活跨 autograd 用 bf16 类型的 `Float8Tensor`**（`_data` + `_scale` 在里面），`unwrap_fp8_activation()` 拆开；op 的参数只收裸张量。
5. **dgrad 的权重形状**：BF16 用 `nn` kernel 或 `trans_b=False`；FP8 SM90 要 K-major 的 `W^T` 与转置 SF，今天每次 backward `transpose(1, 2).contiguous()`，新后端应把它做成每步一次的缓存。
6. **wgrad 的 x^T / dy^T 预量化**：FP8 路径在 forward 里就把 `x^T` 按组转置量化并保存（省显存，不存 bf16 的 x），backward 只量化 `dy`。EA-128 下 `expand_128x` 是恒等，可直接用 `trans_quant_kgrouped(x, starts, counts)`。
7. **空输入与空组**：`capacity == 0` 与某组 0 行都不能靠 host 判断；DeepGEMM 适配器目前仍有 `x.shape[0] > 0` 的 Python 特判，新后端把它收进 op（fake impl 返回空张量）。
8. **确定性**：wgrad 累加序固定；`XTUNER_DETERMINISTIC=true` 时选确定性变体，没有就 `supports()` 返回 False。

### 12.3 新后端适配器模板（从分支的 DeepGEMM 适配器抽出）

```python
# xtuner/v1/ops/moe/cuda/group_gemm_<name>.py
from typing import Any
import torch
from torch import Tensor

REQUIRED_ALIGNMENT = 128          # 或 1；写进 GemmCaps.required_alignment

def _out_buffer(x: Tensor, n: int, dtype=torch.bfloat16) -> Tensor:
    # 零初始化：padding 行与容量尾部保持有限零（§12.2-1）
    return torch.zeros(x.shape[0], n, device=x.device, dtype=dtype)

# ---- 1. 三个无 autograd 的 op：custom_op + fake；参数只有张量与 Python 标量，op 内不看张量内容分支 ----
@torch.library.custom_op("moe::<name>_fwd", mutates_args=())
def name_fwd(x: Tensor, w: Tensor, starts: Tensor, compute_counts: Tensor) -> Tensor:
    out = _out_buffer(x, w.shape[1])
    _launch_m_grouped(x, w, starts, compute_counts, out, trans_b=True)      # D[M,N] = grouped(x @ w[g]^T)
    return out

@name_fwd.register_fake
def _(x: Tensor, w: Tensor, starts: Tensor, compute_counts: Tensor) -> Tensor:
    return x.new_empty(x.shape[0], w.shape[1])

@torch.library.custom_op("moe::<name>_dgrad", mutates_args=())
def name_dgrad(dy: Tensor, w: Tensor, starts: Tensor, compute_counts: Tensor) -> Tensor:
    out = _out_buffer(dy, w.shape[2])
    _launch_m_grouped(dy, w, starts, compute_counts, out, trans_b=False)    # dX[M,K] = grouped(dy @ w[g])
    return out

@name_dgrad.register_fake
def _(dy: Tensor, w: Tensor, starts: Tensor, compute_counts: Tensor) -> Tensor:
    return dy.new_empty(dy.shape[0], w.shape[2])

@torch.library.custom_op("moe::<name>_wgrad", mutates_args=("out",))
def name_wgrad(dy: Tensor, x: Tensor, starts: Tensor, compute_counts: Tensor, out: Tensor, beta: int) -> None:
    # out[g] = beta * out[g] + dy_g^T @ x_g；0 行的组不改写 out[g]（§3.4）
    _launch_k_grouped(dy, x, starts, compute_counts, out, beta)

@name_wgrad.register_fake
def _(dy: Tensor, x: Tensor, starts: Tensor, compute_counts: Tensor, out: Tensor, beta: int) -> None:
    return None

# ---- 2. autograd 在这里组装；三个 op 可以来自不同后端 ----
class _NameGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, w: Tensor, starts: Tensor, compute_counts: Tensor) -> Tensor:
        ctx.save_for_backward(x, w, starts, compute_counts)
        return name_fwd(x, w, starts, compute_counts)

    @staticmethod
    def backward(ctx: Any, dy: Tensor):
        x, w, starts, compute_counts = ctx.saved_tensors
        dy = dy.contiguous()
        dx = name_dgrad(dy, w, starts, compute_counts)
        dw = torch.empty_like(w)
        name_wgrad(dy, x, starts, compute_counts, dw, 0)
        return dx, dw, None, None

# ---- 3. 入口：吃 ExpertRows，布局能力只在这里 assert（注册表落地后由 supports() 接管）----
def name_group_gemm(x: Tensor, w: Tensor, rows: "ExpertRows") -> Tensor:
    assert rows.alignment % REQUIRED_ALIGNMENT == 0, f"{rows.alignment=} not a multiple of {REQUIRED_ALIGNMENT}"
    return _NameGemm.apply(x, w, rows.starts, rows.compute_counts)
```

今天接进 `GroupedLinear` 的最小改动是两处：`ops/moe/__init__.py::get_expert_gemm_backend` 的合法值加 `<name>`，`moe_group_linear.py::GroupedLinear.forward` 的分支点按 backend 名分派（照 `use_deepgemm` 写）。注册表（K1）落地后这两处合并成 `registry.plan()`，适配器只需声明 `GemmCaps` 并实现 `supports / fwd / dgrad / wgrad`。

FP8 适配器多三件事：权重是 `Float8Tensor`，取 `_data` / `_scale`；激活用 `unwrap_fp8_activation(x, x_scale)` 拆成 `(x_fp8, x_sf)` 或 `(x_bf16, None)`，没量化时 `per_tile_quant(x)`；forward 里把 `x^T` 按组量化保存（`trans_quant_kgrouped(x, starts, counts)`），backward 只量化 `dy`。`_DeepGemmFP8` 是完整例子。

### 12.4 wgrad 三种落点的参考实现（K4）

```python
# Triton k-grouped 今天返回新张量；#2056 的 k_grouped_gemm_out 直写；累加用 out.add_ 过渡，目标是 kernel 内 acc=
from xtuner.v1.ops.moe.cuda.triton_kernels import k_grouped_gemm

def wgrad_into(dy: Tensor, x: Tensor, rows: "ExpertRows", spec: "OutSpec") -> Tensor | None:
    dw = k_grouped_gemm(dy, x, rows.compute_counts)                      # [G, N, K] bf16
    if spec.out is None:
        return dw if spec.dtype == dw.dtype else dw.to(spec.dtype)      # autograd 路径
    if spec.beta == 0:
        spec.out.copy_(dw)                                               # overwrite：MoonEP 单段 alias / UltraEP replica_grad
    else:
        spec.out.add_(dw)                                                # add：MetaMoE 共享池；空组 dw[g]==0，不清零 out[g]
    return None                                                          # external：autograd 返回 None
```

在 `autograd.Function.backward` 里，每段按 `ProjectionWeight.grad` 决定返回 dW 还是 `None`；两段权重时 wgrad 过渡实现是单段 kernel 算 `[M+R]` 再 `split(first_slot)` 分发到各段的 `OutSpec`。`beta=1` 的正确性验收是"累加两次等于单次 ×2"与"空组不清零已有梯度"（§8.1）。

### 12.5 op 级 harness 骨架（K6，`tests/ops/grouped_gemm/test_backend.py`）

```python
import pytest, torch
from xtuner.v1.module.dispatcher import rows_from_counts, rows_from_psum

def make_case(valid: list[int], alignment: int, K: int, N: int, *, tail: int = 0, nan_padding: bool = False):
    """同一份 x / w / rows 喂所有后端。valid 可含 0（空组）；tail 是不属于任何组的容量尾部。"""
    v = torch.tensor(valid, device="cuda", dtype=torch.int32)
    if alignment == 1:
        rows = rows_from_counts(v)
    else:
        aligned = ((v + alignment - 1) // alignment) * alignment
        starts = torch.cumsum(aligned, 0) - aligned
        rows = rows_from_psum(starts + v, v, alignment=alignment, padding_zeroed=not nan_padding)
    total = int((rows.starts + rows.compute_counts)[-1])
    x = torch.zeros(total + tail, K, device="cuda", dtype=torch.bfloat16)
    valid_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)
    for s, c in zip(rows.starts.tolist(), valid):                        # 测试里允许 host sync
        x[s : s + c].normal_(); valid_mask[s : s + c] = True
    if nan_padding:
        x[:total][~valid_mask[:total]] = float("nan")                    # 验证 mask / 清零，不能靠约定
    w = torch.randn(len(valid), N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    return x.requires_grad_(), w.requires_grad_(), rows, valid_mask, total

def reference(x, w, rows):
    out = torch.zeros(x.shape[0], w.shape[1], device="cuda", dtype=torch.float32)
    counts = rows.valid_counts if rows.valid_counts is not None else rows.compute_counts
    for g, (s, c) in enumerate(zip(rows.starts.tolist(), counts.tolist())):
        out[s : s + c] = x[s : s + c].float() @ w[g].float().T
    return out

@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "deepgemm"])                # 注册表落地后改为 registry.names()
@pytest.mark.parametrize("valid,alignment,tail", [([200, 0, 130, 77], 1, 0), ([200, 0, 130, 77], 128, 256)])
def test_fwd_bwd_matches_reference(backend, valid, alignment, tail):
    x, w, rows, mask, total = make_case(valid, alignment, K=2048, N=1536, tail=tail)
    out = run_backend(backend, x, w, rows)                                # 适配器入口，如 name_group_gemm
    torch.testing.assert_close(out.float()[mask], reference(x, w, rows)[mask], rtol=2e-2, atol=2e-2)   # L1
    assert torch.all(out[:total][~mask[:total]] == 0)                     # padding 行精确零
    (out.float()[mask] ** 2).mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
    assert torch.all(w.grad[[i for i, c in enumerate(valid) if c == 0]] == 0)   # 空组 dW 为零
```

L0（bit-exact）沿用跨树对比脚本，不在这里做；`beta=1` 与两段权重的用例在 K4 / K5 落地时加到同一文件。

### 12.6 性能矩阵脚本骨架（K6）

```python
import json, time, torch

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters                                      # ms

SHAPES = {"qwen3_w13": (2048, 1536), "qwen3_w2": (768, 2048)}            # (K, N)
BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]                          # expected_m 桶

table = {}
for name, (K, N) in SHAPES.items():
    for m in BUCKETS:
        x, w, rows, *_ = make_case([m] * 128, 128, K, N)                  # 128 组、均匀 m 行
        for backend in ["triton", "deepgemm"]:
            ms = bench(lambda: run_backend(backend, x, w, rows))
            tflops = 2 * 128 * m * K * N / ms / 1e9
            table[f"sm90|fwd|bf16|EA|{N}|{K}|{m}|1"] = table.get(...)   # 取最快后端名，写离线 autotune 表
json.dump(table, open("grouped_gemm_autotune_sm90.json", "w"), indent=1)
```

表的 key 与 §5.2 一致；dgrad / wgrad 各跑一遍；两段权重与 FP8 加维度。端到端验证仍以 8 卡训练吞吐为准。

### 12.7 提交与评审清单

- op 是 `custom_op` + `register_fake`，`tests/module/test_moe_block_rows.py::test_*_compiles_fullgraph` 式的编译测试通过。
- nsys 下 GEMM 路径零 `cudaStreamSynchronize`、零 `.cpu()`（R2）。
- padding 行精确零、容量尾部不读、空组不清零（§12.2）三条各有测试。
- `GemmCaps` 声明与实际行为一致，`supports()` 对不满足的 `GemmSpec` 返回 False 而不是运行期抛错。
- 跨树逐位对比对现有后端仍 BIT-EXACT（登记进注册表不改数值）。
- 文档：本文 §6 的行更新状态；新增环境变量或配置写进 `docs/`。

---

## 13. 独立成库的考虑（不现在拆，但现在就按这个方向写）

直接拆成独立库步子太大：注册表还没有实现，第二个消费框架也还没出现。但这份设计的分层本来就朝着"一个库、多个框架"走，只要现在守住两条规则（§13.2），将来拆包就是搬目录加一个 `setup.py`；守不住，做到 S3 时会发现它长在 XTuner 里拔不出来。

### 13.1 哪些天然可移植，哪些是 XTuner 的胶水

| 层 | 可移植性 | 说明 |
|---|---|---|
| 行布局与布局家族（`ExpertRows`、`Layout`） | 高 | 其他框架的分组信息都能映射进来：Megatron 的 host `tokens_per_expert` 列表就是 `rows_from_counts` 加 `host_counts`；torchtitan 给 `torch._grouped_mm` 的 `offs` 就是 `rows_from_psum` 的 E0 情形；vLLM / SGLang 的 `moe_align_block_size` 产出的就是按 block 对齐的 EA；decode 的 masked GEMM 就是 M。这层是真正的公共语言 |
| 三个无 autograd 的 op 加 `register_fake` | 高 | 只依赖 torch，任何框架都能调 |
| `GemmCaps` / `GemmSpec` / 注册表 / 选择器 / autotune 表 | 高 | 与框架无关，只要 `GemmSpec` 由框架侧拼 |
| 量化辅助（`per_tile_quant`、`trans_quant_kgrouped`、`quantize_reference`）与 SF 格式约定 | 中 | 现在住在 `xtuner.v1.float8`，独立成库时必须搬进库里 |
| FP8 激活的表示 | 低 | 现在是 XTuner 的 `Float8Tensor` 包装类；库的 op 边界应只收裸的 `(data, scale)` 两个张量，包装留给框架 |
| autograd 组装、`MoEConfig.grouped_gemm`、`XTUNER_EXPERT_GEMM_BACKEND`、对齐回传给 dispatcher | 框架侧 | 设计已明确"autograd 留在框架层"，这部分每个框架各写一份薄适配：XTuner 是 `GroupedLinear`，Megatron 是 `GroupedMLP` / `TEGroupedMLP`，torchtitan 是 `_grouped_mm` 的调用处 |
| 权重段与 `GradBinding`（MoonEP / UltraEP 的 call-local 权重） | 中 | 概念通用，但只有接了均衡器的框架才用到；库里只需支持"多段权重加每段一个 `OutSpec`" |

### 13.2 现在就要守住的两条规则

1. **库目录里不 import 任何 `xtuner.*`。** §3 建议的落点 `xtuner/v1/ops/moe/grouped_gemm/` 是树内路径，但目录内部（契约、适配器、量化核、harness）按独立包来写：op 边界只收裸张量与 Python 标量；配置从参数进来，不读 `MoEConfig` 或 `XTUNER_*` 环境变量，那是 `build_grouped_linear` 的事。只要守住这条，拆包就是把目录搬出去。
2. **契约类型的归属要能翻转。** 今天 `ExpertRows` 与 `Layout` 定义在 `dispatcher/base.py`，GEMM 库 import 它；独立成库时反过来，库拥有 rows 与 layout 的定义，XTuner 从库里 import 并重导出为 `ExpertRows`。字段不变，改的只是 import 方向。为此 `contracts.py` 里的行布局类型要与 `ExpertRows` 字段一一对应，不加只有 XTuner 才懂的字段。

### 13.3 对其他框架的价值与接入要求

- 算子组已经维护两个独立仓库（InternLM/GroupedGEMM、AdaptiveGEMM），DeepGEMM、TE grouped GEMM、SonicMoE 也都是独立库，它们都是"一个 kernel 加一个 Python 入口"。共同缺的正是本文上面两层：跨 kernel 的行布局契约，加按 setup 期静态信息选后端的注册表。这是新库对其他框架的价值，不是"又一个 kernel"。自然形态是第三个仓库，依赖前两个，把 DeepGEMM 与 TE 当可选后端。
- 只有一条接入要求是框架必须配合的：对齐与 SF 格式的协商是双向的，库声明 `required_alignment` 与 `preferred_sf_format`，框架要把它传给 dispatcher 与量化核。不做这条回传，库只能收到 E0 布局，DeepGEMM psum 路径用不上，收益退化成"多了几个后端可选"。这条要写进库的接入说明。
- 均衡器相关的权重段（§3.2）对没有均衡器的框架是可选能力，`segments` 长度为 1 时退化成普通权重。

### 13.4 什么时候拆

S1 到 S2 树内开发，按 §13.2 的规则写；出现第二个消费框架，或者 DeepGEMM / AdaptiveGEMM / TE 的版本钉法开始与 XTuner 镜像节奏冲突时，再拆。拆包时的验收：库的 op 级 harness（§8.1）在库仓库里独立运行，XTuner 侧只剩适配层与跨树逐位对比。

## 附录 A：现有 kernel 入口与源码锚点

| 入口 | 位置 |
|---|---|
| `ExpertRows` / `rows_from_counts` / `rows_from_psum` / `rows_psum` / `check_rows`；`ExpertWeights` / `ProjectionWeight` / `GradBinding` | [xtuner/v1/module/dispatcher/base.py](../../xtuner/v1/module/dispatcher/base.py) |
| `GroupedLinear.forward(x, rows, *, x_scale, weight)`、`use_deepgemm(rows)`、`build_grouped_linear` | [xtuner/v1/module/grouped_linear/moe_group_linear.py](../../xtuner/v1/module/grouped_linear/moe_group_linear.py) |
| `get_expert_gemm_backend()`（`XTUNER_EXPERT_GEMM_BACKEND`）、import 期 `group_gemm` 绑定 | [xtuner/v1/ops/moe/__init__.py](../../xtuner/v1/ops/moe/__init__.py) |
| DeepGEMM 适配器：`moe::deepgemm_m_grouped_{bf16_nt,bf16_nn,fp8_nt}`、`moe::deepgemm_k_grouped_fp8_nt`、`_DeepGemmBF16` / `_DeepGemmFP8`、`deepgemm_group_gemm` / `deepgemm_fp8_group_gemm`、`unwrap_fp8_activation`、`row_major_scales`、`_out_buffer` | [xtuner/v1/ops/moe/cuda/group_gemm_deepgemm.py](../../xtuner/v1/ops/moe/cuda/group_gemm_deepgemm.py) |
| Triton `m_grouped_gemm(A, B, size_per_group, trans_b)` / `k_grouped_gemm(A, B, size_per_group)`；autograd 包装 `GroupedGemm` | [xtuner/v1/ops/moe/cuda/group_gemm.py](../../xtuner/v1/ops/moe/cuda/group_gemm.py)、[xtuner/v1/ops/moe/cuda/triton_kernels/](../../xtuner/v1/ops/moe/cuda/triton_kernels/) |
| legacy FP8：`m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous`、`k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous`（AdaptiveGEMM）；`TileWiseFloat8GroupedLinear.forward(input, rows, *, x_scale, weight)` | [xtuner/v1/float8/float8_gmm_tile_wise.py](../../xtuner/v1/float8/float8_gmm_tile_wise.py) |
| 量化核：`per_tile_quant`、`trans_per_block_quant_expand_128x`、`trans_per_tile_quant_expand_128x`、`trans_quant_kgrouped`、`static_ks_host` | [xtuner/v1/float8/triton_kernels/](../../xtuner/v1/float8/triton_kernels/) |
| 路由权重逐行乘 `row_scale` / `row_scale_bwd` | [xtuner/v1/ops/moe/cuda/triton_kernels/row_scale.py](../../xtuner/v1/ops/moe/cuda/triton_kernels/row_scale.py) |
| 单卡 kernel 路径 harness（8 条路径、fullgraph） | [tests/module/test_moe_block_rows.py](../../tests/module/test_moe_block_rows.py) |
| 契约单测 | [tests/module/dispatcher/test_expert_rows.py](../../tests/module/dispatcher/test_expert_rows.py) |
| `grouped_gemm.backend.gmm(a, b, batch_sizes, trans_a, trans_b, c, num_sm)`（CUTLASS） | InternLM/GroupedGEMM `grouped_gemm/backend.py`；`csrc/grouped_gemm.cu`（Sm80 模板，`kMaxExperts`） |
| `te_grouped_gemm.general_grouped_gemm(..., m_splits: list[int])`；`m_grouped_gemm_dual_weight(A, B_master, B_replica, size_per_group, trans_b)` | PR #2050 `xtuner/v1/ops/moe/cuda/group_gemm_te.py`、`m_grouped_gemm_TMA_triton3_4.py` |
| `swiglu_per_tile_quant_with_trans_per_block` | PR #2089 `xtuner/v1/float8/triton_kernels/swiglu_dual_layout_quant.py` |
| `deep_gemm.m_grouped_{bf16,fp8}_gemm_{nt,nn}_contiguous(a, b, d, grouped_layout, ..., use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout)`；`k_grouped_{bf16,fp8}_gemm_tn_contiguous(a, b, d, ks_cpu, grouped_layout, c, ...)` | DeepGEMM `csrc/apis/gemm.hpp`；`deep_gemm/utils/math.py` 的 `per_token_cast_to_fp8 / per_block_cast_to_fp8` |
| `sonicmoe.functional.moe_general_routing_inputs(...)`；`quack.gemm_interface.gemm / gemm_act / gemm_dact` | Dao-AILab/sonic-moe `sonicmoe/functional/__init__.py` |
| `mindspeed.core.fusions.grouped_matmul.Ops.gmm(x, w, split_sizes, trans_b)` | [xtuner/v1/ops/moe/npu/group_gemm.py](../../xtuner/v1/ops/moe/npu/group_gemm.py) |

## 附录 B：DeepGEMM 在 XTuner main 的实际用途

上游镜像从 v2.1.1.post3 升到 v2.6.1（#2080），树内唯一直接调用在 GLM-5.2 DSA 索引器（`xtuner/v1/ops/sparse_mla/lmdeploy_fp8_index.py` 的 `fp8_mqa_logits` / `fp8_fp4_mqa_logits`）。MoE 的 FP8 grouped GEMM 走 AdaptiveGEMM。`feat/deepepv2` 分支是第一个让 MoE 路径用上 `deep_gemm` psum 布局的地方，用的是本机 `/opt/ep-v2-site` 里的 DeepGEMM（psum 布局在上游 #280 引入，2.1.1 没有）。
