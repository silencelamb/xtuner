# DeepEP V2 + EP/FSDP 解耦：实验分支交接说明（2026-09-28）

本文只存在于实验分支 `exp/deepepv2-decouple`，不进任何要提 PR 的分支。读完它应该能够：看懂几条分支的关系、
搭好环境、复现已有的数、知道接下来做什么。文中的提交号、PR 状态核对于 2026-09-28。

机器用内部编号 2296、1727、2999 指代，都是单机 8 × H200。

## 0. 概要

| 事项 | 状态 |
|---|---|
| Qwen3-30B-A3B（BF16）接入 DeepEP V2 | 完成。单机 8 卡 EP4 mb1 比 DeepEP v1 +9.9% |
| GLM-5.2-30B 缩层版（FP8）接入 DeepEP V2 | 能跑、loss 一致，耦合布局下吞吐只 +0.8%。mb2 在耦合布局下贴显存上限，要配合 EP/FSDP 解耦。解耦加最终修复后 mb1 比耦合快 1.8%、显存少 17.9 GB，mb2 比 mb1 慢 1.1% |
| EP/FSDP 解耦（PR #2093） | OPEN，等 review。head `7c53b7da`，含专家组按层 reshard 与反向预取的修复 |
| 本分支 | 已合入 PR #2093 的 `7c53b7da`（合并提交 `ba26df49`）。合并后测试全部通过（§4.3），并跑了第一批「最终修复 + DeepEP V2」的基准（§5.4） |
| 「最终修复 + DeepEP V2」的结论 | Qwen3：解耦与耦合吞吐持平略好（+0.6%），显存少 8.3 GB。trace 的 all-gather 计数与预期逐项相同 |
| `torch.compile` 兼容 | 完成，所有基准行都开着编译跑 |
| CUDA graph | 只在 2 卡单测里验证了静态模式的 MoE 段可捕获、可重放，没有接进训练步 |
| `feat/deepepv2` 提 PR | 未提。对上游 main 有 3 处冲突，要先 rebase |
| GLM FP8 优化、`cpu_sync=False`、UltraEP | 未开始或做到一半，见 §7 |

名词：**耦合布局**指上游现有的做法，EP 与 FSDP 的 shard 维绑在一起，dense 参数被复制 EP 份；
**解耦布局**指 PR #2093 的 `decouple_ep_fsdp=True`，dense 参数在整个 FSDP mesh 上切分，专家在 EP 之上再切 `dp_shard / ep` 份。
**mb** 指层内微批 `intra_layer_micro_batch`。

## 1. 分支

仓库：fork `silencelamb/xtuner`（下文的 `origin`），上游 `InternLM/xtuner`（下文的 `upstream`）。

| 分支 | tip | 内容 | 用途 |
|---|---|---|---|
| `feat/deepepv2` | `5c16724f` | 上游 `7d377424` + `9e557445` [Refactor] 统一 EP 契约 + `5c16724f` [Feature] DeepEP V2 dispatcher 与 DeepGEMM psum | 主题分支，将来提 PR |
| `pr/decouple-ep-fsdp` | `7c53b7da` | 上游 `7d377424` + 14 个提交 | 主题分支，PR #2093 |
| `feat/decouple-ep-fsdp` | `89e00c81` | 同上，多一份 `reports/` | PR #2093 的开发分支，PR 描述链接到它的报告 |
| `exp/deepepv2-decouple` | 本文所在 | 前两条主题分支的合并 + `[Bench]` 提交 + 本文 | 实验分支，只用来跑基准 |
| `archive/deepepv2-parallel-path` | `096f697e` | 早期的并行新路径实现 | 已弃用，留档 |

本分支的历史：

```
*  本文与 DeepEP 补丁                                       [Docs]
*  ba26df49  Merge origin/pr/decouple-ep-fsdp (7c53b7da)    第二次合并，带进 4 个修复提交
*  e0375d9a  [Bench] Wire DECOUPLE_EP_FSDP / HSDP_SHARDING_SIZE into sft_moe_ep_bench.py
*  71695355  Merge origin/pr/decouple-ep-fsdp (212c05ee)    第一次合并
*  5c16724f  feat/deepepv2 的 tip
```

第二次合并带进来的 4 个提交：

| 提交 | 内容 |
|---|---|
| `90f6e5e8` [Fix] | 专家 FSDP 组按层 reshard，不再每次调用后 reshard |
| `1edc86d9` [Fix] | 专家组的反向预取目标指向下一层 |
| `d86987d3` [Test] | 层内微批 + 重算下的专家 all-gather 计数门 |
| `7c53b7da` [Docs] | 设计文档 `docs/design/decouple_ep_fsdp.md` §3.3.1、§5 |

它们修的是解耦布局加层内微批时专家组被反复 all-gather 的问题（mb4 时每层 13 次降到 2 次）。
2026-09-20 到 09-23 在 `e0375d9a` 上跑的解耦行都没有这个修复。

## 2. 分支怎么维护

### 2.1 三类分支

| 类别 | 命名 | 基于 | 放什么 | 能否提 PR |
|---|---|---|---|---|
| 主题分支 | `feat/*`、`fix/*`、`pr/*` | `upstream/main`，或另一条主题分支 | 一个 PR 的全部提交，且只有这些 | 能 |
| 实验分支 | `exp/*` | 若干主题分支的合并 | 合并提交、`[Bench]` 提交、本文 | 不能 |
| 备份 | `backup/<分支>-<日期>` | — | force-push 之前的旧 tip | 不能 |

### 2.2 实验分支不会自动跟随主题分支

`git merge` 记录的是被合并的那个提交。主题分支之后再推送，包括 PR 更新，本分支里的合并提交不变，要手动同步。
跑基准之前先查一遍：

```bash
git fetch origin
# 本分支合进来的是哪些提交（parents 的第二个就是当时主题分支的 tip）
git log --merges --format='%h parents=%p %s' origin/feat/deepepv2..exp/deepepv2-decouple
# 主题分支比本分支多出的提交；没有输出表示已同步
git log --oneline exp/deepepv2-decouple..origin/pr/decouple-ep-fsdp
git log --oneline exp/deepepv2-decouple..origin/feat/deepepv2
```

### 2.3 同步：只追加就 merge，被重写就重建

先判断主题分支是只追加了提交，还是被 amend、rebase、force-push 重写过。
参数是本分支当时合进来的旧 tip 和主题分支现在的 tip：

```bash
git merge-base --is-ancestor <旧 tip> origin/<主题分支> && echo 追加 || echo 重写
# 2026-09-28 之后对 PR #2093 的判断：
git merge-base --is-ancestor 7c53b7da origin/pr/decouple-ep-fsdp && echo 追加 || echo 重写
```

| 结果 | 做法 | 原因 |
|---|---|---|
| 追加 | `git merge --no-ff origin/<主题分支>` | 旧提交还在新历史里，只会带进新增的提交 |
| 重写 | 重建本分支 | 旧提交与新提交内容相同、哈希不同，merge 会把同一份改动合两遍，产生冲突或重复 |

所以 PR #2093 之后如果是**追加新提交**，再 merge 一次即可；如果为了保持 PR 历史干净而 **amend 或 rebase 了已有提交**，就要重建。
`feat/deepepv2` 的约定是改动 amend 进对应的提交、不新增提交，所以它每次更新都属于重写。

合并之前可以先在内存里试合并，不改任何工作区。第一行是结果树的哈希，有冲突时后面会列出冲突文件：

```bash
git merge-tree --write-tree --name-only exp/deepepv2-decouple origin/pr/decouple-ep-fsdp
```

追加时的同步：

```bash
git switch exp/deepepv2-decouple && git status     # 必须干净；确认没有作业在用这棵树
git merge --no-ff origin/pr/decouple-ep-fsdp
git push origin exp/deepepv2-decouple
```

重写时的重建：

```bash
git switch exp/deepepv2-decouple && git status
# 记下要保留的、只属于本分支的提交：[Bench] 提交，以及本文和补丁的提交
git log --oneline --no-merges --grep='^\[Bench\]' HEAD --not origin/feat/deepepv2 origin/pr/decouple-ep-fsdp
git log --oneline -- HANDOVER.md patch/deepep_v2
git branch backup/exp-deepepv2-decouple-$(date +%m%d) HEAD    # 结果表引用的旧提交要保持可达
git reset --hard origin/feat/deepepv2
git merge --no-ff origin/pr/decouple-ep-fsdp
git cherry-pick <上面记下的提交>
git push --force-with-lease origin exp/deepepv2-decouple
```

同步或重建之后：跑 §4.3 的测试；更新本文 §1 的提交号；结果表里新旧代码的行分开写，标明提交号。

### 2.4 代码改动落在哪条分支

1. 判断改动属于哪个 PR，切到那条主题分支修改、测试、提交、推送。
2. 回到本分支，按 §2.3 同步，再跑基准。
3. 在本分支上调试时临时改的代码，定稿前搬回主题分支：在主题分支 `git cherry-pick <提交>`，
   再把本分支退回合并前的位置、重新合并主题分支。
4. 只为做 A/B 的环境变量开关用 `[Bench]` 标签，留在 `exp/*`，不进主题分支。

提交与 PR 的规则在 `.claude/CLAUDE.md`：提交信息 `[Tag] 描述`；一个 PR 一件事；bug 修复带回归测试；
diff 最小化，不改没动到的行的格式；动了前向、反向或通信路径的给改动前后的基准数。

### 2.5 各自单独提 PR

PR 从主题分支提。本分支里的合并提交不会流回主题分支，所以两个 PR 的 diff 互不包含。
两条主题分支只有 `xtuner/v1/model/moe/moe.py` 一个文件都改，自动合并无冲突。提 PR 之前检查：

```bash
git fetch upstream '+refs/heads/main:refs/remotes/upstream/main'
git log --oneline upstream/main..origin/feat/deepepv2                               # 只应有这个 PR 的提交
git merge-tree --write-tree --name-only upstream/main origin/feat/deepepv2          # 对上游 main 能否直接合并
```

同时依赖两条线的改动归后合入的那个 PR。现在只有一处：`e0375d9a`，基准配置文件是 `feat/deepepv2` 新增的，开关是 PR #2093 引入的。

### 2.6 叠在另一条主题分支上的工作

UltraEP 接入、静态模式的后续改动等，从 `feat/deepepv2` 开新的主题分支，各自成为单独的 PR。
从 fork 向上游提 PR 只能以上游的分支为 base，底座没合入之前直接提，diff 里会带着底座的全部提交。
这段时间用 fork 内部的比较视图给框架组看，只显示新分支自己的改动：

```
https://github.com/silencelamb/xtuner/compare/feat/deepepv2...feat/<新分支>
```

也可以在 fork 里开一个 draft PR（base 选 `feat/deepepv2`），便于逐行评论。底座合入上游之后，把新分支 rebase 到 `upstream/main` 再提正式 PR。

## 3. 环境

镜像：torch 2.9.1+cu128、Python 3.12、nvcc 12.8。默认 site-packages 里是 DeepEP 1.2.1（legacy `Buffer`）、DeepGEMM 2.1.1（无 psum 布局）、NCCL 2.27.5。
V2 的依赖装在隔离目录，只在跑 V2 时用 `PYTHONPATH` 前置，legacy 路径不受影响。

| 项 | 版本 | 说明 |
|---|---|---|
| NCCL | `nvidia-nccl-cu12==2.31.2`，原地升级到默认 site | DeepEP V2 需 ≥ 2.30.4。`deep_ep._C` 的 RUNPATH 指向默认 site 的 `nvidia/nccl/lib`，所以必须原地升级。`torch.cuda.nccl.version()` 仍显示编译期的 2.27.5 |
| DeepEP V2 | `deepseek-ai/DeepEP` @ `dd758ca` + `patch/deepep_v2/deepep_v2_capped_capacity.patch` | 补丁给 `dispatch` 加 `num_max_expanded_tokens` 与 `overflow_flag`，静态模式的接收容量可以小于最坏值。不打补丁时 `capacity_factor` 不可用，其余功能正常 |
| DeepGEMM | `deepseek-ai/DeepGEMM` @ `88965b0` | `__version__` 显示 2.5.0，有 psum 布局。2.8.0 接口兼容但 H200 上无收益，没有升 |
| AdaptiveGEMM | 镜像自带 | FP8 的 wgrad 用它（§7.3） |

```bash
pip install --no-deps nvidia-nccl-cu12==2.31.2
mkdir -p /opt/ep-v2-site /opt/ep-cache/{deepep-jit,deepgemm-jit,triton,inductor}

git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP && git checkout dd758ca
git apply <本仓库>/patch/deepep_v2/deepep_v2_capped_capacity.patch
pip wheel --no-build-isolation --no-deps -w dist . && pip install --no-deps --upgrade --target /opt/ep-v2-site dist/*.whl

git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git && cd DeepGEMM && git checkout 88965b0 && git submodule update --init --recursive
pip wheel --no-build-isolation --no-deps -w dist . && pip install --no-deps --upgrade --target /opt/ep-v2-site dist/*.whl

# legacy 代码仍用旧的导入名 deep_ep_cpp，给 V2 的扩展名 deep_ep._C 加一个 shim
cat > /opt/ep-v2-site/deep_ep_cpp.py <<'EOF'
from deep_ep._C import *  # noqa: F401,F403
from deep_ep._C import EventHandle  # noqa: F401
EOF
```

验收：

```bash
strings $(python3 -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))")/lib/libnccl.so.2 | grep '^NCCL version'   # 2.31.2
PYTHONPATH=/opt/ep-v2-site python3 -c "import inspect, deep_ep; print('num_max_expanded_tokens' in inspect.signature(deep_ep.ElasticBuffer.dispatch).parameters)"
PYTHONPATH=/opt/ep-v2-site python3 -c "import deep_gemm; print('use_psum_layout' in deep_gemm.m_grouped_bf16_gemm_nt_contiguous.__doc__)"
python3 -c "import torch._inductor.compile_fx"      # torch.compile 可用
```

注意：

- `/opt` 在我们的机器上是容器本地的，换容器要重装。三台机器共享一块网络盘，编好的 wheel 放在共享盘上可以直接复用。
- 在一个容器里遇到过 `import torch._inductor.compile_fx` 报 `AssertionError: duplicate template name`，
  原因是 `torch/_inductor/kernel/` 下有旧版 torch 残留的 4 个文件（`flex_attention.py`、`flex_decoding.py`、`mm_scaled.py`、`unpack_mixed_mm.py`），移走即可。
- GLM-5.2 的稀疏注意力后端用 `SPARSE_MLA_BACKEND=tilelang`。`cudnn_dsa` 需要另装 `nvidia-cutlass-dsl==4.7.0`。

## 4. 怎么跑

### 4.1 一行基准

配置 `examples/v1/config/sft_moe_ep_bench.py` 全部由环境变量驱动。下面是 GLM-5.2、DeepEP V2、解耦、mb2 的一行：

```bash
cd <本仓库>
export PYTHONPATH=/opt/ep-v2-site:$PWD
export EP_DISABLE_GIN=1                       # 单机，不需要网卡数据通路
export EP_JIT_CACHE_DIR=/opt/ep-cache/deepep-jit DG_JIT_CACHE_DIR=/opt/ep-cache/deepgemm-jit
export TRITON_CACHE_DIR=/opt/ep-cache/triton TORCHINDUCTOR_CACHE_DIR=/opt/ep-cache/inductor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MODEL_PATH=<GLM-5.2-30B 缩层检查点> CHAT_TEMPLATE=glm5.2 FP8=1 SPARSE_MLA_BACKEND=tilelang
export ALPACA_PATH=<alpaca_openai.jsonl>
export DISPATCHER=deepep_v2 EP_SIZE=4 TOTAL_STEP=20
export INTRA_LAYER_MICRO_BATCH=2 PACK_MAX_LENGTH=8192 GLOBAL_BATCH_SIZE=16
export XTUNER_EXPERT_GEMM_BACKEND=auto DEEPEP_V2_ALIGN=128 DEEPEP_V2_CPU_SYNC=1 DEEPEP_V2_FP8_DISPATCH=1
export DECOUPLE_EP_FSDP=1
export MODEL_COMPILE=1 TORCH_COMPILE=1 RECOMPUTE_RATIO=1.0 XTUNER_USE_FA3=1
export WORK_DIR=work_dirs/glm52_deepep_v2_mb2_pack8192_ep4_dec1 && mkdir -p $WORK_DIR

torchrun --nproc-per-node 8 --master-port 29650 \
  xtuner/v1/train/cli/sft.py --config examples/v1/config/sft_moe_ep_bench.py 2>&1 | tee $WORK_DIR/stdout.log
```

| 变量 | 取值 | 含义 |
|---|---|---|
| `DISPATCHER` | `all2all` / `deepep` / `deepep_v2` | `deepep` 是 DeepEP v1。跑 legacy 两种时 `PYTHONPATH` 里不要放 `/opt/ep-v2-site` |
| `INTRA_LAYER_MICRO_BATCH`、`PACK_MAX_LENGTH`、`GLOBAL_BATCH_SIZE` | 见 §4.2 | 三个一起改，保持每卡每步 token 数不变 |
| `XTUNER_EXPERT_GEMM_BACKEND` | `auto` / `legacy` / `deepgemm` | `auto`：行布局 128 对齐且 DeepGEMM 可导入时用 DeepGEMM psum |
| `DEEPEP_V2_ALIGN` | `1` / `128` | 接收行的对齐。128 配 DeepGEMM，1 配 legacy kernel |
| `DEEPEP_V2_CPU_SYNC` | `1` / `0` | 0 是静态模式（§7.4） |
| `DEEPEP_V2_FP8_DISPATCH` | `0` / `1` | dispatch 之前量化，通信传 fp8。只对 FP8 模型有意义 |
| `DEEPEP_V2_CAPACITY_FACTOR` | 浮点数 | 只在静态模式下有效，需要打过补丁的 DeepEP |
| `DECOUPLE_EP_FSDP` | `0` / `1` | 解耦布局 |
| `HSDP_SHARDING_SIZE` | 整数 | 解耦布局下开 HSDP |
| `NUM_HIDDEN_LAYERS` | 整数 | 裁层，配 `STRICT_LOAD=0` |
| `PROFILE_TIME=1 PROFILE_STEP=8` | | 导出第 8 步的 torch profiler trace，配 `TOTAL_STEP=10` |

Qwen3：`MODEL_PATH=<Qwen3-30B-A3B> CHAT_TEMPLATE=qwen3 FP8=0`，不设 `SPARSE_MLA_BACKEND` 与 `DEEPEP_V2_FP8_DISPATCH`。

多机：每台机器跑同一条命令，`torchrun` 加 `--nnodes N --node-rank R --master-addr <node0>`，`WORK_DIR` 放在共享盘上，
另设跨机的 NCCL 环境变量，去掉 `EP_DISABLE_GIN=1`。

### 4.2 口径

- 单机 8 卡，20 步，取第 5 步及之后的均值，只读 rank 0 的日志行。
- 每卡每步 token 数固定。16K 口径：mb1 = pack 16384 × GBS 8，mb2 = pack 8192 × GBS 16。
  64K 口径：pack 16384 × GBS 32 × mb4，或 pack 32768 × GBS 16 × mb2。
- 日志每步一行，含 `time:`（步时）、`tgs:`（**单卡**每秒 token）、`max_memory: … GB`（峰值 allocated）、`reduced_llm_loss:`。
- 步时的 std 用来识别显存抖动或被干扰的行，std 大的行不能拿来比较。
- A/B 只在同一批、交错跑的行之间比，n ≥ 2。跨机、跨天的批次漂移有 0.3% ~ 3.7%，与要测的效应同量级。

### 4.3 测试

```bash
# 契约与行布局，CPU 即可
PYTHONPATH=$PWD python3 -m pytest -q tests/module/dispatcher/test_expert_rows.py
# DeepEP V2 与 DeepGEMM 路径：单卡与 2 卡
EP_DISABLE_GIN=1 PYTHONPATH=/opt/ep-v2-site:$PWD python3 -m pytest -q \
  tests/module/test_moe_block_rows.py tests/module/dispatcher/test_deepep_v2.py
# legacy dispatcher 回归（DeepEP v1，不要前置 /opt/ep-v2-site）
PYTHONPATH=$PWD python3 -m pytest -q tests/module/dispatcher/test_deepep.py
# 解耦布局：mesh 与参数摆放（需要 1 张卡）、8 卡的 L1~L3 门与 all-gather 计数门
PYTHONPATH=$PWD python3 -m pytest -q tests/model/test_decoupled_ep_fsdp_mesh.py
PYTHONPATH=$PWD python3 -m pytest -q tests/engine/test_decoupled_ep_fsdp_train_engine.py
```

各自最近一次的结果：

| 代码 | 测试 | 结果 |
|---|---|---|
| `feat/deepepv2` @ `483c7c42`（tip `5c16724f` 相对它只改了基准配置 5 行） | 前三组，加上对上游 `7d377424` 的逐位对比 66 例 | 2026-09-20，1727：逐位对比 66/66，legacy 测试 14 通过 1 跳过，契约与 V2 测试 27 通过 |
| `pr/decouple-ep-fsdp` @ `7c53b7da` | 解耦相关的测试 | 2026-09-27，2999：51 通过；all-gather 计数门另外连跑 5 次，全部通过 |
| 本分支 @ `adec879a`（合并之后） | 对上游 `7d377424` 的逐位对比，EP1 / EP2 / EP8 | 2026-09-28，2296：66 / 66 逐位相等 |
| 本分支 @ `adec879a` | legacy 测试；契约与 V2 测试 | 2026-09-28，2296：14 通过 1 跳过；27 通过 |
| 本分支 @ `adec879a` | 解耦相关的测试；all-gather 计数门单独重复 3 次 | 2026-09-28，2296：51 通过；3 / 3 通过 |

本分支合并后的树与合并前在内存里试合并得到的树哈希相同（`4629cb81`），`moe.py` 是自动合并、无冲突。
文本上无冲突不等于行为正确，所以合并之后跑了上表的测试和 §5.4 的基准。现有测试里没有「解耦 + DeepEP V2 同时打开」的用例，
这个组合靠 §5.4 的基准行和 trace 验证。以后再合并或重建本分支，都按这个顺序重做一遍。

## 5. 已有数据

全部为实测。V2 行 = `XTUNER_EXPERT_GEMM_BACKEND=auto DEEPEP_V2_ALIGN=128 DEEPEP_V2_CPU_SYNC=1`，GLM 另加 `DEEPEP_V2_FP8_DISPATCH=1`。
tgs 为单卡每秒 token，显存为 rank 0 峰值 allocated。

每批数据基于哪份代码：

| 数据 | 代码 | 含 DeepEP V2 | 含解耦 | 专家组 reshard 修复 |
|---|---|---|---|---|
| 耦合布局的行（09-17、09-20） | `feat/deepepv2` | 是 | 否 | 不涉及 |
| 解耦 + V2，无后缀的行（09-20 起） | 本分支 @ `e0375d9a` | 是 | PR #2093 @ `212c05ee` | 无 |
| 解耦 + V2，修复的 A/B（09-23） | 本分支 @ `e0375d9a` 之上加 `[Bench]` 开关提交的本地分支，未推送 | 是 | 同上 | 环境变量开关版 |
| PR #2093 最终代码的行（09-24、09-27） | `pr/decouple-ep-fsdp` 与它的修复前版本 `212c05ee` | 否，DeepEP v1 | 是 | 最终版，无开关 |
| 本分支 @ `adec879a`（09-28） | 合并后 | 是 | PR #2093 @ `7c53b7da` | 最终版，无开关 |

09-28 之前「解耦 + DeepEP V2」的数据要么在修复前的代码上，要么在开关版的修复上。最终修复与 V2 合在一起的数只有 §5.4 这一批。

### 5.1 Qwen3-30B-A3B，BF16

EP4 耦合，16K token/卡/步，1727，2026-09-20，`feat/deepepv2` @ `5c16724f`：

| dispatcher | mb1 | mb2 | 显存 |
|---|---:|---:|---:|
| DeepEP v1 | 9165 | 9390 | 69 GB |
| DeepEP V2 + DeepGEMM | 10074（+9.9%） | 10278 | 69 GB |

只去掉框架侧的 permute（`DEEPEP_V2_ALIGN=1`，legacy kernel 不变）就有 +8.2%，再换 DeepGEMM 共 +11.0%（2999，2026-09-17）。

EP4，DeepEP V2，64K token/卡/步（pack 16384 × GBS 32 × mb4），2999，2026-09-23，同批 n=3。
代码是修复的环境变量版本（本地实验分支，未推送），逻辑与最终修复相同：

| 布局 | tgs | 显存 |
|---|---:|---:|
| 耦合 | 12243 | 80.3 GB |
| 解耦，修复前 | 11608（−5.2%） | 83.3 GB |
| 解耦，按层 reshard + 反向预取 | 12324（+0.7%） | 72.0 GB |

PR #2093 最终代码，DeepEP v1，同批对照（单机 n=3 或 2，两机 n=2）：

| 工作点 | 耦合 | 解耦，修复前 | 解耦，修复后 |
|---|---|---|---|
| 单机 8 卡，EP4 mb4 64K | 11510 / 79.4 GB | 10900 / 82.6 GB | 11575 / 71.1 GB |
| 单机 8 卡，EP8 mb4 64K | 10088 / 91.3 GB | 9922 / 73.6 GB | 10190 / 73.1 GB |
| 两机 16 卡，EP8 mb4 64K | 9853 / 58.4 GB | 9168 / 51.7 GB | 9898 / 49.8 GB |
| 两机 16 卡，EP4 mb4 64K | 11277 / 51.5 GB | 10166 / 54.8 GB | 11303 / 47.7 GB |

结论：解耦是显存与配置自由度的工作，不是吞吐工作。修复之后吞吐与耦合持平，显存更少。

### 5.2 GLM-5.2-30B 缩层版，FP8

5 层（3 层 dense + 2 层 MoE）加 1 层 MTP，256 专家 top-8。

EP4，16K token/卡/步，1727，2026-09-20。耦合行 `5c16724f`，解耦行 `e0375d9a`（**修复前**）：

| dispatcher | 布局 | mb1 | mb2 | 显存 mb1 / mb2 |
|---|---|---:|---:|---|
| DeepEP v1 | 耦合 | 6366 | 3722（显存抖动，std 1.05 s） | 99.9 / 103.4 GB |
| DeepEP v1 | 解耦 | 6346 | 6158 | 84.0 / 87.0 GB |
| DeepEP V2 | 耦合 | 6416（+0.8%） | 5827（抖动，std 0.28 s） | 100.1 / 103.5 GB |
| DeepEP V2 | 解耦 | 6389 | 6159 | 84.1 / 87.0 GB |

EP4，mb2 × pack 8192，DeepEP v1，PR #2093 的代码，同批 n=2：

| 机器 | 耦合 | 解耦，修复前 | 解耦，修复后 |
|---|---|---|---|
| 单机 8 卡（2999，09-24） | 5624 / 103.4 GB | 6163 / 87.0 GB | 6435 / 83.2 GB |
| 两机 16 卡（2999 + 2296，09-27） | 5956 / 66.8 GB | 6042 / 59.2 GB | 6328 / 55.1 GB |

DeepEP V2 + 解耦 + 最终修复的 mb1 与 mb2 同批对照在 §5.4。

### 5.3 静态模式（`cpu_sync=0`）

2999，2026-09-17，EP4 耦合，容量因子 2.0：

| 模型 | 配置 | 步时 | 显存 | 对动态 V2 |
|---|---|---:|---:|---|
| Qwen3 mb1 | 容量因子 2.0 | 1.70 s | 72.0 GB | 慢 5%（动态 1.62 s） |
| Qwen3 mb2 | 容量因子 2.0 | 1.72 s | 71.6 GB | |
| GLM mb1 | 容量因子 2.0 | 11.9 s | 104.6 GB | 病态 |
| GLM mb1 | 容量因子 2.0 + `EP_AVOID_RECORD_STREAM=1` | 2.57 s | 107.6 GB | 持平 |

### 5.4 最终修复 + DeepEP V2：合并后的第一批数（2026-09-28）

代码是本分支 @ `adec879a`。测试在 2296，基准在 2999；基准的比较只在同一批、同一台机器的行之间做，n=2，按轮次交错。
每行运行中每 15 s 检查一次卡上有没有别的进程，12 行全部干净，没有重跑。

Qwen3-30B-A3B，EP4，mb4 × pack 16384 × GBS32 = 64K token/卡/步：

| 布局 | tgs 均值 | 最小~最大 | 对耦合 | 步时 std | 峰值显存 | 末步 loss |
|---|---:|---|---:|---:|---:|---|
| 耦合 | 12235 | 12223~12247 | — | 0.023 s | 80.3 GB | 1.5771~1.5773 |
| 解耦，最终修复 | 12314 | 12291~12336 | +0.6%（区间不重叠） | 0.024 s | 72.0 GB | 1.5775~1.5776 |

与 09-23 同机开关版修复的结果（12324 / 72.0 GB）相差 0.1%。

GLM-5.2-30B 缩层版，EP4，16K token/卡/步：

| 微批 | 布局 | tgs 均值 | 最小~最大 | 对耦合 mb1 | 对解耦 mb1 | 步时 std | 峰值显存 | 末步 loss |
|---|---|---:|---|---:|---:|---:|---:|---|
| mb1 | 耦合 | 6401 | 6378~6423 | — | | 0.033 s | 100.1 GB | 8.1804~8.1808 |
| mb1 | 解耦，最终修复 | 6519 | 6517~6521 | +1.8%（不重叠） | — | 0.011 s | 82.2 GB | 8.1806~8.1809 |
| mb2 | 解耦，最终修复 | 6446 | 6436~6457 | +0.7% | −1.1%（不重叠） | 0.012 s | 83.3 GB | 8.1975~8.1976 |

trace（解耦，最终修复，第 8 步，rank 0）：

| 每步 | Qwen3 mb4 | GLM mb2 |
|---|---:|---:|
| 单步时长 | 5356 ms | 2631 ms |
| 计算 kernel 忙碌占比 | 90.2% | 93.3% |
| 专家 all-gather，前向 / 重算加反向 | 48 / 48 | 3 / 2 |
| 其中发出后从未被使用 | 1 | 0 |
| FSDP all-gather 合计 | 196 | 26 |
| all-gather 总时长 / 暴露 | 261 / 38.0 ms | 89.0 / 34.7 ms |
| reduce-scatter 总时长 / 暴露 | 174 / 12.0 ms | 89.4 / 22.6 ms |

「暴露」指集合通信在跑、同时没有任何计算 kernel 在跑的时间。修复前 Qwen3 同配置是 715 次 all-gather，其中 46 次从未被使用，暴露 180 ms。
计数与 PR #2093 最终代码在 DeepEP v1 上的 trace 逐项相同。剩下的那 1 次来自首层专家组沿用默认预取目标，见 `docs/design/decouple_ep_fsdp.md` §3.3.1。

结论：

1. 最终修复在 DeepEP V2 上生效，效果与通信库无关。解耦之后吞吐不低于耦合，显存更少。
2. GLM 缩层版上修复把 mb2 相对 mb1 的差距从 −3.6% 收到 −1.1%，mb2 仍然略慢。这个模型只有 2 层 MoE，层内微批能掩盖的通信少；
   剩下的差距归因于每个微批的固定开销，按代码推断，未 profile 验证。mb2 的价值要在完整层数的模型上看。
3. 机器是多人共用的，别人的作业随时会落下来。当天在 2296 上起的第一行基准就在第 2 步被挤到 OOM。
   跑基准时每行起之前等本机空闲，运行中持续检查卡上有没有多出来的进程，出现过就整行作废重跑。

## 6. `torch.compile` 与 CUDA graph

**为兼容做的改动**（都在 `feat/deepepv2`）

| 改动 | 位置 | 原因 |
|---|---|---|
| DeepGEMM 的 4 个调用注册成 `torch.library.custom_op`，各配一个 `register_fake` | `xtuner/v1/ops/moe/cuda/group_gemm_deepgemm.py` | DeepGEMM 是 pybind 函数，dynamo trace 不了。注册后专家块的前向和反向都能进 `fullgraph=True` 的图 |
| `PostDispatchResult` 的键固定，没有内容时填 `None` 或空对象 | `xtuner/v1/module/dispatcher/base.py` | `torch.compile` 不接受带可选键的 TypedDict |
| GEMM 后端按 `rows.alignment` 分支，它是 Python 常量 | `xtuner/v1/ops/moe/__init__.py`、`grouped_linear/moe_group_linear.py` | 分支在编译期折叠；运行期不看行数，不引入 host sync |
| 行布局 `ExpertRows` 的起点与计数都是 GPU 张量，host 计数是可选项 | `xtuner/v1/module/dispatcher/base.py` | graph 捕获区内不能有 host 张量 |
| 静态模式下不对 token 维调 `torch._dynamo.mark_dynamic` | `xtuner/v1/module/decoder_layer/moe_decoder_layer.py` | 动态模式沿用上游做法；静态模式形状固定，按静态形状编译 |

**测试**

| 测试 | 卡数 | 验证了什么 |
|---|---|---|
| `test_deepep_v2.py::TestDeepEPV2GraphAndCompile::test_static_segment_is_cuda_graph_capturable` | 2 卡，EP2 | 静态模式下 dispatch → experts → combine 的前向和反向录进同一个 `torch.cuda.CUDAGraph`；把新输入拷进静态缓冲后重放，输出、dX、路由权重梯度与 eager 对比，容差 2e-2 |
| `test_deepep_v2.py::TestDeepEPV2GraphAndCompile::test_experts_compile_fullgraph_static_and_dynamic` | 2 卡，EP2 | 专家块 `fullgraph=True` 编译后接在静态 dispatcher 后面，token 数 256 和 128 各跑一次 |
| `test_moe_block_rows.py::test_deepgemm_fp8_path_compiles_fullgraph`（2 例） | 单卡 | DeepGEMM FP8 路径的专家块以 `fullgraph=True, dynamic=False` 编译，前向加反向 |

CUDA graph 那个测试的条件和范围：

- 配置是 BF16、`alignment=1`（Triton GEMM）、最坏容量、`EP_AVOID_RECORD_STREAM=1`。捕获前在另一条 stream 上预热 3 次。
- 没有覆盖：DeepGEMM 对齐布局、FP8、容量因子、层内微批的异步路径、attention 与 router、FSDP 集合通信、激活重算、优化器。
- 两个编译测试只断言结果有限，数值正确性由不编译的 `test_moe_block_matches_reference` 保证。

**训练步里的现状**

- 所有基准行都是 `MODEL_COMPILE=1 TORCH_COMPILE=1`。编译边界与上游相同（`moe.py` 的 `MOE_EP_COMPILE_CFG`）：
  专家块、attention 前后几段、shared experts。dispatcher 六段和 `_DispatchV2` / `_CombineV2` 在 eager 区。
- `torch.compile` 不会自动用 CUDA graph：编译表只写了 `fullgraph=True`，用默认 mode，Inductor 的 cudagraphs 没开。
- CUDA graph 的收益上限按 GLM-5.2 缩层版 8 卡 EP4、DeepEP v1、第 8 步的 trace 估：单步 1.579 s 里 GPU 完全空闲 56 ms，即 3.6%，
  实际能拿到 1% ~ 3%。同一步里有 831 次 `cudaStreamSynchronize`，先清这些同步点比上 graph 收益大，而且上 graph 本来就要求先清掉它们。

## 7. 后续工作

### 7.1 GLM 上 mb2 为什么仍然略慢

合并、测试和第一批基准已经做完（§4.3、§5.4）。留下的问题是 GLM 缩层版上解耦 mb2 比 mb1 慢 1.1%：

1. 同批各抓一份 mb1 与 mb2 的 trace（`TOTAL_STEP=10 PROFILE_TIME=1 PROFILE_STEP=8`），按类别比较 kernel 时间，看多出来的时间落在哪一类。
   候选是 tilelang 稀疏注意力的 indexer、MTP、dispatch 握手，以及 §7.3 里 FP8 每次调用的临时张量。
2. 用完整层数或更多 MoE 层的配置重测 mb1 对 mb2。缩层版只有 2 层 MoE，结论不能外推。
3. 16 卡两机上补「最终修复 + DeepEP V2」的行。专家组跨机之后修复的收益更大（DeepEP v1 上 Qwen3 EP4 单机 +6.2%、两机 +11.2%）。

### 7.2 `feat/deepepv2` rebase 到上游 main，准备提 PR

上游 main 在 2026-09-24 是 `e7299bbc`，比 `7d377424` 多 25 个提交，碰到本分支改动区域的有三个：

| 上游提交 | 内容 | 影响 |
|---|---|---|
| `e7299bbc` | 每个 EP rank 的负载比指标，在 `dispatch_postprocess` 之后取 grouped-GEMM 行数 | 改了 `moe_decoder_layer.py` 7 行，落在契约重构改写过的区域里，3 处冲突都来自它。静态模式下 host 上没有行数，这个指标怎么取要设计 |
| `6c6aeb3f` | DeepEP v1 的同步路径改用 `async_finish=False`，不再 `record_stream` | 自动合并。与 §7.4 是同一个根因 |
| `39d978dd` | MTP 的端到端 TV loss | 只改 `moe.py`，自动合并 |

冲突的解法是把 `e7299bbc` 那 7 行重新加到 `_moe_forward` 里对应的位置。rebase 之后重跑 §4.3 的前三组，
并对新的上游提交重做逐位对比：契约重构那个提交的承诺是 legacy dispatcher 的输出、dX 与全部参数梯度与上游逐位相等。

合入顺序建议：契约重构（`9e557445`，无行为变化）单独一个 PR 先进，DeepEP V2（`5c16724f`）第二个。
PR #2056（MoonEP）与 PR #2050（UltraEP）都改了同一批公共文件，谁先合入其余的都要 rebase，要和框架组先对齐顺序。

### 7.3 GLM-5.2 FP8 优化

**wgrad 为什么用 AdaptiveGEMM 而不是 DeepGEMM。** wgrad 是 `dW[e] = dy_e^T @ x_e`，输出形状固定为 `[E, N, K]`，
被收缩掉的维度是每个专家的 token 数。代码在 `group_gemm_deepgemm.py` 的 `_DeepGemmFP8.backward`：
装了 AdaptiveGEMM 就用它的 `k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous`，否则退回 DeepGEMM 的 `k_grouped_fp8_gemm_nt_contiguous`。
前向和 dgrad 两条路径都是 DeepGEMM psum。

| | DeepGEMM k-grouped（SM90） | AdaptiveGEMM dW |
|---|---|---|
| 输出 | 只能 fp32，且是累加语义 | bf16 直出 |
| GLM EP4 w13 的 dW 大小（64 × 3072 × 6144） | 4.8 GB | 2.4 GB |
| 每次调用的显存流量 | 清零 + 写 + 读 + 转 bf16，约 17 GB（按形状计算） | 写一遍 |
| 每组 K 的来源 | device 张量，另要一个 host 列表，用常量列表 `static_ks_host` 绕开 | device 计数 |
| 操作数布局 | SM90 只有 NT，要用 `trans_quant_kgrouped` 把两个操作数转置重量化成每组首尾相接的布局 | 128 对齐的 expand 布局 |
| 单卡微基准，真实形状，含转置重量化（实测，09-17） | 7.26 ms | 3.37 ms |

H200 上这一步受显存带宽限制，fp32 输出把要搬的字节翻倍再加一次类型转换，这是差 2 倍的主因。
DeepGEMM 2.8.0 的反向提速针对它自己的 k-grouped 路径和 SM100，SM90 的 k-grouped 仍然没有 psum 布局，结论不变。

要复核的：

1. 这份微基准的脚本没有保存下来，7.26 / 3.37 ms 目前无法复现。重写一份，形状取 GLM EP4 的 w13 与 w2。
2. 两条路径的 dW 相对差 4e-2，来自各自的重量化舍入。哪一条离 fp32 参考更近没有记录，微基准里一并测。
3. 下面三处是读代码看到的每次调用的开销，**按代码推断，未 profile 验证**，形状按 GLM EP4、16K token/卡、mb1（每 rank 约 131K 行，hidden 6144）估：
   - FP8 dispatch 打开且走 AdaptiveGEMM dW 时，`_DeepGemmFP8.forward` 把收到的 fp8 激活反量化成 bf16 再做转置量化，
     中间有一个 fp32 的 `[M, K]` 临时张量（约 3.2 GB）和一份 bf16 拷贝（约 1.6 GB）。DeepGEMM 回退路径已经在 kernel 内反量化，没有这两份。
   - dgrad 每次调用把 `W` 转置并 `contiguous()`（w13 约 1.2 GB 的 fp8 拷贝，外加 scale），legacy 路径也是这样做的。
   - `_out_buffer` 每个 GEMM 输出先 `torch.zeros`，为了让对齐 padding 行是有限零。
4. V2 在 GLM 上只 +0.8% 的原因：缩层版只有 2 层 MoE，稀疏注意力与 MTP 占比大，DeepGEMM psum 只换掉了前向和 dgrad。
   优化之前先用 profile 确认 MoE 段在单步里的占比，这决定了 FP8 GEMM 优化的上限。

给算子组的需求：wgrad 接口要支持 bf16 直出和 device 计数，见 `docs/design/unified_grouped_gemm_v1.0.md`。

### 7.4 `cpu_sync=False`（静态模式）

**它解决什么。** `cpu_sync=True` 时每次 dispatch 后 host 要等一次接收计数；`False` 时计数留在 GPU，输出形状静态，
是接 CUDA graph 的前提。它本身不提速：Qwen3 上静态模式比动态 V2 慢 5%。

**代价。** DeepEP V2 原生的静态容量是最坏值 = token/卡 × topk × EP，是均衡期望的 EP 倍。
所有按行的 kernel 和临时张量同比放大，EP 越大越不可承受。

**现在能用的配置**（Qwen3）：`DEEPEP_V2_CPU_SYNC=0 DEEPEP_V2_ALIGN=128 DEEPEP_V2_CAPACITY_FACTOR=2.0`。

- 容量因子依赖打过补丁的 DeepEP（§3）。
- 容量因子扫描（Qwen3 mb1，10 步）：1.5 第 4 步溢出，2.0 稳定，2.5 起开始抖，3.0 病态。溢出目前是直接报错。
- FP8 必须配 `DEEPEP_V2_ALIGN=128`。AdaptiveGEMM 的 m-grouped kernel 从激活行数推 M，吃不了静态容量，配置期会拒绝。

**病态变慢的根因**（实测，profile）：DeepEP 用 `record_stream` 把接收张量标到通信流，PyTorch 缓存分配器缓存不命中时要
`cudaEventSynchronize` 等这些事件；容量越大越容易不命中，一个 rank 在 host 上卡住，其余 rank 的通信 kernel 跟着自旋。
静态模式一步里有 268 次、共 7.5 s，动态模式为 0。2 卡隔离微基准里静态模式只比动态慢 30%，问题在训练循环里的调度，不在 DeepEP 的 kernel。
上游 `6c6aeb3f` 在 DeepEP v1 的同步路径上修的是同一个问题。

待办，按依赖顺序：

1. **`EP_AVOID_RECORD_STREAM=1` 下的张量生命周期持有。** 关掉 `record_stream` 后由调用方保证跨流张量在通信 kernel 结束前不被释放。
   反向里传给通信 kernel 的临时梯度张量还没有持有，这是 GLM 静态模式正式可用的前提。
   反向的通信调用在 `xtuner/v1/module/dispatcher/deepep_v2.py` 的 `_DispatchV2.backward` 与 `_CombineV2.backward`。
2. **把容量因子和溢出信号提给 DeepEP 上游**或算子组，不应长期依赖本地补丁。
3. **溢出策略**：现在是报错并要求调大因子，应用层的丢弃或重试没做。
4. **容量上界从均衡器来。** 容量因子 2.0 是因为训练早期路由不均衡。接上负载均衡之后容量因子可以压到接近 1，
   静态模式的显存代价才真正消失。所以这一项与 §7.5 是同一条线上的两段。
5. **CUDA graph 接进训练步。** 现状见 §6，收益上限约 3.6%，排在前面几项之后。要补的：
   - 第 1 项的生命周期持有是前提，捕获时必须开 `EP_AVOID_RECORD_STREAM=1`。
   - 把单测扩到 DeepGEMM 对齐布局和 FP8，这两条才是 V2 的主线配置。
   - 容量因子与 graph 的组合没测过。溢出检查 `_check_overflow` 是 Python 侧逻辑，graph 重放时不会执行，
     要改成重放之后在图外读 device 上的 flag（按代码推断，未验证）。
   - 层内微批的异步路径靠 Python 侧的事件钩子排序，在 graph 下没有测过。
   - 捕获范围从 MoE 段扩到整层，要求 pack 定长、router 输出形状固定；FSDP 的集合通信和激活重算留在图外。
6. rebase 之后上游新增的 EP 负载比指标需要 host 上的行数，静态模式下要改成从 device 计数延迟读取。

### 7.5 UltraEP 接入

设计上的接法见 `docs/design/unified_ep_contract_v1.0.md`。那份文档写的时候 PR #2050 的 head 是 `ae7d6f55`，
PR 在 2026-09-18 推进到 `83a2ce9c`（12 个提交），结构有变化，开工前先按新 head 把设计文档里 UltraEP 那一节核对一遍。
新 head 里有、`ae7d6f55` 里没有的：

| 位置 | 内容 |
|---|---|
| `xtuner/v1/module/ultraep/dispatcher.py`（新文件） | `UltraEPDispatcher`，六段式，包住一个 `inner_dispatcher`（`deepep` / `all2all` / `agrs`） |
| `xtuner/v1/module/dispatcher/__init__.py` | 定义了 `EPExecutionRuntime` Protocol 与 `NoEPExecutionRuntime` |

两个 head 都有的：`ultraep/{config,runtime,fsdp_expert_binding}.py`、`patch/ultraep/` 下的两个原生库补丁；梯度类型是 bf16。
以上只核对了文件和类是否存在，代码逻辑没有重读。

取 PR 的代码：

```bash
git fetch upstream '+refs/pull/2050/head:refs/remotes/upstream/pr/2050'
git worktree add --detach ../pr-2050-ultraep upstream/pr/2050
```

**与 `feat/deepepv2` 的关系。** 两边各有一份名为 `EPExecutionRuntime` 的定义（本分支在 `dispatcher/base.py`），
试合并有 10 个文件冲突，都是公共文件。所以接入方式是移植而不是合并：

1. 从 `feat/deepepv2` 开新的主题分支。
2. 把 `xtuner/v1/module/ultraep/` 整个包搬过来，让它的 runtime 实现 `base.py` 里的 `EPExecutionRuntime` / `LayerEPExecution`，
   副本权重走 `ExpertWeights` 的第二段。当前 `ExpertWeights` 只实现了 1 段，
   `moe_decoder_layer.py` 的 `_call_local_weights` 对多于 1 段直接抛 `NotImplementedError`，两段寻址要先补上。
3. `inner_dispatcher` 加上 `deepep_v2`。换 dispatcher 只换 token 通信这一条流量，副本权重分发、副本梯度归约、
   负载 all-gather 仍走 UltraEP 原生库。
4. 首期范围收窄：拒绝 `intra_layer_micro_batch > 1`、梯度走 UltraEP 上游原生的 fp32 路径再 cast，这样两个原生库补丁都不用打。

**环境。** `ultra_ep` 包在我们的环境里没有安装，要先装 UltraEP 原生库及其 NVSHMEM 依赖，做法参照 §3 的隔离目录。

**与框架组的沟通。** 两份同名定义要尽早统一成一份，否则契约 PR 和 PR #2050 无论谁先合，另一个都要大改。展示方式见 §2.6。

## 8. 已知问题与注意事项

- DeepEP V2 不能与「torch 确定性算法 + `fill_uninitialized_memory=True`」同时开，dispatcher 构造时会报错。
  确定性测试里要设 `torch.utils.deterministic.fill_uninitialized_memory = False`。
- DeepEP V2 要求 `hidden_size % 256 == 0`、top-k ≤ 32、每 rank 本地专家数 ≤ 256。
- `ElasticBuffer` 固定按 BF16 尺寸创建，即使开了 FP8 dispatch：combine 的反向是同一个 buffer 上对 BF16 梯度的 cached dispatch。
- FP8 dispatch 时，交给通信流的事件必须在量化 kernel 之后重新捕获，否则计算流忙的时候 dispatch 会读到未量化的内存。
- 多微批的交错顺序必须与 legacy 一致：先排完所有 attention 与 `dispatch_preprocess`，再逐个 dispatch → experts，所有 combine 排在第一个 `combine_postprocess` 之前。
- `torch_all2all` 的异步路径（mb ≥ 2）偶发 NaN 梯度范数，是上游已有的问题，DeepEP 路径不受影响。
- 日志里的 step time 不包含 swap optimizer 的 H2D。基准行都是 `SWAP_OPTIMIZER=0`，不受影响；开了之后要用真实周期反推。
- 共享网络盘上的工作区被多台机器同时使用。动工作区之前确认没有作业在用；队列在跑的时候不要改它的脚本，bash 是边读边执行的。
- 长队列用 `setsid nohup ... &` 启动，日志落在实验目录里。

## 9. 代码地图

| 文件 | 内容 | 来自 |
|---|---|---|
| `xtuner/v1/module/dispatcher/base.py` | 契约类型：`ExpertRows`、`Layout`、`ExpertWeights`、`DispatcherCaps`、`EPCall`、`LayerEPExecution`、`EPExecutionRuntime` | `feat/deepepv2` |
| `xtuner/v1/module/dispatcher/deepep_v2.py` | `DeepEPV2Config`、六段 dispatcher、`_DispatchV2` / `_CombineV2` / `_RowScale` | `feat/deepepv2` |
| `xtuner/v1/ops/comm/deepep_v2_op.py` | `ElasticBuffer` 的薄封装与注册表 | `feat/deepepv2` |
| `xtuner/v1/ops/moe/cuda/group_gemm_deepgemm.py` | DeepGEMM psum 的 BF16 / FP8 前向、dgrad、wgrad | `feat/deepepv2` |
| `xtuner/v1/ops/moe/cuda/triton_kernels/row_scale.py` | 路由权重逐行乘的融合 kernel | `feat/deepepv2` |
| `xtuner/v1/float8/triton_kernels/trans_quant_kgrouped.py` | DeepGEMM k-grouped wgrad 回退路径用的转置重量化 | `feat/deepepv2` |
| `xtuner/v1/module/decoder_layer/moe_decoder_layer.py` | 单批与多批合成一个交错核心 `_moe_forward` | `feat/deepepv2` |
| `xtuner/v1/model/moe/moe.py` | 解耦布局的 mesh、专家组的 FSDP 包装、按层 reshard 与反向预取 | 两边都改 |
| `xtuner/_testing/decoupled_ep_fsdp.py` | 解耦门测试的公共部分、all-gather 计数 | PR #2093 |
| `examples/v1/config/sft_moe_ep_bench.py` | 基准配置 | `feat/deepepv2`，解耦开关在 `e0375d9a` |
| `docs/design/unified_ep_contract_v1.0.md` | 契约设计与三个后端的接入方式 | `feat/deepepv2` |
| `docs/design/unified_grouped_gemm_v1.0.md` | 统一 Grouped GEMM 算子库设计 | `feat/deepepv2` |
| `docs/design/decouple_ep_fsdp.md` | 解耦布局设计 | PR #2093 |
| `patch/deepep_v2/deepep_v2_capped_capacity.patch` | DeepEP V2 的容量上限补丁 | 本分支 |
