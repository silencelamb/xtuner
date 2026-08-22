# GLM-5.2-30B on 8×H200: legacy vs. decoupled EP/FSDP (pt29 container)

Environment: torch 2.9.1+cu128, transformers 5.14.1, tilelang 0.1.11, cuDNN frontend 1.26 (DSA), DeepEP;
NGC 25.03 base image, container `748c0954d2aa`. Every pair below differs **only** in `DECOUPLE_EP_FSDP`.
All launches go through `examples/v1/config/sft_glm5p2.py` / `xtuner/v1/train/cli/sft.py` with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `DEBUG_SKIP_SAVE=1`, `SWAP_OPTIMIZER=0`, no CPU offload.
`max_memory` / `reserved` are rank 0's `torch.cuda.max_memory_allocated / max_memory_reserved` in GiB
(XTuner prints them as "GB" but divides by 1024³). Rank-0 step logs of every run are kept in
`reports/glm52_pt29_logs/`.

## 1. Production profile — the task's target experiment

`GLM-5.2-30B` (MTP), global batch 8 × 16K-token packs, `INTRA_LAYER_MICRO_BATCH=1`, DeepEP, tile-wise fp8,
`MODEL_COMPILE=1`, `SPARSE_MLA_BACKEND=cudnn_dsa`, `XTUNER_DSA_TOPK_OFFLOAD=1`, AdamW lr 1e-6, 10 steps
(same recipe as `work_dirs/sft_glm5p2/his_gbs8_mb1_deepep`, whose reference run did 1.47 s/step at 99.4 GiB).

| run | exit | steady step time (s, steps 6-9) | max_memory (GiB) | reserved (GiB) | llm loss (last) | mtp loss (last) | grad_norm (step 1) |
|---|---|---|---|---|---|---|---|
| prod_ep4_legacy | exit=0 | 1.688 | 99.50 | 128.57 | 9.906144 | 0.923589 | 38.2750 |
| prod_ep4_decouple | exit=0 | 1.658 | 83.49 | 104.95 | 9.905488 | 0.923488 | 38.1379 |
| prod_ep8_legacy | exit=0 | 11.835 | 120.03 | 132.44 | 9.906259 | 0.923688 | 38.1843 |
| prod_ep8_decouple | exit=0 | 1.636 | 76.32 | 94.44 | 9.905033 | 0.923618 | 38.1688 |

| EP | legacy step / peak | decoupled step / peak | Δ step | Δ peak |
|---|---|---|---|---|
| 4 | 1.688 s / 99.50 GiB | 1.658 s / 83.49 GiB | -1.8% | -16.0 GiB (-16%) |
| 8 | 11.835 s / 120.03 GiB | 1.636 s / 76.32 GiB | -86.2% | -43.7 GiB (-36%) |

- **EP=4**: −16.0 GiB at the same step time — "显存减少、耗时与 EP=4 一致" ✅.
- **EP=8**: −43.7 GiB; decoupled EP=8 (1.636 s) is the fastest configuration measured, slightly
  faster than decoupled EP=4. Legacy EP=8 sits at 120.0 GiB allocated / 132.4 GiB reserved (the card has 140.4 GiB; torch can reserve ~133 GiB once CUDA context, NCCL/DeepEP buffers and kernel workspaces are taken) and runs
  at 11.8 s/step for the whole 10-step run — see §2 for what happens over 40 steps.
- Loss / grad-norm: the four runs agree step by step (llm loss within 1.5e-3, grad norm within 0.15 at step 1),
  i.e. fp8 + DeepEP run-to-run noise (`reports/L3.md`).

## 2. Does legacy EP=8 recover? 40-step check

| run | exit | steady step time (s, steps 20-40) | max_memory (GiB) | reserved (GiB) | llm loss (last) | mtp loss (last) | grad_norm (step 1) |
|---|---|---|---|---|---|---|---|
| prod_ep8_legacy_40 | exit=0 | 11.235 | 119.92 | 132.50 | 7.382744 | 0.655078 | 38.1843 |
| prod_ep8_decouple_40 | exit=0 | 1.617 | 76.35 | 96.38 | 7.382671 | 0.655091 | 38.1686 |

Per-step time (s):

| step | prod_ep8_legacy_40 | prod_ep8_decouple_40 |
|---|---|---|
| 1 | 58.555 | 60.583 |
| 2 | 11.440 | 3.486 |
| 3 | 10.743 | 2.286 |
| 4 | 11.209 | 1.638 |
| 5 | 11.055 | 1.738 |
| 6 | 12.611 | 1.603 |
| 7 | 11.124 | 2.043 |
| 8 | 11.607 | 1.597 |
| 9 | 10.438 | 1.579 |
| 10 | 11.407 | 1.591 |
| 11 | 11.830 | 1.889 |
| 12 | 12.933 | 1.589 |
| 13 | 11.933 | 1.591 |
| 14 | 13.396 | 1.595 |
| 15 | 11.504 | 1.996 |
| 16 | 11.240 | 1.601 |
| 17 | 11.411 | 1.648 |
| 18 | 12.900 | 1.587 |
| 19 | 11.127 | 1.831 |
| 20 | 10.707 | 1.588 |
| 21 | 11.063 | 1.748 |
| 22 | 11.004 | 1.579 |
| 23 | 11.521 | 1.598 |
| 24 | 11.347 | 1.582 |
| 25 | 12.173 | 1.577 |
| 26 | 11.202 | 1.590 |
| 27 | 10.671 | 1.573 |
| 28 | 11.364 | 1.591 |
| 29 | 11.079 | 2.149 |
| 30 | 11.304 | 1.579 |
| 31 | 10.769 | 1.580 |
| 32 | 11.136 | 1.572 |
| 33 | 10.906 | 1.582 |
| 34 | 11.347 | 1.572 |
| 35 | 10.862 | 1.580 |
| 36 | 10.704 | 1.577 |
| 37 | 10.466 | 1.602 |
| 38 | 11.708 | 1.579 |
| 39 | 12.846 | 1.585 |
| 40 | 11.757 | 1.577 |

Legacy EP=8 stays at 10.4–13.4 s for all 40 steps (mean 11.24 s over steps 20–40) versus 1.62 s decoupled; step-40
loss is identical (7.38274 vs 7.38267). The colleague's 200-step legacy EP=4 parity log shows a related pattern at
118 GiB: 13–15 s/step for the first ~50 steps, then 2.43 s from step 55 on, while its legacy EP=2 run (126.9 GiB)
never drops below 13 s in 200 steps.

What separates fast from slow runs is the **reserved** peak, not the allocated peak:

| run | allocated peak (GiB) | reserved peak (GiB) | step time |
|---|---|---|---|
| prod gbs16 EP4 decoupled | 89.1 | 109.9 | fast |
| prod-like mb1 decoupled | 96.5 | 119.0 | fast |
| parity EP4 decoupled | 101.5 | 123.1 | fast (2.75 s) |
| prod-like mb2 decoupled | 107.1 | 132.5 | slow (15.0 s) |
| prod gbs16 EP4 legacy | 108.2 | 132.0 | slow (12.4 s) |
| prod-like mb1 legacy | 111.1 | 132.5 | slow (19.2 s) |
| parity EP4 legacy | 117.4 | 132.3 | slow (16.6 s) |
| prod EP8 legacy | 120.0 | 132.5 | slow (11.2 s) |

Every slow run has its reserved peak pinned at 132.0–132.5 GiB, i.e. at the ceiling the caching allocator can
obtain (140.4 GiB minus non-torch usage); every fast run stays ≤ 123 GiB. The 10–25 GiB between allocated and
reserved is cache/fragmentation, so allocated peaks of ~102–108 GiB are a grey zone rather than a threshold.
This was then measured directly with a wrapper around `Trainer._log_step` that prints
`torch.cuda.memory_stats()` counters and the resident memory right after the optimizer step
(`reports/glm52_pt29_logs/probe_*.rank0.log`, `[MEMPROBE]` lines):

| run (10 steps, steps 2–10) | resident after step (GiB) | peak alloc (GiB) | reserved peak (GiB) | `num_alloc_retries` / step | `num_device_free` / step | `num_sync_all_streams` / step | step time |
|---|---|---|---|---|---|---|---|
| prod EP8 legacy | 82.83 | 118.96 | 132.50 | 2.0 | 24.2 | 2.0 | 11.11 s |
| prod EP8 decoupled | 45.87 | 75.76 | 95.89 | 0.0 | 0.0 | 0.0 | 1.82 s |

Every legacy step hits the caching allocator's release-and-retry path twice (`num_alloc_retries`), each time
freeing ~12 cached segments back to the driver (`num_device_free`) after a full-device synchronization
(`num_sync_all_streams`); the decoupled run never does. The same signature appears in every slow run below
(1.5–2.5 retries/step) and in none of the fast ones. So the slowdown is the allocator thrashing against the
reserved ceiling, not the model or the communication pattern.

## 3. Production profile, `GLOBAL_BATCH_SIZE=16`, EP=4, MB1, DeepEP (10 steps)

Two packs per rank per step (gradient accumulation 2), everything else as in §1.

| run | exit | steady step time (s, steps 6-9) | max_memory (GiB) | reserved (GiB) | llm loss (last) | mtp loss (last) | grad_norm (step 1) |
|---|---|---|---|---|---|---|---|
| prod_gbs16_ep4_legacy | exit=0 | 12.369 | 108.19 | 132.01 | 9.878106 | 0.926278 | 38.0667 |
| prod_gbs16_ep4_decouple | exit=0 | 3.358 | 89.11 | 109.93 | 9.877830 | 0.926023 | 37.9339 |

- step time 12.369 → 3.358 s, peak 108.19 → 89.11 GiB (-19.1 GiB). The decoupled step
  time is 2× the GBS=8 step (1.66 s), as expected for two micro-batches; legacy at 108.2 GiB is again in
  the slow allocator regime.

## 4. Comparison with AutoModel (colleague's EP=4 parity recipe)

Recipe from `/workspace/xtuner/GLM5.2/notes/training/xtuner/glm5.2_readme.md` §2/§4: `GLM-5.2-30B-NoMTP`, Alpaca,
global batch 8 × 16K packs, MB1, DeepEP, `SPARSE_MLA_BACKEND=tilelang`, bf16 (no fp8), no compile, no
offload, activation recompute, AdamW constant lr 1e-6. 20 steps here (memory peaks in the first steps).

| run | exit | steady step time (s, steps 10-20) | max_memory (GiB) | reserved (GiB) | llm loss (last) | grad_norm (step 1) |
|---|---|---|---|---|---|---|
| parity_ep4_legacy | exit=0 | 16.640 | 117.35 | 132.34 | 9.192273 | 32.7078 |
| parity_ep4_decouple | exit=0 | 2.754 | 101.45 | 123.05 | 9.192635 | 32.7005 |

| run | peak memory (GiB) | step time |
|---|---|---|
| AutoModel EP4 (colleague, HybridEP, 200 steps) | 83.79 | 2.589 s (updates 60–200) |
| XTuner EP4 legacy (colleague, 200 steps) | 118.00 | 13-15 s (steps 1–50) → 2.43 s (steps 55–200) |
| XTuner EP4 legacy (this run, 20 steps) | 117.35 | 16.640 s (steps 10–20) |
| XTuner EP4 **decoupled** (this run, 20 steps) | **101.45** | **2.754 s** (steps 10–20, steady from step 3) |

- Decoupling closes about half of the XTuner–AutoModel peak-memory gap on this recipe (117.4 → 101.5 GiB vs
  AutoModel 83.79 GiB). The remaining 17.7 GiB is not parameter/optimizer replication (AutoModel's `ep_shard`
  layout is the same dp2ep layout as ours); §8.2/8.3 show it is first-step Triton-autotune garbage kept alive by XTuner's disabled GC; with
  `XTUNER_GC_ENABLE=1` (or CUTLASS) the decoupled run lands at 82.2 GiB — at AutoModel's level.
- Step time: decoupled 2.75 s vs AutoModel 2.59 s and the colleague's late-phase legacy 2.43 s.
- llm loss after 20 steps: 9.1923 (legacy) vs 9.1926 (decoupled).

## 5. Colleague's prod-like profile: EP=2, SP=2, DeepEP, activation offload, compile, MB1 / MB2

Recipe from the readme §7 (`sft_glm52_prodlike_ep2_sp2_activation_offload_compile.sh`): NoMTP, bf16, tilelang,
`XTUNER_ACTIVATION_OFFLOAD=1`, `LOSS_CHUNK_SIZE=1024`, `XTUNER_GC_ENABLE=1`, constant lr. First run of
`decouple_ep_fsdp` together with sequence parallel. 20 steps.

| run | exit | steady step time (s, steps 10-20) | max_memory (GiB) | reserved (GiB) | llm loss (last) | grad_norm (step 1) |
|---|---|---|---|---|---|---|
| prodlike_ep2sp2_mb1_legacy | exit=0 | 19.223 | 111.08 | 132.49 | 9.178407 | 32.5355 |
| prodlike_ep2sp2_mb1_decouple | exit=0 | 3.163 | 96.51 | 119.00 | 9.178430 | 32.5248 |
| prodlike_ep2sp2_mb2_legacy | exit=0 | 18.951 | 115.96 | 132.40 | 9.177890 | 32.5354 |
| prodlike_ep2sp2_mb2_decouple | exit=0 | 14.965 | 107.05 | 132.49 | 9.178247 | 32.5156 |

| MB | legacy step / peak | decoupled step / peak | Δ peak |
|---|---|---|---|
| 1 | 19.223 s / 111.08 GiB | 3.163 s / 96.51 GiB | -14.6 GiB |
| 2 | 18.951 s / 115.96 GiB | 14.965 s / 107.05 GiB | -8.9 GiB |

Resident (after optimizer step) vs. peak memory from the probe runs (steps 10–20):

| run | resident after step (GiB) | peak alloc (GiB) | peak − resident (GiB) | reserved peak (GiB) | alloc retries / step | step time |
|---|---|---|---|---|---|---|
| MB1 legacy | 51.09 | 111.02 | 59.93 | 132.45 | 1.8 | 18.46 s |
| MB1 decoupled | 45.92 | 96.33 | 50.41 | 119.78 | 0.0 | 3.14 s |
| MB2 legacy | 51.28 | 115.93 | 64.65 | 132.48 | 2.5 | 19.51 s |
| MB2 decoupled | 46.11 | 106.99 | 60.88 | 132.47 | 1.5 | 15.22 s |

Why MB2 saves only 8.9 GiB while MB1 saves 14.6 GiB:

- The **resident** saving is the same in both: 5.2 GiB. It is exactly what de-replication predicts. Let *D*
  be the per-model dense state (fp32 master + AdamW moments + sharded working copies). Legacy EP=*e* holds
  *D*/(8/*e*) per rank, decoupled holds *D*/8, so the saving is *D*(*e*−1)/8. The probes give
  *D* ≈ 42 GiB: EP=2 → 5.3 GiB (measured 5.2), EP=4 → 15.8 GiB (measured 16.0 prod, 15.9 parity), EP=8 →
  36.9 GiB (measured resident 82.83 → 45.87 = 36.96). The model fits all three.
- The rest of the saving is in the **transient** part (peak − resident): −9.5 GiB for MB1 but only −3.8 GiB
  for MB2. With two micro-batches in flight the activation peak is ~5 GiB higher and dominates the peak
  moment, so less of the legacy-only transient overhead (larger fp32 dense-gradient shards alive during
  backward, the EP all-reduce of those gradients) is exposed at the peak. Attributing the transient delta
  exactly would need a `torch.cuda.memory._snapshot`, which was not taken.
- Net effect: MB2 decoupled still lands at 107 GiB allocated / 132.5 GiB reserved — on the allocator ceiling
  (1.5 retries/step) — hence 15 s/step instead of ~3 s.

- MB1: decoupling removes 14.6 GiB and takes the run out of the slow allocator regime (19.2 → 3.16 s/step);
  this is also the first run of `decouple_ep_fsdp` together with sequence parallel (SP=2) — it works unchanged.
- MB2: decoupling removes 8.9 GiB but the run stays pinned at 132.5 GiB reserved, so it only improves from
  19.0 to 15.0 s/step. On this 30B / 16K-pack profile MB2 is activation-bound and needs more than parameter
  de-replication to be fast on one node.
- Loss curves of all four agree step by step (9.1784 vs 9.1784 at step 20).

## 8. CUTLASS group GEMM on top of decoupling

`XTUNER_USE_CUTLASS_GROUP_GEMM=1` (the banner `Using cutlass group gemm` is printed by all 8 ranks) on the
prod-like EP2/SP2 MB1 recipe of §5, 20 steps, probe wrapper enabled:

| run | resident after step (GiB) | peak alloc (GiB) | reserved peak (GiB) | alloc retries / step | step time (steps 10–20) | llm loss (step 20) |
|---|---|---|---|---|---|---|
| MB1 legacy, Triton | 51.09 | 111.02 | 132.45 | 1.8 | 18.46 s | 9.178551 |
| MB1 legacy, CUTLASS | 51.09 | 111.01 | 132.40 | 2.2 | 17.29 s | 9.178483 |
| MB1 decoupled, Triton | 45.92 | 96.33 | 119.78 | 0.0 | 3.14 s | 9.177789 |
| MB1 decoupled, CUTLASS | 45.92 | 96.37 | 119.48 | 0.0 | 3.22 s | 9.178550 |

On this recipe CUTLASS changes neither the resident nor the peak memory (±0.05 GiB) and the step time by
≤ 2.5%; it is orthogonal to decoupling but brings no additional saving here. The colleague's −19 GiB CUTLASS
result was measured on the EP4 parity recipe (no SP, no activation offload, no compile); §8.1 repeats that
A/B on this branch.

### 8.1 CUTLASS on the AutoModel-parity EP4 recipe (§4 settings, 20 steps, probe wrapper)

| run | resident after step (GiB) | peak alloc (GiB) | peak − resident (GiB) | reserved peak (GiB) | alloc retries / step | step time (steps 10–20) | llm loss (step 20) |
|---|---|---|---|---|---|---|---|
| EP4 legacy, Triton | 80.28 | 116.90 | 36.62 | 132.33 | 3.2 | 16.60 s | 9.192291 |
| EP4 legacy, CUTLASS | 61.24 | 97.96 | 36.72 | 127.98 | 0.0 | 2.72 s | 9.191978 |
| EP4 decoupled, Triton | 64.77 | 101.32 | 36.55 | 122.72 | 0.0 | 2.79 s | 9.192810 |
| EP4 **decoupled, CUTLASS** | **45.73** | **82.25** | 36.52 | 103.30 | 0.0 | **2.73 s** | 9.192413 |
| AutoModel EP4 (colleague) | — | 83.79 | — | — | — | 2.59 s | — |

- On this recipe the switch reproduces the colleague's −19 GiB, and the probe shows it is entirely
  **resident** memory; the transient part (peak − resident ≈ 36.5 GiB) is identical in all four runs.

### 8.2 It is not the kernel — it is Python's cyclic GC

XTuner calls `gc.disable()` at startup unless `XTUNER_GC_ENABLE=1` (`trainer.py::_setup_env`). The parity
recipe leaves it disabled; the prod-like recipe of §8 sets it to 1 — which is exactly the recipe where CUTLASS
made no difference. Re-running the Triton path with the collector enabled:

| run (parity EP4, 20 steps) | resident after step (GiB) | peak alloc (GiB) | reserved peak (GiB) | step time (steps 10–20) |
|---|---|---|---|---|
| decoupled, Triton, GC off (default) | 64.77 | 101.32 | 122.72 | 2.79 s |
| decoupled, Triton, **`XTUNER_GC_ENABLE=1`** | **45.73** | **82.23** | 103.95 | 2.74 s |
| decoupled, CUTLASS, GC off | 45.73 | 82.25 | 103.30 | 2.73 s |
| legacy, Triton, GC off | 80.28 | 116.90 | 132.33 | 16.60 s |
| legacy, Triton, `XTUNER_GC_ENABLE=1` | 61.24 | 97.89 | 128.67 | 2.89 s |
| legacy, CUTLASS, GC off | 61.24 | 97.96 | 127.98 | 2.72 s |

Enabling the collector gives byte-identical resident memory to CUTLASS, so the 19 GiB are tensors pinned by
reference cycles that only the cyclic collector can free. Section 8.3 identifies them.

### 8.3 Root cause: Triton autotune exceptions pin the first step's tensors

Probe (`MEMPROBE_GC_DEBUG=1`, Triton path, GC disabled, after step 3): `gc.collect()` with `DEBUG_SAVEALL`
finds **25,728 unreachable objects holding 93 CUDA tensors = 28.6 GiB**. Their types: `traceback` (280),
`frame` (399), `FrameSummary` (3689), `cell` (4784), `function` (3183), `functools.partial`,
`torch._subclasses.fake_tensor.FakeTensorMode` (164), and **40 × `triton.runtime.errors.OutOfResources` with
40 × `triton.backends.nvidia.compiler.CUDAOptions`**. Every `OutOfResources` carries the same traceback:

```
triton/runtime/autotuner.py:_bench > testing.py:do_bench > autotuner.py:kernel_call
  > jit.py:run > compiler.py:launch_metadata > compiler.py:_init_handles > compiler.py:raise_
```

i.e. the grouped-GEMM **autotune**: configurations that exceed shared memory / registers raise
`OutOfResources`, Triton catches them and moves on, but the exception objects survive in a reference cycle
(exception → traceback → `_bench` frame → `args` / closure cells → the kernel argument tensors: the dispatched
activations, the unsharded EP-local expert weights, the dW outputs of that call). Autotuning runs once per
weight shape (`@triton.autotune(key=["N","K"])` / `key=["M","N"]` are weight dimensions), so the cycle is
created **once, in step 1**, and then stays: constant 19 GiB resident, never growing — which is exactly what
the per-step probes show. XTuner disables Python's cyclic GC at startup (`gc.disable()` unless
`XTUNER_GC_ENABLE=1`) and only calls `gc.collect()` every 50 steps, so the garbage lives for 50 steps at a
time. With the collector enabled it is reclaimed right after the autotune; the CUTLASS path has no autotune
and therefore no such garbage; the compiled prod-like recipe (§8) had `XTUNER_GC_ENABLE=1` set, which is why
CUTLASS changed nothing there.

Consequences:

- The colleague's "CUTLASS saves 19 GiB" is a GC artefact of the Triton autotune, not a kernel property; the
  same saving comes from `XTUNER_GC_ENABLE=1` or from a single `gc.collect()` after the first step.
- A cheap, safe fix on the XTuner side is to collect once after step 1 (e.g. `if self.cur_step == 1 or
  self.cur_step % 50 == 0: gc.collect()` in the train loop) so autotune garbage never survives; it is a
  separate one-line PR, not part of the decoupling change.
- Under the allocator-ceiling behaviour of §2 this garbage is also what pushed legacy Triton EP4 over the edge
  (117 GiB → 3.2 alloc retries / step), so the colleague's 13–15 s/step phase for the first 50 steps is the
  autotune garbage being held until the `cur_step % 50` collection at step 50 — matching the jump to 2.43 s
  at step 55 in their log.

So the right comparison with AutoModel at EP4 is **decoupled + `XTUNER_GC_ENABLE=1` (or CUTLASS): 82.2 GiB /
2.74 s vs AutoModel 83.79 GiB / 2.59 s** — parity reached without changing the expert GEMM kernel.

## 6. Per-step llm loss (all runs)

| step | prod_ep4_legacy | prod_ep4_decouple | prod_ep8_legacy | prod_ep8_decouple | prod_gbs16_ep4_legacy | prod_gbs16_ep4_decouple |
|---|---|---|---|---|---|---|
| 1 | 12.242835 | 12.242374 | 12.242205 | 12.242205 | 12.236732 | 12.236374 |
| 2 | 11.496573 | 11.495951 | 11.497021 | 11.497888 | 11.486389 | 11.486962 |
| 3 | 11.353387 | 11.354423 | 11.353814 | 11.353219 | 11.337647 | 11.337610 |
| 4 | 11.160427 | 11.160503 | 11.160210 | 11.160748 | 11.163484 | 11.162803 |
| 5 | 10.934574 | 10.933951 | 10.933998 | 10.934554 | 10.955031 | 10.953431 |
| 6 | 10.724062 | 10.724489 | 10.724340 | 10.724806 | 10.746705 | 10.747047 |
| 7 | 10.523461 | 10.521789 | 10.522464 | 10.522381 | 10.527714 | 10.527598 |
| 8 | 10.323409 | 10.323653 | 10.323524 | 10.323934 | 10.328103 | 10.327041 |
| 9 | 10.050714 | 10.050805 | 10.050767 | 10.051491 | 10.048573 | 10.048362 |
| 10 | 9.906144 | 9.905488 | 9.906259 | 9.905033 | 9.878106 | 9.877830 |

| step | parity_ep4_legacy | parity_ep4_decouple | prodlike_ep2sp2_mb1_legacy | prodlike_ep2sp2_mb1_decouple | prodlike_ep2sp2_mb2_legacy | prodlike_ep2sp2_mb2_decouple |
|---|---|---|---|---|---|---|
| 1 | 12.048020 | 12.048020 | 12.015297 | 12.015297 | 12.015230 | 12.015260 |
| 2 | 12.024372 | 12.024330 | 12.004551 | 12.004889 | 12.004640 | 12.004749 |
| 3 | 11.968227 | 11.968078 | 11.974769 | 11.974870 | 11.974946 | 11.974766 |
| 4 | 11.935795 | 11.935991 | 11.956466 | 11.956565 | 11.956635 | 11.956882 |
| 5 | 11.762895 | 11.762611 | 11.791142 | 11.791558 | 11.791435 | 11.791679 |
| 6 | 11.726269 | 11.726021 | 11.746720 | 11.747422 | 11.746029 | 11.747211 |
| 7 | 11.691442 | 11.692036 | 11.679165 | 11.679321 | 11.678986 | 11.679071 |
| 8 | 11.670940 | 11.671287 | 11.669518 | 11.669571 | 11.669342 | 11.669516 |
| 9 | 11.114158 | 11.114318 | 11.128633 | 11.128882 | 11.128363 | 11.128780 |
| 10 | 11.120523 | 11.120468 | 11.069838 | 11.069673 | 11.068581 | 11.069795 |
| 11 | 11.031364 | 11.031272 | 11.003869 | 11.004610 | 11.005125 | 11.004539 |
| 12 | 11.012856 | 11.013029 | 11.005733 | 11.005695 | 11.005744 | 11.006051 |
| 13 | 10.837400 | 10.838067 | 10.846592 | 10.847128 | 10.846694 | 10.846529 |
| 14 | 10.809042 | 10.809765 | 10.808216 | 10.809036 | 10.809648 | 10.809333 |
| 15 | 10.775974 | 10.776084 | 10.754548 | 10.754716 | 10.755053 | 10.754559 |
| 16 | 10.755434 | 10.755795 | 10.715282 | 10.715774 | 10.716381 | 10.715880 |
| 17 | 9.354899 | 9.354802 | 9.326880 | 9.326507 | 9.326110 | 9.326167 |
| 18 | 9.289428 | 9.289060 | 9.309678 | 9.309234 | 9.309449 | 9.309461 |
| 19 | 9.251739 | 9.251347 | 9.229165 | 9.229153 | 9.228951 | 9.228942 |
| 20 | 9.192273 | 9.192635 | 9.178407 | 9.178430 | 9.177890 | 9.178247 |

## 7. Per-step time (s)

| step | prod_ep4_legacy | prod_ep4_decouple | prod_ep8_legacy | prod_ep8_decouple | prod_gbs16_ep4_legacy | prod_gbs16_ep4_decouple |
|---|---|---|---|---|---|---|
| 1 | 61.615 | 71.797 | 135.281 | 59.375 | 61.494 | 61.261 |
| 2 | 3.928 | 3.895 | 11.946 | 3.053 | 17.459 | 6.021 |
| 3 | 1.842 | 1.643 | 13.227 | 2.367 | 11.758 | 3.358 |
| 4 | 1.651 | 1.836 | 11.846 | 1.637 | 11.608 | 3.594 |
| 5 | 1.790 | 1.644 | 13.804 | 1.986 | 16.748 | 3.220 |
| 6 | 1.649 | 1.704 | 11.950 | 1.603 | 13.620 | 3.308 |
| 7 | 1.806 | 1.643 | 10.998 | 1.768 | 11.402 | 3.219 |
| 8 | 1.649 | 1.641 | 13.293 | 1.594 | 10.916 | 3.683 |
| 9 | 1.647 | 1.643 | 11.099 | 1.581 | 13.539 | 3.222 |
| 10 | 1.648 | 1.645 | 11.222 | 1.599 | 11.836 | 3.760 |

| step | parity_ep4_legacy | parity_ep4_decouple | prodlike_ep2sp2_mb1_legacy | prodlike_ep2sp2_mb1_decouple | prodlike_ep2sp2_mb2_legacy | prodlike_ep2sp2_mb2_decouple |
|---|---|---|---|---|---|---|
| 1 | 84.720 | 39.825 | 82.360 | 65.315 | 65.602 | 75.303 |
| 2 | 13.772 | 3.724 | 16.090 | 7.545 | 15.119 | 12.515 |
| 3 | 16.907 | 2.714 | 18.614 | 4.200 | 21.344 | 18.062 |
| 4 | 16.662 | 2.722 | 20.069 | 3.120 | 18.238 | 11.452 |
| 5 | 16.265 | 3.364 | 17.767 | 3.098 | 21.383 | 16.644 |
| 6 | 16.986 | 2.721 | 17.854 | 3.459 | 20.779 | 14.338 |
| 7 | 16.961 | 2.708 | 16.477 | 3.095 | 21.882 | 14.867 |
| 8 | 14.616 | 2.718 | 20.274 | 3.102 | 18.259 | 13.941 |
| 9 | 14.872 | 2.708 | 19.073 | 4.176 | 21.811 | 14.294 |
| 10 | 15.180 | 2.710 | 18.292 | 3.096 | 18.539 | 17.135 |
| 11 | 17.102 | 3.052 | 19.832 | 3.103 | 18.754 | 15.624 |
| 12 | 17.878 | 2.876 | 19.761 | 3.093 | 19.049 | 13.664 |
| 13 | 17.971 | 2.720 | 17.453 | 3.098 | 19.818 | 13.644 |
| 14 | 15.825 | 2.700 | 21.058 | 3.095 | 17.910 | 14.394 |
| 15 | 18.895 | 2.705 | 19.168 | 3.838 | 19.142 | 15.443 |
| 16 | 16.270 | 2.701 | 18.061 | 3.093 | 17.927 | 13.760 |
| 17 | 15.296 | 2.708 | 20.332 | 3.092 | 19.074 | 17.417 |
| 18 | 16.353 | 2.712 | 20.494 | 3.097 | 20.129 | 15.768 |
| 19 | 15.072 | 2.703 | 18.091 | 3.095 | 15.710 | 14.908 |
| 20 | 17.195 | 2.704 | 18.910 | 3.094 | 22.414 | 12.856 |
