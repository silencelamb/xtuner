# GLM-5.2-30B on 8×H200: re-run of the legacy vs. decoupled EP/FSDP comparison for PR #2093

Re-verification of `reports/GLM52.md` on the rebased branch, run on 2026-09-17 with
`feat/decouple-ep-fsdp` at `6d8ce6f1` (code identical to the PR branch `pr/decouple-ep-fsdp` at `212c05ee`,
base `upstream/main` `6c06f96e`). The rebase brought in upstream's GLM-5.2 changes (the `glm52` module split,
the LMDeploy-compatible FP8 DSA indexer `deep_gemm_fp8`, unified reentrant activation checkpointing), so every
pair below was re-run rather than carried over.

Environment: host `gpu-lg-cmc-h-h200-1727`, torch 2.9.1+cu128, transformers 5.14.1, tilelang 0.1.11, DeepEP 1.2.1,
DeepGEMM 2.1.1, cuDNN frontend **1.10.0**. The `cudnn_dsa` sparse-MLA backend used by `reports/GLM52.md` §1 needs
cuDNN frontend 1.26 and is unavailable in this container (`ensure_cudnn_dsa_runtime_available` raises), so all runs
here use `SPARSE_MLA_BACKEND=tilelang` (the same backend as the CI case `glm5-2-sft-30B-mtp-fp8`). Step times are
therefore not comparable with the §1 reference (tilelang attention is slower); memory, losses and grad norms are.
Every pair differs **only** in `DECOUPLE_EP_FSDP`. Launches go through `examples/v1/config/sft_glm5p2.py` /
`xtuner/v1/train/cli/sft.py` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `DEBUG_SKIP_SAVE=1`,
`SWAP_OPTIMIZER=0`, `XTUNER_ACTIVATION_OFFLOAD=0`, no CPU offload. `max_memory` / `reserved` are rank 0's
`torch.cuda.max_memory_allocated / max_memory_reserved` in GiB, maximum over the run. Rank-0 step logs of every
run are in `reports/glm52_pr_rerun_logs/` (`<run>.rank0.txt`, rank-0 lines plus the XTuner environment dump). The reference column repeats `reports/GLM52.md` (2026-08-22, container
`748c0954d2aa`).

## 1. Production profile, tile-wise FP8 (the target experiment)

`GLM-5.2-30B` (MTP), global batch 8 × 16K-token packs, `INTRA_LAYER_MICRO_BATCH=1`, DeepEP, tile-wise fp8,
`MODEL_COMPILE=1`, `XTUNER_DSA_TOPK_OFFLOAD=1`, AdamW cosine lr 1e-6, 10 steps. The `_dgfp8` pair additionally sets
`INDEXER_BACKEND=deep_gemm_fp8`, the LMDeploy-aligned FP8 indexer path that upstream added in #2076
(GLM-5.2's `index_n_heads=32`, `index_head_dim=128` satisfy its contract).

| run | exit | steady step time (s, steps 6-9) | max_memory (GiB) | reserved (GiB) | llm loss (last) | mtp loss (last) | grad_norm (step 1) | reference (step s / max_memory GiB / llm loss / grad_norm) |
|---|---|---|---|---|---|---|---|---|
| prod_ep4_legacy | exit=0 (10/10) | 2.598 | 100.65 | 129.26 | 9.906231 | 0.923925 | 38.2773 | 1.688 / 99.50 / 9.906144 / 38.2750 |
| prod_ep4_decouple | exit=0 (10/10) | 2.597 | 84.97 | 105.79 | 9.905257 | 0.923413 | 38.1379 | 1.658 / 83.49 / 9.905488 / 38.1379 |
| prod_ep8_legacy | exit=0 (10/10) | 11.582 | 121.60 | 132.48 | 9.906513 | 0.923672 | 38.1807 | 11.835 / 120.03 / 9.906259 / 38.1843 |
| prod_ep8_decouple | exit=0 (10/10) | 2.572 | 77.29 | 97.31 | 9.905258 | 0.923688 | 38.1696 | 1.636 / 76.32 / 9.905033 / 38.1688 |
| prod_ep8_legacy_dgfp8 | exit=0 (10/10) | 11.925 | 121.62 | 132.38 | 9.906261 | 0.923604 | 38.1858 | – |
| prod_ep8_decouple_dgfp8 | exit=0 (10/10) | 2.645 | 77.35 | 96.84 | 9.905219 | 0.923651 | 38.1526 | – |

Per-step `reduced_llm_loss` (rank 0):

| step | prod_ep4_legacy | prod_ep4_decouple | prod_ep8_legacy | prod_ep8_decouple | prod_ep8_legacy_dgfp8 | prod_ep8_decouple_dgfp8 |
|---|---|---|---|---|---|---|
| 1 | 12.2428 | 12.2424 | 12.2422 | 12.2422 | 12.2423 | 12.2423 |
| 2 | 11.4966 | 11.4963 | 11.4974 | 11.4968 | 11.4976 | 11.4983 |
| 3 | 11.3534 | 11.3542 | 11.3521 | 11.3532 | 11.3538 | 11.3533 |
| 4 | 11.1594 | 11.1603 | 11.1602 | 11.1596 | 11.1599 | 11.1609 |
| 5 | 10.9340 | 10.9333 | 10.9348 | 10.9352 | 10.9351 | 10.9352 |
| 6 | 10.7251 | 10.7242 | 10.7246 | 10.7246 | 10.7250 | 10.7245 |
| 7 | 10.5234 | 10.5212 | 10.5221 | 10.5226 | 10.5228 | 10.5224 |
| 8 | 10.3243 | 10.3240 | 10.3232 | 10.3240 | 10.3232 | 10.3237 |
| 9 | 10.0509 | 10.0510 | 10.0515 | 10.0515 | 10.0518 | 10.0513 |
| 10 | 9.9062 | 9.9053 | 9.9065 | 9.9053 | 9.9063 | 9.9052 |

## 2. Production profile, BF16 (same recipe with `FP8=0`)

| run | exit | steady step time (s, steps 6-9) | max_memory (GiB) | reserved (GiB) | llm loss (last) | mtp loss (last) | grad_norm (step 1) | reference (step s / max_memory GiB / llm loss / grad_norm) |
|---|---|---|---|---|---|---|---|---|
| bf16_ep4_legacy | exit=0 (10/10) | 10.043 | 104.87 | 132.71 | 11.349463 | 1.077515 | 38.1811 | – |
| bf16_ep4_decouple | exit=0 (10/10) | 2.794 | 90.76 | 111.35 | 11.349125 | 1.077581 | 38.1478 | – |
| bf16_ep8_legacy | exit=0 (10/10) | 11.645 | 122.09 | 132.40 | 11.349180 | 1.077490 | 38.1672 | – |
| bf16_ep8_decouple | exit=0 (10/10) | 3.013 | 78.16 | 99.40 | 11.349021 | 1.077343 | 38.1361 | – |

Per-step `reduced_llm_loss` (rank 0):

| step | bf16_ep4_legacy | bf16_ep4_decouple | bf16_ep8_legacy | bf16_ep8_decouple |
|---|---|---|---|---|
| 1 | 12.2463 | 12.2463 | 12.2462 | 12.2462 |
| 2 | 12.2154 | 12.2154 | 12.2158 | 12.2159 |
| 3 | 12.1589 | 12.1589 | 12.1589 | 12.1591 |
| 4 | 12.1343 | 12.1345 | 12.1344 | 12.1342 |
| 5 | 11.9639 | 11.9640 | 11.9639 | 11.9637 |
| 6 | 11.9332 | 11.9330 | 11.9334 | 11.9332 |
| 7 | 11.8967 | 11.8967 | 11.8971 | 11.8969 |
| 8 | 11.8869 | 11.8868 | 11.8867 | 11.8869 |
| 9 | 11.3455 | 11.3455 | 11.3451 | 11.3451 |
| 10 | 11.3495 | 11.3491 | 11.3492 | 11.3490 |

## 3. AutoModel-parity profile, BF16 (`reports/GLM52.md` §4 recipe)

`GLM-5.2-30B-NoMTP`, global batch 8 × 16K packs, MB1, DeepEP, tilelang, bf16, no compile, no offload
(`XTUNER_DSA_TOPK_OFFLOAD=0`), activation recompute, AdamW constant lr 1e-6, 20 steps. The reference was produced
with the same backend, so step times are comparable here.

| run | exit | steady step time (s, steps 10-20) | max_memory (GiB) | reserved (GiB) | llm loss (last) | grad_norm (step 1) | reference (step s / max_memory GiB / llm loss / grad_norm) |
|---|---|---|---|---|---|---|---|
| parity_ep4_legacy | exit=0 (20/20) | 15.537 | 118.27 | 132.73 | 9.192342 | 32.7076 | 16.640 / 117.35 / 9.192273 / 32.7078 |
| parity_ep4_decouple | exit=0 (20/20) | 2.376 | 102.39 | 123.68 | 9.192407 | 32.7007 | 2.754 / 101.45 / 9.192635 / 32.7005 |

Per-step `reduced_llm_loss` (rank 0):

| step | parity_ep4_legacy | parity_ep4_decouple |
|---|---|---|
| 1 | 12.0480 | 12.0480 |
| 2 | 12.0245 | 12.0244 |
| 3 | 11.9683 | 11.9683 |
| 4 | 11.9356 | 11.9361 |
| 5 | 11.7626 | 11.7627 |
| 6 | 11.7259 | 11.7261 |
| 7 | 11.6923 | 11.6922 |
| 8 | 11.6713 | 11.6711 |
| 9 | 11.1140 | 11.1146 |
| 10 | 11.1208 | 11.1206 |
| 11 | 11.0314 | 11.0316 |
| 12 | 11.0133 | 11.0123 |
| 13 | 10.8386 | 10.8382 |
| 14 | 10.8096 | 10.8087 |
| 15 | 10.7753 | 10.7752 |
| 16 | 10.7559 | 10.7554 |
| 17 | 9.3548 | 9.3548 |
| 18 | 9.2886 | 9.2888 |
| 19 | 9.2518 | 9.2519 |
| 20 | 9.1923 | 9.1924 |

## 4. Conclusions

All 12 runs finished with exit 0 on the rebased branch. Within every legacy / decoupled pair the two layouts differ
only in memory and step time; their loss curves agree step by step.

- **FP8 production (§1)**: EP4 peak memory 100.65 → 84.97 GiB (−15.7 GiB) at the same step time (2.60 s);
  EP8 121.60 → 77.29 GiB (−44.3 GiB) and 11.58 → 2.57 s per step. Step-1 llm loss agrees to 6e-4, step-10 to 1.3e-3
  (9.9065 vs 9.9053), step-1 grad norm to 0.4%. Against `reports/GLM52.md` §1 (cudnn_dsa, 2026-08-22) the peak memory
  of every run is within 1.6 GiB, the losses within 1e-3 and the grad norms within 0.02; the legacy EP8 run again sits at
  the 132 GiB reserved ceiling and pays the allocator retries (11.6 s vs 11.8 s in the reference). Step times of the
  fast runs are 2.6 s instead of 1.65 s only because of the tilelang attention backend.
- **`deep_gemm_fp8` indexer (§1, `_dgfp8` pair)**: the LMDeploy-aligned FP8 indexer that upstream added in #2076 runs
  on both layouts; decoupled 2.65 s / 77.35 GiB vs. legacy 11.93 s / 121.62 GiB, losses within 1e-3 of the default
  indexer runs.
- **BF16 production (§2)**: EP4 104.87 → 90.76 GiB (−14.1 GiB) and 10.04 → 2.79 s; EP8 122.09 → 78.16 GiB
  (−43.9 GiB) and 11.65 → 3.01 s. Losses agree to 4e-4 at every step. The legacy bf16 EP4 run already touches the reserved
  ceiling (132.7 GiB, vs. 129.3 GiB for its fp8 counterpart), which is why the bf16 EP4 pair shows the 3.6× step-time
  gap that the fp8 EP4 pair does not.
- **AutoModel-parity profile (§3)**: legacy 15.54 s / 118.27 GiB vs. decoupled 2.38 s / 102.39 GiB, final llm loss
  9.192342 vs. 9.192407; the reference pair was 16.64 s / 117.35 GiB vs. 2.754 s / 101.45 GiB and 9.192273 vs. 9.192635.
- The fp8 and bf16 recipes converge differently (fp8 12.24 → 9.91 in 10 steps, bf16 12.25 → 11.35). This is a property
  of the two existing training paths, not of this PR: the pre-decoupling fp8 reference run of 2026-08-10
  (`work_dirs/sft_glm5p2/his_gbs8_mb1_deepep`) goes 12.22 → 11.50 → 9.89, and the bf16 parity runs of 2026-08-22 show the
  slow curve. Within each path legacy and decoupled agree.

Not covered here: the `cudnn_dsa` attention backend (cuDNN frontend 1.10 in this container, 1.26 needed), upstream's
8-GPU tiny GLM-5.2 engine tests in `tests/engine/test_glm52_moe_train_engine.py` (they need the `GLM5_2_TINY_MOE_PATH`
checkpoint, which is not on this box; they run in CI), and multi-node topologies.

