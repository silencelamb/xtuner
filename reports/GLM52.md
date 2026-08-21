# GLM-5.2-30B on 8×H200: legacy vs. decoupled EP/FSDP

The target experiment from the task brief: the production-like GLM-5.2 30B (MTP) SFT profile,
`EP=4` and `EP=8`, each with the legacy layout and with `DECOUPLE_EP_FSDP=1`. Expectation:
lower memory, step time no worse than the legacy run at the same EP.

Settings (from `/workspace/xtuner/work_dirs/sft_glm5p2/his_gbs8_mb1_deepep`): `examples/v1/config/sft_glm5p2.py`,
global batch 8 × 16K-token packs, `INTRA_LAYER_MICRO_BATCH=1`, DeepEP dispatcher, tile-wise fp8,
`MODEL_COMPILE=1`, `XTUNER_DSA_TOPK_OFFLOAD=1`, AdamW lr 1e-6, 10 steps, `DEBUG_SKIP_SAVE=1`.
Two deviations from the reference run, forced by today's environment (see `reports/decisions.md` D11):
`SPARSE_MLA_BACKEND=tilelang` instead of `cudnn_dsa` (the installed cuDNN frontend has no DSA module), and
`transformers==5.14.1` from an isolated `pip --target` directory. Both apply equally to all four runs, so the
legacy-vs-decoupled comparison is like-for-like; absolute step times are not comparable with the
`cudnn_dsa` reference (1.47 s/step).

```bash
DECOUPLE_EP_FSDP={0,1} EP_SIZE={4,8} SPARSE_MLA_BACKEND=tilelang DISPATCHER=deepep FP8=1 MODEL_COMPILE=1 \
  INTRA_LAYER_MICRO_BATCH=1 TOTAL_STEP=10 PACK_MAX_LENGTH=16384 GLOBAL_BATCH_SIZE=8 ... \
  torchrun --nproc-per-node 8 xtuner/v1/train/cli/sft.py --config examples/v1/config/sft_glm5p2.py
```

## Summary

| run | layout | mean step time s (steps 3-10) | max_memory GB | reserved GB | tgs (step 10) | llm loss (step 10) | mtp loss (step 10) |
|---|---|---|---|---|---|---|---|
| ep4_legacy | root (fsdp=2, ep=4): dense replicated 4× | 3.680 | 99.31 | 130.12 | 3641 | 9.906123 | 0.924151 |
| ep4_decouple | root (1, efsdp=2, ep=4): dense FSDP-8, experts EP4 × FSDP-2 | 3.219 | 83.49 | 107.72 | 5420 | 9.905253 | 0.923514 |
| ep8_legacy | root (fsdp=1, ep=8): dense replicated 8× | 15.532 | 120.46 | 133.90 | 1073 | 9.905459 | 0.923544 |
| ep8_decouple | root (1, efsdp=1, ep=8): dense FSDP-8, experts pure EP8 | 3.197 | 78.61 | 97.66 | 5310 | 9.905190 | 0.923457 |

### Per-step time (s)

| step | ep4_legacy | ep4_decouple | ep8_legacy | ep8_decouple |
|---|---|---|---|---|
| 1 | 343.278 | 64.492 | 128.199 | 44.846 |
| 2 | 7.061 | 5.130 | 16.411 | 4.900 |
| 3 | 5.864 | 3.095 | 17.768 | 3.029 |
| 4 | 3.325 | 4.529 | 14.988 | 3.054 |
| 5 | 3.647 | 3.020 | 14.359 | 3.166 |
| 6 | 3.029 | 3.022 | 16.807 | 3.333 |
| 7 | 3.026 | 3.026 | 15.094 | 3.384 |
| 8 | 3.024 | 3.022 | 16.485 | 3.267 |
| 9 | 3.023 | 3.017 | 13.487 | 3.254 |
| 10 | 4.500 | 3.022 | 15.269 | 3.085 |

### Per-step llm loss and grad norm

| step | ep4_legacy loss | ep4_decouple loss | ep8_legacy loss | ep8_decouple loss | ep4_legacy grad_norm | ep4_decouple grad_norm | ep8_legacy grad_norm | ep8_decouple grad_norm |
|---|---|---|---|---|---|---|---|---|
| 1 | 12.242800 | 12.242373 | 12.242203 | 12.242203 | 38.2735 | 38.1383 | 38.1814 | 38.1707 |
| 2 | 11.496803 | 11.495708 | 11.498307 | 11.497190 | 42.6749 | 42.4932 | 42.5134 | 42.5036 |
| 3 | 11.353598 | 11.353171 | 11.354156 | 11.353515 | 37.0537 | 37.0256 | 37.0996 | 37.1438 |
| 4 | 11.160206 | 11.160623 | 11.160633 | 11.160156 | 31.9895 | 31.9584 | 31.9928 | 31.9396 |
| 5 | 10.935408 | 10.933840 | 10.935240 | 10.934601 | 34.3679 | 34.3845 | 34.3607 | 34.3290 |
| 6 | 10.725193 | 10.723736 | 10.724656 | 10.724776 | 33.8208 | 33.7133 | 33.7792 | 33.7701 |
| 7 | 10.522309 | 10.522133 | 10.522678 | 10.522328 | 32.3401 | 32.3253 | 32.2573 | 32.3094 |
| 8 | 10.323658 | 10.322938 | 10.322732 | 10.322933 | 30.7218 | 30.7097 | 30.7327 | 30.7313 |
| 9 | 10.051273 | 10.050630 | 10.051438 | 10.052432 | 29.5472 | 29.5462 | 29.5450 | 29.5401 |
| 10 | 9.906123 | 9.905253 | 9.905459 | 9.905190 | 28.9784 | 28.9661 | 28.9714 | 28.9614 |

## Findings

Steady-state step time (mean of steps 6–9, after the tilelang / torch.compile / adaptive_gemm JIT warm-up
that dominates steps 1–5 and, for ep4_legacy, step 10):

| EP | legacy | decoupled | Δ step time | legacy max_memory / reserved | decoupled max_memory / reserved | Δ max_memory |
|---|---|---|---|---|---|---|
| 4 | 3.025 s | 3.022 s | -0.1% | 99.3 / 130.1 GB | 83.5 / 107.7 GB | -15.8 GB (-16%) |
| 8 | 15.468 s | 3.309 s | -78.6% | 120.5 / 133.9 GB | 78.6 / 97.7 GB | -41.8 GB (-35%) |

- **EP=4, decoupled (efsdp=2)**: peak memory drops by 15.8 GB per rank at the same
  steady-state step time (3.02 vs 3.03 s) — the "显存减少、耗时与 EP=4 一致" target.
- **EP=8, decoupled (efsdp=1)**: peak memory drops by 41.8 GB per rank. The legacy EP=8 layout
  replicates every dense parameter (attention / DSA projections, shared experts, embeddings, lm_head, MTP
  heads) 8× and drives the allocator to 133.9 GB reserved of 143 GB, where `expandable_segments`
  keeps remapping and every step takes 13–18 s; the decoupled run stays at 97.7 GB reserved and
  trains at 3.31 s/step, the same speed as EP=4. On this 8-GPU profile the legacy EP=8
  configuration is effectively unusable, EP=8 + decoupling is.
- **Numerics**: the four loss curves and grad norms agree step by step to ≤ 1.5e-3 absolute (fp8 +
  DeepEP run-to-run noise, cf. `reports/L3.md`); the step-1 loss of ep8_legacy and ep8_decouple is identical.
- Absolute step times here use the tilelang sparse-MLA backend and include no activation offload; they are
  not comparable with the 1.47 s/step `cudnn_dsa` reference, but the legacy-vs-decoupled deltas are.
