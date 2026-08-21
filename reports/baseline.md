# Phase 0 — Baseline snapshot (before any decoupling code)

Environment: `torch 2.8.0+cu128`, 8×H200 (single node), branch `feat/decouple-ep-fsdp` cut from
`upstream/main` @ `b934f462`. All runs use `PYTHONPATH=/workspace/xtuner-feat`.

## Existing MoE-related tests

| Test file | Result | Note |
|---|---|---|
| `tests/model/test_moe.py` (8 GPU) | 3 passed | EP/all2all parallel accuracy vs. single-GPU reference |
| `tests/model/test_fsdp_model.py` (8 GPU) | 1 passed | |
| `tests/module/test_grouped_linear.py` | 2 passed | |
| `tests/utils/test_load_spec.py`, `tests/model/test_model_config.py` | all passed | CPU |
| `tests/utils/test_interleaved_shard.py` | 1 failed (pre-existing) | `TestInterleavedShardPostFSDP::test_reconstruct_and_load`: `NotImplementedError: FSDP only supports 1D TP, not (Shard(dim=0), InterleavedShard(...))` — the Expert-TP path needs a newer PyTorch than 2.8; unrelated to EP/FSDP decoupling |
| `tests/engine/test_moe_train_engine.py`, `tests/model/test_qwen3_moe.py` | not run | require `QWEN3_MOE_PATH` (Qwen3-30B-A3B), not available on this machine |
| `tests/model/test_glm52_moe.py` | not run | installed `transformers` lacks `transformers.models.glm_moe_dsa` |

## New L0 test (fake ProcessGroup, single process)

`tests/model/test_decoupled_ep_fsdp_mesh.py::TestLegacyMoEMeshLayout` — 12 passed.

It pins the legacy layout that `decouple_ep_fsdp=False` must keep bit-for-bit:

- root mesh `(fsdp = world / ep, ep)`, `ep` innermost (contiguous ranks), `fsdp` ranks strided by `ep`;
- dense params: `(Shard(0), Replicate())` on `(fsdp, ep)` → local rows = `rows / (world / ep)`
  (the `ep`-fold replication described in DESIGN §1.3);
- routed experts: `(_StridedShard(0, split_factor=ep), Shard(0))` on `(fsdp, ep)` → local rows = `rows / world`;
- `ep=1` → plain 1D FSDP; `hsdp_sharding_size` + `ep=1` → `(Replicate(), Shard(0))` on
  `(hsdp_replicate, hsdp_shard)`; `hsdp_sharding_size` + `ep>1` rejected by `FSDPConfig`.

Verified for `(world, ep, rank)` ∈ {(8,8,0), (8,8,5), (8,4,6), (8,2,3), (16,8,9), (64,8,37)}.
