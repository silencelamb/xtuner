# Decisions not covered by DESIGN.md

Each entry records the options considered and the one taken, before the code was written.

## D1. L0 tests need a CUDA context (not a GPU-free CI job)

`MoE.__init__` unconditionally creates `torch.cuda.Stream()` and the all2all dispatcher allocates a
CUDA buffer in its constructor, so a model cannot even be constructed without a CUDA device.
Options: (a) stub `torch.cuda.*` in the test, (b) change `MoE.__init__`, (c) `skipif(not cuda)`.
Taken: (c). (b) touches the legacy execution path (forbidden); (a) stubs production code and would
hide real breakage. The tests still never communicate (fake ProcessGroup) and never materialize
parameters (`_to_empty_meta` is patched to a no-op), so a single idle GPU is enough.

## D2. Expert TP is out of scope for the decoupled path

`decouple_ep_fsdp=True` with `expert_tp_size > 1` raises `NotImplementedError`. The ETP layout
relies on a 2-placement `(Shard, InterleavedShard)` DTensor that FSDP2 on torch 2.8 already rejects
("FSDP only supports 1D TP"), and DESIGN §4 only specifies the `(replicate, efsdp, ep)` root. Adding
an `etp` dimension is a separate change.

## D3. All sub-meshes derive from one root via `DeviceMesh._flatten`

Options: (a) DESIGN §4.2 — one root `(replicate, efsdp, ep)` and `root[efsdp, ep]._flatten("dp_shard")`
for the dense shard mesh; (b) a second independent `(replicate, dp_shard)` root for dense params
(legal because dense params are plain tensors when `fully_shard` sees them).
Taken: (a). It is the torchtitan #1324 layout, keeps `_mesh_resources.get_root_mesh()` consistent for
every mesh the model exposes, and was verified on torch 2.8 (`root["replicate", "dp_shard"]` slicing
with a flattened name works). The known pitfall — PyTorch caches flattened meshes per root and keys
the cache by mesh equality — is handled in tests by resetting `_mesh_resources` between fake worlds.

## D4. Attribute semantics on the decoupled path

- `fsdp_mesh` keeps meaning "1D shard group of dense params" (= `dp_shard`), so existing consumers
  (`_fsdp_foreach_allgather`, `Float8Handler.build_reduce_mesh`) keep a sensible default;
- `hsdp_mesh` = `root[replicate, dp_shard]` when `replicate > 1`, else `None` (same contract as legacy);
- new `expert_fsdp_mesh` = `root[efsdp]` or `root[replicate, efsdp]`, only set on the decoupled path.
- The `ep` mesh dimension keeps the legacy name `f"{mesh_prefix}.ep"`: the `ep_mesh` created in
  `MoE.__init__` is re-parented onto the FSDP root through PyTorch's mesh-equality hashing, exactly as
  the legacy `_init_device_mesh` comment documents. Renaming it would break that re-parenting.

## D5. Expert FSDP unit = `MoEBlock` (`layer.experts`), wrapped before checkpoint wrapping

XTuner has no `experts` sub-module containing only `GroupedLinear`s *and* nothing else except
`MoEBlock` itself (`fused_w1w3`, `fused_w2`, a parameter-free activation). `fully_shard` is applied to
every `MoEBlock` found under a decoder layer (this also covers MTP layers, which embed a
`MoEDecoderLayer`) using `expert_fsdp_mesh`, then the layer is wrapped as before. This mirrors
torchtitan's `fully_shard(transformer_block.moe.experts, mesh=dp_mod_ep_mesh)`. The expert group
inherits the layer's `reshard_after_forward`.

## D6. Gradient scaling: explicit `grad.div_(ep_size)` on experts, FSDP handles everything else

Options: (a) torchtitan #1551 `set_gradient_divide_factor`; (b) AutoModel-style explicit division.
Taken: (b), as DESIGN §4.4 recommends. On the decoupled path `scale_and_reduce_grad`:
- `.experts` params: `div_(ep_size)` (reduce-scatter over `efsdp` already averaged over `efsdp`;
  HSDP all-reduce averages over `replicate`; the remaining factor is `ep`);
- FSDP-managed params (any `Shard` placement): nothing — FSDP already averages over `dp_shard`
  (and `replicate`); the legacy manual all-reduce over `ep` is dropped;
- DTensors that are `Replicate` on every mesh dim (fp32 params ignored by FSDP through
  `fp32_keys_pattern`): the legacy coalesced all-reduce is kept, since nobody else reduces them.

## D7. Float8 padding / tile-wise reduce meshes — deferred to PR-3

`Float8Handler.pad_for_fsdp` pads every weight for a single `fsdp_mesh.size(-1)`; with decoupling,
experts are sharded `efsdp` ways and dense params `dp_shard` ways. Supporting FP8 on the decoupled
path means per-parameter-class shard sizes and reduce meshes; it is handled in PR-3 together with
the HF adapter, after the bf16 path is validated (L1/L2).

## D8. `_fsdp_foreach_allgather` (RL weight sync) — deferred to PR-3

It gathers with `fsdp_mesh.get_group()`; on the decoupled path expert params are sharded on the
`efsdp` group instead, so their FSDP shard would be preserved rather than gathered. Needs a
per-spec gather group; tracked for PR-3.

## D9. Commit layout

Four commits: (1) Phase-0 L0 tests pinning the legacy layout + baseline report; (2) PR-1 basic
decoupling (config, mesh, two-level `fully_shard`, gradient scaling, L0/L1); (3) PR-2 HSDP
coexistence + L2; (4) PR-3 ecosystem (HF/DCP, fp8) + L3.

Note on PR-2: the `replicate` mesh dimension is produced by the same `init_device_mesh` call as
`efsdp`/`ep`, and FSDP2 handles the `(Replicate, Shard)` placements itself, so there is no
HSDP-specific execution code to split out of PR-1. PR-2 is therefore the validation commit: the
HSDP fake-PG placement tests live in PR-1's test file, and PR-2 adds `reports/L2.md` (efsdp > 1,
HSDP + EP, DeepEP and empty-expert smokes, legacy regression).

## D10. L2 on a single node: scaled-down topologies instead of 16 emulated ranks

Only one 8-GPU node is available. NCCL refuses two ranks on the same GPU inside one communicator,
and the MoE model needs CUDA (streams, Triton grouped GEMM), so a 16-rank torchrun cannot be
emulated with 2 processes per GPU. Options: (a) wait for a second node; (b) gloo/CPU (FSDP2 works
on CPU but the MoE kernels do not); (c) run the *same mesh structures* at 8 ranks.
Taken: (c) plus the fake-PG L0 tests which already pin the exact 16- and 64-rank mesh shapes:

| DESIGN §6-L2 scenario (16 ranks) | 8-rank equivalent | structure exercised |
|---|---|---|
| `ep=8, decouple, dp_shard=16` (efsdp=2) | `ep=4, decouple` (efsdp=2) | experts also FSDP-sharded (`_StridedShard` over efsdp) |
| `ep=8, hsdp_sharding_size=8` (replicate=2) | `ep=4, hsdp_sharding_size=4` (replicate=2, efsdp=1) | the formerly asserted HSDP+EP combination |
| — | `ep=2, hsdp_sharding_size=4` (replicate=2, efsdp=2) | replicate and efsdp both > 1 |

Reference curves come from the legacy path at the same `ep` (bit-for-bit the same expert math)
and from the `ep=1` FSDP-8 baseline.

## D7 / D8 resolution (PR-3)

- fp8: `Float8Handler.pad_for_fsdp` takes an optional `expert_fsdp_mesh`; grouped-linear (expert)
  weights are padded for `efsdp` FSDP chunks, everything else for `dp_shard`. `build_reduce_mesh`
  builds two sets of tile-wise "reduce max" meshes on the decoupled path, each with the rank stride
  of its class (dense: stride 1 along the flattened `dp_shard`; experts: stride `ep` along `efsdp`),
  and `precompute_tilewise_float8_scale_for_fsdp` is called once per class. The legacy path still
  takes the original single-mesh code path (`expert_fsdp_mesh is None`). Tensor-wise fp8 only
  applies to dense linears, whose shard group is still `fsdp_mesh`, so it needs no change.
- RL weight sync: `_fsdp_foreach_allgather` picks the gather group per `LoadSpec` — the `efsdp`
  group for specs that carry a shard on it, `fsdp_mesh`'s group otherwise — so EP-local expert
  slices are still reconstructed with an FSDP-only gather.
- `expert_fsdp_mesh` moved to `BaseModel` (always `None` for dense models) so the float8 handler
  wiring in `BaseModel.float8_handler` needs no MoE-specific branch.
- Training is not bit-deterministic run to run (see `reports/L3.md`); acceptance of DCP resume and
  cross-layout comparisons is therefore judged against the measured run-to-run noise floor.

## D11. Environment facts that shaped the fp8 / GLM-5.2 validation

- The tiny (hidden 512, moe_inter 256) Qwen3-MoE crashes in the fp8 grouped GEMM kernel
  (`adaptive_gemm` TMA descriptor assertion → SIGSEGV) on the **legacy** `ep=8` path, reproduced on
  the Phase-0 commit (`2098db9e`) with the same command. It is a kernel shape limitation unrelated
  to this work; fp8 numerics are therefore validated on the 3.4B "medium" model and on GLM-5.2-30B.
- `transformers` on this machine is an editable 4.57.0 checkout without `glm_moe_dsa`, while
  `pyproject.toml` pins `transformers==5.14.1`. GLM-5.2 runs use an isolated
  `pip install --no-deps --target <dir> transformers==5.14.1 huggingface_hub==1.5.0 "safetensors>=0.8" "regex>=2025.10.22"`
  prepended to `PYTHONPATH`; the global environment is left untouched.

## D12. Re-validation in the intended `pt29` container

The first pass of every report was produced in a container started from the NGC 25.03 / torch
2.8.0 image of another project (same NFS home, same hostname — only `/usr/local` differed), which
is also why `transformers`/`tilelang`/cuDNN-DSA had to be improvised (D11). All L0–L3 runs, the
fp8 numerics, the legacy regression and the GLM-5.2 runs were then repeated unchanged in the
`pt29` container (torch 2.9.1+cu128, transformers 5.14.1, tilelang 0.1.11, cuDNN frontend 1.26 with
DSA, DeepEP). `reports/L1.md`, `L2.md`, `L3.md` and `GLM52.md` now carry the torch-2.9 numbers;
conclusions are unchanged. Note for memory comparisons: XTuner's logged `max_memory` is
`max_memory_allocated() / 1024**3`, i.e. GiB despite the "GB" label, so it is directly comparable
with AutoModel's `mem … GiB`.
