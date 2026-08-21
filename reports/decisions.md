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
