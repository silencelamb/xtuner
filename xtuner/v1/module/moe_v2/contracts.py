# Copyright (c) OpenMMLab. All rights reserved.
"""Unified EP contract v0.

Data contracts shared by token dispatchers, expert GEMM backends and EP execution runtimes. This module
imports no communication or GEMM library: every backend maps its native metadata onto these types.

Design notes (see ``xtuner_统一EP接口_v0.3`` in the planning repo):

* Row layout is explicit. A dispatcher publishes ``ExpertRows`` describing, per local compute slot, the
  start row, the rows a GEMM may touch (``compute_counts``) and the rows that hold real tokens
  (``valid_counts``). ``tokens_per_expert`` is kept as an alias of ``compute_counts`` so counts-only GEMM
  kernels keep working unchanged.
* GPU metadata is the contract. Host mirrors are optional and only for consumers that declare
  ``needs_host_counts``.
* Expert weights are segments. ``None`` means "read the module's own parameters"; a backend that owns
  call-local weight views (replica experts, VMM aliases) fills one or two segments.
* Control state (plans, events, slots) travels in ``EPCall`` and never enters the compiled expert block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import torch
from torch import Tensor
from typing_extensions import TypedDict


Layout = Literal["E0", "EA"]


class ExpertRows(NamedTuple):
    """Row layout of one dispatched expert batch on the local rank.

    Invariants (all tensors are GPU ``int32`` of shape ``[P]`` with ``P`` local compute slots):

    * ``starts[0] == 0`` and ``starts[i + 1] == starts[i] + compute_counts[i]``.
    * Real tokens of slot ``i`` occupy the prefix ``[starts[i], starts[i] + valid_counts[i])``.
    * ``valid_counts <= compute_counts``; ``compute_counts`` is a multiple of ``alignment``.
    * ``starts[-1] + compute_counts[-1] <= hidden_states.shape[0]`` (capacity may exceed it).
    """

    starts: Tensor
    compute_counts: Tensor
    valid_counts: Tensor | None
    alignment: int
    padding_zeroed: bool
    host_counts: Tensor | None = None

    @property
    def layout(self) -> Layout:
        return "E0" if self.alignment == 1 else "EA"

    @property
    def num_slots(self) -> int:
        return int(self.compute_counts.shape[0])


def rows_from_counts(counts: Tensor, *, host_counts: Tensor | None = None) -> ExpertRows:
    """Build an ``E0`` layout from plain per-expert counts (legacy ``tokens_per_expert``).

    Args:
        counts (Tensor): GPU ``[P]`` integer token counts, no padding.
        host_counts (Tensor | None): Optional CPU mirror of ``counts``.

    Returns:
        ExpertRows: Contiguous, unpadded layout with ``valid_counts is compute_counts``.
    """
    counts = counts.to(torch.int32)
    starts = torch.cumsum(counts, 0) - counts
    return ExpertRows(
        starts=starts,
        compute_counts=counts,
        valid_counts=counts,
        alignment=1,
        padding_zeroed=True,
        host_counts=host_counts,
    )


def rows_from_psum(
    psum: Tensor,
    valid_counts: Tensor,
    *,
    alignment: int,
    padding_zeroed: bool,
    capacity: int | None = None,
    host_counts: Tensor | None = None,
) -> ExpertRows:
    """Build an ``EA`` layout from DeepEP-V2 / DeepGEMM style prefix sums.

    ``psum[i]`` is the real end row of slot ``i`` (``start_i + valid_i``) and slot ``i + 1`` starts at
    ``align_up(psum[i], alignment)``. The last slot's compute extent is aligned up as well; when
    ``capacity`` is given the trailing spare rows are *not* folded into any slot.

    Args:
        psum (Tensor): GPU ``[P]`` int32, real end row per slot.
        valid_counts (Tensor): GPU ``[P]`` int32, real token count per slot.
        alignment (int): Segment alignment in rows.
        padding_zeroed (bool): Whether padding rows were zero-filled by the producer.
        capacity (int | None): Total rows of the hidden buffer, for the invariant check only.
        host_counts (Tensor | None): Optional CPU mirror of the compute counts.

    Returns:
        ExpertRows: Aligned layout consumable by both counts-only and psum-aware GEMM kernels.
    """
    del capacity
    psum = psum.to(torch.int32)
    valid_counts = valid_counts.to(torch.int32)
    starts = psum - valid_counts
    ends_aligned = _align_up(psum, alignment)
    compute_counts = ends_aligned - starts
    return ExpertRows(
        starts=starts,
        compute_counts=compute_counts,
        valid_counts=valid_counts,
        alignment=alignment,
        padding_zeroed=padding_zeroed,
        host_counts=host_counts,
    )


def _align_up(x: Tensor, alignment: int) -> Tensor:
    if alignment == 1:
        return x
    return (x + (alignment - 1)) // alignment * alignment


def rows_psum(rows: ExpertRows) -> Tensor:
    """DeepGEMM ``grouped_layout`` for ``use_psum_layout=True``: real end row per slot."""
    valid = rows.valid_counts if rows.valid_counts is not None else rows.compute_counts
    return rows.starts + valid


class ExpertBatch(TypedDict):
    """The one data object passed from ``dispatch_postprocess`` to the expert block and back.

    Keys are fixed (``torch.compile`` rejects optional-key TypedDicts). ``tokens_per_expert`` always
    equals ``rows.compute_counts`` and exists for counts-only kernels.
    """

    hidden_states: Tensor
    hidden_scales: Tensor | None
    tokens_per_expert: Tensor
    rows: ExpertRows
    probs: Tensor | None
    expert_weights: "ExpertWeights"
    tokens_per_expert_cpu: Tensor | None


class GradBinding(NamedTuple):
    path: Literal["autograd", "external"]
    buffer: Tensor | None = None
    write_op: Literal["overwrite", "add"] | None = None


class ProjectionWeight(NamedTuple):
    value: Tensor
    backward_read: Literal["saved", "restore_before_dgrad"] = "saved"
    grad: GradBinding = GradBinding("autograd")


class ExpertWeightSegment(NamedTuple):
    first_slot: int
    w1w3: ProjectionWeight
    w2: ProjectionWeight


class ExpertWeights(NamedTuple):
    """Call-local expert weights. ``segments is None`` means "use the module's own parameters"."""

    segments: tuple[ExpertWeightSegment, ...] | None = None


@dataclass(frozen=True)
class DispatcherCaps:
    """Static capabilities a dispatcher declares for configuration-time validation and scheduling."""

    name: str
    produces_alignment: int = 1
    phys_expert_space: bool = True
    host_counts: bool = True
    host_sync_free: bool = False
    static_shape: bool = False
    combine_applies_probs: bool = False
    fp8_dispatch: bool = False
    hidden_multiple: int = 1
    ep_sizes: tuple[int, ...] | None = None
    supports_recompute: bool = True
    async_stream: bool = True


@dataclass(frozen=True)
class MoESpec:
    """Setup-time description of one MoE layer family, given to dispatchers and GEMM backends once."""

    hidden: int
    num_topk: int
    num_experts: int
    num_local_experts: int
    max_tokens_per_rank: int
    dtype: torch.dtype = torch.bfloat16
    alignment: int = 1
    fp8_dispatch: bool = False


@dataclass
class EPCall:
    """Per-invocation control state. Never passed into the compiled expert block."""

    layer_idx: int
    micro_batch: int = 0
    plan: Any = None
    events: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LayerEPExecution(Protocol):
    """Per-layer hooks of a model-scoped EP runtime. The default implementation is identity."""

    def prepare_layer_inputs(self, inputs: list[Tensor]) -> tuple[list[Tensor], list[EPCall]]: ...

    def prepare_dispatch(
        self, call: EPCall, hidden_states: Tensor, topk_ids: Tensor, topk_weights: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def prepare_experts(self, call: EPCall, batch: ExpertBatch) -> ExpertBatch: ...

    def attach_after_experts(self, call: EPCall, output: Tensor) -> Tensor: ...

    def attach_after_combine(self, call: EPCall, output: Tensor) -> Tensor: ...


class NoOpLayerEPExecution:
    """Identity hooks: builds no autograd nodes and touches no stream."""

    def __init__(self, layer_idx: int = 0) -> None:
        self._layer_idx = layer_idx

    def prepare_layer_inputs(self, inputs: list[Tensor]) -> tuple[list[Tensor], list[EPCall]]:
        return inputs, [EPCall(layer_idx=self._layer_idx, micro_batch=i) for i in range(len(inputs))]

    def prepare_dispatch(
        self, call: EPCall, hidden_states: Tensor, topk_ids: Tensor, topk_weights: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        del call
        return hidden_states, topk_ids, topk_weights

    def prepare_experts(self, call: EPCall, batch: ExpertBatch) -> ExpertBatch:
        del call
        return batch

    def attach_after_experts(self, call: EPCall, output: Tensor) -> Tensor:
        del call
        return output

    def attach_after_combine(self, call: EPCall, output: Tensor) -> Tensor:
        del call
        return output


@runtime_checkable
class EPExecutionRuntime(Protocol):
    """Model-scoped EP runtime lifecycle (four boundaries the model calls unconditionally)."""

    def bind_layer(self, *, layer_fqn: str, layer_idx: int, projections: tuple[Any, Any]) -> LayerEPExecution: ...

    def validate_before_fsdp(self, fsdp_config: Any) -> None: ...

    def install_after_fsdp(self, *, fsdp_root: Any, execution_order: list[str]) -> None: ...

    def close(self) -> None: ...


class NoOpEPExecutionRuntime:
    """Runtime for backends without model-scoped resources."""

    def bind_layer(self, *, layer_fqn: str, layer_idx: int, projections: tuple[Any, Any]) -> LayerEPExecution:
        del layer_fqn, projections
        return NoOpLayerEPExecution(layer_idx)

    def validate_before_fsdp(self, fsdp_config: Any) -> None:
        del fsdp_config

    def install_after_fsdp(self, *, fsdp_root: Any, execution_order: list[str]) -> None:
        del fsdp_root, execution_order

    def close(self) -> None:
        return


def check_rows(rows: ExpertRows, capacity: int) -> None:
    """Debug-only invariant check. Performs host synchronization; never call on the hot path."""
    starts = rows.starts.cpu()
    compute = rows.compute_counts.cpu()
    if starts.numel() == 0:
        return
    assert int(starts[0]) == 0, "first slot must start at row 0"
    ends = starts + compute
    assert torch.equal(ends[:-1], starts[1:]), "slots must be contiguous"
    assert int(ends[-1]) <= capacity, f"rows exceed capacity: {int(ends[-1])} > {capacity}"
    assert bool((compute % rows.alignment == 0).all()), "compute_counts must be multiples of alignment"
    if rows.valid_counts is not None:
        valid = rows.valid_counts.cpu()
        assert bool((valid <= compute).all()) and bool((valid >= 0).all()), "valid_counts out of range"


__all__ = [
    "DispatcherCaps",
    "EPCall",
    "EPExecutionRuntime",
    "ExpertBatch",
    "ExpertRows",
    "ExpertWeightSegment",
    "ExpertWeights",
    "GradBinding",
    "Layout",
    "LayerEPExecution",
    "MoESpec",
    "NoOpEPExecutionRuntime",
    "NoOpLayerEPExecution",
    "ProjectionWeight",
    "check_rows",
    "rows_from_counts",
    "rows_from_psum",
    "rows_psum",
]
