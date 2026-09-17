from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Generic,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import torch
from typing_extensions import TypedDict, override

from xtuner.v1.ops import permute, unpermute

from .expert_tp import ExpertTP


HiddenStates: TypeAlias = torch.Tensor


def _get_backward_pre_hook(backward_previous_event: torch.cuda.Event):
    def _backward_pre_hook(*_):
        torch.cuda.current_stream().wait_event(backward_previous_event)

    return _backward_pre_hook


def _get_backward_hook(backward_finished_event: torch.cuda.Event):
    def _backward_hook(*_):
        backward_finished_event.record()

    return _backward_hook


# ----------------------------------------------------------------------------------------------------------------------
# Unified EP contract: how a dispatcher describes the expert batch it produced, and the seams an EP execution
# runtime (UltraEP / MoonEP style) plugs into. No communication or GEMM library is imported here.
# ----------------------------------------------------------------------------------------------------------------------


class Layout(str, Enum):
    """Row-layout families of a dispatched expert batch.

    ``ExpertRows`` describes ``E0`` and ``EA``; ``R`` and ``M`` are reserved for backends that need a gather index
    or a per-expert mask and are not expressible by ``ExpertRows`` yet.
    """

    E0 = "expert_contig"  # per-expert contiguous rows, no padding (``alignment == 1``)
    EA = "expert_aligned"  # per-expert segments aligned to ``alignment`` rows, zero padding rows in between
    R = "rank_grouped"  # reserved: rows grouped by source rank plus a gather index (SonicMoE-style consumers)
    M = "masked"  # reserved: ``[experts, max_rows, hidden]`` with a per-expert mask (decode / low latency)


class ExpertRows(NamedTuple):
    """Row layout of one dispatched expert batch on the local rank.

    Invariants (all tensors are GPU integer tensors of shape ``[P]``, ``P`` local compute slots):

    * ``starts[0] == 0`` and ``starts[i + 1] == starts[i] + compute_counts[i]``.
    * Real tokens of slot ``i`` occupy the prefix ``[starts[i], starts[i] + valid_counts[i])``.
    * ``valid_counts <= compute_counts``; ``compute_counts`` is a multiple of ``alignment``.
    * ``starts[-1] + compute_counts[-1] <= hidden_states.shape[0]`` (the buffer capacity may exceed it).

    ``alignment == 1`` is the unpadded legacy layout (``E0``); ``alignment > 1`` is the segment-aligned layout
    (``EA``) that psum-style grouped GEMMs consume without a permute. Alignment padding rows are finite zeros when
    ``padding_zeroed`` is set; rows past the last slot are unspecified and must never be reduced over.
    """

    starts: torch.Tensor
    compute_counts: torch.Tensor
    valid_counts: torch.Tensor | None
    alignment: int
    padding_zeroed: bool
    host_counts: torch.Tensor | None = None

    @property
    def num_slots(self) -> int:
        return int(self.compute_counts.shape[0])

    @property
    def layout(self) -> Layout:
        """Layout family derived from ``alignment``; ``ExpertRows`` never carries a stored layout field."""
        return Layout.EA if self.alignment > 1 else Layout.E0


def rows_from_counts(counts: torch.Tensor, *, host_counts: torch.Tensor | None = None) -> ExpertRows:
    """Describe plain per-expert counts (the legacy ``tokens_per_expert``) as an unpadded ``ExpertRows``.

    Args:
        counts (torch.Tensor): GPU ``[P]`` integer token counts, no padding.
        host_counts (torch.Tensor | None): Optional CPU mirror of ``counts``.

    Returns:
        ExpertRows: Contiguous, unpadded layout with ``valid_counts is compute_counts``.
    """
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
    psum: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    alignment: int,
    padding_zeroed: bool,
    host_counts: torch.Tensor | None = None,
) -> ExpertRows:
    """Describe a segment-aligned layout given DeepEP-V2 / DeepGEMM style prefix sums.

    ``psum[i]`` is the real end row of slot ``i`` (``starts[i] + valid_counts[i]``) and slot ``i + 1`` starts at
    ``psum[i]`` rounded up to ``alignment``.

    Args:
        psum (torch.Tensor): GPU ``[P]`` int32, real end row per slot.
        valid_counts (torch.Tensor): GPU ``[P]`` int32, real token count per slot.
        alignment (int): Segment alignment in rows.
        padding_zeroed (bool): Whether the producer zero-filled the alignment padding rows.
        host_counts (torch.Tensor | None): Optional CPU mirror of the compute counts.

    Returns:
        ExpertRows: Aligned layout consumable by counts-only and psum-aware grouped GEMMs alike.
    """
    psum = psum.to(torch.int32)
    valid_counts = valid_counts.to(torch.int32)
    starts = psum - valid_counts
    ends_aligned = psum if alignment == 1 else (psum + (alignment - 1)) // alignment * alignment
    return ExpertRows(
        starts=starts,
        compute_counts=ends_aligned - starts,
        valid_counts=valid_counts,
        alignment=alignment,
        padding_zeroed=padding_zeroed,
        host_counts=host_counts,
    )


def rows_psum(rows: ExpertRows) -> torch.Tensor:
    """Real end row per slot (``starts + valid_counts``): DeepGEMM's ``grouped_layout`` for its psum layout."""
    valid = rows.valid_counts if rows.valid_counts is not None else rows.compute_counts
    return rows.starts + valid


def check_rows(rows: ExpertRows, capacity: int) -> None:
    """Debug-only invariant check; it synchronizes with the host, so never call it on the hot path.

    Args:
        rows (ExpertRows): The layout to check.
        capacity (int): Number of rows of the hidden-states buffer.
    """
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


class GradBinding(NamedTuple):
    path: Literal["autograd", "external"]
    buffer: torch.Tensor | None = None
    write_op: Literal["overwrite", "add"] | None = None


class ProjectionWeight(NamedTuple):
    value: torch.Tensor
    backward_read: Literal["saved", "restore_before_dgrad"] = "saved"
    grad: GradBinding = GradBinding("autograd")


class ExpertWeightSegment(NamedTuple):
    first_slot: int
    w1w3: ProjectionWeight
    w2: ProjectionWeight


class ExpertWeights(NamedTuple):
    """Call-local expert weights; ``segments is None`` means "use the expert module's own parameters"."""

    segments: tuple[ExpertWeightSegment, ...] | None = None


@dataclass(frozen=True)
class DispatcherCaps:
    """Static capabilities a dispatcher declares for configuration-time checks and scheduling decisions."""

    name: str
    produces_alignment: int = 1
    host_counts: bool = True
    host_sync_free: bool = False
    static_shape: bool = False
    combine_applies_probs: bool = False
    fp8_dispatch: bool = False
    hidden_multiple: int = 1

    @property
    def produces_layout(self) -> Layout:
        """Layout family of the batches this dispatcher produces, derived from ``produces_alignment``."""
        return Layout.EA if self.produces_alignment > 1 else Layout.E0


@dataclass
class EPCall:
    """Per-invocation control state of one MoE layer call (plans, events, backend handles).

    It travels through the execution hooks and ``dispatch_preprocess`` and never enters the compiled expert block.
    """

    layer_idx: int
    micro_batch: int = 0
    plan: Any = None
    events: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LayerEPExecution(Protocol):
    """Per-layer hooks of a model-scoped EP execution runtime; the default implementation is identity."""

    def prepare_layer_inputs(self, inputs: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[EPCall]]: ...

    def prepare_dispatch(
        self, call: EPCall, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def prepare_experts(self, call: EPCall, batch: "PostDispatchResult") -> "PostDispatchResult": ...

    def attach_after_experts(self, call: EPCall, output: torch.Tensor) -> torch.Tensor: ...

    def attach_after_combine(self, call: EPCall, output: torch.Tensor) -> torch.Tensor: ...


class NoOpLayerEPExecution:
    """Identity hooks: no autograd nodes, no stream work.

    Args:
        layer_idx (int): Index of the layer the hooks are bound to.
    """

    def __init__(self, layer_idx: int = 0) -> None:
        self._layer_idx = layer_idx

    def prepare_layer_inputs(self, inputs: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[EPCall]]:
        return inputs, [EPCall(layer_idx=self._layer_idx, micro_batch=i) for i in range(len(inputs))]

    def prepare_dispatch(
        self, call: EPCall, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hidden_states, topk_ids, topk_weights

    def prepare_experts(self, call: EPCall, batch: "PostDispatchResult") -> "PostDispatchResult":
        return batch

    def attach_after_experts(self, call: EPCall, output: torch.Tensor) -> torch.Tensor:
        return output

    def attach_after_combine(self, call: EPCall, output: torch.Tensor) -> torch.Tensor:
        return output


@runtime_checkable
class EPExecutionRuntime(Protocol):
    """Model-scoped EP execution runtime: the four boundaries the model calls unconditionally."""

    def bind_layer(self, *, layer_fqn: str, layer_idx: int, projections: tuple[Any, Any]) -> LayerEPExecution: ...

    def validate_before_fsdp(self, fsdp_config: Any) -> None: ...

    def install_after_fsdp(self, *, fsdp_root: Any, execution_order: list[str]) -> None: ...

    def close(self) -> None: ...


class NoOpEPExecutionRuntime:
    """Runtime for backends without model-scoped resources."""

    def bind_layer(self, *, layer_fqn: str, layer_idx: int, projections: tuple[Any, Any]) -> LayerEPExecution:
        return NoOpLayerEPExecution(layer_idx)

    def validate_before_fsdp(self, fsdp_config: Any) -> None:
        return

    def install_after_fsdp(self, *, fsdp_root: Any, execution_order: list[str]) -> None:
        return

    def close(self) -> None:
        return


class PreDispatchResult(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor


class DispatchResult(TypedDict):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor


class PostDispatchResult(TypedDict):
    """The expert batch a dispatcher hands to ``MoEBlock.forward`` (one object, fixed keys).

    Keys are fixed because ``torch.compile`` rejects optional-key TypedDicts. ``tokens_per_expert`` always equals
    ``rows.compute_counts`` and exists for counts-only grouped GEMM kernels.

    Attributes:
        hidden_states: The dispatched activations, ``[capacity, hidden]`` in the layout described by ``rows``.
        hidden_scales: Per-tile FP8 scales when the dispatcher delivers quantized activations, else ``None``.
        tokens_per_expert: Rows a grouped GEMM may touch per local expert (alias of ``rows.compute_counts``).
        rows: Explicit row layout of the batch, see :class:`ExpertRows`.
        expert_weights: Call-local expert weights, see :class:`ExpertWeights`.
    """

    hidden_states: torch.Tensor
    hidden_scales: torch.Tensor | None
    tokens_per_expert: torch.Tensor
    rows: ExpertRows
    expert_weights: ExpertWeights


class PreCombineResult(TypedDict):
    hidden_states: torch.Tensor


class CombineResult(TypedDict):
    hidden_states: torch.Tensor


class PostCombineResult(TypedDict):
    hidden_states: torch.Tensor


PreDispatch = TypeVar("PreDispatch")
Dispatch = TypeVar("Dispatch")
PostDispatch = TypeVar("PostDispatch")
PreCombine = TypeVar("PreCombine")
Combine = TypeVar("Combine")
PostCombine = TypeVar("PostCombine")
# TODO: add DecodingPostDispatch if needed.


# Not using Protocol here since `__init__` is shared for all dispatchers.
class GenericDispatcher(
    ABC,
    Generic[
        PreDispatch,
        Dispatch,
        PostDispatch,
        PreCombine,
        Combine,
        PostCombine,
    ],
):
    _n_routed_experts: int
    _process_group: torch.distributed.ProcessGroup | None

    def __init__(
        self,
        *,
        n_routed_experts: int,
        process_group: torch.distributed.ProcessGroup | None = None,
        training_dtype: Literal["fp8", "bf16"] = "bf16",
        generate_dtype: Literal["fp8", "bf16"] = "bf16",
    ):
        self._process_group = process_group
        self._n_routed_experts = n_routed_experts
        self._training_dtype = training_dtype
        self._generate_dtype = generate_dtype
        # Legacy dispatchers permute into the unpadded layout and return exact counts on the GPU.
        self.caps = DispatcherCaps(name=type(self).__name__)

    @abstractmethod
    def dispatch(
        self,
        *,
        pre_dispatched: PreDispatch,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> Dispatch: ...

    @abstractmethod
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: PreDispatch,
        dispatched: Dispatch,
        async_op: bool = False,
    ) -> PostDispatch: ...

    @abstractmethod
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        tokens_per_expert: torch.Tensor | None = None,
        layer_state: EPCall | None = None,
    ) -> PreDispatch:
        """Stage 1 of 6. ``tokens_per_expert`` are the router's logical counts and ``layer_state`` the call's
        control state; dispatchers that plan ahead (MoonEP-style) read them, the others ignore them."""

    @abstractmethod
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: PreDispatch,
        dispatched: Dispatch,
        post_dispatched: PostDispatch,
        async_op: bool = False,
        decoding: bool = False,
    ) -> PreCombine: ...

    @abstractmethod
    def combine(
        self,
        *,
        pre_dispatched: PreDispatch,
        dispatched: Dispatch,
        post_dispatched: PostDispatch,
        pre_combined: PreCombine,
        async_op: bool = False,
        decoding: bool = False,
    ) -> CombineResult: ...

    @abstractmethod
    def combine_postprocess(
        self,
        *,
        pre_dispatched: PreDispatch,
        dispatched: Dispatch,
        post_dispatched: PostDispatch,
        pre_combined: PreCombine,
        combined: Combine,
        async_op: bool = False,
    ) -> PostCombine: ...


class DispacherInterface(
    GenericDispatcher[
        PreDispatchResult,
        DispatchResult,
        PostDispatchResult,
        PreCombineResult,
        CombineResult,
        PostCombineResult,
    ],
): ...


class NaivePreDispatchResult(PreDispatchResult):
    # 中文注释：这些 key 必须始终存在；torch.compile 不支持 optional-key TypedDict。
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class NaiveDispatchResult(DispatchResult):
    topk_ids: torch.Tensor
    tp_rank_row_counts: list[int]
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None
    topk_weights_backward_previous_event: torch.cuda.Event | None


class NaivePostDispatchResult(PostDispatchResult):
    row_ids_map: torch.Tensor


class NaivePreCombineResult(PreCombineResult):
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class NaiveCombineResult(CombineResult):
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class NaivePostCombineResult(PostCombineResult): ...


class NaiveDispatcher(
    GenericDispatcher[
        NaivePreDispatchResult,
        NaiveDispatchResult,
        NaivePostDispatchResult,
        NaivePreCombineResult,
        NaiveCombineResult,
        NaivePostCombineResult,
    ]
):
    _comm_stream: torch.cuda.Stream | None = None

    def __init__(
        self,
        *,
        n_routed_experts: int,
        process_group: torch.distributed.ProcessGroup | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        training_dtype: Literal["fp8", "bf16"] = "bf16",
        generate_dtype: Literal["fp8", "bf16"] = "bf16",
    ):
        super().__init__(
            n_routed_experts=n_routed_experts,
            process_group=process_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )
        if self._process_group is not None:
            assert self._process_group.size() == 1, "Naive dispatcher is only for ep=1."
        self._expert_tp = ExpertTP(tp_group) if tp_group is not None and tp_group.size() > 1 else None
        if self._expert_tp is not None and NaiveDispatcher._comm_stream is None:
            NaiveDispatcher._comm_stream = torch.cuda.Stream()

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        tokens_per_expert: torch.Tensor | None = None,  # noqa: ARG002 — contract seam, unused by this dispatcher
        layer_state: EPCall | None = None,  # noqa: ARG002
    ) -> NaivePreDispatchResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")

            forward_finished_event = torch.cuda.Event()
            forward_finished_event.record()
            backward_previous_event = torch.cuda.Event()
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(_get_backward_pre_hook(backward_previous_event))

            return NaivePreDispatchResult(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                forward_finished_event=forward_finished_event,
                backward_previous_event=backward_previous_event,
            )

        return NaivePreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            forward_finished_event=None,
            backward_previous_event=None,
        )

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: NaivePreDispatchResult,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> NaiveDispatchResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")

            forward_previous_event = pre_dispatched["forward_finished_event"]
            backward_finished_event = pre_dispatched["backward_previous_event"]
            assert forward_previous_event is not None, "Use async_op=True for dispatch_preprocess!"
            assert backward_finished_event is not None, "Use async_op=True for dispatch_preprocess!"
            assert self._comm_stream is not None

            tp_rank_row_counts = self._expert_tp.gather_tp_rank_row_counts(pre_dispatched["hidden_states"])
            # 中文注释：dispatch 内部的 TP AllGather 都排在同一个 comm stream，
            # 互相不需要 event 串行化；只在 dispatch 阶段边界记录最终完成事件。
            forward_finished_event = torch.cuda.Event()
            hidden_backward_previous_event = torch.cuda.Event()
            topk_weights_backward_previous_event = torch.cuda.Event()
            topk_weights_backward_finished_event = torch.cuda.Event()
            if topk_weights.grad_fn is not None:
                topk_weights.grad_fn.register_prehook(_get_backward_pre_hook(topk_weights_backward_finished_event))

            hidden_states = self._expert_tp.async_all_gather_rows(
                pre_dispatched["hidden_states"],
                tp_rank_row_counts=tp_rank_row_counts,
                forward_previous_event=forward_previous_event,
                forward_finished_event=None,
                backward_previous_event=hidden_backward_previous_event,
                backward_finished_event=backward_finished_event,
                comm_stream=self._comm_stream,
            )
            topk_ids = self._expert_tp.async_all_gather_row_metadata(
                pre_dispatched["topk_ids"],
                tp_rank_row_counts=tp_rank_row_counts,
                forward_previous_event=None,
                forward_finished_event=None,
                comm_stream=self._comm_stream,
            )
            topk_weights = self._expert_tp.async_all_gather_rows(
                topk_weights,
                tp_rank_row_counts=tp_rank_row_counts,
                forward_previous_event=None,
                forward_finished_event=forward_finished_event,
                backward_previous_event=topk_weights_backward_previous_event,
                backward_finished_event=topk_weights_backward_finished_event,
                comm_stream=self._comm_stream,
            )

            return NaiveDispatchResult(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                tp_rank_row_counts=tp_rank_row_counts,
                forward_finished_event=forward_finished_event,
                backward_previous_event=hidden_backward_previous_event,
                topk_weights_backward_previous_event=topk_weights_backward_previous_event,
            )

        if self._expert_tp is not None:
            hidden_states, tp_rank_row_counts = self._expert_tp.all_gather_rows(pre_dispatched["hidden_states"])
            topk_ids = self._expert_tp.all_gather_row_metadata(pre_dispatched["topk_ids"], tp_rank_row_counts)
            topk_weights = self._expert_tp.all_gather_row_metadata(topk_weights, tp_rank_row_counts)
            return NaiveDispatchResult(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                tp_rank_row_counts=tp_rank_row_counts,
                forward_finished_event=None,
                backward_previous_event=None,
                topk_weights_backward_previous_event=None,
            )

        return NaiveDispatchResult(
            hidden_states=pre_dispatched["hidden_states"],
            topk_ids=pre_dispatched["topk_ids"],
            topk_weights=topk_weights,
            tp_rank_row_counts=[],
            forward_finished_event=None,
            backward_previous_event=None,
            topk_weights_backward_previous_event=None,
        )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: NaivePreDispatchResult,
        dispatched: NaiveDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> NaivePostDispatchResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")
            forward_finished_event = dispatched["forward_finished_event"]
            assert forward_finished_event is not None, "Use async_op=True for dispatch!"
            torch.cuda.current_stream().wait_event(forward_finished_event)

        topk_ids = dispatched["topk_ids"] if self._expert_tp is not None else pre_dispatched["topk_ids"]
        hidden_states, row_id_maps = permute(
            dispatched["hidden_states"],
            topk_ids.to(torch.int32),
        )
        tokens_per_expert = torch.histc(topk_ids, bins=self._n_routed_experts, min=0, max=self._n_routed_experts)
        if async_op:
            backward_previous_event = dispatched["backward_previous_event"]
            assert backward_previous_event is not None, "Use async_op=True for dispatch!"
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_hook(_get_backward_hook(backward_previous_event))

        if decoding:
            raise NotImplementedError
        else:
            return NaivePostDispatchResult(
                hidden_states=hidden_states,
                hidden_scales=None,
                row_ids_map=row_id_maps,
                tokens_per_expert=tokens_per_expert,
                rows=rows_from_counts(tokens_per_expert),
                expert_weights=ExpertWeights(),
            )

    @override
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: NaivePreDispatchResult,
        dispatched: NaiveDispatchResult,
        post_dispatched: NaivePostDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> NaivePreCombineResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")

        hidden_states = unpermute(
            input_act=hidden_states,
            row_id_map=post_dispatched["row_ids_map"],
            probs=dispatched["topk_weights"],
        )
        if async_op:
            backward_previous_event = torch.cuda.Event()
            forward_finished_event = torch.cuda.Event()
            forward_finished_event.record()
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(_get_backward_pre_hook(backward_previous_event))
                topk_weights_backward_previous_event = dispatched["topk_weights_backward_previous_event"]
                assert topk_weights_backward_previous_event is not None, "Use async_op=True for dispatch!"
                hidden_states.grad_fn.register_hook(_get_backward_hook(topk_weights_backward_previous_event))
        else:
            backward_previous_event = None
            forward_finished_event = None

        if decoding:
            raise NotImplementedError("NaiveDispatcher does not support decoding.")
        else:
            return NaivePreCombineResult(
                hidden_states=hidden_states,
                backward_previous_event=backward_previous_event,
                forward_finished_event=forward_finished_event,
            )

    @override
    def combine(
        self,
        *,
        pre_dispatched: NaivePreDispatchResult,
        dispatched: NaiveDispatchResult,
        post_dispatched: NaivePostDispatchResult,
        pre_combined: NaivePreCombineResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> NaiveCombineResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")

        if decoding:
            raise NotImplementedError
        else:
            if self._expert_tp is not None:
                if async_op:
                    forward_previous_event = pre_combined["forward_finished_event"]
                    backward_finished_event = pre_combined["backward_previous_event"]
                    assert forward_previous_event is not None, "Use async_op=True for combine_preprocess!"
                    assert backward_finished_event is not None, "Use async_op=True for combine_preprocess!"
                    assert self._comm_stream is not None

                    forward_finished_event = torch.cuda.Event()
                    backward_previous_event = torch.cuda.Event()
                    hidden_states = self._expert_tp.async_reduce_scatter_rows_sum(
                        pre_combined["hidden_states"],
                        tp_rank_row_counts=dispatched["tp_rank_row_counts"],
                        forward_previous_event=forward_previous_event,
                        forward_finished_event=forward_finished_event,
                        backward_previous_event=backward_previous_event,
                        backward_finished_event=backward_finished_event,
                        comm_stream=self._comm_stream,
                    )
                    return NaiveCombineResult(
                        hidden_states=hidden_states,
                        forward_finished_event=forward_finished_event,
                        backward_previous_event=backward_previous_event,
                    )

                hidden_states = self._expert_tp.reduce_scatter_rows_sum(
                    pre_combined["hidden_states"],
                    dispatched["tp_rank_row_counts"],
                )
                return NaiveCombineResult(
                    hidden_states=hidden_states,
                    forward_finished_event=None,
                    backward_previous_event=None,
                )

            return NaiveCombineResult(
                hidden_states=pre_combined["hidden_states"],
                forward_finished_event=None,
                backward_previous_event=None,
            )

    @override
    def combine_postprocess(
        self,
        *,
        pre_dispatched: NaivePreDispatchResult,
        dispatched: NaiveDispatchResult,
        post_dispatched: NaivePostDispatchResult,
        pre_combined: NaivePreCombineResult,
        combined: NaiveCombineResult,
        async_op: bool = False,
    ) -> PostCombineResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")
            forward_finished_event = combined["forward_finished_event"]
            backward_previous_event = combined["backward_previous_event"]
            assert forward_finished_event is not None, "Use async_op=True for combine!"
            assert backward_previous_event is not None, "Use async_op=True for combine!"
            torch.cuda.current_stream().wait_event(forward_finished_event)
            hidden_states = combined["hidden_states"].view_as(combined["hidden_states"])
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_hook(_get_backward_hook(backward_previous_event))
            return PostCombineResult(hidden_states=hidden_states)

        return PostCombineResult(hidden_states=combined["hidden_states"])
