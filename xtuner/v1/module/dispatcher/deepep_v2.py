# Copyright (c) OpenMMLab. All rights reserved.
"""DeepEP V2 (``ElasticBuffer``) dispatcher.

Layout: expanded (``do_expand=True``) with ``expert_alignment`` (1 or 128): one row per (token, local expert) slot,
segments aligned to the GEMM's alignment, no framework-side permute. ``dispatch_postprocess`` is zero-copy: the
received tensor *is* the expert input and its row layout comes straight from the ``EPHandle`` as ``ExpertRows``.

Modes:

* ``cpu_sync=True``: exact received counts on the host (dynamic shapes, like the legacy dispatchers).
* ``cpu_sync=False``: counts stay on the GPU, output capacity is the worst case (static shapes; the whole MoE segment
  is CUDA-graph capturable together with ``EP_AVOID_RECORD_STREAM=1``).
* ``fp8_dispatch=True``: activations are quantized (1x128 per-tile) *before* dispatch and travel as FP8 + scales.

Routing weights are applied in ``combine_preprocess`` (the library combine is a pure sum) with fused Triton kernels.
Backward is the transpose: dispatch-bwd = ``combine(grad, handle, topk_weights=grad_probs)``, combine-bwd =
cached ``dispatch(grad, handle=handle)``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Literal

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from typing_extensions import override

from xtuner.v1.ops.comm.deepep_v2_op import (
    ElasticBufferRegistry,
    ElasticBufferSpec,
    capture_event,
    combine_forward,
    dispatch_cached,
    dispatch_forward,
    require_deepep_v2,
)
from xtuner.v1.ops.moe.cuda.triton_kernels import row_scale, row_scale_bwd
from xtuner.v1.utils import get_logger

from .base import (
    CombineResult,
    DispatcherCaps,
    DispatchResult,
    EPCall,
    ExpertWeights,
    GenericDispatcher,
    PostCombineResult,
    PostDispatchResult,
    PreCombineResult,
    PreDispatchResult,
    check_rows,
    rows_from_psum,
)


logger = get_logger()


class DeepEPV2Config(BaseModel):
    """Options of the DeepEP V2 dispatcher (``MoEConfig.dispatcher == "deepep_v2"``).

    Args:
        expert_alignment (Literal[1, 128]): Row alignment of the expanded layout. ``128`` lets the expert GEMMs run on
            DeepGEMM's psum kernels without a permute; ``1`` gives the unpadded layout for the legacy kernels.
        cpu_sync (bool): ``True`` returns exact received counts to the host (dynamic shapes); ``False`` keeps counts
            on the GPU and allocates the worst-case capacity (static shapes, CUDA-graph friendly).
        fp8_dispatch (bool): Quantize activations to FP8 (1x128 per-tile) before dispatch. Requires FP8 experts.
        max_tokens_per_rank (int | None): Upper bound of tokens one rank sends per dispatch, used to size the
            communication buffer; ``None`` sizes it collectively on the first call.
        deterministic (bool): Deterministic receive order.
        num_sms (int): SMs used by the DeepEP kernels; ``0`` lets the library decide.
        allow_hybrid_mode (bool): Hierarchical RDMA + NVLink mode for multi-node EP groups.
        zero_padding (bool): Zero-fill the alignment padding rows (required by the counts-based wgrad kernels).
        capacity_factor (float | None): Static mode only (``cpu_sync=False``): size the receive rows to this multiple
            of the balanced expectation (``max_tokens_per_rank x min(top-k, local experts)``) instead of the worst
            case (``x ep_size``). Overflow is detected one call later through a device flag and raises.
        check_rows (bool): Debug: verify the row-layout invariants with a host synchronization on every call.
    """

    model_config = ConfigDict(extra="forbid")

    expert_alignment: Literal[1, 128] = 128
    cpu_sync: bool = True
    fp8_dispatch: bool = False
    max_tokens_per_rank: int | None = None
    capacity_factor: float | None = None
    deterministic: bool = False
    num_sms: int = 0
    allow_hybrid_mode: bool = True
    zero_padding: bool = True
    check_rows: bool = False


class DeepEPV2PreDispatchResult(PreDispatchResult):
    topk_weights: Tensor
    forward_finished_event: Any
    backward_previous_event: Any


class DeepEPV2DispatchResult(DispatchResult):
    handle: Any
    forward_finished_event: Any


class DeepEPV2PreCombineResult(PreCombineResult):
    forward_finished_event: Any
    backward_previous_event: Any


class DeepEPV2CombineResult(CombineResult):
    forward_finished_event: Any
    backward_previous_event: Any


def _wait_hook(event_holder: Any) -> Callable[..., None]:
    def _hook(*_: Any) -> None:
        if event_holder.event is not None:
            event_holder.current_stream_wait()

    return _hook


def _record_hook(event_holder: Any) -> Callable[..., None]:
    def _hook(*_: Any) -> None:
        event_holder.event = capture_event().event

    return _hook


class _DispatchV2(torch.autograd.Function):
    """Forward: (quantize +) expanded dispatch. Backward: pure-sum combine carrying ``grad_probs``."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        dispatcher: "DeepEPV2Dispatcher",
        forward_previous_event: Any,
        backward_finished_event: Any,
    ) -> tuple[Tensor, Tensor, Any, Any]:
        is_async = forward_previous_event is not None
        buffer = dispatcher._get_buffer(hidden_states.shape[0])
        x: Tensor | tuple[Tensor, Tensor] = hidden_states
        if dispatcher.cfg.fp8_dispatch:
            from xtuner.v1.float8.triton_kernels import per_tile_quant

            x = per_tile_quant(hidden_states)
            if is_async:
                # The event from ``dispatch_preprocess`` predates the quantization kernels; the comm stream must
                # wait for them too, otherwise dispatch can read unquantized memory when compute is busy.
                forward_previous_event = capture_event()
        recv_x, _, recv_topk_weights, handle, event = dispatch_forward(
            buffer,
            x,
            topk_ids,
            topk_weights,
            num_experts=dispatcher.num_experts,
            expert_alignment=dispatcher.alignment,
            do_cpu_sync=dispatcher.cfg.cpu_sync,
            do_zero_padding=dispatcher.cfg.zero_padding,
            use_tma_aligned_col_major_sf=dispatcher.tma_aligned_sf,
            num_max_expanded_tokens=dispatcher._capacity,
            overflow_flag=dispatcher._overflow_flag,
            num_sms=dispatcher.cfg.num_sms,
            previous_event=forward_previous_event,
            async_finish=is_async,
        )
        if not is_async and event is not None and event.event is not None:
            event.current_stream_wait()
        ctx.dispatcher = dispatcher
        ctx.handle = handle
        ctx.is_async = is_async
        ctx.backward_finished_event = backward_finished_event
        if isinstance(recv_x, tuple):
            # Keep FP8 activations behind a bf16-typed wrapper: autograd casts incoming gradients to the *input*
            # dtype of the consumer, so a raw fp8 tensor on an autograd edge would silently quantize dX.
            from xtuner.v1.float8.config import ScalingGranularity
            from xtuner.v1.float8.float8_tensor import Float8Tensor

            recv_hidden = Float8Tensor(recv_x[0], recv_x[1], torch.bfloat16, ScalingGranularity.TILEWISE, 128)
        else:
            recv_hidden = recv_x
        return recv_hidden, recv_topk_weights, handle, event

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_recv_x: Tensor, grad_recv_topk_weights: Tensor, *_: Any
    ) -> tuple[Tensor, None, Tensor | None, None, None, None]:
        dispatcher: DeepEPV2Dispatcher = ctx.dispatcher
        buffer = dispatcher._buffer
        previous_event = capture_event()
        grad_x, grad_topk_weights, event = combine_forward(
            buffer,
            grad_recv_x.contiguous().to(torch.bfloat16),
            ctx.handle,
            topk_weights=grad_recv_topk_weights.contiguous().float() if grad_recv_topk_weights is not None else None,
            num_sms=dispatcher.cfg.num_sms,
            previous_event=previous_event,
            async_finish=ctx.is_async,
        )
        if ctx.is_async:
            ctx.backward_finished_event.event = event.event
        elif event is not None and event.event is not None:
            event.current_stream_wait()
        return grad_x, None, grad_topk_weights, None, None, None


class _CombineV2(torch.autograd.Function):
    """Forward: pure-sum combine. Backward: cached expanded dispatch of the incoming gradient."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        hidden_states: Tensor,
        handle: Any,
        dispatcher: "DeepEPV2Dispatcher",
        forward_previous_event: Any,
        backward_previous_event: Any,
        backward_finished_event: Any,
    ) -> tuple[Tensor, Any]:
        is_async = forward_previous_event is not None
        combined, _, event = combine_forward(
            dispatcher._buffer,
            hidden_states,
            handle,
            topk_weights=None,
            num_sms=dispatcher.cfg.num_sms,
            previous_event=forward_previous_event,
            async_finish=is_async,
        )
        if not is_async and event is not None and event.event is not None:
            event.current_stream_wait()
        ctx.dispatcher = dispatcher
        ctx.handle = handle
        ctx.is_async = is_async
        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event
        return combined, event

    @staticmethod
    def backward(ctx: Any, grad_combined: Tensor, *_: Any) -> tuple[Tensor, None, None, None, None, None]:  # type: ignore[override]
        dispatcher: DeepEPV2Dispatcher = ctx.dispatcher
        previous_event = ctx.backward_previous_event if ctx.is_async else capture_event()
        if previous_event is not None and getattr(previous_event, "event", None) is None:
            previous_event = capture_event()
        grad_x, event = dispatch_cached(
            dispatcher._buffer,
            grad_combined.contiguous().to(torch.bfloat16),
            ctx.handle,
            do_zero_padding=True,
            num_sms=dispatcher.cfg.num_sms,
            previous_event=previous_event,
            async_finish=ctx.is_async,
        )
        if ctx.is_async:
            ctx.backward_finished_event.event = event.event
        elif event is not None and event.event is not None:
            event.current_stream_wait()
        return grad_x, None, None, None, None, None


class _RowScale(torch.autograd.Function):
    """``y = x * w[:, None]`` with fp32 math and bf16 storage (fused Triton kernels, no fp32 temporaries)."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, w: Tensor) -> Tensor:  # type: ignore[override]
        w = w.float().contiguous()
        ctx.save_for_backward(x, w)
        return row_scale(x, w)

    @staticmethod
    def backward(ctx: Any, dy: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        x, w = ctx.saved_tensors
        dx, dw = row_scale_bwd(dy, x, w)
        return dx, dw


class DeepEPV2Dispatcher(
    GenericDispatcher[
        DeepEPV2PreDispatchResult,
        DeepEPV2DispatchResult,
        PostDispatchResult,
        DeepEPV2PreCombineResult,
        DeepEPV2CombineResult,
        PostCombineResult,
    ]
):
    """Six-stage dispatcher over DeepEP V2's ``ElasticBuffer``.

    Args:
        n_routed_experts (int): Global number of routed experts.
        process_group (dist.ProcessGroup): EP process group.
        hidden_size (int): Model hidden size (DeepEP V2 requires a multiple of 256).
        num_experts_per_tok (int): Router top-k.
        cfg (DeepEPV2Config): Dispatcher options.
        training_dtype (Literal["fp8", "bf16"]): Expert GEMM dtype; ``fp8_dispatch`` needs ``"fp8"``.
        generate_dtype (Literal["fp8", "bf16"]): Unused, kept for interface compatibility.
    """

    def __init__(
        self,
        *,
        n_routed_experts: int,
        process_group: dist.ProcessGroup,
        hidden_size: int,
        num_experts_per_tok: int,
        cfg: DeepEPV2Config,
        training_dtype: Literal["fp8", "bf16"] = "bf16",
        generate_dtype: Literal["fp8", "bf16"] = "bf16",
    ) -> None:
        require_deepep_v2()
        super().__init__(
            n_routed_experts=n_routed_experts,
            process_group=process_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )
        if torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory:
            raise RuntimeError(
                "DeepEP V2 cannot run with torch deterministic algorithms *and* "
                "`torch.utils.deterministic.fill_uninitialized_memory=True` (the fill kernel races with the "
                "communication stream). Set `torch.utils.deterministic.fill_uninitialized_memory = False`."
            )
        if n_routed_experts % process_group.size() != 0:
            raise ValueError(f"num_experts {n_routed_experts} must be divisible by ep_size {process_group.size()}")
        if hidden_size % 256 != 0:
            raise ValueError(f"DeepEP V2 combine requires hidden_size % 256 == 0, got {hidden_size}")
        if num_experts_per_tok > 32:
            raise ValueError("DeepEP V2 supports at most 32 top-k experts")
        if n_routed_experts // process_group.size() > 256:
            raise ValueError("DeepEP V2 supports at most 256 local experts per rank")
        if cfg.fp8_dispatch and training_dtype != "fp8":
            raise ValueError("fp8_dispatch requires FP8 expert GEMMs (Float8Config.scaling_granularity_grouped_gemm)")
        if cfg.capacity_factor is not None and (cfg.cpu_sync or cfg.capacity_factor <= 0):
            raise ValueError("capacity_factor applies to cpu_sync=False only and must be positive")
        if not cfg.cpu_sync and training_dtype == "fp8" and cfg.expert_alignment != 128:
            # The AdaptiveGEMM FP8 kernels derive M from the activation and need sum(counts) == M, so they cannot run
            # on the worst-case capacity of the static mode; only the DeepGEMM psum path (128-aligned) can.
            raise ValueError("cpu_sync=False with FP8 experts requires expert_alignment=128 (DeepGEMM path)")
        self.group = process_group
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.num_topk = num_experts_per_tok
        self.num_experts = n_routed_experts
        self.num_local_experts = n_routed_experts // process_group.size()
        self.alignment = cfg.expert_alignment
        # DeepGEMM (SM90) consumes MN-major TMA-aligned scale factors and the dispatcher can emit them directly; the
        # AdaptiveGEMM path (alignment 1) wants plain row-major scales.
        self.tma_aligned_sf = bool(cfg.fp8_dispatch) and cfg.expert_alignment == 128
        self.caps = DispatcherCaps(
            name="deepep_v2",
            produces_alignment=cfg.expert_alignment,
            host_counts=cfg.cpu_sync,
            host_sync_free=not cfg.cpu_sync,
            static_shape=not cfg.cpu_sync,
            combine_applies_probs=False,
            fp8_dispatch=cfg.fp8_dispatch,
            hidden_multiple=256,
        )
        self._buffer: Any = None
        self._max_tokens_per_rank = int(cfg.max_tokens_per_rank or 0)
        # Capped static capacity (see ``DeepEPV2Config.capacity_factor``): rows, device flag, pinned mirror + event
        # for a check that never stalls the host (the flag written by call i is read at call i+1).
        self._capacity: int | None = None
        self._overflow_flag: Tensor | None = None
        self._overflow_host: Tensor | None = None
        self._overflow_event: torch.cuda.Event | None = None

    def _get_buffer(self, num_tokens: int) -> Any:
        if self._buffer is not None:
            if num_tokens > self._max_tokens_per_rank:
                raise RuntimeError(
                    f"DeepEP V2 buffer was sized for {self._max_tokens_per_rank} tokens/rank but got {num_tokens}; "
                    "set DeepEPV2Config.max_tokens_per_rank to the pack length."
                )
            return self._buffer
        # One-time, collective sizing so every rank holds the same bound (a DeepEP requirement).
        bound = torch.tensor([max(self._max_tokens_per_rank, num_tokens)], device="cuda", dtype=torch.int64)
        dist.all_reduce(bound, op=dist.ReduceOp.MAX, group=self.group)
        self._max_tokens_per_rank = int(bound.item())
        if self.cfg.capacity_factor is not None:
            expected = self._max_tokens_per_rank * min(self.num_topk, self.num_local_experts)
            self._capacity = int(math.ceil(self.cfg.capacity_factor * expected))
            self._overflow_flag = torch.zeros(1, dtype=torch.int32, device="cuda")
            self._overflow_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
            logger.info(
                f"[DeepEP V2] static receive capacity capped at {self._capacity} rows "
                f"({self.cfg.capacity_factor}x of {expected}; worst case {expected * self.group.size()})"
            )
        self._buffer = ElasticBufferRegistry.get(
            self.group,
            ElasticBufferSpec(
                hidden=self.hidden_size,
                num_topk=self.num_topk,
                num_max_tokens_per_rank=self._max_tokens_per_rank,
                use_fp8_dispatch=self.cfg.fp8_dispatch,
                deterministic=self.cfg.deterministic,
                allow_hybrid_mode=self.cfg.allow_hybrid_mode,
            ),
        )
        return self._buffer

    def _check_overflow(self) -> None:
        # Read the flag of the previous call (its copy has long completed, so ``query`` is true without a stall),
        # then enqueue the copy for this call on the current stream, which already waited for the dispatch.
        if self._overflow_flag is None or self._overflow_host is None:
            return
        if self._overflow_event is not None and self._overflow_event.query() and int(self._overflow_host[0]) != 0:
            raise RuntimeError(
                f"DeepEP V2 receive capacity overflow ({self._capacity} rows): raise DeepEPV2Config.capacity_factor"
            )
        self._overflow_host.copy_(self._overflow_flag, non_blocking=True)
        self._overflow_event = torch.cuda.Event()
        self._overflow_event.record()

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        async_op: bool = False,
        tokens_per_expert: Tensor | None = None,  # noqa: ARG002 — contract seam, unused by this dispatcher
        layer_state: EPCall | None = None,  # noqa: ARG002
    ) -> DeepEPV2PreDispatchResult:
        import deep_ep

        # DeepEP exports its compile-time index dtype (``EP_NUM_TOPK_IDX_BITS``); default to int64.
        topk_ids = topk_ids.to(getattr(deep_ep, "topk_idx_t", torch.int64)).contiguous()
        topk_weights = topk_weights.float().contiguous()
        backward_previous_event = None
        if async_op:
            from deep_ep import EventOverlap

            backward_previous_event = EventOverlap(None)
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(_wait_hook(backward_previous_event))
            if topk_weights.grad_fn is not None:
                topk_weights.grad_fn.register_prehook(_wait_hook(backward_previous_event))
        forward_finished_event = capture_event() if async_op else None
        return DeepEPV2PreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            forward_finished_event=forward_finished_event,
            backward_previous_event=backward_previous_event,
        )

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: DeepEPV2PreDispatchResult,
        topk_weights: Tensor,  # noqa: ARG002 — already cast and stashed in pre_dispatched
        async_op: bool = False,
        decoding: bool = False,
    ) -> DeepEPV2DispatchResult:
        if decoding:
            raise NotImplementedError
        recv_x, recv_topk_weights, handle, event = _DispatchV2.apply(
            pre_dispatched["hidden_states"],
            pre_dispatched["topk_ids"],
            pre_dispatched["topk_weights"],
            self,
            pre_dispatched["forward_finished_event"],
            pre_dispatched["backward_previous_event"],
        )
        return DeepEPV2DispatchResult(
            hidden_states=recv_x,
            topk_weights=recv_topk_weights,
            handle=handle,
            forward_finished_event=event if async_op else None,
        )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: DeepEPV2PreDispatchResult,  # noqa: ARG002
        dispatched: DeepEPV2DispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> PostDispatchResult:
        if decoding:
            raise NotImplementedError
        if async_op and dispatched["forward_finished_event"] is not None:
            dispatched["forward_finished_event"].current_stream_wait()
        self._check_overflow()
        handle = dispatched["handle"]
        rows = rows_from_psum(
            handle.psum_num_recv_tokens_per_expert,
            handle.num_unaligned_recv_tokens_per_expert,
            alignment=self.alignment,
            padding_zeroed=self.cfg.zero_padding,
        )
        if self.cfg.check_rows:
            check_rows(rows, dispatched["hidden_states"].shape[0])
        # FP8 activations arrive as a bf16-typed ``Float8Tensor`` wrapper (data + scales inside), see ``_DispatchV2``.
        return PostDispatchResult(
            hidden_states=dispatched["hidden_states"],
            hidden_scales=None,
            tokens_per_expert=rows.compute_counts,
            rows=rows,
            expert_weights=ExpertWeights(),
        )

    @override
    def combine_preprocess(
        self,
        *,
        hidden_states: Tensor,
        pre_dispatched: DeepEPV2PreDispatchResult,  # noqa: ARG002
        dispatched: DeepEPV2DispatchResult,
        post_dispatched: PostDispatchResult,  # noqa: ARG002
        async_op: bool = False,
        decoding: bool = False,
    ) -> DeepEPV2PreCombineResult:
        if decoding:
            raise NotImplementedError
        # Routing weights are applied here (the library combine is a pure sum); grad_probs flows back 1D per row.
        weighted = _RowScale.apply(hidden_states, dispatched["topk_weights"])
        backward_previous_event = None
        forward_finished_event = None
        if async_op:
            from deep_ep import EventOverlap

            backward_previous_event = EventOverlap(None)
            forward_finished_event = capture_event()
            if weighted.grad_fn is not None:
                weighted.grad_fn.register_prehook(_wait_hook(backward_previous_event))
        return DeepEPV2PreCombineResult(
            hidden_states=weighted,
            forward_finished_event=forward_finished_event,
            backward_previous_event=backward_previous_event,
        )

    @override
    def combine(
        self,
        *,
        pre_dispatched: DeepEPV2PreDispatchResult,  # noqa: ARG002
        dispatched: DeepEPV2DispatchResult,
        post_dispatched: PostDispatchResult,  # noqa: ARG002
        pre_combined: DeepEPV2PreCombineResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> DeepEPV2CombineResult:
        if decoding:
            raise NotImplementedError
        backward_previous_event = None
        if async_op:
            from deep_ep import EventOverlap

            backward_previous_event = EventOverlap(None)
        combined, event = _CombineV2.apply(
            pre_combined["hidden_states"],
            dispatched["handle"],
            self,
            pre_combined["forward_finished_event"],
            backward_previous_event,
            pre_combined["backward_previous_event"],
        )
        return DeepEPV2CombineResult(
            hidden_states=combined,
            forward_finished_event=event if async_op else None,
            backward_previous_event=backward_previous_event,
        )

    @override
    def combine_postprocess(
        self,
        *,
        pre_dispatched: DeepEPV2PreDispatchResult,  # noqa: ARG002
        dispatched: DeepEPV2DispatchResult,  # noqa: ARG002
        post_dispatched: PostDispatchResult,  # noqa: ARG002
        pre_combined: DeepEPV2PreCombineResult,  # noqa: ARG002
        combined: DeepEPV2CombineResult,
        async_op: bool = False,
    ) -> PostCombineResult:
        hidden_states = combined["hidden_states"].view_as(combined["hidden_states"])
        if async_op:
            if hidden_states.grad_fn is not None and combined["backward_previous_event"] is not None:
                hidden_states.grad_fn.register_hook(_record_hook(combined["backward_previous_event"]))
            if combined["forward_finished_event"] is not None:
                combined["forward_finished_event"].current_stream_wait()
        return PostCombineResult(hidden_states=hidden_states)


__all__ = ["DeepEPV2Config", "DeepEPV2Dispatcher"]
