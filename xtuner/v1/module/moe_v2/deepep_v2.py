# Copyright (c) OpenMMLab. All rights reserved.
"""DeepEP V2 (``ElasticBuffer``) dispatcher speaking the unified EP contract.

Layout: expanded (``do_expand=True``) with ``expert_alignment`` — one row per (token, local expert) slot, segments
aligned to the GEMM's alignment, no framework-side permute. ``dispatch_postprocess`` is zero-copy: the received
tensor *is* ``ExpertBatch.hidden_states`` and the row layout comes straight from the ``EPHandle``.

Modes:

* ``cpu_sync=True``: exact received counts on the host (dynamic shapes, legacy-like).
* ``cpu_sync=False``: counts stay on the GPU, output capacity is the worst case (static shapes; the whole segment
  is CUDA-graph capturable when combined with ``EP_AVOID_RECORD_STREAM=1``).
* ``fp8_dispatch=True``: activations are quantized (1x128 per-tile) *before* dispatch and travel as FP8 + scales.

Backward is the transpose: dispatch-bwd = ``combine(grad, handle, topk_weights=grad_probs)``, combine-bwd =
cached ``dispatch(grad, handle=handle)``.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

import torch
import torch.distributed as dist
from torch import Tensor

from xtuner.v1.ops.comm.deepep_v2_op import (
    ElasticBufferRegistry,
    ElasticBufferSpec,
    capture_event,
    combine_forward,
    dispatch_cached,
    dispatch_forward,
    require_deepep_v2,
)
from xtuner.v1.utils import get_logger

from .config import MoEV2Config
from .contracts import DispatcherCaps, EPCall, ExpertBatch, ExpertWeights, MoESpec, rows_from_psum
from .row_scale import row_scale, row_scale_bwd


logger = get_logger()


class DeepEPV2PreDispatchResult(TypedDict):
    hidden_states: Tensor
    topk_ids: Tensor
    topk_weights: Tensor
    forward_finished_event: Any
    backward_previous_event: Any


class DeepEPV2DispatchResult(TypedDict):
    hidden_states: Tensor
    hidden_scales: Tensor | None
    topk_weights: Tensor  # 1D, one weight per expanded row (zero on padding rows)
    handle: Any
    forward_finished_event: Any


class DeepEPV2PreCombineResult(TypedDict):
    hidden_states: Tensor
    forward_finished_event: Any
    backward_previous_event: Any


class DeepEPV2CombineResult(TypedDict):
    hidden_states: Tensor
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
            num_experts=dispatcher.spec.num_experts,
            expert_alignment=dispatcher.spec.alignment,
            do_cpu_sync=dispatcher.cfg.cpu_sync,
            do_zero_padding=dispatcher.cfg.zero_padding,
            use_tma_aligned_col_major_sf=dispatcher.tma_aligned_sf,
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


class DeepEPV2Dispatcher:
    """Six-stage dispatcher over DeepEP V2's ``ElasticBuffer``.

    Args:
        spec (MoESpec): Layer family description (hidden, top-k, expert counts, alignment, per-rank token bound).
        group (dist.ProcessGroup): EP process group.
        cfg (MoEV2Config): Runtime options (cpu_sync, fp8_dispatch, deterministic, num_sms, zero_padding).
    """

    def __init__(
        self, *, spec: MoESpec, group: dist.ProcessGroup, cfg: MoEV2Config, tma_aligned_sf: bool = False
    ) -> None:
        require_deepep_v2()
        if torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory:
            raise RuntimeError(
                "DeepEP V2 cannot run with torch deterministic algorithms *and* "
                "`torch.utils.deterministic.fill_uninitialized_memory=True` (the fill kernel races with the "
                "communication stream). Set `torch.utils.deterministic.fill_uninitialized_memory = False`."
            )
        if spec.num_experts % group.size() != 0:
            raise ValueError(f"num_experts {spec.num_experts} must be divisible by ep_size {group.size()}")
        self.spec = spec
        self.group = group
        self.cfg = cfg
        self.num_local_experts = spec.num_experts // group.size()
        # DeepGEMM (SM90) consumes MN-major TMA-aligned scale factors and the dispatcher can emit them directly;
        # AdaptiveGEMM wants plain row-major fp32 scales. The decoder picks this from the selected GEMM backend.
        self.tma_aligned_sf = bool(spec.fp8_dispatch) and tma_aligned_sf
        self.caps = DispatcherCaps(
            name="deepep_v2",
            produces_alignment=spec.alignment,
            phys_expert_space=True,
            host_counts=cfg.cpu_sync,
            host_sync_free=not cfg.cpu_sync,
            static_shape=not cfg.cpu_sync,
            combine_applies_probs=False,
            fp8_dispatch=cfg.fp8_dispatch,
            hidden_multiple=256,
        )
        self._buffer: Any = None
        self._max_tokens_per_rank = int(spec.max_tokens_per_rank or 0)

    # ------------------------------------------------------------------------------------------------------------
    def _get_buffer(self, num_tokens: int) -> Any:
        if self._buffer is not None:
            if num_tokens > self._max_tokens_per_rank:
                raise RuntimeError(
                    f"DeepEP V2 buffer was sized for {self._max_tokens_per_rank} tokens/rank but got {num_tokens}; "
                    "set MoEV2Config.max_tokens_per_rank to the pack length."
                )
            return self._buffer
        # One-time, collective sizing so every rank holds the same bound (a DeepEP requirement).
        bound = torch.tensor([max(self._max_tokens_per_rank, num_tokens)], device="cuda", dtype=torch.int64)
        dist.all_reduce(bound, op=dist.ReduceOp.MAX, group=self.group)
        self._max_tokens_per_rank = int(bound.item())
        self._buffer = ElasticBufferRegistry.get(
            self.group,
            ElasticBufferSpec(
                hidden=self.spec.hidden,
                num_topk=self.spec.num_topk,
                num_max_tokens_per_rank=self._max_tokens_per_rank,
                use_fp8_dispatch=self.cfg.fp8_dispatch,
                deterministic=self.cfg.deterministic,
                allow_hybrid_mode=self.cfg.allow_hybrid_mode,
            ),
        )
        return self._buffer

    # ------------------------------------------------------------------------------------------------------------
    def dispatch_preprocess(
        self,
        *,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        tokens_per_expert: Tensor,
        layer_state: EPCall,
        async_op: bool = False,
    ) -> DeepEPV2PreDispatchResult:
        del tokens_per_expert, layer_state
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

    def dispatch(
        self,
        *,
        pre_dispatched: DeepEPV2PreDispatchResult,
        topk_weights: Tensor,
        layer_state: EPCall,
        async_op: bool = False,
    ) -> DeepEPV2DispatchResult:
        del topk_weights  # already cast and stashed in ``pre_dispatched``
        recv_x, recv_topk_weights, handle, event = _DispatchV2.apply(
            pre_dispatched["hidden_states"],
            pre_dispatched["topk_ids"],
            pre_dispatched["topk_weights"],
            self,
            pre_dispatched["forward_finished_event"],
            pre_dispatched["backward_previous_event"],
        )
        layer_state.plan = handle
        return DeepEPV2DispatchResult(
            hidden_states=recv_x,
            hidden_scales=None,
            topk_weights=recv_topk_weights,
            handle=handle,
            forward_finished_event=event if async_op else None,
        )

    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: DeepEPV2PreDispatchResult,
        dispatched: DeepEPV2DispatchResult,
        layer_state: EPCall,
        async_op: bool = False,
    ) -> ExpertBatch:
        del pre_dispatched, layer_state
        if async_op and dispatched["forward_finished_event"] is not None:
            dispatched["forward_finished_event"].current_stream_wait()
        handle = dispatched["handle"]
        rows = rows_from_psum(
            handle.psum_num_recv_tokens_per_expert,
            handle.num_unaligned_recv_tokens_per_expert,
            alignment=self.spec.alignment,
            padding_zeroed=self.cfg.zero_padding,
        )
        return ExpertBatch(
            hidden_states=dispatched["hidden_states"],
            hidden_scales=dispatched["hidden_scales"],
            tokens_per_expert=rows.compute_counts,
            rows=rows,
            probs=dispatched["topk_weights"],
            expert_weights=ExpertWeights(),
            tokens_per_expert_cpu=None,
        )

    def combine_preprocess(
        self,
        *,
        hidden_states: Tensor,
        pre_dispatched: DeepEPV2PreDispatchResult,
        dispatched: DeepEPV2DispatchResult,
        layer_state: EPCall,
        async_op: bool = False,
    ) -> DeepEPV2PreCombineResult:
        del pre_dispatched, layer_state
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

    def combine(
        self,
        *,
        pre_dispatched: DeepEPV2PreDispatchResult,
        dispatched: DeepEPV2DispatchResult,
        pre_combined: DeepEPV2PreCombineResult,
        layer_state: EPCall,
        async_op: bool = False,
    ) -> DeepEPV2CombineResult:
        del pre_dispatched, layer_state
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

    def combine_postprocess(
        self,
        *,
        pre_dispatched: DeepEPV2PreDispatchResult,
        dispatched: DeepEPV2DispatchResult,
        pre_combined: DeepEPV2PreCombineResult,
        combined: DeepEPV2CombineResult,
        layer_state: EPCall,
        async_op: bool = False,
    ) -> Tensor:
        del pre_dispatched, dispatched, pre_combined
        layer_state.plan = None
        hidden_states = combined["hidden_states"].view_as(combined["hidden_states"])
        if async_op:
            if hidden_states.grad_fn is not None and combined["backward_previous_event"] is not None:
                hidden_states.grad_fn.register_hook(_record_hook(combined["backward_previous_event"]))
            if combined["forward_finished_event"] is not None:
                combined["forward_finished_event"].current_stream_wait()
        return hidden_states


__all__ = ["DeepEPV2Dispatcher"]
