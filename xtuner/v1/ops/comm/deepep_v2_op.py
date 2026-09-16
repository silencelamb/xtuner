# Copyright (c) OpenMMLab. All rights reserved.
"""Thin functional wrapper over DeepEP V2 (``deep_ep.ElasticBuffer``).

Kept separate from the legacy ``deepep_op.py`` (DeepEP v1 ``Buffer``). One ``ElasticBuffer`` is created per
(process group, spec) and grown when a larger per-rank token bound shows up. All dispatch / combine calls go
through the four transposed primitives below so the dispatcher's autograd Functions stay small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from xtuner.v1.utils import get_logger


logger = get_logger()

try:
    from deep_ep import ElasticBuffer, EPHandle, EventOverlap  # DeepEP V2

    DEEPEP_V2_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - depends on the installed DeepEP
    ElasticBuffer = EPHandle = EventOverlap = None  # type: ignore[assignment,misc]
    DEEPEP_V2_AVAILABLE = False
    _IMPORT_ERROR: BaseException | None = exc
else:
    _IMPORT_ERROR = None


def require_deepep_v2() -> None:
    if not DEEPEP_V2_AVAILABLE:
        raise ImportError(
            "DeepEP V2 (`deep_ep.ElasticBuffer`) is not available. Install DeepEP >= 2.0 built against NCCL >= 2.30.4."
        ) from _IMPORT_ERROR


@dataclass(frozen=True)
class ElasticBufferSpec:
    hidden: int
    num_topk: int
    num_max_tokens_per_rank: int
    use_fp8_dispatch: bool
    deterministic: bool
    allow_hybrid_mode: bool


class ElasticBufferRegistry:
    """Process-wide ``ElasticBuffer`` cache keyed by process group; grows the buffer when a larger spec arrives."""

    _buffers: dict[int, tuple[ElasticBufferSpec, Any]] = {}

    @classmethod
    def get(cls, group: dist.ProcessGroup, spec: ElasticBufferSpec) -> Any:
        require_deepep_v2()
        key = id(group)
        cached = cls._buffers.get(key)
        if cached is not None:
            old_spec, buffer = cached
            if (
                old_spec.num_max_tokens_per_rank >= spec.num_max_tokens_per_rank
                and old_spec.hidden >= spec.hidden
                and old_spec.num_topk >= spec.num_topk
                and old_spec.use_fp8_dispatch == spec.use_fp8_dispatch
                and old_spec.deterministic == spec.deterministic
            ):
                return buffer
            logger.info(f"[DeepEP V2] growing ElasticBuffer {old_spec} -> {spec}")
            buffer.destroy()
            cls._buffers.pop(key)
        # DeepEP reuses PyTorch's NCCL communicator; make sure the (sub)group is initialized before it asks for it.
        _warm_up_group(group)
        buffer = ElasticBuffer(
            group,
            num_max_tokens_per_rank=spec.num_max_tokens_per_rank,
            hidden=spec.hidden,
            num_topk=spec.num_topk,
            # Size for BF16 even when the forward dispatch is FP8: combine's backward is a cached dispatch of BF16
            # gradients through the same buffer, and the BF16 token layout is the larger of the two.
            use_fp8_dispatch=False,
            deterministic=spec.deterministic,
            allow_hybrid_mode=spec.allow_hybrid_mode,
            allow_multiple_reduction=True,  # required to carry ``topk_weights`` gradients through combine
            explicitly_destroy=True,
        )
        cls._buffers[key] = (spec, buffer)
        if dist.get_rank() == 0:
            logger.info(f"[DeepEP V2] ElasticBuffer created: {spec}, ranks={group.size()}")
        return buffer

    @classmethod
    def destroy_all(cls) -> None:
        for _, buffer in cls._buffers.values():
            buffer.destroy()
        cls._buffers.clear()


def _warm_up_group(group: dist.ProcessGroup) -> None:
    t = torch.ones(1, device="cuda")
    dist.all_reduce(t, group=group)
    torch.cuda.synchronize()


def capture_event() -> Any:
    """Capture an event on the current stream, wrapped in ``EventOverlap`` (waitable from Python)."""
    return EventOverlap(ElasticBuffer.capture())


def _raw_event(event: Any) -> Any:
    """The C++ runtime takes the bare ``EventHandle``; Python-side code carries the ``EventOverlap`` wrapper."""
    if event is None:
        return None
    return event.event if hasattr(event, "event") else event


def dispatch_forward(
    buffer: Any,
    x: Tensor | tuple[Tensor, Tensor],
    topk_idx: Tensor,
    topk_weights: Tensor,
    *,
    num_experts: int,
    expert_alignment: int,
    do_cpu_sync: bool,
    do_zero_padding: bool,
    use_tma_aligned_col_major_sf: bool,
    num_sms: int,
    previous_event: Any,
    async_finish: bool,
) -> tuple[Tensor | tuple[Tensor, Tensor], Tensor, Tensor, Any, Any]:
    """Expanded-layout dispatch: one output row per (token, local expert) slot, segments aligned to ``expert_alignment``."""
    return buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        expert_alignment=expert_alignment,
        num_sms=num_sms,
        previous_event=_raw_event(previous_event),
        async_with_compute_stream=async_finish,
        allocate_on_comm_stream=previous_event is not None,
        do_cpu_sync=do_cpu_sync,
        do_expand=True,
        do_zero_padding=do_zero_padding,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
    )


def dispatch_cached(
    buffer: Any,
    x: Tensor,
    handle: Any,
    *,
    do_zero_padding: bool,
    num_sms: int,
    previous_event: Any,
    async_finish: bool,
) -> tuple[Tensor, Any]:
    """Replay a dispatch with a cached handle (used as the backward of combine). No notify, no CPU sync."""
    recv_x, _, _, _, event = buffer.dispatch(
        x,
        handle=handle,
        num_experts=handle.num_experts,  # DeepEP derives the SM count before unpacking the handle
        expert_alignment=handle.expert_alignment,
        num_max_tokens_per_rank=handle.num_max_tokens_per_rank,
        num_sms=num_sms,
        previous_event=_raw_event(previous_event),
        async_with_compute_stream=async_finish,
        allocate_on_comm_stream=previous_event is not None,
        do_cpu_sync=None,
        do_expand=True,
        do_zero_padding=do_zero_padding,
    )
    return recv_x, event


def combine_forward(
    buffer: Any,
    x: Tensor,
    handle: Any,
    *,
    topk_weights: Tensor | None,
    num_sms: int,
    previous_event: Any,
    async_finish: bool,
) -> tuple[Tensor, Tensor | None, Any]:
    """Pure reduction combine. ``topk_weights`` (1D per expanded row) is only carried back, never applied."""
    return buffer.combine(
        x,
        handle,
        topk_weights=topk_weights,
        num_sms=num_sms,
        previous_event=_raw_event(previous_event),
        async_with_compute_stream=async_finish,
        allocate_on_comm_stream=previous_event is not None,
    )


__all__ = [
    "DEEPEP_V2_AVAILABLE",
    "ElasticBufferRegistry",
    "ElasticBufferSpec",
    "capture_event",
    "combine_forward",
    "dispatch_cached",
    "dispatch_forward",
    "require_deepep_v2",
]
