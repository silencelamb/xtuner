# Copyright (c) OpenMMLab. All rights reserved.
"""Adapter exposing a legacy six-stage dispatcher through the V2 contract.

Legacy dispatchers (``all2all`` / ``deepep`` v1 / naive) produce ``PostDispatchResult{hidden_states,
tokens_per_expert}``. The adapter turns that into an ``E0`` ``ExpertBatch`` and stores the legacy result in
``EPCall.extras`` for the combine stages. It exists so that the contract-based decoder can be validated bit-exactly
against the legacy decoder before any new communication backend is introduced.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from xtuner.v1.module.dispatcher.base import GenericDispatcher

from .contracts import DispatcherCaps, EPCall, ExpertBatch, ExpertWeights, rows_from_counts


_LEGACY_POST = "legacy_post_dispatched"


class LegacyDispatcherAdapter:
    """Wrap a :class:`GenericDispatcher` so it speaks the V2 stage protocol.

    Args:
        inner (GenericDispatcher): The legacy dispatcher instance.
        name (str): Backend name used in capability declarations.
    """

    def __init__(self, inner: GenericDispatcher, name: str) -> None:
        self.inner = inner
        self.caps = DispatcherCaps(
            name=name,
            produces_alignment=1,
            host_counts=False,
            host_sync_free=False,
            static_shape=False,
            combine_applies_probs=False,
        )

    def dispatch_preprocess(
        self,
        *,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        tokens_per_expert: Tensor,
        layer_state: EPCall,
        async_op: bool = False,
    ) -> Any:
        del tokens_per_expert, layer_state
        return self.inner.dispatch_preprocess(
            hidden_states=hidden_states, topk_ids=topk_ids, topk_weights=topk_weights, async_op=async_op
        )

    def dispatch(
        self, *, pre_dispatched: Any, topk_weights: Tensor, layer_state: EPCall, async_op: bool = False
    ) -> Any:
        del layer_state
        return self.inner.dispatch(pre_dispatched=pre_dispatched, topk_weights=topk_weights, async_op=async_op)

    def dispatch_postprocess(
        self, *, pre_dispatched: Any, dispatched: Any, layer_state: EPCall, async_op: bool = False
    ) -> ExpertBatch:
        post = self.inner.dispatch_postprocess(pre_dispatched=pre_dispatched, dispatched=dispatched, async_op=async_op)
        layer_state.extras[_LEGACY_POST] = post
        counts = post["tokens_per_expert"]
        return ExpertBatch(
            hidden_states=post["hidden_states"],
            hidden_scales=None,
            tokens_per_expert=counts,
            rows=rows_from_counts(counts),
            probs=None,
            expert_weights=ExpertWeights(),
            tokens_per_expert_cpu=None,
        )

    def combine_preprocess(
        self,
        *,
        hidden_states: Tensor,
        pre_dispatched: Any,
        dispatched: Any,
        layer_state: EPCall,
        async_op: bool = False,
    ) -> Any:
        return self.inner.combine_preprocess(
            hidden_states=hidden_states,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=layer_state.extras[_LEGACY_POST],
            async_op=async_op,
        )

    def combine(
        self, *, pre_dispatched: Any, dispatched: Any, pre_combined: Any, layer_state: EPCall, async_op: bool = False
    ) -> Any:
        return self.inner.combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=layer_state.extras[_LEGACY_POST],
            pre_combined=pre_combined,
            async_op=async_op,
        )

    def combine_postprocess(
        self,
        *,
        pre_dispatched: Any,
        dispatched: Any,
        pre_combined: Any,
        combined: Any,
        layer_state: EPCall,
        async_op: bool = False,
    ) -> Tensor:
        post = self.inner.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=layer_state.extras[_LEGACY_POST],
            pre_combined=pre_combined,
            combined=combined,
            async_op=async_op,
        )
        layer_state.extras.pop(_LEGACY_POST, None)
        return post["hidden_states"]


def build_legacy_adapter(
    *,
    dispatcher: str,
    n_routed_experts: int,
    ep_group: torch.distributed.ProcessGroup | None,
    tp_group: torch.distributed.ProcessGroup | None,
    ep_tp_group: torch.distributed.ProcessGroup | None,
    float8: bool,
) -> LegacyDispatcherAdapter:
    """Build a legacy dispatcher through the unchanged legacy factory and wrap it."""
    from xtuner.v1.module.dispatcher import build_dispatcher

    inner = build_dispatcher(
        dispatcher=dispatcher,  # type: ignore[arg-type]
        n_routed_experts=n_routed_experts,
        ep_group=ep_group,
        tp_group=tp_group,
        ep_tp_group=ep_tp_group,
        training_dtype="fp8" if float8 else "bf16",
    )
    return LegacyDispatcherAdapter(inner, name=dispatcher)


__all__ = ["LegacyDispatcherAdapter", "build_legacy_adapter"]
