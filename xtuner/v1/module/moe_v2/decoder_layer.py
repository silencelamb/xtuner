# Copyright (c) OpenMMLab. All rights reserved.
"""MoE decoder layer driving the unified EP contract (v0).

The layer keeps attention, norms, router and shared experts from :class:`MoEDecoderLayer` and replaces the MoE
segment: a contract-speaking dispatcher produces an :class:`ExpertBatch`, :class:`GroupedExpertsV2` consumes it,
and a :class:`LayerEPExecution` runtime gets five unconditional hooks (identity by default). Both the single and
the intra-layer micro-batch paths call the same hooks in the same order.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.config.generate import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8 import Float8Config
from xtuner.v1.module import (
    AttnOutputs,
    GatedDeltaNetConfig,
    GreedyRouterConfig,
    MHAConfig,
    MLAConfig,
    NoAuxRouterConfig,
    RouterResults,
)
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEActFnConfig,
    MoEDecoderLayer,
    MoEDecoderLayerMicroBatchOutput,
    MoEDecoderLayerOutput,
)
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.utils import ForwardState, get_logger

from .config import MoEV2Config
from .contracts import EPCall, ExpertBatch, LayerEPExecution, MoESpec, NoOpLayerEPExecution, check_rows
from .experts import GroupedExpertsV2
from .legacy_adapter import LegacyDispatcherAdapter


logger = get_logger()


class MoEDecoderLayerV2(MoEDecoderLayer):
    """Contract-based MoE decoder layer.

    Args:
        moe_v2_cfg (MoEV2Config): Options of the V2 path (dispatcher backend, GEMM backend, alignment, ...).
        max_tokens_per_rank (int | None): Per-rank token upper bound used to size static communication buffers.
        **kwargs: Same as :class:`MoEDecoderLayer`; ``dispatcher`` is interpreted by the V2 path.
    """

    def __init__(
        self,
        *,
        moe_v2_cfg: MoEV2Config,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        mlp_bias: bool = False,
        gate_bias: bool = False,
        moe_bias: bool = False,
        hidden_act: str,
        rms_norm_eps: float = 1e-6,
        rms_norm_type: Literal["default", "zero_centered"] = "default",
        num_experts_per_tok: int,
        n_routed_experts: int,
        n_shared_experts: int,
        with_shared_expert_gate: bool = False,
        hidden_factor: float = 1.0,
        attention_config: MHAConfig | MLAConfig | GatedDeltaNetConfig,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        generate_config: GenerateConfig | None = None,
        router_config: GreedyRouterConfig | NoAuxRouterConfig,
        router_compute_dtype: Literal["float32", "native"] = "float32",
        moe_act_fn_cfg: MoEActFnConfig,
        float8_cfg: Float8Config | None = None,
        layer_idx: int = 0,
        dispatcher: str | None,
        ep_mesh: DeviceMesh | None = None,
        expert_tp_mesh: DeviceMesh | None = None,
        ep_tp_mesh: DeviceMesh | None = None,
    ) -> None:
        del dispatcher  # the V2 path reads the backend from ``moe_v2_cfg``
        legacy_name = moe_v2_cfg.legacy_dispatcher if moe_v2_cfg.dispatcher == "legacy" else None
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            moe_intermediate_size=moe_intermediate_size,
            mlp_bias=mlp_bias,
            gate_bias=gate_bias,
            moe_bias=moe_bias,
            hidden_act=hidden_act,
            rms_norm_eps=rms_norm_eps,
            rms_norm_type=rms_norm_type,
            num_experts_per_tok=num_experts_per_tok,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            with_shared_expert_gate=with_shared_expert_gate,
            hidden_factor=hidden_factor,
            attention_config=attention_config,
            rope_scaling_cfg=rope_scaling_cfg,
            layer_type=layer_type,
            generate_config=generate_config,
            router_config=router_config,
            router_compute_dtype=router_compute_dtype,
            moe_act_fn_cfg=moe_act_fn_cfg,
            float8_cfg=float8_cfg,
            layer_idx=layer_idx,
            dispatcher=legacy_name,  # type: ignore[arg-type]
            ep_mesh=ep_mesh,
            expert_tp_mesh=expert_tp_mesh,
            ep_tp_mesh=ep_tp_mesh,
        )
        self.moe_v2_cfg = moe_v2_cfg
        ep_group = ep_mesh.get_group() if ep_mesh is not None else None
        ep_size = ep_group.size() if ep_group is not None else 1
        self._ep_enabled = ep_size > 1

        # Expert compute reading the contract (same parameter names as ``MoEBlock``).
        self.experts = GroupedExpertsV2(
            gemm_backend=moe_v2_cfg.gemm_backend,
            expert_alignment=moe_v2_cfg.expert_alignment if moe_v2_cfg.dispatcher == "deepep_v2" else 1,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=n_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=ep_mesh,
            expert_tp_mesh=expert_tp_mesh,
            float8_cfg=float8_cfg,
            moe_act_fn_cfg=moe_act_fn_cfg,
            ep_tp_mesh=ep_tp_mesh,
        )

        # Transport axis.
        if moe_v2_cfg.dispatcher == "legacy":
            legacy_inner = self.dispatcher  # built by ``MoEDecoderLayer.__init__`` through the legacy factory
            self.dispatcher = LegacyDispatcherAdapter(legacy_inner, name=str(legacy_name))  # type: ignore[assignment]
        else:
            from .deepep_v2 import DeepEPV2Dispatcher

            assert ep_group is not None, "DeepEP V2 dispatcher requires ep_size > 1"
            spec = MoESpec(
                hidden=hidden_size,
                num_topk=num_experts_per_tok,
                num_experts=n_routed_experts,
                num_local_experts=n_routed_experts // ep_size,
                max_tokens_per_rank=moe_v2_cfg.max_tokens_per_rank or 0,
                dtype=torch.bfloat16,
                alignment=self.experts.gemm.caps.required_alignment
                if self.experts.gemm.caps.required_alignment > 1
                else moe_v2_cfg.expert_alignment,
                fp8_dispatch=moe_v2_cfg.fp8_dispatch,
            )
            self.dispatcher = DeepEPV2Dispatcher(  # type: ignore[assignment]
                spec=spec, group=ep_group, cfg=moe_v2_cfg, tma_aligned_sf=self.experts.gemm.caps.name == "deepgemm"
            )
        self.caps = self.dispatcher.caps
        if self.caps.static_shape and not self.experts.gemm.caps.static_shape_friendly:
            raise ValueError(
                f"GEMM backend {self.experts.gemm.caps.name!r} cannot consume the static (worst-case capacity) layout of "
                f"dispatcher {self.caps.name!r}; use cpu_sync=True or gemm_backend='deepgemm' / 'triton'."
            )
        # Execution axis (identity until a runtime binds this layer).
        self.ep_exec: LayerEPExecution = NoOpLayerEPExecution(layer_idx)
        # Token dimension is dynamic unless the dispatcher guarantees static output shapes.
        self._dynamic_tokens = not self.caps.static_shape
        self._check_rows = moe_v2_cfg.check_rows

    def bind_ep_execution(self, ep_exec: LayerEPExecution) -> None:
        """Attach a model-scoped runtime's per-layer hooks (UltraEP / MoonEP later)."""
        self.ep_exec = ep_exec

    # ------------------------------------------------------------------------------------------------------------
    # single forward
    # ------------------------------------------------------------------------------------------------------------
    def _forward(  # type: ignore[override]
        self,
        hidden_states: Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[Tensor, Tensor],
        attention_kwargs: dict[str, object] | None = None,
    ) -> MoEDecoderLayerOutput:
        [hidden_states], [call] = self.ep_exec.prepare_layer_inputs([hidden_states])
        residual, hidden_states, router_results, attn_outputs = self._pre_moe_forward(
            hidden_states=hidden_states,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
            state=ForwardState.TRAINING,
            attention_kwargs=attention_kwargs,
        )
        origin_shape = hidden_states.shape
        async_op = self._ep_enabled

        pre, topk_weights = self._pre_dispatch_stage(hidden_states, router_results, call, async_op=async_op)
        dispatched, batch = self._dispatch_stage(pre, topk_weights, call, async_op=async_op)
        out = self._experts_stage(batch, call)
        pre_combined = self.dispatcher.combine_preprocess(
            hidden_states=out, pre_dispatched=pre, dispatched=dispatched, layer_state=call, async_op=async_op
        )
        combined = self.dispatcher.combine(
            pre_dispatched=pre, dispatched=dispatched, pre_combined=pre_combined, layer_state=call, async_op=async_op
        )
        # Shared experts run on the compute stream while the combine is in flight.
        shared_experts_out = (
            self._shared_experts_forward(hidden_states=hidden_states) if self.n_shared_experts > 0 else None
        )
        y = self.dispatcher.combine_postprocess(
            pre_dispatched=pre,
            dispatched=dispatched,
            pre_combined=pre_combined,
            combined=combined,
            layer_state=call,
            async_op=async_op,
        )
        y = self.ep_exec.attach_after_combine(call, y).view(*origin_shape)
        hidden_states = self._post_moe_forward(
            combined_hidden_states=y, residual=residual, shared_experts_out=shared_experts_out
        )
        return self._build_output(
            hidden_states=hidden_states, router_results=router_results, attn_outputs=attn_outputs
        )

    # ------------------------------------------------------------------------------------------------------------
    # intra-layer micro-batch (domino) forward: same hooks, same stage order, async everywhere
    # ------------------------------------------------------------------------------------------------------------
    def _micro_batch_forward(  # type: ignore[override]
        self,
        hidden_states_list: list[Tensor],
        seq_ctx_list: list[SequenceContext],
        position_embeddings_list: list[tuple[Tensor, Tensor]],
        attention_kwargs_list: list[dict[str, object]] | None = None,
    ) -> MoEDecoderLayerMicroBatchOutput:
        # Same interleaving as the legacy decoder: all attentions are queued before the first dispatch (which may
        # block the host in cpu_sync mode), experts of MB i are queued before dispatch of MB i+1, and every combine
        # is queued before the first combine_postprocess, so comm of one micro-batch overlaps compute of the other.
        n = len(hidden_states_list)
        if attention_kwargs_list is None:
            attention_kwargs_list = [{} for _ in range(n)]
        hidden_states_list, calls = self.ep_exec.prepare_layer_inputs(hidden_states_list)

        residual_list: list[Tensor] = []
        router_results_list: list[RouterResults] = []
        attn_outputs_list: list[AttnOutputs] = []
        pre_moe_out_list: list[Tensor] = []
        pre_list: list[Any] = []
        topk_weights_list: list[Tensor] = []

        # Stage 1: attention + router + dispatch_preprocess for every micro-batch.
        for i in range(n):
            residual, hs, router_results, attn_outputs = self._pre_moe_forward(
                hidden_states=hidden_states_list[i],
                seq_ctx=seq_ctx_list[i],
                position_embeddings=position_embeddings_list[i],
                state=ForwardState.TRAINING,
                attention_kwargs=attention_kwargs_list[i],
            )
            pre, topk_weights = self._pre_dispatch_stage(hs, router_results, calls[i], async_op=True)
            residual_list.append(residual)
            router_results_list.append(router_results)
            attn_outputs_list.append(attn_outputs)
            pre_moe_out_list.append(hs)
            pre_list.append(pre)
            topk_weights_list.append(topk_weights)

        # Stage 2: dispatch + experts + combine_preprocess; dispatch of MB i+1 overlaps experts of MB i.
        dispatched_list: list[Any] = []
        pre_combined_list: list[Any] = []
        for i in range(n):
            dispatched, batch = self._dispatch_stage(pre_list[i], topk_weights_list[i], calls[i], async_op=True)
            out = self._experts_stage(batch, calls[i])
            pre_combined = self.dispatcher.combine_preprocess(
                hidden_states=out,
                pre_dispatched=pre_list[i],
                dispatched=dispatched,
                layer_state=calls[i],
                async_op=True,
            )
            dispatched_list.append(dispatched)
            pre_combined_list.append(pre_combined)

        # Stage 3: combine launches; combine of MB i overlaps experts of MB i+1 / the shared experts.
        combined_list: list[Any] = [
            self.dispatcher.combine(
                pre_dispatched=pre_list[i],
                dispatched=dispatched_list[i],
                pre_combined=pre_combined_list[i],
                layer_state=calls[i],
                async_op=True,
            )
            for i in range(n)
        ]

        shared_out_list: list[Tensor | None] = [
            self._shared_experts_forward(hidden_states=hs) if self.n_shared_experts > 0 else None
            for hs in pre_moe_out_list
        ]

        # Stage 4: collect combines, residual add.
        hidden_out_list: list[Tensor] = []
        for i in range(n):
            y = self.dispatcher.combine_postprocess(
                pre_dispatched=pre_list[i],
                dispatched=dispatched_list[i],
                pre_combined=pre_combined_list[i],
                combined=combined_list[i],
                layer_state=calls[i],
                async_op=True,
            )
            y = self.ep_exec.attach_after_combine(calls[i], y).view(*pre_moe_out_list[i].shape)
            hidden_out_list.append(
                self._post_moe_forward(
                    combined_hidden_states=y, residual=residual_list[i], shared_experts_out=shared_out_list[i]
                )
            )
        return self._build_micro_batch_output(
            hidden_states_list=hidden_out_list,
            router_results_list=router_results_list,
            attn_outputs_list=attn_outputs_list,
        )

    # ------------------------------------------------------------------------------------------------------------
    # stages shared by both paths
    # ------------------------------------------------------------------------------------------------------------
    def _pre_dispatch_stage(
        self, hidden_states: Tensor, router_results: RouterResults, call: EPCall, *, async_op: bool
    ) -> tuple[Any, Tensor]:
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        flat, topk_ids, topk_weights = self.ep_exec.prepare_dispatch(
            call, flat, router_results["topk_ids"], router_results["topk_weights"]
        )
        pre = self.dispatcher.dispatch_preprocess(
            hidden_states=flat,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            tokens_per_expert=router_results["topkens_per_expert"],
            layer_state=call,
            async_op=async_op,
        )
        return pre, topk_weights

    def _dispatch_stage(
        self, pre: Any, topk_weights: Tensor, call: EPCall, *, async_op: bool
    ) -> tuple[Any, ExpertBatch]:
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre, topk_weights=topk_weights, layer_state=call, async_op=async_op
        )
        batch = self.dispatcher.dispatch_postprocess(
            pre_dispatched=pre, dispatched=dispatched, layer_state=call, async_op=async_op
        )
        batch = self.ep_exec.prepare_experts(call, batch)
        if self._check_rows:
            check_rows(batch["rows"], batch["hidden_states"].shape[0])
        return dispatched, batch

    def _experts_stage(self, batch: ExpertBatch, call: EPCall) -> Tensor:
        if self._dynamic_tokens and self._ep_enabled:
            # Only the routed-token dimension varies across layers / micro-batches; keep one compiled graph.
            torch._dynamo.mark_dynamic(batch["hidden_states"], 0)
            if batch["hidden_scales"] is not None:
                torch._dynamo.mark_dynamic(batch["hidden_scales"], 0)
        out = self.experts(batch)
        return self.ep_exec.attach_after_experts(call, out)


_V2_CLASSES: dict[type, type] = {}


def make_v2_decoder_cls(base_cls: type[MoEDecoderLayer]) -> type[MoEDecoderLayerV2]:
    """Compose the V2 MoE stages with a model-specific decoder class.

    Model families override ``forward`` / ``_build_output`` (e.g. GLM-5.2 threads DSA top-k ids through
    ``attention_kwargs``) but keep ``_forward`` / ``_micro_batch_forward``; the V2 mixin replaces exactly those two,
    so ``type(name, (MoEDecoderLayerV2, base_cls), {})`` gives the model's wrapper with the contract-based MoE segment.

    Args:
        base_cls (type[MoEDecoderLayer]): The model's decoder layer class.

    Returns:
        type[MoEDecoderLayerV2]: ``MoEDecoderLayerV2`` itself for the generic layer, otherwise a cached subclass.
    """
    if base_cls is MoEDecoderLayer or issubclass(base_cls, MoEDecoderLayerV2):
        return MoEDecoderLayerV2
    if base_cls not in _V2_CLASSES:
        _V2_CLASSES[base_cls] = type(f"{base_cls.__name__}V2", (MoEDecoderLayerV2, base_cls), {})
    return _V2_CLASSES[base_cls]


__all__ = ["MoEDecoderLayerV2", "make_v2_decoder_cls"]
