# Copyright (c) OpenMMLab. All rights reserved.
"""Contract-consuming expert block: ``ExpertBatch`` in, ``[capacity, H]`` out."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from xtuner.v1.float8 import Float8Config
from xtuner.v1.float8.distributed_utils import tensor_already_casted_to_fp8
from xtuner.v1.float8.float8_gmm_tile_wise import (
    TileWiseFloat8GroupedLinear,
    slice_weight,
    view_weight,
    weight_to_per_block_float8_dynamic,
)
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock

from .contracts import ExpertBatch, ExpertWeights
from .gemm_backends import GemmBackend, build_gemm_backend


class GroupedExpertsV2(MoEBlock):
    """Expert block of the unified-contract MoE path.

    It keeps ``MoEBlock``'s parameters and names (``fused_w1w3`` / ``fused_w2`` / ``moe_act``) so FSDP sharding,
    FP8 weight casting, HF load/save and the EP gradient scaling in the model stay untouched, and only replaces
    the compute: the two grouped GEMMs read the row layout from ``ExpertBatch.rows`` through a pluggable GEMM
    backend selected once at construction.

    Args:
        gemm_backend (str): Backend name, see :func:`build_gemm_backend`.
        expert_alignment (int): Row alignment the dispatcher produces.
        **moe_block_kwargs: Forwarded to :class:`MoEBlock`.
    """

    def __init__(
        self,
        *,
        gemm_backend: str = "auto",
        expert_alignment: int = 1,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        moe_bias: bool = False,
        ep_mesh: DeviceMesh | None = None,
        expert_tp_mesh: DeviceMesh | None = None,
        float8_cfg: Float8Config | None = None,
        moe_act_fn_cfg: MoEActFnConfig,
        ep_tp_mesh: DeviceMesh | None = None,
    ) -> None:
        if moe_bias:
            raise NotImplementedError("GroupedExpertsV2 does not support routed-expert bias yet")
        if expert_tp_mesh is not None and expert_tp_mesh.size() > 1:
            raise NotImplementedError("GroupedExpertsV2 does not support expert TP yet")
        super().__init__(
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
        self.fp8 = isinstance(self.fused_w1w3, TileWiseFloat8GroupedLinear)
        self.expert_alignment = expert_alignment
        self.gemm: GemmBackend = build_gemm_backend(gemm_backend, fp8=self.fp8, alignment=expert_alignment)
        self.local_num_experts = self.fused_w1w3.local_num_routed_experts

    # The compiled entry point (registered by the V2 decoder's ``compile_cfg``).
    def forward(self, batch: ExpertBatch) -> Tensor:  # type: ignore[override]
        w13, w2 = self._resolve_weights(batch["expert_weights"])
        rows = batch["rows"]
        h = self.gemm.gemm(batch["hidden_states"], batch["hidden_scales"], w13, rows, trans_b=True)
        a = self.moe_act(h, split_dim=-1)
        return self.gemm.gemm(a, None, w2, rows, trans_b=True)

    def _resolve_weights(self, weights: ExpertWeights) -> tuple[Any, Any]:
        if weights.segments is not None:
            if len(weights.segments) != 1:
                raise NotImplementedError("multi-segment expert weights are not supported yet")
            seg = weights.segments[0]
            return seg.w1w3.value, seg.w2.value
        if self.fp8:
            return _fp8_weight(self.fused_w1w3), _fp8_weight(self.fused_w2)
        return _bf16_weight(self.fused_w1w3), _bf16_weight(self.fused_w2)


def _bf16_weight(linear: Any) -> Tensor:
    weight = linear.weight.to_local() if isinstance(linear.weight, DTensor) else linear.weight
    return weight.view(-1, linear.local_out_features, linear.local_in_features)


def _fp8_weight(linear: TileWiseFloat8GroupedLinear) -> Any:
    """Mirror ``TileWiseFloat8GroupedLinear.forward``'s weight resolution without its GEMM."""
    weight = linear.weight.to_local() if isinstance(linear.weight, DTensor) else linear.weight
    linear._check_shape(weight)
    if tensor_already_casted_to_fp8(weight):
        weight_fp8 = slice_weight.apply(weight, linear.ori_local_shape) if linear.is_padded else weight
        return view_weight.apply(weight_fp8, linear.ori_local_shape)
    weight = weight.view(*linear.ori_local_shape)
    return weight_to_per_block_float8_dynamic.apply(weight, torch.float8_e4m3fn, 128)


__all__ = ["GroupedExpertsV2"]
