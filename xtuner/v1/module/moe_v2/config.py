# Copyright (c) OpenMMLab. All rights reserved.
"""Configuration and configuration-time validation for the MoE V2 (unified EP contract) path."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


GemmBackendName = Literal["auto", "triton", "adaptive_gemm", "deepgemm"]
V2DispatcherName = Literal["legacy", "deepep_v2"]


class MoEV2Config(BaseModel):
    """Options of the contract-based MoE path (``MoEConfig.dispatcher == "deepep_v2"`` or legacy adapters).

    Args:
        dispatcher (V2DispatcherName): ``"deepep_v2"`` uses the DeepEP V2 ``ElasticBuffer`` dispatcher;
            ``"legacy"`` wraps the legacy dispatcher named by ``legacy_dispatcher`` behind the contract
            (used for bit-exact validation of the contract itself).
        legacy_dispatcher (str | None): Legacy dispatcher name for ``dispatcher="legacy"``.
        gemm_backend (GemmBackendName): Expert GEMM backend. ``"auto"`` picks DeepGEMM when the row layout is
            aligned and DeepGEMM is importable, otherwise Triton (BF16) / AdaptiveGEMM (FP8).
        expert_alignment (int): Row alignment requested from the dispatcher. ``128`` is required by DeepGEMM
            on SM90; ``1`` gives the unpadded legacy layout.
        max_tokens_per_rank (int | None): Upper bound of tokens one rank sends per dispatch. Required by
            DeepEP V2 to size its buffer; defaults to the trainer's pack length when ``None``.
        cpu_sync (bool): DeepEP V2 ``do_cpu_sync``. ``True`` returns exact received counts to the host
            (dynamic shapes); ``False`` keeps counts on the GPU and allocates worst-case capacity
            (static shapes, CUDA-graph friendly).
        fp8_dispatch (bool): Quantize activations to FP8 (1x128 per-tile) before dispatch. Requires FP8
            experts (``Float8Config.scaling_granularity_grouped_gemm``).
        deterministic (bool): DeepEP V2 deterministic receive order.
        num_sms (int): SMs used by DeepEP V2 kernels; ``0`` lets the library decide.
        allow_hybrid_mode (bool): DeepEP V2 hierarchical RDMA + NVLink mode for multi-node EP groups.
        zero_padding (bool): Ask the dispatcher to zero-fill alignment padding rows.
        check_rows (bool): Debug: verify layout invariants with host synchronization on every call.
    """

    model_config = ConfigDict(extra="forbid")

    dispatcher: V2DispatcherName = "deepep_v2"
    legacy_dispatcher: str | None = None
    gemm_backend: GemmBackendName = "auto"
    expert_alignment: int = 128
    max_tokens_per_rank: int | None = None
    cpu_sync: bool = True
    fp8_dispatch: bool = False
    deterministic: bool = False
    num_sms: int = 0
    allow_hybrid_mode: bool = True
    zero_padding: bool = True
    check_rows: bool = False


def validate_moe_v2_config(moe_cfg: Any, v2_cfg: MoEV2Config) -> None:
    """Reject unsupported combinations at model-construction time.

    Args:
        moe_cfg (MoEConfig): The model config (typed loosely to avoid an import cycle).
        v2_cfg (MoEV2Config): The V2 options.
    """
    if getattr(moe_cfg, "expert_tp_size", 1) > 1:
        raise NotImplementedError("MoE V2 path does not support expert TP yet")
    if getattr(moe_cfg, "moe_bias", False):
        raise NotImplementedError("MoE V2 path does not support routed-expert bias yet")
    fp8 = _fp8_experts_enabled(moe_cfg)
    if v2_cfg.fp8_dispatch and not fp8:
        raise ValueError("fp8_dispatch requires FP8 grouped GEMM (Float8Config.scaling_granularity_grouped_gemm)")
    if v2_cfg.expert_alignment not in (1, 128):
        raise ValueError(f"expert_alignment must be 1 or 128, got {v2_cfg.expert_alignment}")
    if v2_cfg.gemm_backend == "deepgemm" and v2_cfg.expert_alignment != 128:
        raise ValueError("DeepGEMM backend requires expert_alignment=128 on SM90")
    if v2_cfg.dispatcher == "deepep_v2":
        if moe_cfg.hidden_size % 256 != 0:
            raise ValueError(f"DeepEP V2 combine requires hidden_size % 256 == 0, got {moe_cfg.hidden_size}")
        if moe_cfg.num_experts_per_tok > 32:
            raise ValueError("DeepEP V2 supports at most 32 top-k experts")
        ep_size = getattr(moe_cfg, "ep_size", 1)
        if ep_size > 1 and moe_cfg.n_routed_experts // ep_size > 256:
            raise ValueError("DeepEP V2 supports at most 256 local experts per rank")
    if v2_cfg.dispatcher == "legacy" and v2_cfg.legacy_dispatcher is None:
        raise ValueError("dispatcher='legacy' requires legacy_dispatcher")


def _fp8_experts_enabled(moe_cfg: Any) -> bool:
    float8_cfg = getattr(moe_cfg, "float8_cfg", None)
    return float8_cfg is not None and getattr(float8_cfg, "scaling_granularity_grouped_gemm", None) is not None


__all__ = ["GemmBackendName", "MoEV2Config", "V2DispatcherName", "validate_moe_v2_config"]
