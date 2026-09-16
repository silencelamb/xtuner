# Copyright (c) OpenMMLab. All rights reserved.
"""Expert GEMM backends that consume ``ExpertRows`` (contract v0).

Every backend exposes ``gemm(x, x_scale, weight, rows, ...)`` as a differentiable op producing ``[capacity, N]``.
Backends never call ``.cpu()`` / ``.item()`` on the hot path; they only read ``rows.compute_counts`` (and
``rows.starts`` / ``rows_psum`` for psum-aware kernels). Alignment padding rows are assumed zero-filled by the
producer (``rows.padding_zeroed``); the decoder rejects backends that cannot handle unzeroed padding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

from .contracts import ExpertRows, rows_psum


@dataclass(frozen=True)
class GemmCaps:
    name: str
    fp8: bool
    required_alignment: int
    accepts_padding_rows: bool
    needs_host_counts: bool = False
    static_shape_friendly: bool = True


class GemmBackend(Protocol):
    caps: GemmCaps

    def gemm(
        self,
        x: Tensor,
        x_scale: Tensor | None,
        weight: Any,
        rows: ExpertRows,
        *,
        trans_b: bool,
    ) -> Tensor: ...


# ----------------------------------------------------------------------------------------------------------------------
# BF16 Triton (counts-only, kernel-internal 128-row virtual alignment). Numerically identical to the legacy
# ``triton_group_gemm`` path when ``rows.alignment == 1``.
# ----------------------------------------------------------------------------------------------------------------------


class _TritonGroupedGemmRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, compute_counts: Tensor) -> Tensor:  # type: ignore[override]
        from xtuner.v1.ops.moe.cuda.triton_kernels import m_grouped_gemm

        ctx.save_for_backward(x, w, compute_counts)
        if x.shape[0] == 0:
            return x.new_empty((0, w.shape[1]))
        return m_grouped_gemm(x, w, compute_counts, trans_b=True)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        from xtuner.v1.ops.moe.cuda.triton_kernels import k_grouped_gemm, m_grouped_gemm

        x, w, compute_counts = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if x.shape[0] == 0:
            return torch.empty_like(x), torch.zeros_like(w), None
        dx = m_grouped_gemm(grad_output, w, compute_counts, trans_b=False)
        dw = k_grouped_gemm(grad_output, x, compute_counts)
        return dx, dw, None


class TritonBF16Backend:
    caps = GemmCaps(name="triton", fp8=False, required_alignment=1, accepts_padding_rows=True)

    def gemm(self, x: Tensor, x_scale: Tensor | None, weight: Tensor, rows: ExpertRows, *, trans_b: bool) -> Tensor:
        assert x_scale is None, "Triton BF16 backend does not take FP8 activations"
        assert trans_b, "weights are always [E, N, K]"
        return _TritonGroupedGemmRows.apply(x, weight, rows.compute_counts)


# ----------------------------------------------------------------------------------------------------------------------
# FP8 AdaptiveGEMM (legacy XTuner FP8 recipe: 1x128 per-tile activations, 128x128 per-block weights).
# ``rows.compute_counts`` replaces ``tokens_per_expert``; in EA-128 the ``expand_128x`` helpers become identity.
# ----------------------------------------------------------------------------------------------------------------------


class _AdaptiveGemmFP8Rows(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        x_scale: Tensor | None,
        w_fp8: Any,
        compute_counts: Tensor,
    ) -> Tensor:
        from adaptive_gemm import m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous

        from xtuner.v1.float8.triton_kernels import per_tile_quant, trans_per_block_quant_expand_128x

        ne, dout, din = w_fp8.shape
        ctx.input_shape = (x.shape[0], din)
        ctx.weight_shape = (ne, dout, din)
        ctx.zero_token = x.shape[0] == 0
        if ctx.zero_token:
            ctx.save_for_backward(compute_counts)
            ctx.w_fp8 = w_fp8
            return x.new_empty((0, dout), dtype=torch.bfloat16)

        if x_scale is None:
            # BF16 activation: quantize here (legacy behaviour).
            x_fp8, x_sf = per_tile_quant(x)
            x_bf16_for_wgrad = x
        else:
            # Pre-quantized activation (FP8 dispatch): dequantize once for the wgrad-side re-quantization.
            x_fp8, x_sf = x, x_scale
            x_bf16_for_wgrad = _dequant_per_tile(x_fp8, x_sf)
        x_t_fp8, x_t_sf, _ = trans_per_block_quant_expand_128x(
            x_bf16_for_wgrad, compute_counts, group_size=128, dtype=torch.float8_e4m3fn
        )
        out = m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous(
            (x_fp8, x_sf), (w_fp8._data, w_fp8._scale), compute_counts
        )
        ctx.save_for_backward(x_t_fp8, x_t_sf, compute_counts)
        ctx.w_fp8 = w_fp8
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        from adaptive_gemm import (
            k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous,
            m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous,
        )

        from xtuner.v1.float8.triton_kernels import per_tile_quant, trans_per_tile_quant_expand_128x

        w_fp8 = ctx.w_fp8
        if ctx.zero_token:
            dx = grad_output.new_empty(ctx.input_shape, dtype=torch.bfloat16)
            dw = grad_output.new_zeros(ctx.weight_shape, dtype=torch.bfloat16)
            return dx, None, dw, None

        x_t_fp8, x_t_sf, compute_counts = ctx.saved_tensors
        ne, dout, din = ctx.weight_shape
        grad_output = grad_output.contiguous()
        g_fp8, g_sf = per_tile_quant(grad_output)
        dx = m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous(
            (g_fp8, g_sf),
            (w_fp8._data.transpose(1, 2).contiguous(), w_fp8._scale.transpose(1, 2).contiguous()),
            compute_counts,
        )
        g_t_fp8, g_t_sf, counts_expand = trans_per_tile_quant_expand_128x(grad_output, compute_counts)
        dw = grad_output.new_empty((ne, dout, din), dtype=torch.bfloat16)
        k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous(g_t_fp8, g_t_sf, x_t_fp8, x_t_sf, dw, counts_expand.int())
        # ``x`` may be pre-quantized FP8 (no gradient) or BF16 (gradient flows back through the dispatcher).
        return dx, None, dw, None


def _dequant_per_tile(x_fp8: Tensor, x_sf: Tensor) -> Tensor:
    m, k = x_fp8.shape
    return (x_fp8.view(m, k // 128, 128).to(torch.float32) * x_sf.view(m, k // 128, 1)).view(m, k).to(torch.bfloat16)


class AdaptiveGemmFP8Backend:
    caps = GemmCaps(name="adaptive_gemm", fp8=True, required_alignment=1, accepts_padding_rows=True)

    def gemm(self, x: Tensor, x_scale: Tensor | None, weight: Any, rows: ExpertRows, *, trans_b: bool) -> Tensor:
        assert trans_b
        # The legacy FP8 helpers (``trans_*_quant_expand_128x`` / AdaptiveGEMM) require int64 counts.
        return _AdaptiveGemmFP8Rows.apply(x, x_scale, weight, rows.compute_counts.to(torch.int64))


# ----------------------------------------------------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------------------------------------------------


def build_gemm_backend(name: str, *, fp8: bool, alignment: int) -> GemmBackend:
    """Pick a GEMM backend once at setup time (never per call).

    Args:
        name (str): ``auto`` / ``triton`` / ``adaptive_gemm`` / ``deepgemm``; ``XTUNER_MOE_V2_GEMM`` overrides.
        fp8 (bool): Whether experts run in FP8.
        alignment (int): Row alignment the dispatcher will produce.

    Returns:
        GemmBackend: The selected backend.
    """
    name = os.environ.get("XTUNER_MOE_V2_GEMM", name).lower()
    if name == "auto":
        name = (
            "deepgemm" if (alignment == 128 and _deepgemm_psum_available()) else ("adaptive_gemm" if fp8 else "triton")
        )
    if name == "deepgemm":
        from .gemm_deepgemm import DeepGemmBackend

        if alignment != 128:
            raise ValueError("DeepGEMM backend needs expert_alignment=128")
        return DeepGemmBackend(fp8=fp8)
    if name == "triton":
        if fp8:
            raise ValueError("Triton backend is BF16 only; use adaptive_gemm or deepgemm for FP8 experts")
        return TritonBF16Backend()
    if name == "adaptive_gemm":
        if not fp8:
            raise ValueError("adaptive_gemm backend is FP8 only")
        return AdaptiveGemmFP8Backend()
    raise ValueError(f"unknown MoE V2 GEMM backend {name!r}")


def _deepgemm_psum_available() -> bool:
    try:
        import deep_gemm  # noqa: F401

        return hasattr(deep_gemm, "m_grouped_bf16_gemm_nt_contiguous")
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "AdaptiveGemmFP8Backend",
    "GemmBackend",
    "GemmCaps",
    "TritonBF16Backend",
    "build_gemm_backend",
    "rows_psum",
]
