# Copyright (c) OpenMMLab. All rights reserved.
"""DeepGEMM (>= 2.6, psum layout) expert GEMM backend for the ``EA`` (128-aligned) row layout.

Forward / dgrad use ``m_grouped_*_gemm_*_contiguous(..., use_psum_layout=True)`` and read the segment layout
directly from ``rows`` (``grouped_layout = psum``): no permute, no host counts. FP8 wgrad uses DeepGEMM's k-grouped
NT GEMM on the same footing: the operands are re-quantized transposed into its k-grouped contiguous layout by
``trans_quant_kgrouped`` (device offsets), the real per-group K travels in the device ``ks_tensor`` and the host ``ks``
list is a shape-only constant (``static_ks_host``), so the whole backend stays free of host syncs. BF16 wgrad uses the
Triton ``k_grouped_gemm`` (DeepGEMM has no BF16 k-grouped kernel). Alignment padding rows are assumed zero-filled
(``rows.padding_zeroed``); outputs are zero-initialised so padding rows stay finite.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .contracts import ExpertRows, rows_psum
from .gemm_backends import GemmCaps, row_major_scales, unwrap_fp8_activation
from .kgrouped_quant import static_ks_host, trans_quant_kgrouped


def _dg():
    import deep_gemm

    if not hasattr(deep_gemm, "m_grouped_bf16_gemm_nt_contiguous"):
        raise ImportError("DeepGEMM >= 2.6 with psum layout support is required (found an older deep_gemm)")
    return deep_gemm


def _out_buffer(x: Tensor, n: int, static: bool) -> Tensor:
    # Alignment-padding rows (and, in static mode, rows past the last segment) are not written by the psum kernel.
    # They must still be finite zeros: the activation and the counts-based wgrad kernels read them, and a NaN times
    # a zero gradient row would poison dW. A memset per GEMM output is the price until the kernel zero-fills itself.
    del static
    return torch.zeros((x.shape[0], n), dtype=torch.bfloat16, device=x.device)


# ----------------------------------------------------------------------------------------------------------------------
# BF16
# ----------------------------------------------------------------------------------------------------------------------


@torch.library.custom_op("moe_v2::dg_m_grouped_bf16_nt", mutates_args=())
def dg_m_grouped_bf16_nt(x: Tensor, w: Tensor, psum: Tensor, static: bool) -> Tensor:
    """``D[M, N] = grouped(x[M, K] @ w[G, N, K]^T)`` with DeepGEMM psum layout."""
    d = _out_buffer(x, w.shape[1], static)
    if x.shape[0] > 0:
        _dg().m_grouped_bf16_gemm_nt_contiguous(x, w, d, psum, use_psum_layout=True)
    return d


@dg_m_grouped_bf16_nt.register_fake
def _(x: Tensor, w: Tensor, psum: Tensor, static: bool) -> Tensor:
    return torch.empty((x.shape[0], w.shape[1]), dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op("moe_v2::dg_m_grouped_bf16_nn", mutates_args=())
def dg_m_grouped_bf16_nn(dy: Tensor, w: Tensor, psum: Tensor, static: bool) -> Tensor:
    """``dX[M, K] = grouped(dy[M, N] @ w[G, N, K])`` (weights consumed as-is, MN-major B)."""
    d = _out_buffer(dy, w.shape[2], static)
    if dy.shape[0] > 0:
        _dg().m_grouped_bf16_gemm_nn_contiguous(dy, w, d, psum, use_psum_layout=True)
    return d


@dg_m_grouped_bf16_nn.register_fake
def _(dy: Tensor, w: Tensor, psum: Tensor, static: bool) -> Tensor:
    return torch.empty((dy.shape[0], w.shape[2]), dtype=torch.bfloat16, device=dy.device)


class _DeepGemmBF16Rows(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, w: Tensor, psum: Tensor, compute_counts: Tensor, static: bool) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x, w, psum, compute_counts)
        ctx.static = static
        return dg_m_grouped_bf16_nt(x, w, psum, static)

    @staticmethod
    def backward(ctx: Any, dy: Tensor):  # type: ignore[override]
        from xtuner.v1.ops.moe.cuda.triton_kernels import k_grouped_gemm

        x, w, psum, compute_counts = ctx.saved_tensors
        dy = dy.contiguous()
        dx = dg_m_grouped_bf16_nn(dy, w, psum, ctx.static)
        # wgrad over GPU counts (padding rows are zero on both operands, so they contribute nothing).
        dw = k_grouped_gemm(dy, x, compute_counts) if x.shape[0] > 0 else torch.zeros_like(w)
        return dx, dw, None, None, None


# ----------------------------------------------------------------------------------------------------------------------
# FP8 (activations 1x128 per-tile e4m3 + fp32 SF; weights 128x128 per-block e4m3 + fp32 SF)
# ----------------------------------------------------------------------------------------------------------------------


@torch.library.custom_op("moe_v2::dg_m_grouped_fp8_nt", mutates_args=())
def dg_m_grouped_fp8_nt(x: Tensor, x_sf: Tensor, w: Tensor, w_sf: Tensor, psum: Tensor, static: bool) -> Tensor:
    d = _out_buffer(x, w.shape[1], static)
    if x.shape[0] > 0:
        _dg().m_grouped_fp8_gemm_nt_contiguous((x, x_sf), (w, w_sf), d, psum, use_psum_layout=True)
    return d


@dg_m_grouped_fp8_nt.register_fake
def _(x: Tensor, x_sf: Tensor, w: Tensor, w_sf: Tensor, psum: Tensor, static: bool) -> Tensor:
    return torch.empty((x.shape[0], w.shape[1]), dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op("moe_v2::dg_k_grouped_fp8_nt", mutates_args=())
def dg_k_grouped_fp8_nt(
    a: Tensor, a_sf: Tensor, b: Tensor, b_sf: Tensor, ks_host: list[int], ks: Tensor, num_groups: int, m: int, n: int
) -> Tensor:
    """``D[g] = A_g @ B_g^T`` (fp32) over DeepGEMM's k-grouped contiguous layout; ``ks`` (device) holds the real per-group K."""
    d = torch.zeros((num_groups, m, n), dtype=torch.float32, device=a.device)
    _dg().k_grouped_fp8_gemm_nt_contiguous((a, a_sf), (b, b_sf), d, ks_host, ks, c=d)
    return d


@dg_k_grouped_fp8_nt.register_fake
def _(
    a: Tensor, a_sf: Tensor, b: Tensor, b_sf: Tensor, ks_host: list[int], ks: Tensor, num_groups: int, m: int, n: int
) -> Tensor:
    return torch.empty((num_groups, m, n), dtype=torch.float32, device=a.device)


class _DeepGemmFP8Rows(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: Tensor,
        x_scale: Tensor | None,
        w_fp8: Any,
        psum: Tensor,
        starts: Tensor,
        compute_counts: Tensor,
        static: bool,
    ) -> Tensor:
        from xtuner.v1.float8.triton_kernels import per_tile_quant

        x, x_scale = unwrap_fp8_activation(x, x_scale)
        counts32 = compute_counts.to(torch.int32)
        starts32 = starts.to(torch.int32)  # psum is the *unaligned* segment end (starts + valid), not starts + compute
        # x^T per group in DeepGEMM's k-grouped layout, saved for wgrad (fp8 + one scale per 128-row tile). A
        # pre-quantized input (FP8 dispatch) is dequantized inside the kernel: no bf16 copy of the received tokens.
        if x_scale is None:
            x_fp8, x_sf = per_tile_quant(x)
            x_t_fp8, x_t_sf = trans_quant_kgrouped(x, starts32, counts32)
        else:
            x_fp8, x_sf = x, x_scale  # scales may be in DeepEP's TMA-aligned layout; DeepGEMM accepts both
            x_t_fp8, x_t_sf = trans_quant_kgrouped(x_fp8, starts32, counts32, row_major_scales(x_fp8, x_sf))
        out = dg_m_grouped_fp8_nt(x_fp8, x_sf, w_fp8._data, w_fp8._scale, psum, static)
        ctx.save_for_backward(x_t_fp8, x_t_sf, psum, counts32, starts32)
        ctx.w_fp8 = w_fp8
        ctx.static = static
        ctx.input_shape = (x.shape[0], w_fp8.shape[2])
        return out

    @staticmethod
    def backward(ctx: Any, dy: Tensor):  # type: ignore[override]
        from xtuner.v1.float8.triton_kernels import per_tile_quant

        x_t_fp8, x_t_sf, psum, counts32, starts32 = ctx.saved_tensors
        w_fp8 = ctx.w_fp8
        ne, dout, din = w_fp8.shape
        if dy.shape[0] == 0:
            return (
                dy.new_empty(ctx.input_shape, dtype=torch.bfloat16),
                None,
                dy.new_zeros((ne, dout, din), dtype=torch.bfloat16),
                None,
                None,
                None,
                None,
            )
        dy = dy.contiguous()
        g_fp8, g_sf = per_tile_quant(dy)
        # SM90 FP8 needs K-major B: materialize W^T (and its block scales) per call, as the legacy path does.
        w_t = w_fp8._data.transpose(1, 2).contiguous()
        w_t_sf = w_fp8._scale.transpose(1, 2).contiguous()
        dx = dg_m_grouped_fp8_nt(g_fp8, g_sf, w_t, w_t_sf, psum, ctx.static)
        # dW[e] = dy_e^T @ x_e as DeepGEMM k-grouped NT: A_e = dy_e^T [dout, k_e], B_e = x_e^T [din, k_e], fp32 D.
        g_t_fp8, g_t_sf = trans_quant_kgrouped(dy, starts32, counts32)
        dw32 = dg_k_grouped_fp8_nt(
            g_t_fp8, g_t_sf, x_t_fp8, x_t_sf, static_ks_host(x_t_sf.shape[1] * 128, ne), counts32, ne, dout, din
        )
        return dx, None, dw32.to(torch.bfloat16), None, None, None, None


class DeepGemmBackend:
    """DeepGEMM psum backend; ``fp8`` selects the FP8 recipe."""

    def __init__(self, *, fp8: bool) -> None:
        self.caps = GemmCaps(name="deepgemm", fp8=fp8, required_alignment=128, accepts_padding_rows=True)
        self.fp8 = fp8
        _dg()

    def gemm(self, x: Tensor, x_scale: Tensor | None, weight: Any, rows: ExpertRows, *, trans_b: bool) -> Tensor:
        assert trans_b
        assert rows.alignment == 128, "DeepGEMM psum backend needs 128-aligned segments"
        psum = rows_psum(rows)
        # capacity > compute extent only in the static (no-cpu-sync) mode; in cpu_sync mode they are equal.
        static = True
        if self.fp8:
            return _DeepGemmFP8Rows.apply(x, x_scale, weight, psum, rows.starts, rows.compute_counts, static)
        return _DeepGemmBF16Rows.apply(x, weight, psum, rows.compute_counts, static)


__all__ = ["DeepGemmBackend"]
