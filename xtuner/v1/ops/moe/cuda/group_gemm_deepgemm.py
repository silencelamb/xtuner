# Copyright (c) OpenMMLab. All rights reserved.
"""DeepGEMM (>= 2.6, psum layout) grouped GEMMs over the segment-aligned (``alignment == 128``) row layout.

Forward and dgrad use ``m_grouped_*_gemm_*_contiguous(..., use_psum_layout=True)`` and read the segment layout
straight from ``ExpertRows`` (``grouped_layout = starts + valid_counts``): no permute, no host counts. Both wgrads run
on GPU counts only. BF16 wgrad uses the Triton ``k_grouped_gemm``. FP8 wgrad uses AdaptiveGEMM's k-grouped dW kernel
when it is installed (bf16 output; at GLM-5.2 EP4 shapes about 2x faster than the alternative), otherwise DeepGEMM's
k-grouped NT GEMM, whose SM90 kernel accumulates into an fp32 D: the operands are re-quantized transposed into its
k-grouped contiguous layout by ``trans_quant_kgrouped`` (device offsets), the real per-group K travels in the device
``ks`` tensor and the host ``ks`` list is a shape-only constant (``static_ks_host``), so that path is free of host
syncs as well. Alignment padding rows are assumed zero-filled by the producer; outputs are zero-initialised so padding
rows stay finite. All GEMM calls are registered custom ops so ``torch.compile(fullgraph=True)`` traces forward and
backward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor


if TYPE_CHECKING:
    from xtuner.v1.module.dispatcher.base import ExpertRows


def deepgemm_available() -> bool:
    """Whether a DeepGEMM with the psum grouped layout is importable."""
    try:
        import deep_gemm  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return hasattr(deep_gemm, "m_grouped_bf16_gemm_nt_contiguous")


def unwrap_fp8_activation(x: Tensor, x_scale: Tensor | None) -> tuple[Tensor, Tensor | None]:
    """Return ``(payload, scales)`` for a BF16 tensor, an ``(fp8, scales)`` pair or a ``Float8Tensor`` wrapper."""
    from xtuner.v1.float8.float8_tensor import Float8Tensor

    if isinstance(x, Float8Tensor):
        return x._data, x._scale
    return x, x_scale


def row_major_scales(x_fp8: Tensor, x_sf: Tensor) -> Tensor:
    """Normalize per-tile scales to a contiguous ``[M, K/128]`` view (DeepEP may hand out the MN-major TMA layout)."""
    m, k = x_fp8.shape
    if x_sf.shape != (m, k // 128):
        x_sf = x_sf.t()
    return x_sf.contiguous()


def _dg():
    import deep_gemm

    if not hasattr(deep_gemm, "m_grouped_bf16_gemm_nt_contiguous"):
        raise ImportError("DeepGEMM >= 2.6 with psum layout support is required (found an older deep_gemm)")
    return deep_gemm


def _out_buffer(x: Tensor, n: int) -> Tensor:
    # Alignment-padding rows (and, on a static capacity, rows past the last segment) are not written by the psum
    # kernel. They must still be finite zeros: the activation and the wgrad kernels read them, and a NaN times a
    # zero gradient row would poison dW. A memset per GEMM output is the price until the kernel zero-fills itself.
    return torch.zeros((x.shape[0], n), dtype=torch.bfloat16, device=x.device)


# ----------------------------------------------------------------------------------------------------------------------
# BF16
# ----------------------------------------------------------------------------------------------------------------------


@torch.library.custom_op("moe::deepgemm_m_grouped_bf16_nt", mutates_args=())
def deepgemm_m_grouped_bf16_nt(x: Tensor, w: Tensor, psum: Tensor) -> Tensor:
    """``D[M, N] = grouped(x[M, K] @ w[G, N, K]^T)`` with DeepGEMM's psum layout."""
    d = _out_buffer(x, w.shape[1])
    if x.shape[0] > 0:
        _dg().m_grouped_bf16_gemm_nt_contiguous(x, w, d, psum, use_psum_layout=True)
    return d


@deepgemm_m_grouped_bf16_nt.register_fake
def _(x: Tensor, w: Tensor, psum: Tensor) -> Tensor:
    return torch.empty((x.shape[0], w.shape[1]), dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op("moe::deepgemm_m_grouped_bf16_nn", mutates_args=())
def deepgemm_m_grouped_bf16_nn(dy: Tensor, w: Tensor, psum: Tensor) -> Tensor:
    """``dX[M, K] = grouped(dy[M, N] @ w[G, N, K])`` (weights consumed as-is, MN-major B)."""
    d = _out_buffer(dy, w.shape[2])
    if dy.shape[0] > 0:
        _dg().m_grouped_bf16_gemm_nn_contiguous(dy, w, d, psum, use_psum_layout=True)
    return d


@deepgemm_m_grouped_bf16_nn.register_fake
def _(dy: Tensor, w: Tensor, psum: Tensor) -> Tensor:
    return torch.empty((dy.shape[0], w.shape[2]), dtype=torch.bfloat16, device=dy.device)


class _DeepGemmBF16(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, w: Tensor, psum: Tensor, compute_counts: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x, w, psum, compute_counts)
        return deepgemm_m_grouped_bf16_nt(x, w, psum)

    @staticmethod
    def backward(ctx: Any, dy: Tensor):  # type: ignore[override]
        from .triton_kernels import k_grouped_gemm

        x, w, psum, compute_counts = ctx.saved_tensors
        dy = dy.contiguous()
        dx = deepgemm_m_grouped_bf16_nn(dy, w, psum)
        # wgrad over GPU counts (padding rows are zero on both operands, so they contribute nothing).
        dw = k_grouped_gemm(dy, x, compute_counts) if x.shape[0] > 0 else torch.zeros_like(w)
        return dx, dw, None, None


def deepgemm_group_gemm(x: Tensor, w: Tensor, rows: "ExpertRows") -> Tensor:
    """BF16 grouped GEMM ``[capacity, K] x [E, N, K]^T`` over a 128-aligned row layout.

    Args:
        x (Tensor): ``[capacity, K]`` bf16 activations in the layout described by ``rows``.
        w (Tensor): ``[E, N, K]`` bf16 weights.
        rows (ExpertRows): Row layout with ``alignment == 128``.

    Returns:
        Tensor: ``[capacity, N]`` bf16.
    """
    from xtuner.v1.module.dispatcher.base import rows_psum

    assert rows.alignment == 128, "DeepGEMM psum GEMMs need 128-aligned segments"
    return _DeepGemmBF16.apply(x, w, rows_psum(rows), rows.compute_counts)


# ----------------------------------------------------------------------------------------------------------------------
# FP8 (activations 1x128 per-tile e4m3 + fp32 SF; weights 128x128 per-block e4m3 + fp32 SF)
# ----------------------------------------------------------------------------------------------------------------------


@torch.library.custom_op("moe::deepgemm_m_grouped_fp8_nt", mutates_args=())
def deepgemm_m_grouped_fp8_nt(x: Tensor, x_sf: Tensor, w: Tensor, w_sf: Tensor, psum: Tensor) -> Tensor:
    d = _out_buffer(x, w.shape[1])
    if x.shape[0] > 0:
        _dg().m_grouped_fp8_gemm_nt_contiguous((x, x_sf), (w, w_sf), d, psum, use_psum_layout=True)
    return d


@deepgemm_m_grouped_fp8_nt.register_fake
def _(x: Tensor, x_sf: Tensor, w: Tensor, w_sf: Tensor, psum: Tensor) -> Tensor:
    return torch.empty((x.shape[0], w.shape[1]), dtype=torch.bfloat16, device=x.device)


_FP32_WORKSPACES: dict[tuple[int, int, int, int], Tensor] = {}


def _fp32_workspace(num_groups: int, m: int, n: int, device: torch.device) -> Tensor:
    # The SM90 k-grouped kernel accumulates into an fp32 D. At model scale that is gigabytes per call
    # (local_experts x out x in x 4 B), so one persistent buffer per shape is reused across layers instead of
    # allocating, zero-filling and freeing a fresh one each time (allocator churn dominated the step time).
    key = (num_groups, m, n, device.index if device.index is not None else -1)
    ws = _FP32_WORKSPACES.get(key)
    if ws is None:
        ws = torch.empty((num_groups, m, n), dtype=torch.float32, device=device)
        _FP32_WORKSPACES[key] = ws
    return ws


@torch.library.custom_op("moe::deepgemm_k_grouped_fp8_nt", mutates_args=())
def deepgemm_k_grouped_fp8_nt(
    a: Tensor, a_sf: Tensor, b: Tensor, b_sf: Tensor, ks_host: list[int], ks: Tensor, num_groups: int, m: int, n: int
) -> Tensor:
    """``D[g] = A_g @ B_g^T`` over DeepGEMM's k-grouped contiguous layout, returned in bf16; ``ks`` (device) holds the
    real per-group K, accumulation happens in a reused fp32 workspace."""
    ws = _fp32_workspace(num_groups, m, n, a.device)
    ws.zero_()
    _dg().k_grouped_fp8_gemm_nt_contiguous((a, a_sf), (b, b_sf), ws, ks_host, ks, c=ws)
    return ws.to(torch.bfloat16)


@deepgemm_k_grouped_fp8_nt.register_fake
def _(
    a: Tensor, a_sf: Tensor, b: Tensor, b_sf: Tensor, ks_host: list[int], ks: Tensor, num_groups: int, m: int, n: int
) -> Tensor:
    return torch.empty((num_groups, m, n), dtype=torch.bfloat16, device=a.device)


def _adaptive_gemm_dw_available() -> bool:
    try:
        from adaptive_gemm import k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous  # noqa: F401
    except ImportError:
        return False
    return True


class _DeepGemmFP8(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, x: Tensor, x_scale: Tensor | None, w_fp8: Any, psum: Tensor, starts: Tensor, compute_counts: Tensor
    ) -> Tensor:
        from xtuner.v1.float8.triton_kernels import (
            per_tile_quant,
            trans_per_block_quant_expand_128x,
            trans_quant_kgrouped,
        )

        x, x_scale = unwrap_fp8_activation(x, x_scale)
        counts32 = compute_counts.to(torch.int32)
        starts32 = starts.to(torch.int32)
        if x_scale is None:
            x_fp8, x_sf = per_tile_quant(x)
            x_bf16 = x
        else:
            x_fp8, x_sf = x, x_scale  # scales may be in DeepEP's TMA-aligned layout; DeepGEMM accepts both
            x_bf16 = None
        out = deepgemm_m_grouped_fp8_nt(x_fp8, x_sf, w_fp8._data, w_fp8._scale, psum)
        # x^T per group, re-quantized for the wgrad kernel and saved instead of x.
        ctx.adaptive_dw = _adaptive_gemm_dw_available()
        if ctx.adaptive_dw:
            # AdaptiveGEMM's dW kernel wants per-block (128x128) x^T in its expand layout (identity on aligned rows).
            if x_bf16 is None:
                m, k = x_fp8.shape
                sf = row_major_scales(x_fp8, x_sf)
                x_bf16 = (x_fp8.view(m, k // 128, 128).float() * sf.view(m, k // 128, 1)).view(m, k).to(torch.bfloat16)
            x_t_fp8, x_t_sf, _ = trans_per_block_quant_expand_128x(
                x_bf16, counts32.to(torch.int64), group_size=128, dtype=torch.float8_e4m3fn
            )
        elif x_bf16 is not None:
            x_t_fp8, x_t_sf = trans_quant_kgrouped(x_bf16, starts32, counts32)
        else:
            # Pre-quantized input is dequantized inside the kernel: no bf16 copy of the received tokens.
            x_t_fp8, x_t_sf = trans_quant_kgrouped(x_fp8, starts32, counts32, row_major_scales(x_fp8, x_sf))
        ctx.save_for_backward(x_t_fp8, x_t_sf, psum, counts32, starts32)
        ctx.w_fp8 = w_fp8
        ctx.input_shape = (x.shape[0], w_fp8.shape[2])
        return out

    @staticmethod
    def backward(ctx: Any, dy: Tensor):  # type: ignore[override]
        from xtuner.v1.float8.triton_kernels import (
            per_tile_quant,
            static_ks_host,
            trans_per_tile_quant_expand_128x,
            trans_quant_kgrouped,
        )

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
            )
        dy = dy.contiguous()
        g_fp8, g_sf = per_tile_quant(dy)
        # SM90 FP8 needs K-major B: materialize W^T (and its block scales) per call, as the legacy path does.
        w_t = w_fp8._data.transpose(1, 2).contiguous()
        w_t_sf = w_fp8._scale.transpose(1, 2).contiguous()
        dx = deepgemm_m_grouped_fp8_nt(g_fp8, g_sf, w_t, w_t_sf, psum)
        if ctx.adaptive_dw:
            from adaptive_gemm import k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous

            g_t_fp8, g_t_sf, counts_expand = trans_per_tile_quant_expand_128x(dy, counts32.to(torch.int64))
            dw = dy.new_empty((ne, dout, din), dtype=torch.bfloat16)
            k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous(g_t_fp8, g_t_sf, x_t_fp8, x_t_sf, dw, counts_expand.int())
        else:
            # dW[e] = dy_e^T @ x_e as DeepGEMM k-grouped NT: A_e = dy_e^T [dout, k_e], B_e = x_e^T [din, k_e].
            g_t_fp8, g_t_sf = trans_quant_kgrouped(dy, starts32, counts32)
            dw = deepgemm_k_grouped_fp8_nt(
                g_t_fp8, g_t_sf, x_t_fp8, x_t_sf, static_ks_host(x_t_sf.shape[1] * 128, ne), counts32, ne, dout, din
            )
        return dx, None, dw, None, None, None


def deepgemm_fp8_group_gemm(x: Tensor, x_scale: Tensor | None, w_fp8: Any, rows: "ExpertRows") -> Tensor:
    """FP8 grouped GEMM over a 128-aligned row layout with per-tile activations and per-block weights.

    Args:
        x (Tensor): ``[capacity, K]`` bf16 activations, or fp8 e4m3 data when ``x_scale`` is given, or a
            ``Float8Tensor`` wrapper carrying both.
        x_scale (Tensor | None): Per-tile (1x128) fp32 scales of an fp8 ``x``.
        w_fp8 (Float8Tensor): ``[E, N, K]`` per-block (128x128) quantized weights.
        rows (ExpertRows): Row layout with ``alignment == 128``.

    Returns:
        Tensor: ``[capacity, N]`` bf16.
    """
    from xtuner.v1.module.dispatcher.base import rows_psum

    assert rows.alignment == 128, "DeepGEMM psum GEMMs need 128-aligned segments"
    return _DeepGemmFP8.apply(x, x_scale, w_fp8, rows_psum(rows), rows.starts, rows.compute_counts)


__all__ = [
    "deepgemm_available",
    "deepgemm_fp8_group_gemm",
    "deepgemm_group_gemm",
    "row_major_scales",
    "unwrap_fp8_activation",
]
