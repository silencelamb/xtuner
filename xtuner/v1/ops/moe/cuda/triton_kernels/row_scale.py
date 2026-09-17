# Copyright (c) OpenMMLab. All rights reserved.
"""Fused per-row scaling ``y = x * w[:, None]`` (bf16 in/out, fp32 math) and its backward, as Triton ops.

DeepEP V2's combine is a pure sum, so the routing weights are applied to the expert output right before it. Doing
that with plain torch ops materialises fp32 copies of the whole ``[rows, hidden]`` tensor (three passes forward,
five backward); these kernels read each operand once and never allocate an fp32 temporary.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


__all__ = ["row_scale", "row_scale_bwd"]

_BLOCK_M = 16
_BLOCK_H = 512


@triton.jit
def _row_scale_fwd_kernel(
    x_ptr, w_ptr, y_ptr, M, H, stride_xm, stride_ym, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    w = tl.load(w_ptr + rows, mask=mask_m, other=0.0).to(tl.float32)
    rows64 = rows.to(tl.int64)
    for h0 in tl.range(0, H, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        mask = mask_m[:, None] & (cols < H)[None, :]
        x = tl.load(x_ptr + rows64[:, None] * stride_xm + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        tl.store(
            y_ptr + rows64[:, None] * stride_ym + cols[None, :], (x * w[:, None]).to(y_ptr.dtype.element_ty), mask=mask
        )


@triton.jit
def _row_scale_bwd_kernel(
    dy_ptr,
    x_ptr,
    w_ptr,
    dx_ptr,
    dw_ptr,
    M,
    H,
    stride_dym,
    stride_xm,
    stride_dxm,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    w = tl.load(w_ptr + rows, mask=mask_m, other=0.0).to(tl.float32)
    rows64 = rows.to(tl.int64)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for h0 in tl.range(0, H, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        mask = mask_m[:, None] & (cols < H)[None, :]
        dy = tl.load(dy_ptr + rows64[:, None] * stride_dym + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + rows64[:, None] * stride_xm + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        tl.store(
            dx_ptr + rows64[:, None] * stride_dxm + cols[None, :],
            (dy * w[:, None]).to(dx_ptr.dtype.element_ty),
            mask=mask,
        )
        acc += tl.sum(dy * x, axis=1)
    tl.store(dw_ptr + rows, acc, mask=mask_m)


@triton_op("moe::row_scale", mutates_args={})
def row_scale(x: Tensor, w: Tensor) -> Tensor:
    """``y = (x.float() * w[:, None]).to(x.dtype)`` without fp32 temporaries.

    Args:
        x (Tensor): ``[M, H]`` bf16 rows (last dim contiguous).
        w (Tensor): ``[M]`` fp32 weight per row.

    Returns:
        Tensor: ``[M, H]`` in ``x.dtype``.
    """
    M, H = x.shape
    y = torch.empty_like(x)
    if M == 0 or H == 0:
        return y
    grid = (triton.cdiv(M, _BLOCK_M),)
    wrap_triton(_row_scale_fwd_kernel)[grid](
        x, w, y, M, H, x.stride(0), y.stride(0), BLOCK_M=_BLOCK_M, BLOCK_H=_BLOCK_H
    )
    return y


@triton_op("moe::row_scale_bwd", mutates_args={})
def row_scale_bwd(dy: Tensor, x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """Backward of :func:`row_scale`: ``dx = dy * w[:, None]`` (in ``x.dtype``) and ``dw = sum(dy * x, -1)`` (fp32).

    Args:
        dy (Tensor): ``[M, H]`` upstream gradient.
        x (Tensor): ``[M, H]`` forward input.
        w (Tensor): ``[M]`` fp32 weight per row.

    Returns:
        tuple[Tensor, Tensor]: ``(dx, dw)``.
    """
    M, H = x.shape
    dx = torch.empty_like(x)
    dw = torch.empty((M,), dtype=torch.float32, device=x.device)
    if M == 0 or H == 0:
        dw.zero_()
        return dx, dw
    dy = dy.contiguous() if dy.stride(-1) != 1 else dy
    grid = (triton.cdiv(M, _BLOCK_M),)
    wrap_triton(_row_scale_bwd_kernel)[grid](
        dy, x, w, dx, dw, M, H, dy.stride(0), x.stride(0), dx.stride(0), BLOCK_M=_BLOCK_M, BLOCK_H=_BLOCK_H
    )
    return dx, dw
