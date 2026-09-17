# Copyright (c) OpenMMLab. All rights reserved.
"""Transposed per-tile FP8 quantization into DeepGEMM's k-grouped contiguous layout, with device-side group offsets.

``x[K, N]`` holds the rows of ``G`` groups back to back: group ``g`` is rows ``[starts[g], starts[g] + counts[g])``
and every ``counts[g]`` is a multiple of 128 (the ``EA`` layout, alignment padding rows zero). The output is what
``deep_gemm.k_grouped_fp8_gemm_nt_contiguous`` consumes:

* ``out`` (fp8, ``N * k_alloc`` elements): group ``g`` is one contiguous ``[N, counts[g]]`` block starting at element
  ``N * starts[g]``, i.e. ``out[N * starts[g] + n * counts[g] + m] = q(x[starts[g] + m, n])``;
* ``scales`` (fp32, ``[N, k_alloc // 128]``): one scale per (column, 128-row tile), tile index ``(starts[g] + m) // 128``.

No host involvement: offsets come from ``starts`` / ``counts`` on the GPU and ``k_alloc`` (``x.shape[0]`` rounded up
to 128) is a pure shape. Regions past the last group are left unwritten and are never read by the GEMM, which takes
the real per-group K from the device tensor.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


__all__ = ["trans_quant_kgrouped", "static_ks_host"]


@triton.jit
def _trans_quant_kgrouped_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    starts_ptr,
    counts_ptr,
    in_scale_ptr,
    stride_xm,
    stride_xn,
    stride_scale_n,
    stride_in_scale_m,
    fmax: tl.constexpr,
    fmin: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_IN_SCALE: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    g = tl.program_id(axis=1)
    start = tl.load(starts_ptr + g).to(tl.int64)
    count = tl.load(counts_ptr + g).to(tl.int64)
    if count <= 0:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_n64 = offs_n.to(tl.int64)
    offs_m = tl.arange(0, 128).to(tl.int64)
    out_group = out_ptr + N * start + offs_n64[None, :] * count
    scale_row = scale_ptr + offs_n64 * stride_scale_n + start // 128
    col_block = (pid_n * BLOCK_N) // 128  # BLOCK_N divides 128, so one tile never straddles two scale columns
    for i in tl.range(0, tl.cdiv(count, 128)):
        rows = start + i * 128 + offs_m
        xb = tl.load(
            x_ptr + rows[:, None] * stride_xm + offs_n64[None, :] * stride_xn, mask=mask_n[None, :], other=0.0
        ).to(tl.float32)
        if HAS_IN_SCALE:
            s_in = tl.load(in_scale_ptr + rows * stride_in_scale_m + col_block).to(tl.float32)
            xb = xb * s_in[:, None]
        s = tl.clamp(tl.max(tl.abs(xb), 0) / fmax, 1e-12, 3e38)
        tl.store(scale_row + i, s, mask=mask_n)
        q = tl.clamp(xb / s[None, :], fmin, fmax).to(out_ptr.dtype.element_ty)
        tl.store(out_group + (i * 128 + offs_m)[:, None], q, mask=mask_n[None, :])


@triton_op("float8::trans_quant_kgrouped", mutates_args={})
def trans_quant_kgrouped(
    x: Tensor, starts: Tensor, counts: Tensor, x_scales: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Quantize ``x[K, N]`` transposed, per (column, 128-row tile), into the k-grouped contiguous layout.

    ``x`` is bf16, or fp8 together with its per-tile scales ``x_scales`` (fp32 ``[K, N // 128]``, one per
    (row, 128-column block), row-major): the dequantization then happens in-kernel instead of through a bf16 copy.

    Args:
        x (Tensor): ``[K, N]`` activations or gradients in the ``EA`` row layout (groups back to back).
        starts (Tensor): ``[G]`` int32 first row of each group (``ExpertRows.starts``).
        counts (Tensor): ``[G]`` int32 rows of each group, multiples of 128 (``ExpertRows.compute_counts``).
        x_scales (Tensor | None): Per-tile scales of an fp8 ``x``; ``None`` for a bf16 ``x``.

    Returns:
        tuple[Tensor, Tensor]: ``(out, scales)``; ``out`` is flat fp8 e4m3 with ``N * k_alloc`` elements and ``scales``
        is fp32 ``[N, k_alloc // 128]`` with ``k_alloc = ceil(K / 128) * 128``.
    """
    k_rows, n = x.shape
    k_alloc = triton.cdiv(k_rows, 128) * 128
    out = x.new_empty((n * k_alloc,), dtype=torch.float8_e4m3fn)
    scales = x.new_empty((n, k_alloc // 128), dtype=torch.float32)
    if k_rows == 0 or n == 0:
        return out, scales
    finfo = torch.finfo(torch.float8_e4m3fn)
    if x_scales is not None:
        assert x_scales.shape == (k_rows, n // 128) and x_scales.stride(1) == 1, (
            "x_scales must be row-major [K, N/128]"
        )
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]), starts.shape[0])  # noqa: E731
    wrap_triton(_trans_quant_kgrouped_kernel)[grid](
        x,
        out,
        scales,
        starts,
        counts,
        x_scales if x_scales is not None else scales,
        x.stride(0),
        x.stride(1),
        scales.stride(0),
        x_scales.stride(0) if x_scales is not None else 0,
        fmax=finfo.max,
        fmin=finfo.min,
        N=n,
        BLOCK_N=64,
        HAS_IN_SCALE=x_scales is not None,
    )
    return out, scales


def static_ks_host(k_alloc: int, num_groups: int) -> list[int]:
    """Host-side ``ks`` for DeepGEMM's k-grouped GEMM on a static capacity.

    DeepGEMM's kernel reads the real per-group K from the device ``ks_tensor``; the host list only sizes the TMA
    descriptors and steers the kernel configuration (``max(ks)`` is the expected K per group, so a single huge entry
    picks a pipeline whose shared memory exceeds the SM90 limit). A balanced list summing to the allocated K keeps
    the call free of host syncs and the configuration sane.

    Args:
        k_alloc (int): Allocated K (``scales.shape[1] * 128``), a multiple of 128.
        num_groups (int): Number of groups (local experts).

    Returns:
        list[int]: ``num_groups`` multiples of 128 summing to ``k_alloc``.
    """
    base = (k_alloc // num_groups) // 128 * 128
    ks = [base] * num_groups
    ks[-1] += k_alloc - base * num_groups
    return ks
