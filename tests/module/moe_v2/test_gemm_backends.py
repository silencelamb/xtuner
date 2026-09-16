"""Single-GPU checks of the contract GEMM backends on synthetic E0 / EA-128 batches against a fp32 reference."""

import pytest
import torch

from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig
from xtuner.v1.module.moe_v2.contracts import ExpertBatch, ExpertWeights, rows_from_counts, rows_from_psum
from xtuner.v1.module.moe_v2.experts import GroupedExpertsV2


pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

H, I, E = 256, 128, 4


def _experts(gemm: str, fp8: bool, alignment: int) -> GroupedExpertsV2:
    torch.manual_seed(0)
    f8 = Float8Config(scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE) if fp8 else None
    m = (
        GroupedExpertsV2(
            gemm_backend=gemm,
            expert_alignment=alignment,
            hidden_size=H,
            moe_intermediate_size=I,
            n_routed_experts=E,
            float8_cfg=f8,
            moe_act_fn_cfg=MoEActFnConfig(),
        )
        .cuda()
        .bfloat16()
    )
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.02)
    return m


def _batch(valid: list[int], alignment: int, fp8_input: bool, tail: int = 0):
    """Build hidden [capacity, H] with zero padding rows and the matching rows; capacity adds a spare tail."""
    valid_t = torch.tensor(valid, device="cuda", dtype=torch.int32)
    if alignment == 1:
        rows = rows_from_counts(valid_t)
    else:
        ends = torch.cumsum(((valid_t + alignment - 1) // alignment) * alignment, 0)
        starts = ends - ((valid_t + alignment - 1) // alignment) * alignment
        rows = rows_from_psum(starts + valid_t, valid_t, alignment=alignment, padding_zeroed=True)
    total = int((rows.starts + rows.compute_counts)[-1])
    capacity = total + tail
    x = torch.zeros(capacity, H, device="cuda", dtype=torch.bfloat16)
    valid_mask = torch.zeros(capacity, device="cuda", dtype=torch.bool)
    for s, v in zip(rows.starts.tolist(), valid):
        x[s : s + v] = torch.randn(v, H, device="cuda", dtype=torch.bfloat16)
        valid_mask[s : s + v] = True
    x.requires_grad_()
    hidden = x
    if fp8_input:
        from xtuner.v1.float8.float8_tensor import Float8Tensor
        from xtuner.v1.float8.triton_kernels import per_tile_quant

        data, sf = per_tile_quant(x.detach())
        hidden = Float8Tensor(data, sf, torch.bfloat16, ScalingGranularity.TILEWISE, 128).requires_grad_()
    batch = ExpertBatch(
        hidden_states=hidden,
        hidden_scales=None,
        tokens_per_expert=rows.compute_counts,
        rows=rows,
        probs=None,
        expert_weights=ExpertWeights(),
        tokens_per_expert_cpu=None,
    )
    return batch, x, valid_mask, total


def _reference(m: GroupedExpertsV2, x: torch.Tensor, rows) -> torch.Tensor:
    w13 = m.fused_w1w3.weight.detach().float().view(E, 2 * I, H)
    w2 = m.fused_w2.weight.detach().float().view(E, H, I)
    out = torch.zeros(x.shape[0], H, device="cuda", dtype=torch.float32)
    starts, counts = (
        rows.starts.tolist(),
        (rows.valid_counts if rows.valid_counts is not None else rows.compute_counts).tolist(),
    )
    for g, (s, c) in enumerate(zip(starts, counts)):
        h = x[s : s + c].float() @ w13[g].T
        gate, up = h.chunk(2, dim=-1)
        out[s : s + c] = (torch.nn.functional.silu(gate) * up) @ w2[g].T
    return out


@pytest.mark.parametrize(
    "gemm,fp8,alignment,fp8_input",
    [
        ("triton", False, 1, False),
        ("triton", False, 128, False),
        ("adaptive_gemm", True, 1, False),
        ("adaptive_gemm", True, 128, False),
        ("adaptive_gemm", True, 128, True),
        ("deepgemm", False, 128, False),
        ("deepgemm", True, 128, False),
        ("deepgemm", True, 128, True),
    ],
)
def test_backend_matches_reference(gemm, fp8, alignment, fp8_input):
    if gemm == "deepgemm":
        dg = pytest.importorskip("deep_gemm")
        if not hasattr(dg, "m_grouped_bf16_gemm_nt_contiguous"):
            pytest.skip("DeepGEMM without psum layout")
    m = _experts(gemm, fp8, alignment)
    # Static-shape-friendly backends must tolerate a worst-case capacity tail that belongs to no expert.
    tail = 128 if m.gemm.caps.static_shape_friendly else 0
    batch, x, valid_mask, total = _batch([200, 0, 130, 77], alignment, fp8_input, tail=tail)
    out = m(batch)
    ref = _reference(m, x.detach(), batch["rows"])
    tol = 6e-2 if fp8 else 2e-2
    torch.testing.assert_close(out.float()[valid_mask], ref[valid_mask], rtol=tol, atol=tol)
    # Rows inside the compute extent must be finite; alignment padding rows must be exact zeros (they are reduced
    # over by the counts-based wgrad kernels). Rows past the compute extent (the capacity tail) are unspecified.
    inside = out[:total]
    assert torch.isfinite(inside).all()
    assert torch.all(inside[~valid_mask[:total]].float() == 0), (
        "alignment padding rows of the expert output are not zero"
    )
    (out.float()[valid_mask] ** 2).mean().backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


@pytest.mark.parametrize("fp8_input", [False, True])
def test_deepgemm_fp8_backend_compiles_fullgraph(fp8_input):
    """The whole expert forward *and* its backward must be traceable by dynamo (raw pybind calls are not)."""
    dg = pytest.importorskip("deep_gemm")
    if not hasattr(dg, "m_grouped_bf16_gemm_nt_contiguous"):
        pytest.skip("DeepGEMM without psum layout")
    m = _experts("deepgemm", True, 128)
    m.forward = torch.compile(m.forward, fullgraph=True, dynamic=False)
    batch, x, valid_mask, total = _batch([200, 0, 130, 77], 128, fp8_input, tail=128)
    out = m(batch)
    (out.float()[valid_mask] ** 2).mean().backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
