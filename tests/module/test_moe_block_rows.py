"""Single-GPU checks of ``MoEBlock`` over unpadded and 128-aligned row layouts against a fp32 reference.

Covers the legacy kernels (Triton BF16, AdaptiveGEMM FP8) and the DeepGEMM psum path, with BF16 or pre-quantized
FP8 activations, including alignment padding rows and a spare capacity tail.
"""

import pytest
import torch

from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock
from xtuner.v1.module.dispatcher import ExpertWeights, PostDispatchResult, rows_from_counts, rows_from_psum


pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

H, I, E = 256, 128, 4


def _deepgemm_or_skip():
    dg = pytest.importorskip("deep_gemm")
    if not hasattr(dg, "m_grouped_bf16_gemm_nt_contiguous"):
        pytest.skip("DeepGEMM without psum layout")


def _experts(fp8: bool) -> MoEBlock:
    torch.manual_seed(0)
    f8 = Float8Config(scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE) if fp8 else None
    m = (
        MoEBlock(
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
        aligned = ((valid_t + alignment - 1) // alignment) * alignment
        starts = torch.cumsum(aligned, 0) - aligned
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
    batch = PostDispatchResult(
        hidden_states=hidden,
        hidden_scales=None,
        tokens_per_expert=rows.compute_counts,
        rows=rows,
        expert_weights=ExpertWeights(),
    )
    return batch, x, valid_mask, total


def _reference(m: MoEBlock, x: torch.Tensor, rows) -> torch.Tensor:
    w13 = m.fused_w1w3.weight.detach().float().view(E, 2 * I, H)
    w2 = m.fused_w2.weight.detach().float().view(E, H, I)
    out = torch.zeros(x.shape[0], H, device="cuda", dtype=torch.float32)
    counts = rows.valid_counts if rows.valid_counts is not None else rows.compute_counts
    for g, (s, c) in enumerate(zip(rows.starts.tolist(), counts.tolist())):
        h = x[s : s + c].float() @ w13[g].T
        gate, up = h.chunk(2, dim=-1)
        out[s : s + c] = (torch.nn.functional.silu(gate) * up) @ w2[g].T
    return out


@pytest.mark.parametrize(
    "backend,fp8,alignment,fp8_input",
    [
        ("legacy", False, 1, False),
        ("legacy", False, 128, False),
        ("legacy", True, 1, False),
        ("legacy", True, 128, False),
        ("legacy", True, 128, True),
        ("deepgemm", False, 128, False),
        ("deepgemm", True, 128, False),
        ("deepgemm", True, 128, True),
    ],
)
def test_moe_block_matches_reference(monkeypatch, backend, fp8, alignment, fp8_input):
    if backend == "deepgemm":
        _deepgemm_or_skip()
    monkeypatch.setenv("XTUNER_EXPERT_GEMM_BACKEND", backend)
    m = _experts(fp8)
    # The AdaptiveGEMM FP8 kernels derive M from the activation, so only the other paths get a capacity tail that
    # belongs to no expert (the static, no-cpu-sync dispatch layout).
    tail = 0 if (backend == "legacy" and fp8) else 128
    batch, x, valid_mask, total = _batch([200, 0, 130, 77], alignment, fp8_input, tail=tail)
    out = m(batch)
    ref = _reference(m, x.detach(), batch["rows"])
    tol = 6e-2 if fp8 else 2e-2
    torch.testing.assert_close(out.float()[valid_mask], ref[valid_mask], rtol=tol, atol=tol)
    # Rows inside the compute extent must be finite; alignment padding rows must be exact zeros (the counts-based
    # wgrad kernels reduce over them). Rows past the compute extent (the capacity tail) are unspecified.
    inside = out[:total]
    assert torch.isfinite(inside).all()
    assert torch.all(inside[~valid_mask[:total]].float() == 0), "alignment padding rows are not zero"
    (out.float()[valid_mask] ** 2).mean().backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


@pytest.mark.parametrize("fp8_input", [False, True])
def test_deepgemm_fp8_path_compiles_fullgraph(monkeypatch, fp8_input):
    """The expert forward *and* its backward must be traceable by dynamo (raw DeepGEMM pybind calls are not)."""
    _deepgemm_or_skip()
    monkeypatch.setenv("XTUNER_EXPERT_GEMM_BACKEND", "deepgemm")
    m = _experts(fp8=True)
    m.forward = torch.compile(m.forward, fullgraph=True, dynamic=False)
    batch, x, valid_mask, total = _batch([200, 0, 130, 77], 128, fp8_input, tail=128)
    out = m(batch)
    (out.float()[valid_mask] ** 2).mean().backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
