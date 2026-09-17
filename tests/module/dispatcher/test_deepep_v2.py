"""DeepEP V2 dispatcher in ``MoEDecoderLayer``, checked against the all2all dispatcher with identical weights.

Not bit-exact by construction (DeepEP V2 reduces partial sums in a different order and applies routing weights
outside the kernel), so the layer checks are tolerance based. The static-shape mode is additionally captured and
replayed as one CUDA graph (forward + backward), and the expert block is compiled with ``fullgraph=True``.
"""

import inspect
import os

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.module import GreedyRouterConfig, MHAConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEDecoderLayer
from xtuner.v1.module.dispatcher.deepep_v2 import DeepEPV2Config


pytestmark = pytest.mark.gpu
deep_ep = pytest.importorskip("deep_ep")
if not hasattr(deep_ep, "ElasticBuffer"):
    pytest.skip("DeepEP V2 (ElasticBuffer) is not installed", allow_module_level=True)

HIDDEN, INTER, MOE_INTER, EXPERTS, TOPK, HEADS, KV_HEADS, HEAD_DIM = 256, 512, 128, 8, 2, 4, 2, 64


def _layer(*, ep_mesh, dispatcher: str, cfg: DeepEPV2Config | None, fp8: bool) -> MoEDecoderLayer:
    torch.manual_seed(0)
    layer = MoEDecoderLayer(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        moe_intermediate_size=MOE_INTER,
        hidden_act="silu",
        num_experts_per_tok=TOPK,
        n_routed_experts=EXPERTS,
        n_shared_experts=1,
        attention_config=MHAConfig(
            num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM, qk_norm=True
        ),
        router_config=GreedyRouterConfig(scoring_func="softmax", router_scaling_factor=1.0, norm_topk_prob=True),
        moe_act_fn_cfg=MoEActFnConfig(),
        float8_cfg=Float8Config(scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE) if fp8 else None,
        layer_idx=0,
        dispatcher=dispatcher,  # type: ignore[arg-type]
        deepep_v2_cfg=cfg,
        ep_mesh=ep_mesh,
    ).to(device="cuda", dtype=torch.bfloat16)
    g = torch.Generator(device="cpu").manual_seed(1234)
    with torch.no_grad():
        for _, p in sorted(layer.named_parameters(), key=lambda kv: kv[0]):
            local = p.to_local() if hasattr(p, "to_local") else p
            full = torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.02
            if hasattr(p, "to_local"):
                from torch.distributed.tensor import distribute_tensor

                full = distribute_tensor(full.to(local.device), p.device_mesh, p.placements).to_local()
            local.copy_(full.to(local.dtype))
    return layer


def _inputs(nmb: int, seq_len: int):
    torch.manual_seed(7)
    hidden = [
        torch.randn(1, seq_len, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(nmb)
    ]
    pos = [
        (
            torch.ones(1, seq_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16),
            torch.zeros(1, seq_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16),
        )
        for _ in range(nmb)
    ]
    ctx = [SequenceContext.from_input_ids((torch.arange(seq_len).view(1, -1),), device="cuda") for _ in range(nmb)]
    return hidden, pos, ctx


def _run(layer: MoEDecoderLayer, nmb: int, seq_len: int):
    hidden, pos, ctx = _inputs(nmb, seq_len)
    if nmb == 1:
        outs = [layer(hidden[0], seq_ctx=ctx[0], position_embeddings=pos[0])["hidden_states"]]
    else:
        outs = list(layer(hidden, seq_ctx=ctx, position_embeddings=pos)["hidden_states"])
    sum((o.float() ** 2).mean() for o in outs).backward()
    grads = {}
    for name, p in layer.named_parameters():
        if p.grad is not None:
            g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            grads[name] = g.detach().clone()
    return [o.detach() for o in outs], [h.grad.detach() for h in hidden], grads


def _assert_close(ref, got, *, tol: float, tag: str) -> None:
    ref_outs, ref_in, ref_g = ref
    got_outs, got_in, got_g = got
    for i, (a, b) in enumerate(zip(ref_outs, got_outs)):
        torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol, msg=lambda m: f"{tag} out[{i}]: {m}")
    for i, (a, b) in enumerate(zip(ref_in, got_in)):
        torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol, msg=lambda m: f"{tag} dX[{i}]: {m}")
    assert ref_g.keys() == got_g.keys()
    for k in ref_g:
        torch.testing.assert_close(
            ref_g[k].float(), got_g[k].float(), rtol=tol, atol=tol, msg=lambda m: f"{tag} grad {k}: {m}"
        )


class _Setup(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def _mesh(self):
        self.create_pg("cuda")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # The deterministic test harness turns on fill_uninitialized_memory, which DeepEP V2 rejects.
        torch.utils.deterministic.fill_uninitialized_memory = False
        return init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("ep",))


class TestDeepEPV2DecoderLayer(_Setup):
    def _check(
        self,
        *,
        fp8: bool,
        fp8_dispatch: bool,
        cpu_sync: bool,
        align: int,
        seq: int = 256,
        capacity_factor: float | None = None,
    ) -> None:
        ep_mesh = self._mesh()
        cfg = DeepEPV2Config(
            expert_alignment=align,  # type: ignore[arg-type]
            cpu_sync=cpu_sync,
            fp8_dispatch=fp8_dispatch,
            max_tokens_per_rank=seq,
            check_rows=cpu_sync,
            capacity_factor=capacity_factor,
        )
        ref_layer = _layer(ep_mesh=ep_mesh, dispatcher="all2all", cfg=None, fp8=fp8)
        v2_layer = _layer(ep_mesh=ep_mesh, dispatcher="deepep_v2", cfg=cfg, fp8=fp8)
        for nmb in (1, 2):
            ref = _run(ref_layer, nmb, seq)
            got = _run(v2_layer, nmb, seq)
            _assert_close(ref, got, tol=3e-2, tag=f"fp8={fp8} cpu_sync={cpu_sync} A={align} mb={nmb}")
        dist.barrier()

    def test_bf16_cpu_sync_unpadded(self):
        self._check(fp8=False, fp8_dispatch=False, cpu_sync=True, align=1)

    def test_bf16_static_unpadded(self):
        self._check(fp8=False, fp8_dispatch=False, cpu_sync=False, align=1)

    def test_bf16_cpu_sync_aligned_deepgemm(self):
        pytest.importorskip("deep_gemm")
        self._check(fp8=False, fp8_dispatch=False, cpu_sync=True, align=128)

    def test_bf16_static_aligned_deepgemm(self):
        pytest.importorskip("deep_gemm")
        self._check(fp8=False, fp8_dispatch=False, cpu_sync=False, align=128)

    def test_fp8_cpu_sync_unpadded_adaptive_gemm(self):
        self._check(fp8=True, fp8_dispatch=False, cpu_sync=True, align=1)

    def test_fp8_dispatch_static_aligned_deepgemm(self):
        pytest.importorskip("deep_gemm")
        self._check(fp8=True, fp8_dispatch=True, cpu_sync=False, align=128)

    def test_bf16_static_capped_capacity(self):
        # 1.5x the balanced expectation instead of the ep_size x worst case (needs the capped-capacity DeepEP build).
        pytest.importorskip("deep_gemm")
        if "num_max_expanded_tokens" not in inspect.signature(deep_ep.ElasticBuffer.dispatch).parameters:
            pytest.skip("DeepEP without the capped receive capacity")
        self._check(fp8=False, fp8_dispatch=False, cpu_sync=False, align=128, capacity_factor=1.5)


class TestDeepEPV2CappedCapacity(_Setup):
    def test_overflow_is_flagged_and_raised(self):
        """Rows beyond the capped capacity are dropped, the device flag is set, and the next call raises."""
        if "num_max_expanded_tokens" not in inspect.signature(deep_ep.ElasticBuffer.dispatch).parameters:
            pytest.skip("DeepEP without the capped receive capacity")
        ep_mesh = self._mesh()
        cfg = DeepEPV2Config(expert_alignment=1, cpu_sync=False, max_tokens_per_rank=256, capacity_factor=0.05)
        layer = _layer(ep_mesh=ep_mesh, dispatcher="deepep_v2", cfg=cfg, fp8=False)
        d = layer.dispatcher
        hidden, ids, w = TestDeepEPV2GraphAndCompile._segment_inputs(256, seed=5)
        pre = d.dispatch_preprocess(hidden_states=hidden, topk_ids=ids, topk_weights=w)
        dispatched = d.dispatch(pre_dispatched=pre, topk_weights=w)
        post = d.dispatch_postprocess(pre_dispatched=pre, dispatched=dispatched)  # enqueues the flag copy
        torch.cuda.synchronize()
        assert post["hidden_states"].shape[0] == d._capacity + (cfg.expert_alignment - 1) * d.num_local_experts
        assert int(d._overflow_flag.item()) == 1
        with pytest.raises(RuntimeError, match="capacity overflow"):
            d.dispatch_postprocess(pre_dispatched=pre, dispatched=dispatched)
        dist.barrier()


class _MoESegment(nn.Module):
    """dispatch -> experts -> combine of one layer with tensor-only inputs (what a CUDA graph can capture)."""

    def __init__(self, layer: MoEDecoderLayer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        d = self.layer.dispatcher
        pre = d.dispatch_preprocess(hidden_states=hidden, topk_ids=topk_ids, topk_weights=topk_weights)
        dispatched = d.dispatch(pre_dispatched=pre, topk_weights=topk_weights)
        post = d.dispatch_postprocess(pre_dispatched=pre, dispatched=dispatched)
        out = self.layer.experts(post)
        pre_c = d.combine_preprocess(
            hidden_states=out, pre_dispatched=pre, dispatched=dispatched, post_dispatched=post
        )
        combined = d.combine(pre_dispatched=pre, dispatched=dispatched, post_dispatched=post, pre_combined=pre_c)
        return d.combine_postprocess(
            pre_dispatched=pre, dispatched=dispatched, post_dispatched=post, pre_combined=pre_c, combined=combined
        )["hidden_states"]


class TestDeepEPV2GraphAndCompile(_Setup):
    def _static_layer(self, *, align: int) -> MoEDecoderLayer:
        ep_mesh = self._mesh()
        cfg = DeepEPV2Config(expert_alignment=align, cpu_sync=False, max_tokens_per_rank=256)  # type: ignore[arg-type]
        return _layer(ep_mesh=ep_mesh, dispatcher="deepep_v2", cfg=cfg, fp8=False)

    @staticmethod
    def _segment_inputs(seq: int, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        hidden = torch.randn(seq, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
        topk_ids = torch.stack([torch.randperm(EXPERTS, device="cuda", generator=g)[:TOPK] for _ in range(seq)])
        topk_weights = torch.softmax(torch.randn(seq, TOPK, device="cuda", generator=g), dim=-1).requires_grad_()
        return hidden, topk_ids.to(torch.int64), topk_weights

    def test_static_segment_is_cuda_graph_capturable(self):
        os.environ["EP_AVOID_RECORD_STREAM"] = "1"
        seg = _MoESegment(self._static_layer(align=1))
        hidden, ids, w = self._segment_inputs(256, seed=1)
        static_out = [None]

        def step():
            out = seg(hidden, ids, w)
            out.float().pow(2).mean().backward()
            static_out[0] = out

        # warm-up on a side stream (buffer creation, JIT, allocator), then capture forward + backward in one graph
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        # The graph accumulates into the gradient buffers that exist at capture time: keep them, zero in place.
        hidden.grad.zero_()
        w.grad.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        torch.cuda.synchronize()

        def eager_reference(seed: int):
            h2, i2, w2 = self._segment_inputs(256, seed=seed)
            out = seg(h2, i2, w2)
            out.float().pow(2).mean().backward()
            return out.detach().clone(), h2.grad.clone(), w2.grad.clone()

        for seed in (1, 2):
            # new routing / activations written into the static input buffers, then replay
            h2, i2, w2 = self._segment_inputs(256, seed=seed)
            with torch.no_grad():
                hidden.copy_(h2)
                ids.copy_(i2)
                w.copy_(w2)
                hidden.grad.zero_()
                w.grad.zero_()
            graph.replay()
            torch.cuda.synchronize()
            ref_out, ref_dx, ref_dw = eager_reference(seed)
            torch.testing.assert_close(static_out[0].float(), ref_out.float(), rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(hidden.grad.float(), ref_dx.float(), rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(w.grad.float(), ref_dw.float(), rtol=2e-2, atol=2e-2)
        dist.barrier()

    def test_experts_compile_fullgraph_static_and_dynamic(self):
        layer = self._static_layer(align=1)
        layer.experts.forward = torch.compile(layer.experts.forward, fullgraph=True)  # type: ignore[method-assign]
        seg = _MoESegment(layer)
        for seq in (256, 128):
            hidden, ids, w = self._segment_inputs(seq, seed=3)
            out = seg(hidden, ids, w)
            out.float().pow(2).mean().backward()
            assert torch.isfinite(out).all()
        dist.barrier()
