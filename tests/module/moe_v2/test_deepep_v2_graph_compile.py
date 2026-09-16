"""CUDA-graph capture and ``torch.compile`` coverage of the DeepEP V2 static-shape path.

The MoE segment (dispatch -> experts -> combine, forward + backward) is graphed with
``torch.cuda.make_graphed_callables``; this only works because ``cpu_sync=False`` keeps every shape static and all
counts on the GPU. The expert block is additionally compiled with ``fullgraph=True`` the same way the model does it.
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.module.moe_v2 import MoEV2Config
from xtuner.v1.module.moe_v2.experts import GroupedExpertsV2

from . import parity_tools as pt


pytestmark = pytest.mark.gpu
deep_ep = pytest.importorskip("deep_ep")
if not hasattr(deep_ep, "ElasticBuffer"):
    pytest.skip("DeepEP V2 (ElasticBuffer) is not installed", allow_module_level=True)


class _MoESegment(nn.Module):
    """dispatch -> experts -> combine of a V2 layer with tensor-only inputs (what a graph can capture)."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        layer = self.layer
        [hidden], [call] = layer.ep_exec.prepare_layer_inputs([hidden])
        router_results = {"topk_ids": topk_ids, "topk_weights": topk_weights, "topkens_per_expert": None}
        pre, topk_weights = layer._pre_dispatch_stage(hidden, router_results, call, async_op=False)
        dispatched, batch = layer._dispatch_stage(pre, topk_weights, call, async_op=False)
        out = layer._experts_stage(batch, call)
        pre_c = layer.dispatcher.combine_preprocess(
            hidden_states=out, pre_dispatched=pre, dispatched=dispatched, layer_state=call
        )
        combined = layer.dispatcher.combine(
            pre_dispatched=pre, dispatched=dispatched, pre_combined=pre_c, layer_state=call
        )
        return layer.dispatcher.combine_postprocess(
            pre_dispatched=pre, dispatched=dispatched, pre_combined=pre_c, combined=combined, layer_state=call
        )


class TestDeepEPV2GraphAndCompile(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def _layer(self, *, gemm: str, align: int, compile_experts: bool):
        self.create_pg("cuda")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # The deterministic test harness turns on fill_uninitialized_memory, which DeepEP V2 rejects.
        torch.utils.deterministic.fill_uninitialized_memory = False
        ep_mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("ep",))
        cfg = MoEV2Config(
            dispatcher="deepep_v2", gemm_backend=gemm, expert_alignment=align, cpu_sync=False, max_tokens_per_rank=256
        )
        _, v2 = pt.build_pair(ep_mesh=ep_mesh, legacy_dispatcher="all2all", v2_cfg=cfg)
        if compile_experts:
            GroupedExpertsV2.forward = torch.compile(
                GroupedExpertsV2.forward.__wrapped__
                if hasattr(GroupedExpertsV2.forward, "__wrapped__")
                else GroupedExpertsV2.forward,
                fullgraph=True,
            )  # type: ignore[method-assign]
        return v2

    def _inputs(self, seq: int, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        hidden = torch.randn(seq, pt.HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g, requires_grad=True)
        topk_ids = torch.stack(
            [torch.randperm(pt.EXPERTS, device="cuda", generator=g)[: pt.TOPK] for _ in range(seq)]
        ).to(torch.int64)
        topk_weights = torch.softmax(torch.randn(seq, pt.TOPK, device="cuda", generator=g), dim=-1).requires_grad_()
        return hidden, topk_ids, topk_weights

    def test_static_segment_is_cuda_graph_capturable(self):
        os.environ["EP_AVOID_RECORD_STREAM"] = "1"
        layer = self._layer(gemm="triton", align=1, compile_experts=False)
        seg = _MoESegment(layer)
        hidden, ids, w = self._inputs(256, seed=1)
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
            h2, i2, w2 = self._inputs(256, seed=seed)
            out = seg(h2, i2, w2)
            out.float().pow(2).mean().backward()
            return out.detach().clone(), h2.grad.clone(), w2.grad.clone()

        for seed in (1, 2):
            # new routing / activations written into the static input buffers, then replay
            h2, i2, w2 = self._inputs(256, seed=seed)
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
        layer = self._layer(gemm="triton", align=1, compile_experts=True)
        try:
            for seq in (256, 128):
                hidden, ids, w = self._inputs(seq, seed=3)
                seg = _MoESegment(layer)
                out = seg(hidden, ids, w)
                out.float().pow(2).mean().backward()
                assert torch.isfinite(out).all()
        finally:
            GroupedExpertsV2.forward = (
                GroupedExpertsV2.forward._torchdynamo_orig_callable
                if hasattr(GroupedExpertsV2.forward, "_torchdynamo_orig_callable")
                else GroupedExpertsV2.forward
            )  # type: ignore[method-assign]
        dist.barrier()
