"""Bit-exact parity of the contract-based decoder (``MoEDecoderLayerV2`` + legacy adapter) against the legacy decoder.

Both layers share weights and inputs; outputs, input gradients and every parameter gradient must be identical for the
``E0`` layout, because the adapter only re-describes the legacy ``tokens_per_expert`` and the GEMM kernels are the same.
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.module.moe_v2 import MoEV2Config

from . import parity_tools as pt


pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestMoEDecoderV2ParitySingleRank:
    def test_ep1_naive_dispatcher_bit_exact(self):
        v2_cfg = MoEV2Config(
            dispatcher="legacy", legacy_dispatcher=None, gemm_backend="triton", expert_alignment=1, check_rows=True
        )
        legacy, v2 = pt.build_pair(ep_mesh=None, legacy_dispatcher=None, v2_cfg=v2_cfg)
        ref = pt.run_layer(legacy, *pt.build_inputs(1, seq_len=64))
        got = pt.run_layer(v2, *pt.build_inputs(1, seq_len=64))
        pt.compare(ref, got, exact=True, tag="ep1")


class TestMoEDecoderV2ParityEP(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def _check_parity(self, legacy_name: str, *, fp8: bool) -> None:
        self.create_pg("cuda")
        os.environ["LOCAL_RANK"] = str(dist.get_rank() % torch.cuda.device_count())
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # DeepEP >= 2 (also its legacy Buffer) rejects deterministic mode with fill_uninitialized_memory on.
        torch.utils.deterministic.fill_uninitialized_memory = False
        ep_mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("ep",))
        f8 = Float8Config(scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE) if fp8 else None
        v2_cfg = MoEV2Config(
            dispatcher="legacy",
            legacy_dispatcher=legacy_name,
            gemm_backend="adaptive_gemm" if fp8 else "triton",
            expert_alignment=1,
            check_rows=True,
        )
        legacy, v2 = pt.build_pair(ep_mesh=ep_mesh, legacy_dispatcher=legacy_name, v2_cfg=v2_cfg, float8_cfg=f8)
        for nmb in (1, 2):
            ref = pt.run_layer(legacy, *pt.build_inputs(nmb, seq_len=256 if fp8 else 64))
            got = pt.run_layer(v2, *pt.build_inputs(nmb, seq_len=256 if fp8 else 64))
            pt.compare(ref, got, exact=True, tag=f"{legacy_name} fp8={fp8} mb={nmb}")

    def test_all2all_bf16(self):
        self._check_parity("all2all", fp8=False)

    def test_all2all_fp8(self):
        self._check_parity("all2all", fp8=True)

    def test_deepep_v1_bf16(self):
        pytest.importorskip("deep_ep")
        self._check_parity("deepep", fp8=False)
