"""DeepEP V2 dispatcher through the contract path, checked against the legacy all2all decoder.

Not bit-exact by construction (DeepEP V2 reduces partial sums in a different order and applies routing weights
outside the kernel), so the checks are tolerance based; every mode also verifies the row-layout invariants.
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
deep_ep = pytest.importorskip("deep_ep")
if not hasattr(deep_ep, "ElasticBuffer"):
    pytest.skip("DeepEP V2 (ElasticBuffer) is not installed", allow_module_level=True)


class TestDeepEPV2Decoder(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def _check(self, *, fp8: bool, fp8_dispatch: bool, cpu_sync: bool, align: int, gemm: str, seq: int = 256) -> None:
        self.create_pg("cuda")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # The deterministic test harness turns on fill_uninitialized_memory, which DeepEP V2 rejects.
        torch.utils.deterministic.fill_uninitialized_memory = False
        ep_mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("ep",))
        f8 = Float8Config(scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE) if fp8 else None
        v2_cfg = MoEV2Config(
            dispatcher="deepep_v2",
            gemm_backend=gemm,
            expert_alignment=align,
            cpu_sync=cpu_sync,
            fp8_dispatch=fp8_dispatch,
            max_tokens_per_rank=seq,
            check_rows=cpu_sync,
        )
        legacy, v2 = pt.build_pair(ep_mesh=ep_mesh, legacy_dispatcher="all2all", v2_cfg=v2_cfg, float8_cfg=f8)
        for nmb in (1, 2):
            ref = pt.run_layer(legacy, *pt.build_inputs(nmb, seq_len=seq))
            got = pt.run_layer(v2, *pt.build_inputs(nmb, seq_len=seq))
            pt.compare(
                ref,
                got,
                exact=False,
                rtol=3e-2,
                atol=3e-2,
                tag=f"fp8={fp8} cpu_sync={cpu_sync} A={align} {gemm} mb={nmb}",
            )
        dist.barrier()

    def test_bf16_cpu_sync_e0_triton(self):
        self._check(fp8=False, fp8_dispatch=False, cpu_sync=True, align=1, gemm="triton")

    def test_bf16_static_e0_triton(self):
        self._check(fp8=False, fp8_dispatch=False, cpu_sync=False, align=1, gemm="triton")

    def test_bf16_cpu_sync_ea128_deepgemm(self):
        pytest.importorskip("deep_gemm")
        self._check(fp8=False, fp8_dispatch=False, cpu_sync=True, align=128, gemm="deepgemm")

    def test_bf16_static_ea128_deepgemm(self):
        pytest.importorskip("deep_gemm")
        self._check(fp8=False, fp8_dispatch=False, cpu_sync=False, align=128, gemm="deepgemm")

    def test_fp8_cpu_sync_e0_adaptive(self):
        self._check(fp8=True, fp8_dispatch=False, cpu_sync=True, align=1, gemm="adaptive_gemm")

    def test_fp8_dispatch_static_ea128_deepgemm(self):
        pytest.importorskip("deep_gemm")
        self._check(fp8=True, fp8_dispatch=True, cpu_sync=False, align=128, gemm="deepgemm")
