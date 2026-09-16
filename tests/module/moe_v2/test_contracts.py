"""CPU tests for the unified EP contract v0 (``xtuner.v1.module.moe_v2.contracts``)."""

import pytest
import torch

from xtuner.v1.module.moe_v2.config import MoEV2Config, validate_moe_v2_config
from xtuner.v1.module.moe_v2.contracts import (
    EPCall,
    ExpertBatch,
    ExpertWeights,
    NoOpEPExecutionRuntime,
    check_rows,
    rows_from_counts,
    rows_from_psum,
    rows_psum,
)


class TestExpertRows:
    def test_rows_from_counts_is_e0(self):
        counts = torch.tensor([3, 0, 5, 2], dtype=torch.int64)
        rows = rows_from_counts(counts)
        assert rows.layout == "E0"
        assert rows.alignment == 1 and rows.padding_zeroed
        assert rows.starts.tolist() == [0, 3, 3, 8]
        assert rows.compute_counts.tolist() == [3, 0, 5, 2]
        assert rows.valid_counts is not None and rows.valid_counts.tolist() == [3, 0, 5, 2]
        assert rows.compute_counts.dtype == torch.int32
        check_rows(rows, capacity=10)
        assert rows_psum(rows).tolist() == [3, 3, 8, 10]

    def test_rows_from_psum_matches_deepep_v2_expand_layout(self):
        # E_local=3, A=128, n=[200, 0, 130] -> starts [0, 256, 256], psum [200, 256, 386]
        valid = torch.tensor([200, 0, 130], dtype=torch.int32)
        psum = torch.tensor([200, 256, 386], dtype=torch.int32)
        rows = rows_from_psum(psum, valid, alignment=128, padding_zeroed=True)
        assert rows.layout == "EA"
        assert rows.starts.tolist() == [0, 256, 256]
        assert rows.compute_counts.tolist() == [256, 0, 256]
        assert rows.valid_counts.tolist() == [200, 0, 130]
        assert rows_psum(rows).tolist() == [200, 256, 386]
        check_rows(rows, capacity=512)
        with pytest.raises(AssertionError):
            check_rows(rows, capacity=500)

    def test_check_rows_rejects_gaps(self):
        rows = rows_from_counts(torch.tensor([2, 2]))
        bad = rows._replace(starts=torch.tensor([0, 3], dtype=torch.int32))
        with pytest.raises(AssertionError):
            check_rows(bad, capacity=8)


class TestExpertBatchAndHooks:
    def test_noop_hooks_are_identity(self):
        runtime = NoOpEPExecutionRuntime()
        ep_exec = runtime.bind_layer(layer_fqn="layers.0.experts", layer_idx=0, projections=(None, None))
        x = torch.randn(4, 8)
        outs, calls = ep_exec.prepare_layer_inputs([x, x])
        assert outs[0] is x and len(calls) == 2 and calls[1].micro_batch == 1
        h, ids, w = ep_exec.prepare_dispatch(calls[0], x, torch.zeros(4, 2, dtype=torch.long), torch.ones(4, 2))
        assert h is x
        counts = torch.tensor([2, 2])
        batch = ExpertBatch(
            hidden_states=x,
            hidden_scales=None,
            tokens_per_expert=counts,
            rows=rows_from_counts(counts),
            probs=None,
            expert_weights=ExpertWeights(),
            tokens_per_expert_cpu=None,
        )
        assert ep_exec.prepare_experts(calls[0], batch) is batch
        assert ep_exec.attach_after_experts(calls[0], x) is x
        assert ep_exec.attach_after_combine(calls[0], x) is x
        runtime.validate_before_fsdp(None)
        runtime.install_after_fsdp(fsdp_root=None, execution_order=[])
        runtime.close()

    def test_epcall_carries_backend_state(self):
        call = EPCall(layer_idx=3, micro_batch=1)
        call.extras["handle"] = object()
        assert call.layer_idx == 3 and "handle" in call.extras


class _Cfg:
    def __init__(self, **kw):
        self.expert_tp_size = 1
        self.moe_bias = False
        self.float8_cfg = None
        self.hidden_size = 2048
        self.num_experts_per_tok = 8
        self.ep_size = 8
        self.n_routed_experts = 128
        self.__dict__.update(kw)


class TestConfigValidation:
    def test_defaults_pass(self):
        validate_moe_v2_config(_Cfg(), MoEV2Config())

    def test_fp8_dispatch_requires_fp8_experts(self):
        with pytest.raises(ValueError, match="fp8_dispatch"):
            validate_moe_v2_config(_Cfg(), MoEV2Config(fp8_dispatch=True))

    def test_hidden_multiple_for_deepep_v2(self):
        with pytest.raises(ValueError, match="256"):
            validate_moe_v2_config(_Cfg(hidden_size=2000), MoEV2Config())

    def test_legacy_requires_name(self):
        with pytest.raises(ValueError, match="legacy_dispatcher"):
            validate_moe_v2_config(_Cfg(), MoEV2Config(dispatcher="legacy"))

    def test_expert_tp_rejected(self):
        with pytest.raises(NotImplementedError):
            validate_moe_v2_config(_Cfg(expert_tp_size=2), MoEV2Config())
