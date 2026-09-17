"""CPU tests for the unified EP contract in ``xtuner.v1.module.dispatcher.base``: row layouts and identity hooks."""

import pytest
import torch

from xtuner.v1.module.dispatcher import (
    EPCall,
    ExpertWeights,
    NoOpEPExecutionRuntime,
    PostDispatchResult,
    check_rows,
    rows_from_counts,
    rows_from_psum,
    rows_psum,
)


class TestExpertRows:
    def test_rows_from_counts_is_unpadded(self):
        counts = torch.tensor([3, 0, 5, 2], dtype=torch.int64)
        rows = rows_from_counts(counts)
        assert rows.alignment == 1 and rows.padding_zeroed
        assert rows.starts.tolist() == [0, 3, 3, 8]
        assert rows.compute_counts is counts
        assert rows.valid_counts is counts
        check_rows(rows, capacity=10)
        assert rows_psum(rows).tolist() == [3, 3, 8, 10]

    def test_rows_from_psum_matches_expanded_aligned_layout(self):
        # 3 local experts, alignment 128, valid [200, 0, 130] -> starts [0, 256, 256], psum [200, 256, 386]
        valid = torch.tensor([200, 0, 130], dtype=torch.int32)
        psum = torch.tensor([200, 256, 386], dtype=torch.int32)
        rows = rows_from_psum(psum, valid, alignment=128, padding_zeroed=True)
        assert rows.alignment == 128
        assert rows.starts.tolist() == [0, 256, 256]
        assert rows.compute_counts.tolist() == [256, 0, 256]
        assert rows.valid_counts is not None and rows.valid_counts.tolist() == [200, 0, 130]
        assert rows_psum(rows).tolist() == [200, 256, 386]
        check_rows(rows, capacity=512)
        with pytest.raises(AssertionError):
            check_rows(rows, capacity=500)

    def test_check_rows_rejects_gaps(self):
        rows = rows_from_counts(torch.tensor([2, 2]))
        bad = rows._replace(starts=torch.tensor([0, 3]))
        with pytest.raises(AssertionError):
            check_rows(bad, capacity=8)


class TestExecutionHooks:
    def test_noop_hooks_are_identity(self):
        runtime = NoOpEPExecutionRuntime()
        ep_exec = runtime.bind_layer(layer_fqn="layers.0", layer_idx=0, projections=(None, None))
        x = torch.randn(4, 8)
        outs, calls = ep_exec.prepare_layer_inputs([x, x])
        assert outs[0] is x and len(calls) == 2 and calls[1].micro_batch == 1
        h, ids, w = ep_exec.prepare_dispatch(calls[0], x, torch.zeros(4, 2, dtype=torch.long), torch.ones(4, 2))
        assert h is x
        counts = torch.tensor([2, 2])
        batch = PostDispatchResult(
            hidden_states=x,
            hidden_scales=None,
            tokens_per_expert=counts,
            rows=rows_from_counts(counts),
            expert_weights=ExpertWeights(),
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
