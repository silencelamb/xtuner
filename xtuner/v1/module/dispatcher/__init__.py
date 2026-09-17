import os
from typing import TYPE_CHECKING, Literal


XTUNER_DISPATCHER_DEBUG = os.getenv("XTUNER_DISPATCHER_DEBUG", "0") == "1"

import torch.distributed as dist

from xtuner.v1.utils import get_logger, log_rank0

from .agrs import MoEAGRSDispatcher
from .base import (
    CombineResult,
    DispacherInterface,
    DispatcherCaps,
    DispatchResult,
    EPCall,
    EPExecutionRuntime,
    ExpertRows,
    ExpertWeights,
    LayerEPExecution,
    Layout,
    NaiveDispatcher,
    NoOpEPExecutionRuntime,
    NoOpLayerEPExecution,
    PostCombineResult,
    PostDispatchResult,
    PreCombineResult,
    PreDispatchResult,
    check_rows,
    rows_from_counts,
    rows_from_psum,
    rows_psum,
)
from .torch_all2all import TorchAll2AllDispatcher


if TYPE_CHECKING:
    from .deepep_v2 import DeepEPV2Config

logger = get_logger()


# TODO: (yehaochen) This interface declaration does not follow the Liskov Substitution Principle.
# Maybe we should find a better way to handle the dispatchers.
def build_dispatcher(
    dispatcher: Literal["deepep", "all2all", "agrs", "deepep_v2"] | None,
    n_routed_experts: int,
    ep_group: dist.ProcessGroup | None = None,
    tp_group: dist.ProcessGroup | None = None,
    ep_tp_group: dist.ProcessGroup | None = None,
    training_dtype: Literal["bf16", "fp8"] = "bf16",
    generate_dtype: Literal["bf16", "fp8"] = "bf16",
    *,
    hidden_size: int | None = None,
    num_experts_per_tok: int | None = None,
    deepep_v2_cfg: "DeepEPV2Config | None" = None,
) -> DispacherInterface:
    if ep_group is None or ep_group.size() == 1:
        if dispatcher is not None:
            log_rank0.warning(f"{dispatcher} will not be used because the ep group is None.")
        return NaiveDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            tp_group=tp_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )  # type: ignore[return-value]

    if dispatcher is None:
        dispatcher = "all2all"

    if dispatcher == "deepep":
        # TODO: remove ignore
        from .deepep import DeepEPDispatcher  # type: ignore[attr-defined]

        assert ep_group is not None, "DeepEPDispatcher requires a non-null process group."
        # When expert TP is enabled, fuse EP dispatch + TP replication into a single DeepEP
        # collective: the dispatcher operates on the combined (ep × tp) group with each
        # physical expert virtualized into ``tp_size`` copies (see ``DeepEPDispatcher``).
        tp_size = tp_group.size() if tp_group is not None else 1
        if tp_size > 1:
            assert ep_tp_group is not None, (
                "DeepEPDispatcher with expert TP requires the combined (ep × tp) process group; "
                "pass ``ep_tp_group`` from ``ep_tp_mesh._flatten().get_group()``."
            )
            process_group = ep_tp_group
        else:
            process_group = ep_group
        # TODO: remove type ignore here
        return DeepEPDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=process_group,
            tp_size=tp_size,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )  # type: ignore
    elif dispatcher == "deepep_v2":
        from .deepep_v2 import DeepEPV2Config, DeepEPV2Dispatcher

        assert ep_group is not None, "DeepEPV2Dispatcher requires a non-null process group."
        assert hidden_size is not None and num_experts_per_tok is not None, (
            "DeepEPV2Dispatcher needs hidden_size and num_experts_per_tok to size its buffer."
        )
        if tp_group is not None and tp_group.size() > 1:
            raise NotImplementedError("DeepEPV2Dispatcher does not support expert TP yet.")
        return DeepEPV2Dispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            hidden_size=hidden_size,
            num_experts_per_tok=num_experts_per_tok,
            cfg=deepep_v2_cfg or DeepEPV2Config(),
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )  # type: ignore[return-value]
    elif dispatcher == "all2all":
        assert ep_group is not None, "TorchAll2AllDispatcher requires a non-null ep_group."
        return TorchAll2AllDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            tp_group=tp_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )  # type: ignore[return-value]
    elif dispatcher == "agrs":
        assert ep_group is not None, "MoEAGRSDispatcher requires a non-null process group."
        return MoEAGRSDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )  # type: ignore[return-value]
    else:
        raise ValueError(
            f"Unknown dispatcher name: {dispatcher}, name must be one of 'deepep', 'deepep_v2', 'all2all' or 'agrs'."
        )


__all__ = [
    "DispacherInterface",
    "NaiveDispatcher",
    "TorchAll2AllDispatcher",
    "MoEAGRSDispatcher",
    "build_dispatcher",
    "PreDispatchResult",
    "DispatchResult",
    "PostDispatchResult",
    "PreCombineResult",
    "CombineResult",
    "PostCombineResult",
    "DispatcherCaps",
    "EPCall",
    "EPExecutionRuntime",
    "ExpertRows",
    "ExpertWeights",
    "LayerEPExecution",
    "Layout",
    "NoOpEPExecutionRuntime",
    "NoOpLayerEPExecution",
    "check_rows",
    "rows_from_counts",
    "rows_from_psum",
    "rows_psum",
]
