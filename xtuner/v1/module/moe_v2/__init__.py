# Copyright (c) OpenMMLab. All rights reserved.
"""Unified EP contract (v0) and the contract-based MoE path.

Nothing here modifies the legacy dispatcher / ``MoEBlock`` code paths; ``MoEConfig.dispatcher == "deepep_v2"``
(or an explicit ``moe_v2_cfg``) switches a model onto :class:`MoEDecoderLayerV2`.
"""

from .config import MoEV2Config, validate_moe_v2_config
from .contracts import (
    DispatcherCaps,
    EPCall,
    EPExecutionRuntime,
    ExpertBatch,
    ExpertRows,
    ExpertWeights,
    LayerEPExecution,
    MoESpec,
    NoOpEPExecutionRuntime,
    NoOpLayerEPExecution,
    rows_from_counts,
    rows_from_psum,
    rows_psum,
)
from .decoder_layer import MoEDecoderLayerV2, make_v2_decoder_cls
from .experts import GroupedExpertsV2


__all__ = [
    "DispatcherCaps",
    "EPCall",
    "EPExecutionRuntime",
    "ExpertBatch",
    "ExpertRows",
    "ExpertWeights",
    "GroupedExpertsV2",
    "LayerEPExecution",
    "MoEDecoderLayerV2",
    "make_v2_decoder_cls",
    "MoESpec",
    "MoEV2Config",
    "NoOpEPExecutionRuntime",
    "NoOpLayerEPExecution",
    "rows_from_counts",
    "rows_from_psum",
    "rows_psum",
    "validate_moe_v2_config",
]
