"""SFT benchmark config for the EP dispatchers (DeepEP V2 vs the legacy all2all / DeepEP v1).

Everything is env-driven so one config serves both reference workloads:

* GLM-5.2-30B (MTP, FP8):  ``MODEL_PATH=/mnt/nvme1n1/models/GLM-5.2-30B CHAT_TEMPLATE=glm5.2 FP8=1``
* Qwen3-30B-A3B (BF16):    ``MODEL_PATH=/mnt/nvme1n1/models/Qwen3-30B-A3B CHAT_TEMPLATE=qwen3``

Dispatcher switch: ``DISPATCHER=all2all|deepep|deepep_v2``. DeepEP V2 knobs (only read for ``deepep_v2``):
``DEEPEP_V2_ALIGN=1|128``, ``DEEPEP_V2_CPU_SYNC=0|1``, ``DEEPEP_V2_FP8_DISPATCH=0|1``, ``DEEPEP_V2_DETERMINISTIC=0|1``,
``DEEPEP_V2_NUM_SMS=<int>``, ``DEEPEP_V2_CAPACITY_FACTOR=<float>`` (static mode receive capacity). The expert GEMM backend is picked by ``XTUNER_EXPERT_GEMM_BACKEND=auto|legacy|deepgemm``.
``NUM_HIDDEN_LAYERS=<n>`` overrides the decoder depth (layer-cut proxy runs).
"""

import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.train import TrainerConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


MODEL_PATH = os.environ["MODEL_PATH"]
work_dir = os.environ.get("WORK_DIR", "work_dirs/moe_ep_bench")
ep_size = int(os.environ.get("EP_SIZE", "4"))
sp_size = int(os.environ.get("SP_SIZE", "1"))
intra_layer_micro_batch = int(os.environ.get("INTRA_LAYER_MICRO_BATCH", "1"))
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", os.environ.get("WORLD_SIZE", "8")))
sample_max_length = int(os.environ.get("SAMPLE_MAX_LENGTH", "4096"))
pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", "16384"))
total_step = int(os.environ.get("TOTAL_STEP", "20"))
dispatcher = os.environ.get("DISPATCHER", "deepep_v2").lower()
fp8 = _bool("FP8", False)

model_cfg = get_model_config_from_hf(MODEL_PATH)
# NUM_HIDDEN_LAYERS=<n> trims the decoder to a proxy depth (weights of the dropped layers are not loaded;
# pair it with STRICT_LOAD=0). Unset keeps the checkpoint's own depth.
if os.environ.get("NUM_HIDDEN_LAYERS"):
    model_cfg.num_hidden_layers = int(os.environ["NUM_HIDDEN_LAYERS"])
model_cfg.dispatcher = dispatcher
model_cfg.ep_size = ep_size
model_cfg.compile_cfg = _bool("MODEL_COMPILE", True)
model_cfg.float8_cfg = (
    Float8Config(
        scaling_granularity_gemm=ScalingGranularity.TILEWISE,
        scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
    )
    if fp8
    else None
)
if hasattr(model_cfg.attention, "sparse_mla_backend"):
    model_cfg.attention.sparse_mla_backend = os.environ.get("SPARSE_MLA_BACKEND", "cudnn_dsa")
if dispatcher == "deepep_v2":
    from xtuner.v1.module.dispatcher.deepep_v2 import DeepEPV2Config

    model_cfg.deepep_v2_cfg = DeepEPV2Config(
        expert_alignment=int(os.environ.get("DEEPEP_V2_ALIGN", "128")),  # type: ignore[arg-type]
        max_tokens_per_rank=pack_max_length // sp_size,
        cpu_sync=_bool("DEEPEP_V2_CPU_SYNC", True),
        fp8_dispatch=_bool("DEEPEP_V2_FP8_DISPATCH", False),
        deterministic=_bool("DEEPEP_V2_DETERMINISTIC", False),
        num_sms=int(os.environ.get("DEEPEP_V2_NUM_SMS", "0")),
        capacity_factor=float(os.environ["DEEPEP_V2_CAPACITY_FACTOR"])
        if os.environ.get("DEEPEP_V2_CAPACITY_FACTOR")
        else None,
    )

loss_cfg = CELossConfig(
    mode=os.environ.get("LOSS_MODE", "chunk"), chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "1024"))
)
model_cfg.lm_loss_cfg = loss_cfg

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="alpaca",
            anno_path=os.environ["ALPACA_PATH"],
            sample_ratio=float(os.environ.get("DATASET_SAMPLE_RATIO", "1.0")),
            cache_dir=os.path.join(work_dir, "jsonl_cache"),
            cache_tag=os.environ.get(
                "CACHE_TAG", f"moe_ep_{os.environ.get('CHAT_TEMPLATE', 'qwen3')}_{sample_max_length}"
            ),
        ),
        "tokenize_fn": OpenaiTokenizeFunctionConfig(
            chat_template=os.environ.get("CHAT_TEMPLATE", "qwen3"),
            max_length=sample_max_length,
        ),
    }
]
dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_level=os.environ.get("PACK_LEVEL", "soft"),
    pack_max_length=pack_max_length,
    pack_chunk_size=int(os.environ.get("PACK_CHUNK_SIZE", "10000")),
    pack_workers=int(os.environ.get("PACK_WORKERS", "4")),
    global_pack=_bool("GLOBAL_PACK", True),
    group_by_length=_bool("GROUP_BY_LENGTH", True),
    num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
)

optim_cfg = AdamWConfig(
    lr=float(os.environ.get("LR", "1e-6")), foreach=False, swap_optimizer=_bool("SWAP_OPTIMIZER", False)
)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0.0)
hsdp_sharding_size = os.environ.get("HSDP_SHARDING_SIZE")
fsdp_cfg = FSDPConfig(
    cpu_offload=False,
    ep_size=ep_size,
    torch_compile=_bool("TORCH_COMPILE", True),
    recompute_ratio=float(os.environ.get("RECOMPUTE_RATIO", "1.0")),
    # DECOUPLE_EP_FSDP=1: dp2ep layout (PR #2093) -- dense params sharded over the full FSDP mesh instead of
    # being replicated ep_size times; experts sharded dp_shard / ep_size ways on top of EP.
    decouple_ep_fsdp=_bool("DECOUPLE_EP_FSDP", False),
    hsdp_sharding_size=int(hsdp_sharding_size) if hsdp_sharding_size else None,
)

trainer = TrainerConfig(
    model_cfg=model_cfg,
    load_from=MODEL_PATH,
    tokenizer_path=MODEL_PATH,
    strict_load=_bool("STRICT_LOAD", True),
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    fsdp_cfg=fsdp_cfg,
    global_batch_size=global_batch_size,
    total_step=total_step,
    intra_layer_micro_batch=intra_layer_micro_batch,
    sp_size=sp_size,
    load_checkpoint_cfg=LoadCheckpointConfig(checkpoint_path=os.environ.get("LOAD_CHECKPOINT_PATH")),
    checkpoint_interval=10_000,
    hf_interval=10_000,
    work_dir=work_dir,
    profile_memory=_bool("PROFILE_MEMORY", False),
    profile_time=_bool("PROFILE_TIME", False),
    profile_step=[int(x) for x in os.environ.get("PROFILE_STEP", "8").split(",") if x],
    debug_skip_save=True,
)
