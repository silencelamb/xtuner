"""Helpers shared by MoE V2 parity tests: build a tiny legacy / V2 decoder pair with identical weights."""

from __future__ import annotations

import torch
import torch.distributed as dist

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import GreedyRouterConfig, MHAConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEDecoderLayer
from xtuner.v1.module.moe_v2 import MoEDecoderLayerV2, MoEV2Config


HIDDEN = 256
INTER = 512
MOE_INTER = 128
EXPERTS = 8
TOPK = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64


def layer_kwargs(*, ep_mesh, dispatcher, float8_cfg=None, n_shared_experts=1):
    return dict(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        moe_intermediate_size=MOE_INTER,
        hidden_act="silu",
        num_experts_per_tok=TOPK,
        n_routed_experts=EXPERTS,
        n_shared_experts=n_shared_experts,
        attention_config=MHAConfig(
            num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM, qk_norm=True
        ),
        router_config=GreedyRouterConfig(scoring_func="softmax", router_scaling_factor=1.0, norm_topk_prob=True),
        moe_act_fn_cfg=MoEActFnConfig(),
        float8_cfg=float8_cfg,
        layer_idx=0,
        dispatcher=dispatcher,
        ep_mesh=ep_mesh,
    )


def build_pair(
    *,
    ep_mesh,
    legacy_dispatcher: str | None,
    v2_cfg: MoEV2Config,
    device="cuda",
    dtype=torch.bfloat16,
    float8_cfg=None,
    n_shared_experts=1,
):
    """Return (legacy_layer, v2_layer) with identical parameters on ``device``."""
    torch.manual_seed(0)
    legacy = MoEDecoderLayer(
        **layer_kwargs(
            ep_mesh=ep_mesh, dispatcher=legacy_dispatcher, float8_cfg=float8_cfg, n_shared_experts=n_shared_experts
        )
    ).to(device=device, dtype=dtype)
    torch.manual_seed(0)
    v2 = MoEDecoderLayerV2(
        moe_v2_cfg=v2_cfg,
        **layer_kwargs(
            ep_mesh=ep_mesh, dispatcher=legacy_dispatcher, float8_cfg=float8_cfg, n_shared_experts=n_shared_experts
        ),
    ).to(device=device, dtype=dtype)
    _init_same(legacy)
    _copy_params(legacy, v2)
    return legacy, v2


def _init_same(module: torch.nn.Module) -> None:
    g = torch.Generator(device="cpu").manual_seed(1234)
    with torch.no_grad():
        for name, p in sorted(module.named_parameters(), key=lambda kv: kv[0]):
            local = p.to_local() if hasattr(p, "to_local") else p
            if dist.is_initialized() and hasattr(p, "to_local"):
                # EP-sharded expert weights: every rank draws the full tensor and slices its shard deterministically.
                full = torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.02
                from torch.distributed.tensor import distribute_tensor

                shard = distribute_tensor(full.to(local.device), p.device_mesh, p.placements).to_local()
                local.copy_(shard.to(local.dtype))
            else:
                local.copy_((torch.randn(local.shape, generator=g, dtype=torch.float32) * 0.02).to(local.dtype))


def _copy_params(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    src_params = dict(src.named_parameters())
    with torch.no_grad():
        for name, p in dst.named_parameters():
            s = src_params[name]
            pl = p.to_local() if hasattr(p, "to_local") else p
            sl = s.to_local() if hasattr(s, "to_local") else s
            pl.copy_(sl)


def build_inputs(num_micro_batches: int, seq_len: int, device="cuda", dtype=torch.bfloat16, seed: int = 7):
    torch.manual_seed(seed)
    hidden = [
        torch.randn(1, seq_len, HIDDEN, device=device, dtype=dtype, requires_grad=True)
        for _ in range(num_micro_batches)
    ]
    pos = [
        (
            torch.ones(1, seq_len, HEAD_DIM, device=device, dtype=dtype),
            torch.zeros(1, seq_len, HEAD_DIM, device=device, dtype=dtype),
        )
        for _ in range(num_micro_batches)
    ]
    ctx = [
        SequenceContext.from_input_ids((torch.arange(seq_len).view(1, -1),), device=device)
        for _ in range(num_micro_batches)
    ]
    return hidden, pos, ctx


def run_layer(layer, hidden, pos, ctx):
    if len(hidden) == 1:
        out = layer(hidden[0], seq_ctx=ctx[0], position_embeddings=pos[0])
        outs = [out["hidden_states"]]
    else:
        out = layer(hidden, seq_ctx=ctx, position_embeddings=pos)
        outs = list(out["hidden_states"])
    loss = sum((o.float() ** 2).mean() for o in outs)
    loss.backward()
    grads = {}
    for name, p in layer.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
        grads[name] = g.detach().clone()
    in_grads = [h.grad.detach().clone() for h in hidden]
    return [o.detach().clone() for o in outs], in_grads, grads


def compare(ref, got, *, exact: bool, rtol=2e-2, atol=2e-2, tag=""):
    ref_outs, ref_in, ref_g = ref
    got_outs, got_in, got_g = got

    def _cmp(a, b, what):
        if exact:
            assert torch.equal(a, b), (
                f"{tag} {what}: not bit-exact, max diff {(a.float() - b.float()).abs().max().item()}"
            )
        else:
            torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol, msg=lambda m: f"{tag} {what}: {m}")

    for i, (a, b) in enumerate(zip(ref_outs, got_outs)):
        _cmp(a, b, f"out[{i}]")
    for i, (a, b) in enumerate(zip(ref_in, got_in)):
        _cmp(a, b, f"dX[{i}]")
    assert ref_g.keys() == got_g.keys(), f"param set differs: {ref_g.keys() ^ got_g.keys()}"
    for k in ref_g:
        _cmp(ref_g[k], got_g[k], f"grad {k}")
