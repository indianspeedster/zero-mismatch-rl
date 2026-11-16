"""TP-sliced LoRA export (``lora_expert_tp_slice``): each rollout TP rank gets only its shard of the expert factors.

Reference: tag ``[tp_rank, tp_size]``, then the whole export with ``w{1,3}.lora_B`` (dim 1) and ``w2.lora_A``
(dim 2) narrowed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.k31.vllm import checkpoint as ck

E, H, I, R = 8, 6, 12, 4  # experts, hidden, intermediate, LoRA rank
PFX = "base_model.model.model.layers.3.mlp.experts"

# (native, param, peft, kind) as _lora_entries yields them (shapes as the K31 MoE LoRA stores them)
_SHAPES = [
    ("model.layers.3.self_attn.q_proj.lora_A", (R, H), "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight", "dense"),
    ("model.layers.3.self_attn.q_proj.lora_B", (H, R), "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight", "dense"),
    ("lora_w1_A", (1, R, H), f"{PFX}.w1.lora_A.weight", "expert_shared"),
    ("lora_w1_B", (E, I, R), f"{PFX}.w1.lora_B.weight", "expert_local"),
    ("lora_w3_A", (1, R, H), f"{PFX}.w3.lora_A.weight", "expert_shared"),
    ("lora_w3_B", (E, I, R), f"{PFX}.w3.lora_B.weight", "expert_local"),
    ("lora_w2_A", (E, R, I), f"{PFX}.w2.lora_A.weight", "expert_local"),
    ("lora_w2_B", (1, H, R), f"{PFX}.w2.lora_B.weight", "expert_shared"),
]


@pytest.fixture
def export(monkeypatch):
    g = torch.Generator().manual_seed(7)
    items = [(n, torch.randn(s, generator=g, dtype=torch.float32), p, k) for n, s, p, k in _SHAPES]
    monkeypatch.setattr(ck, "K31WeightSpec", lambda cfg, ps: None)
    monkeypatch.setattr(ck, "_models", lambda chunks: [None])
    monkeypatch.setattr(ck, "_lora_entries", lambda model, spec: iter(items))
    cfg = SimpleNamespace(num_experts=E)
    ps = SimpleNamespace(tp_size=1, etp_size=1, pp_size=1, ep_size=1)

    def run(**kw):
        return list(ck.export_hf_weights(None, cfg, ps, cpu=True, **kw))

    return run


def _reference_tp_slice(weights, tp_rank, tp_size):
    """Slice a whole export on the send side (reference)."""
    sharded = {".experts.w1.lora_B.weight": 1, ".experts.w3.lora_B.weight": 1, ".experts.w2.lora_A.weight": 2}
    out = [("__k31_lora_expert_tp_slice__", torch.tensor([tp_rank, tp_size], dtype=torch.int64))]
    for name, t in weights:
        dim = next((d for s, d in sharded.items() if name.endswith(s)), None)
        if dim is not None:
            s = t.shape[dim] // tp_size
            t = t.narrow(dim, tp_rank * s, s).contiguous()
        out.append((name, t))
    return out


def test_tag_name_matches_the_rollout_receiver():
    assert ck.LORA_EXPERT_TP_SLICE_TAG == "__k31_lora_expert_tp_slice__"
    assert ck.SUPPORTS_LORA_EXPERT_TP_SLICE is True


@pytest.mark.parametrize("tp_size", [2, 4])
def test_matches_verl_send_side_slicing_bitwise(export, tp_size):
    whole = export()
    for tp_rank in range(tp_size):
        got = export(lora_expert_tp_slice=(tp_rank, tp_size))
        want = _reference_tp_slice(whole, tp_rank, tp_size)
        assert [n for n, _ in got] == [n for n, _ in want]
        for (name, a), (_, b) in zip(got, want):
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), name
            assert a.is_contiguous(), name


def test_shards_reassemble_the_whole_adapter(export):
    whole = dict(export())
    shards = [dict(export(lora_expert_tp_slice=(r, 4))) for r in range(4)]
    for name, t in whole.items():
        dim = ck._tp_sharded_dim(name)
        if dim is None:
            assert all(torch.equal(s[name], t) for s in shards), name
        else:
            assert shards[0][name].shape[dim] == t.shape[dim] // 4
            assert torch.equal(torch.cat([s[name] for s in shards], dim=dim), t), name


def test_unsliced_export_is_unchanged(export):
    names = [n for n, _ in export()]
    assert ck.LORA_EXPERT_TP_SLICE_TAG not in names and ck.LORA_EXPERT_SLICE_TAG not in names
    assert names == [p for _n, _s, p, _k in _SHAPES]


@pytest.mark.parametrize(
    "kw,error",
    [
        ({"lora_expert_tp_slice": (0, 1)}, ValueError),
        ({"lora_expert_tp_slice": (4, 4)}, ValueError),
        ({"lora_expert_tp_slice": (-1, 4)}, ValueError),
        ({"lora_expert_tp_slice": (0, 5)}, ValueError),  # I=12 not divisible by 5
        ({"lora_expert_tp_slice": (0, 2), "lora_expert_slice": (0, 2)}, ValueError),
    ],
    ids=["tp1", "rank-out-of-range", "negative-rank", "indivisible", "both-slices"],
)
def test_rejects_bad_requests(export, kw, error):
    with pytest.raises(error):
        export(**kw)


def test_rejects_non_3d_expert_factor():
    with pytest.raises(ValueError, match="3D expert factor"):
        ck._tp_narrow(f"{PFX}.w1.lora_B.weight", torch.zeros(I, R), 0, 2)
