"""mLite engine ``set_rollout_weight_quantizer``: applied to full-weight exports only; adapter exports stay BF16."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))

pytestmark = pytest.mark.optional

WEIGHTS = [
    ("model.layers.0.mlp.down_proj.weight", torch.ones(2, 2, dtype=torch.bfloat16))
]


@pytest.fixture(autouse=True)
def _require_verl() -> None:
    pytest.importorskip(
        "verl", reason="VERL is required for this optional example test."
    )


def _engine(monkeypatch, *, peft_config, qat=None):
    from verl_mlite.engine import mlite_engine
    from verl_mlite.engine.config import MegatronLiteEngineConfig

    engine = mlite_engine.MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path="/tmp/k31", hf_config={"model_type": "kimi_k31"}, mtp=None
        ),
        engine_config=MegatronLiteEngineConfig(
            custom_backend_module=None, impl_cfg={}, qat=qat or {}
        ),
        optimizer_config=SimpleNamespace(),
        checkpoint_config={},
    )
    monkeypatch.setattr(
        type(engine), "is_param_offload_enabled", property(lambda self: False)
    )
    monkeypatch.setattr(engine, "_require_initialized", lambda: None)
    monkeypatch.setattr(engine, "_resolve_model_name", lambda: "kimi_k31")
    monkeypatch.setattr(
        mlite_engine.qat_export, "export_qat_weights", lambda w, cfg: [("qat", w)]
    )
    proto = (
        SimpleNamespace(export_peft_config=lambda chunks: peft_config)
        if peft_config is not None
        else None
    )
    engine._initial_sync_cache_cleared = True
    engine.handle = SimpleNamespace(_extras={"protocol": proto}, _model=None)
    engine.runtime = SimpleNamespace(export_weights=lambda handle, **kw: list(WEIGHTS))
    return engine


def _quantizer(calls):
    def quantize(weights):
        calls.append(weights)
        return [("quantized", weights)]

    return quantize


def test_default_has_no_quantizer(monkeypatch):
    engine = _engine(monkeypatch, peft_config=None)
    assert engine.get_per_tensor_param() == (WEIGHTS, None)


def test_adapter_export_is_not_quantized(monkeypatch):
    peft = {"r": 4}
    engine = _engine(monkeypatch, peft_config=peft)
    calls = []
    engine.set_rollout_weight_quantizer(_quantizer(calls))
    assert engine.get_per_tensor_param() == (WEIGHTS, peft)
    assert calls == []


def test_full_export_is_quantized(monkeypatch):
    engine = _engine(monkeypatch, peft_config=None)
    calls = []
    engine.set_rollout_weight_quantizer(_quantizer(calls))
    assert engine.get_per_tensor_param() == ([("quantized", WEIGHTS)], None)
    assert calls == [WEIGHTS]


def test_mlite_qat_export_is_not_quantized_twice(monkeypatch):
    engine = _engine(monkeypatch, peft_config=None, qat={"enable": True})
    calls = []
    engine.set_rollout_weight_quantizer(_quantizer(calls))
    assert engine.get_per_tensor_param() == ([("qat", WEIGHTS)], None)
    assert calls == []
