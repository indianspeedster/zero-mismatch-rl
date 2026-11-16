"""CPU tests of the ROCm K31 backend: backend seam, retired switches, contract, matmul_np dispatch, VTP helpers, and
(TRITON_INTERPRET=1 subprocess) the fixed-order segmented sum and the explicit DSA backward. Bitwise trainer == rollout
is checked by GPU tests, not here."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from _tiny import tiny_config
from megatron.lite.model.k31.vllm import protocol
from megatron.lite.model.k31.vllm import runtime_metadata as rm
from megatron.lite.primitive.kernels import k31_backend, rocm_k31, vllm_k31
from test_vllm_skeleton import _impl


@pytest.fixture
def backend(monkeypatch):
    """Select a backend for one test (``k31_backend.backend`` is cached per process)."""

    def select(name: str | None) -> None:
        if name is None:
            monkeypatch.delenv("MLITE_K31_BACKEND", raising=False)
        else:
            monkeypatch.setenv("MLITE_K31_BACKEND", name)
        k31_backend.backend.cache_clear()

    yield select
    k31_backend.backend.cache_clear()


# --------------------------------------------------------------------------- backend seam
def test_default_backend_follows_torch_build(backend, monkeypatch) -> None:
    backend(None)
    monkeypatch.setattr(torch.version, "hip", "7.0", raising=False)
    k31_backend.backend.cache_clear()
    assert k31_backend.backend() == "rocm" and k31_backend.is_rocm()
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    k31_backend.backend.cache_clear()
    assert k31_backend.backend() == "vllm_cuda" and not k31_backend.is_rocm()


def test_unknown_backend_fails_closed(backend) -> None:
    backend("cuda")
    with pytest.raises(ValueError, match="MLITE_K31_BACKEND"):
        k31_backend.backend()


ROCM_ADAPTERS = {"BILinearAdapter", "BIRouterAdapter", "LoRALeafAdapter", "RMSNormAdapter", "SiTUAndMulAdapter",
                 "AttnResAdapter", "TokenLogprobAdapter", "TemperatureAdapter"}


def test_adapters_resolve_to_active_backend(backend) -> None:
    backend("rocm")
    for name in k31_backend.ADAPTERS:
        want = getattr(rocm_k31, name) if name in ROCM_ADAPTERS else getattr(vllm_k31, name)
        assert getattr(k31_backend, name) is want, name
    assert k31_backend.moe_kernels() is rocm_k31
    backend("vllm_cuda")
    for name in k31_backend.ADAPTERS:
        assert getattr(k31_backend, name) is getattr(vllm_k31, name), name
    with pytest.raises(AttributeError):
        k31_backend.NotAnAdapter  # noqa: B018


def test_rocm_adapter_set_is_what_rocm_k31_defines() -> None:
    defined = {n for n in k31_backend.ADAPTERS if n in vars(rocm_k31) and isinstance(vars(rocm_k31)[n], type)}
    assert defined == ROCM_ADAPTERS


@pytest.mark.parametrize("value,want", [("", False), ("0", False), ("1", True), ("2", True)])
def test_bi_follows_vllm_batch_invariant(monkeypatch, value, want) -> None:
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", value)
    assert k31_backend.bi() is want


# --------------------------------------------------------------------------- contract
def test_no_deepep_is_rocm_only(backend) -> None:
    backend("rocm")
    protocol._validate_contract(tiny_config(), _impl(use_deepep=False))
    backend("vllm_cuda")
    with pytest.raises(NotImplementedError, match="DeepEP"):
        protocol._validate_contract(tiny_config(), _impl(use_deepep=False))


# --------------------------------------------------------------------------- retired switches
@pytest.fixture
def clean_env(monkeypatch):
    for k in (*rm._ROCM_RETIRED, *rm._ROCM_RETIRED_BI, *rm._PLUGIN_PINNED_BI, "MLITE_K31_BI_VTP", "ROLLOUT_TP"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_clean_env_passes(clean_env) -> None:
    rm._check_rocm_env(True)
    rm._check_rocm_env(False)


def test_retired_lists_are_disjoint() -> None:
    assert not set(rm._ROCM_RETIRED) & set(rm._ROCM_RETIRED_BI)
    assert len(rm._ROCM_RETIRED) == len(set(rm._ROCM_RETIRED))


@pytest.mark.parametrize("name", rm._ROCM_RETIRED)
@pytest.mark.parametrize("bi", [True, False])
def test_retired_switch_fails_closed(clean_env, name, bi) -> None:
    clean_env.setenv(name, "1")
    with pytest.raises(RuntimeError, match=name):
        rm._check_rocm_env(bi)


@pytest.mark.parametrize("name,fixed", sorted(rm._ROCM_RETIRED_BI.items()))
def test_retired_bi_switch_accepts_only_its_fixed_value_under_bi(clean_env, name, fixed) -> None:
    clean_env.setenv(name, fixed)
    rm._check_rocm_env(True)  # old launchers that name the fixed behavior keep working
    with pytest.raises(RuntimeError, match=name):
        rm._check_rocm_env(False)  # ... but not with BI off, where that behavior is not what runs
    clean_env.setenv(name, fixed + "x")
    with pytest.raises(RuntimeError, match=name):
        rm._check_rocm_env(True)


@pytest.mark.parametrize("name,fixed", sorted(rm._PLUGIN_PINNED_BI.items()))
def test_plugin_switch_pinned_under_bi(clean_env, name, fixed) -> None:
    """Under BI a mirrored rollout-plugin switch accepts only its fixed value; with BI off it is not checked."""
    clean_env.setenv(name, fixed)
    rm._check_rocm_env(True)
    clean_env.setenv(name, "0")
    with pytest.raises(RuntimeError, match=name):
        rm._check_rocm_env(True)
    rm._check_rocm_env(False)


@pytest.mark.parametrize("scaling,ok", [(2.0, True), (0.5, True), (1.0, True), (1.5, False), (3.0, False)])
def test_bi_expert_lora_scaling_must_be_power_of_two(monkeypatch, scaling, ok) -> None:
    """Under BI a non-power-of-two expert-LoRA scaling fails before the BI MoE runs."""
    from megatron.lite.primitive.alignment import rocm_mxfp4_moe as rm_moe

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setitem(sys.modules, "k31lora.bi_moe", None)   # stop right after the check
    with pytest.raises(ImportError if ok else ValueError):
        rm_moe.bi_moe_slot_leaves(None, None, None, None, scaling, lora=(None,) * 6)


def test_all_bad_switches_are_reported_together(clean_env) -> None:
    clean_env.setenv("MLITE_K31_TRAIN", "1")
    clean_env.setenv("MLITE_K31_DSA_BI_BWD", "atomic")
    with pytest.raises(RuntimeError) as err:
        rm._check_rocm_env(True)
    assert "MLITE_K31_TRAIN" in str(err.value) and "MLITE_K31_DSA_BI_BWD" in str(err.value)


@pytest.mark.parametrize(
    "vtp,tp,bi,error",
    [
        ("", "8", True, None),
        ("1", "8", True, None),
        ("8", "8", True, None),
        ("8", "", True, None),
        ("8", "4", True, "differs from ROLLOUT_TP"),
        ("eight", "", True, "rollout's TP size"),
        ("eight", "", False, None),  # VTP is read only under BI
    ],
)
def test_vtp_must_mirror_rollout_tp(clean_env, vtp, tp, bi, error) -> None:
    clean_env.setenv("MLITE_K31_BI_VTP", vtp)
    clean_env.setenv("ROLLOUT_TP", tp)
    if error is None:
        rm._check_rocm_env(bi)
    else:
        with pytest.raises(RuntimeError, match=error):
            rm._check_rocm_env(bi)


# --------------------------------------------------------------------------- matmul_np dispatch (trainer only)
@pytest.fixture
def fake_bi(monkeypatch):
    """Stub vllm.model_executor.determinism.batch_invariant and record which GEMM each call takes."""

    from megatron.lite.primitive.kernels import k31_bi_mm_np as NP

    calls = []
    mod = types.ModuleType("vllm.model_executor.determinism.batch_invariant")
    mod.matmul_persistent = lambda a, b, bias=None: calls.append("persistent") or "persistent"
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.determinism"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    sys.modules["vllm.model_executor.determinism"].batch_invariant = mod
    monkeypatch.setattr(NP, "_ORIG", {})
    monkeypatch.setattr(NP, "matmul_np", lambda a, b, bias=None: calls.append("np") or "np")
    return NP, mod, calls


def test_matmul_np_thresholds(fake_bi) -> None:
    NP, mod, calls = fake_bi
    wrapper = NP.install(min_m=1024)
    assert mod.matmul_persistent is wrapper
    bf, f32 = torch.bfloat16, torch.float32

    def mm(m, k, n, dt=bf):
        return mod.matmul_persistent(torch.empty(m, k, dtype=dt), torch.empty(k, n, dtype=dt))

    cases = [((1024, 256, 256), "np"), ((1023, 256, 256), "persistent"), ((4096, 255, 4096), "persistent"),
             ((4096, 4096, 255), "persistent"), ((4096, 32, 4096), "persistent"), ((16384, 7168, 8448), "np")]
    for shape, want in cases:
        assert mm(*shape) == want, shape
    assert mm(4096, 4096, 4096, f32) == "persistent"  # non-BF16 keeps the original path
    assert NP.install(min_m=1) is wrapper  # idempotent: no double wrap, first thresholds stay
    assert mm(1023, 256, 256) == "persistent"
    assert calls == [w for _, w in cases] + ["persistent", "persistent"]  # each call reached exactly one GEMM


# --------------------------------------------------------------------------- VTP / reduction helpers
@pytest.mark.parametrize("vtp,bi,want", [("8", "1", 8), ("8", "0", 1), ("1", "1", 1), ("", "1", 1), ("0", "1", 1)])
def test_vtp_size_needs_bi(monkeypatch, vtp, bi, want) -> None:
    monkeypatch.setenv("MLITE_K31_BI_VTP", vtp)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", bi)
    assert rocm_k31.vtp_size() == want


def test_vtp_slices_partition_k() -> None:
    sl = rocm_k31._vtp_slices(7168, 8)
    assert [s.start for s in sl] == list(range(0, 7168, 896)) and sl[-1].stop == 7168
    with pytest.raises(ValueError, match="does not divide"):
        rocm_k31._vtp_slices(100, 8)


def test_tree16_root_is_the_pairwise_tree() -> None:
    g = torch.Generator().manual_seed(3)
    leaves = torch.randn(16, 33, generator=g)
    level = list(leaves)
    while len(level) > 1:
        level = [level[i] + level[i + 1] for i in range(0, len(level), 2)]
    assert torch.equal(rocm_k31.tree16_root(leaves), level[0])
    for bad in (leaves.double(), leaves[:8], leaves[0]):
        with pytest.raises(ValueError):
            rocm_k31.tree16_root(bad)


def test_mm_f32out_cpu() -> None:
    a = torch.randn(5, 7).bfloat16()
    b = torch.randn(7, 3).bfloat16()
    out = rocm_k31.mm_f32out(a, b)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, a.float() @ b.float())


# --------------------------------------------------------------------------- Triton kernels (interpreter, subprocess)
@pytest.mark.parametrize("case", ["segsum_bitwise", "segsum_edges", "dsa_bwd", "dsa_core"])
def test_interpreted_kernel_case(case) -> None:
    pytest.importorskip("triton")
    here = Path(__file__).parent
    lite_root = Path(protocol.__file__).parents[5]  # experimental/lite
    env = {**os.environ, "TRITON_INTERPRET": "1",
           "PYTHONPATH": os.pathsep.join([str(lite_root), os.environ.get("PYTHONPATH", "")])}
    res = subprocess.run([sys.executable, str(here / "_rocm_interp_cases.py"), case], env=env, capture_output=True,
                         text=True, timeout=900, check=False)
    assert res.returncode == 0 and f"OK {case}" in res.stdout, res.stdout[-2000:] + res.stderr[-4000:]


# --------------------------------------------------------------------------- ROCm BI init
@pytest.fixture
def rocm_init(backend, clean_env, monkeypatch):
    """ROCm init with no BI cuBLAS env at startup or now; returns the list of k31_bi_mm_np.install calls."""

    from megatron.lite.primitive.kernels import k31_bi_mm_np as NP

    backend("rocm")
    monkeypatch.setattr(rm, "_INITIALIZED", False)
    monkeypatch.setattr(rm, "_STARTUP_CUBLAS_ENV", dict.fromkeys(rm._CUBLAS_ENV))
    for name in (*rm._CUBLAS_ENV, *rm.RETIRED_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    installs = []
    monkeypatch.setattr(NP, "install", lambda **kw: installs.append(kw))
    return installs


def _stub_plugins(monkeypatch, present=("k31kdabi", "k31lora", "k31lora.bi_moe")):
    for name in ("k31kdabi", "k31lora", "k31lora.bi_moe"):
        if name in present:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        else:
            monkeypatch.setitem(sys.modules, name, None)  # import raises ImportError


def test_rocm_init_bi1_installs_trainer_gemm(rocm_init, monkeypatch) -> None:
    _stub_plugins(monkeypatch)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    rm.initialize_k31_vllm_batch_invariance()
    assert rm._INITIALIZED is True
    assert rocm_init == [{"min_m": 1024}]
    assert torch.backends.cuda.matmul.allow_tf32 is False


@pytest.mark.parametrize("missing", ["k31kdabi", "k31lora.bi_moe"])
def test_rocm_init_bi1_needs_rollout_plugins(rocm_init, monkeypatch, missing) -> None:
    _stub_plugins(monkeypatch, present=[m for m in ("k31kdabi", "k31lora", "k31lora.bi_moe") if m != missing])
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(RuntimeError, match=missing.replace(".", r"\.")):
        rm.initialize_k31_vllm_batch_invariance()
    assert rm._INITIALIZED is False and rocm_init == []


def test_rocm_init_bi0_runs_without_plugins(rocm_init, monkeypatch) -> None:
    _stub_plugins(monkeypatch, present=())
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    rm.initialize_k31_vllm_batch_invariance()
    assert rm._INITIALIZED is True and rocm_init == []
    assert torch.backends.cuda.matmul.allow_tf32 is True  # BI off: left alone


def test_rocm_init_bi0_refuses_bi_cublas_env(rocm_init, monkeypatch) -> None:
    name = next(iter(rm._CUBLAS_ENV))
    monkeypatch.setenv(name, rm._CUBLAS_ENV[name] or "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    with pytest.raises(RuntimeError, match="BI cuBLAS environment"):
        rm.initialize_k31_vllm_batch_invariance()
    assert rm._INITIALIZED is False


def test_rocm_init_refuses_retired_switch_first(rocm_init, monkeypatch) -> None:
    _stub_plugins(monkeypatch)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("MLITE_K31_ROCM_DGRAD", "1")
    with pytest.raises(RuntimeError, match="MLITE_K31_ROCM_DGRAD"):
        rm.initialize_k31_vllm_batch_invariance()
    assert rm._INITIALIZED is False and rocm_init == []


# --------------------------------------------------------------------------- DSA through the base attention_core
def test_rocm_dsa_metadata_fields() -> None:
    from megatron.lite.model.k31.vllm.primitive import dsa_rocm as D

    md = D.build_dsa_rocm_metadata([5, 70, 1], device="cpu", topk=16)
    assert torch.equal(md.block_table, torch.tensor([0, 5, 75])) and md.workspace_starts is md.block_table
    assert torch.equal(md.slot_mapping, torch.arange(76))
    assert md.mla_cache.shape == (2, 64, D.DS_MLA_ROW_BYTES) and D.DS_MLA_ROW_BYTES == 656
    assert md.indexer is md and md.cache_lens is None and md.dummy_block_table is None
    assert torch.equal(md.positions, torch.cat((torch.arange(5), torch.arange(70), torch.arange(1))))
    with pytest.raises(ValueError):
        D.build_dsa_rocm_metadata([3, 0], device="cpu")


def test_rocm_dsa_indices_equal_packed_rows() -> None:
    """attention_core's sparse and backward index maps equal the packed rows (request start + local index, -1 kept)."""
    from megatron.lite.model.k31.vllm.primitive import dsa_rocm as D

    g = torch.Generator().manual_seed(2)
    md = D.build_dsa_rocm_metadata([7, 12, 4], device="cpu", topk=6)
    T = 23
    logical = torch.full((T, 6), -1, dtype=torch.int32)
    for t in range(T):
        n = int(md.positions[t]) + 1
        logical[t, : min(n, 6)] = torch.randperm(n, generator=g)[:6].to(torch.int32)
    starts = md.cu_seqlens[:-1].index_select(0, md.req_id_per_token)
    packed = torch.where(logical >= 0, logical.long() + starts[:, None], -1)
    conv = D.DSASparseMLAAdapter.convert_indices(md.req_id_per_token, md.block_table, logical, block_size=64)
    ws = torch.where(logical >= 0, logical + md.workspace_starts.index_select(0, md.req_id_per_token.long())[:, None],
                     logical)  # attention_core's workspace_topk expression
    assert torch.equal(conv, packed) and torch.equal(ws, packed)


def test_ds_mla_dequant_layout() -> None:
    from _rocm_interp_cases import ds_mla_write_rows
    from megatron.lite.model.k31.vllm.primitive import dsa_rocm as D

    g = torch.Generator().manual_seed(4)
    kv = (torch.randn(9, 512, generator=g) * torch.tensor([1e-6, 1.0, 30.0, 0.0]).repeat_interleave(128)).bfloat16()
    pe = torch.randn(9, 64, generator=g).bfloat16()
    out = D.ds_mla_dequant(ds_mla_write_rows(kv, pe))
    assert out.dtype == torch.bfloat16 and out.shape == (9, 576)
    assert torch.equal(out[:, 512:], pe)  # rope tail raw
    assert torch.equal(out[:, 384:512], torch.zeros(9, 128, dtype=torch.bfloat16))  # zero tile stays zero
    torch.testing.assert_close(out[:, 128:384].float(), kv[:, 128:384].float(), rtol=0.07, atol=0)  # e4m3 rounding


def test_default_dsa_adapters_follow_backend(backend) -> None:
    from megatron.lite.model.k31.vllm.primitive import attention as A
    from megatron.lite.model.k31.vllm.primitive import dsa_rocm as D

    backend("rocm")
    ad = A.default_dsa_adapters()
    assert ad.backward is D.sparse_backward and isinstance(ad.sparse, D.DSASparseMLAAdapter)
    assert isinstance(ad.upconvert, D.DSAFP8KVUpconvertAdapter) and isinstance(ad.bmm, D.DSAAbsorbBmmAdapter)
    backend("vllm_cuda")
    ad = A.default_dsa_adapters()
    assert ad.backward is None and type(ad.sparse).__module__.endswith("vllm_k31")
