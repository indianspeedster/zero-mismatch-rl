"""ROCm adapters for the K31 model (``MLITE_K31_BACKEND=rocm``), with :mod:`vllm_k31`'s names and signatures.

Under ``VLLM_BATCH_INVARIANT=1`` each adapter calls the BI ROCm rollout's kernel, so forward values are bitwise equal.
"""

from __future__ import annotations

import functools
import importlib
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.lite.primitive.kernels import k31_backend
from megatron.lite.primitive.kernels.vllm_k31 import (
    ATTN_RES_HIDDEN,
    _Adapter,
    _output_boundary,
    _same_device,
    _tensor,
)


@functools.lru_cache(maxsize=None)
def _c_op(name: str):
    """``torch.ops._C.<name>`` from vLLM; raises if it is not registered."""

    importlib.import_module("vllm._C")
    op = getattr(torch.ops._C, name, None) if hasattr(torch.ops, "_C") else None
    if op is None:
        raise RuntimeError(f"K31 ROCm needs vllm._C {name} (the rollout's kernel)")
    return op


# ---------------------------------------------------------------------------
# dense linear / router / LoRA
# ---------------------------------------------------------------------------
def mm_f32out(a: Tensor, b: Tensor) -> Tensor:
    """``a @ b`` with BF16 operands and an FP32 result (no intermediate BF16 rounding)."""

    if a.is_cuda:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a.float(), b.float())  # CPU (tests)


def dense_bi() -> bool:
    """``VLLM_BATCH_INVARIANT=1``: call the BI rollout's kernels."""

    return k31_backend.bi()


@functools.lru_cache(maxsize=None)
def _bi_lib():
    """vLLM's batch-invariant module with its tuned configs resolved (aten overrides not installed)."""

    m = importlib.import_module("vllm.model_executor.determinism.batch_invariant")
    for name in ("resolve_tuned_matmul_configs", "_resolve_block_sizes"):
        fn = getattr(m, name, None)
        if fn is not None:
            fn()
    return m


@functools.lru_cache(maxsize=None)
def _bi_lora_epilogue() -> str:
    """The rollout's punica expand epilogue: under BI ``k31lora.bi_lora`` (one rounding), else vLLM's default."""

    if not dense_bi():
        return "default"
    try:
        bi_lora = importlib.import_module("k31lora.bi_lora")
    except ImportError as exc:
        raise RuntimeError("VLLM_BATCH_INVARIANT=1: the trainer needs the k31lora plugin on PYTHONPATH") from exc
    return bi_lora.install()


def vtp_size() -> int:
    """Virtual TP size ``MLITE_K31_BI_VTP`` (BI only; must equal the rollout's TP): row-parallel layers are computed
    as per-rank K-slices summed in fixed rank order."""

    n = int(os.environ.get("MLITE_K31_BI_VTP", "0") or 0)
    return n if n > 1 and dense_bi() else 1


def vtp_fixed_sum(partials: Tensor) -> Tensor:
    """``all_reduce_batch_invariant``'s fixed-order sum over stacked partials ``[N, ...]``."""

    lib = _bi_lib()
    n = partials.shape[0]
    gathered = partials.reshape(n, -1).contiguous()
    numel = gathered.shape[1]
    out = torch.empty(numel, dtype=partials.dtype, device=partials.device)
    import triton

    lib._fixed_order_sum_kernel[(triton.cdiv(numel, 1024),)](
        gathered, out, numel, numel, WORLD_SIZE=n, BLOCK_SIZE=1024, LARGE=gathered.numel() > 2**31
    )
    return out.view(partials.shape[1:])


def _vtp_slices(k: int, n: int):
    if k % n:
        raise ValueError(f"virtual TP={n} does not divide K={k}")
    s = k // n
    return [slice(r * s, (r + 1) * s) for r in range(n)]


def vtp_row_partials(value: Tensor, weight: Tensor, n: int, lora=None) -> Tensor:
    """Per-rank partials ``[n, T, out]`` of a row-parallel linear (and LoRA) before its all-reduce."""

    parts = []
    for ks in _vtp_slices(value.shape[-1], n):
        xr = value[:, ks].contiguous()
        wr = weight[:, ks].contiguous()
        if lora is None:
            parts.append(rocm_linear(xr, wr))
        else:
            a_st, b_st, slices, meta = lora
            a_r = tuple(a[..., ks].contiguous() for a in a_st)
            parts.append(_punica_leaf(xr, wr, a_r, b_st, slices, meta))
    return torch.stack(parts)


def _punica_leaf(value, weight, lora_a_stacked, lora_b_stacked, output_slices, meta, specialize_active_lora=False):
    """``PunicaWrapperGPU.add_lora_linear`` on the base output."""

    _bi_lora_epilogue()
    from megatron.lite.primitive.kernels.vllm_k31 import _drop_punica_ptr_cache

    lops = importlib.import_module("vllm.lora.ops.triton_ops")
    y = rocm_linear(value, weight).contiguous()
    tokens = value.shape[0]
    rank = lora_b_stacked[0].size(-1)
    buffer = torch.empty((len(output_slices), tokens, rank), dtype=torch.float32, device=value.device)
    _drop_punica_ptr_cache(lora_a_stacked, lora_b_stacked)
    try:
        lops.lora_shrink(value, lora_a_stacked, buffer, *meta.meta_args(tokens, specialize_active_lora), 1.0)
        lops.lora_expand(buffer, lora_b_stacked, y, *meta.meta_args(tokens, specialize_active_lora), offset_start=0,
                         add_inputs=True)
    finally:
        _drop_punica_ptr_cache(lora_a_stacked, lora_b_stacked)
    return y


def rocm_linear(value: Tensor, weight: Tensor) -> Tensor:
    """BF16 ``x @ W^T``; under BI vLLM's ``linear_batch_invariant``."""

    if dense_bi():
        return _bi_lib().linear_batch_invariant(value, weight, None)
    return F.linear(value, weight)


class BILinearAdapter(_Adapter):
    """BF16 linear; ``row_parallel`` layers follow the virtual TP (``vtp_size``)."""

    row_parallel = False

    def forward(self, value: Tensor, weight: Tensor) -> Tensor:
        _tensor(value, "x", dtype=torch.bfloat16, contiguous=False)
        _tensor(weight, "weight", ndim=2, dtype=torch.bfloat16, contiguous=False)
        if value.shape[-1] != weight.shape[1]:
            raise ValueError("x and weight inner dimensions differ")
        _same_device({"x": value, "weight": weight})
        n = vtp_size() if self.row_parallel else 1
        if n > 1:
            x2 = value.reshape(-1, value.shape[-1])
            out = vtp_fixed_sum(vtp_row_partials(x2, weight, n)).reshape(*value.shape[:-1], weight.shape[0])
            return _output_boundary(out, value, weight)
        return _output_boundary(rocm_linear(value, weight), value, weight)


class BIRouterAdapter(_Adapter):
    """Router logits, FP32 ``[T, E]``."""

    def forward(self, value: Tensor, weight: Tensor) -> Tensor:
        _tensor(value, "x", ndim=2, dtype=torch.bfloat16, contiguous=False)
        _tensor(weight, "weight", ndim=2, dtype=torch.bfloat16)
        _same_device({"x": value, "weight": weight})
        if dense_bi():
            # BI: the rollout's router GEMM is the BI linear in BF16, then cast to FP32
            out = rocm_linear(value, weight).to(torch.float32)
        else:
            out = mm_f32out(value, weight.t())
        return _output_boundary(out, value, weight)


class LoRALeafAdapter(_Adapter):
    """Base linear plus the rollout's punica ``add_lora_linear`` (FP32 shrink, expand with add_inputs) as one leaf."""

    @staticmethod
    def fused_lora() -> bool:
        return True

    def fused_forward(self, value, weight, lora_a_stacked, lora_b_stacked, output_slices, meta,
                      specialize_active_lora: bool = False):
        _tensor(value, "x", ndim=2, dtype=torch.bfloat16)
        _tensor(weight, "weight", ndim=2, dtype=torch.bfloat16, contiguous=False)
        if weight.shape[0] != sum(output_slices):
            raise ValueError("LoRA slices must cover the base output")
        n = vtp_size() if getattr(self, "row_parallel", False) else 1
        if n > 1:
            y = vtp_fixed_sum(vtp_row_partials(value, weight, n,
                                               lora=(lora_a_stacked, lora_b_stacked, output_slices, meta)))
        else:
            y = _punica_leaf(value, weight, lora_a_stacked, lora_b_stacked, output_slices, meta, specialize_active_lora)
        return _output_boundary(y, value, weight, lora_a_stacked, lora_b_stacked)


# ---------------------------------------------------------------------------
# norms and activations
# ---------------------------------------------------------------------------
class RMSNormAdapter(_Adapter):
    def forward(self, value: Tensor, weight: Tensor, eps: float) -> Tensor:
        _tensor(value, "x", dtype=torch.bfloat16, contiguous=False)
        _tensor(weight, "weight", ndim=1)
        if weight.shape[0] != value.shape[-1]:
            raise ValueError("RMSNorm weight must match the hidden dimension")
        if eps <= 0:
            raise ValueError("eps must be positive")
        _same_device({"x": value, "weight": weight})
        if dense_bi():
            # BI: vLLM's rms_norm_batch_invariant
            out = _bi_lib().rms_norm_batch_invariant(value, weight, float(eps))
        else:
            if weight.dtype != value.dtype:
                raise TypeError("RMSNorm: the rollout's _C.rms_norm takes the weight in the activation dtype")
            x2 = value.reshape(-1, value.shape[-1]).contiguous()
            out = torch.empty_like(x2)
            _c_op("rms_norm")(out, x2, weight.contiguous(), float(eps))
            out = out.view(value.shape)
        return _output_boundary(out, value, weight)


class SiTUAndMulAdapter(_Adapter):
    def forward(self, value: Tensor, beta: float, linear_beta: float | None) -> Tensor:
        _tensor(value, "x", dtype=torch.bfloat16)
        if value.shape[-1] % 2:
            raise ValueError("SiTU input must hold gate and up halves")
        d = value.shape[-1] // 2
        out = torch.empty(value.shape[:-1] + (d,), dtype=value.dtype, device=value.device)
        _c_op("situ_and_mul")(out, value, float(beta), -1.0 if linear_beta is None else float(linear_beta))
        return _output_boundary(out, value)


# ---------------------------------------------------------------------------
# AttnRes
# ---------------------------------------------------------------------------
class AttnResAdapter(_Adapter):
    """The rollout ``attn_res`` wrapper semantics (mutates ``prefix`` and the bank in place)."""

    def forward(
        self,
        prefix: Tensor,
        delta: Tensor | None,
        blocks: Tensor,
        norm_weight: Tensor,
        qk_weight: Tensor,
        output_norm_weight: Tensor | None,
        num_blocks: int,
        block_write_idx: int,
        eps: float,
        output_norm_eps: float,
    ) -> Tensor:
        _tensor(prefix, "prefix", ndim=2, dtype=torch.bfloat16)
        if delta is not None:
            _tensor(delta, "delta", ndim=2, dtype=torch.bfloat16)
            if delta.shape != prefix.shape:
                raise ValueError("delta must match prefix")
        _tensor(blocks, "blocks", ndim=3, dtype=torch.bfloat16)
        tokens, hidden = prefix.shape
        if hidden != ATTN_RES_HIDDEN:
            raise ValueError(f"K31 AttnRes expects hidden={ATTN_RES_HIDDEN}")
        if blocks.shape[0] != tokens or blocks.shape[2] != hidden:
            raise ValueError("blocks must be [tokens, slots, hidden]")
        if not 0 <= num_blocks <= blocks.shape[1]:
            raise ValueError(f"num_blocks={num_blocks} outside the bank")
        out = _vllm_attn_res(
            prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight,
            num_blocks, block_write_idx, eps, output_norm_eps,
        )
        return _output_boundary(
            out, prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight
        )


def _vllm_attn_res(*args) -> Tensor:
    """vLLM's ROCm ``attn_res`` op (BI: the rollout's pinned tiling)."""

    if dense_bi():
        return _bi_attn_res()(*args)
    for mod in ("vllm.models.kimi_k3.amd.ops.attn_res", "vllm.models.kimi_k3.common.ops.attn_res"):
        try:
            fn = getattr(importlib.import_module(mod), "attn_res")
        except (ImportError, AttributeError):
            continue
        return fn(*args)
    raise RuntimeError("K31 ROCm needs the image's attn_res op (vllm.models.kimi_k3.amd.ops.attn_res)")


@functools.lru_cache(maxsize=None)
def _bi_attn_res():
    """``attn_res`` with the batch-size-independent launch tiling of ``k31lora.bi_attnres`` (as the BI rollout)."""

    bi = None
    for name in ("k31lora.bi_attnres", "k31bi_attnres"):
        try:
            bi = importlib.import_module(name)
            break
        except ImportError:
            continue
    if bi is None:
        raise RuntimeError("VLLM_BATCH_INVARIANT=1: the trainer needs k31lora.bi_attnres on PYTHONPATH")
    return bi._pinned_attn_res(importlib.import_module("vllm.models.kimi_k3.amd.ops.attn_res"))


# ---------------------------------------------------------------------------
# head
# ---------------------------------------------------------------------------
class TokenLogprobAdapter(_Adapter):
    """Selected-token log-softmax in FP32 (``[T, n]`` for ``token_ids [T, n]``)."""

    def forward(self, logits: Tensor, token_ids: Tensor) -> Tensor:
        _tensor(logits, "logits", ndim=2, dtype=(torch.bfloat16, torch.float32))
        _tensor(token_ids, "token_ids", ndim=2, contiguous=False)
        if dense_bi():
            # BI: the rollout sampler's log_softmax kernel
            lp = _bi_lib()._log_softmax_batch_invariant(logits.float(), -1, False)
        else:
            lp = torch.log_softmax(logits.float(), dim=-1)
        return _output_boundary(lp.gather(-1, token_ids.long()), logits)


class TemperatureAdapter(_Adapter):
    def forward(self, logits: Tensor, temperature: float) -> Tensor:
        _tensor(logits, "logits", ndim=2, dtype=(torch.bfloat16, torch.float32))
        return logits.detach().float() / float(temperature)


# ---------------------------------------------------------------------------
# MoE helpers (replace the vllm_k31_moe entries)
# ---------------------------------------------------------------------------
class TorchK31Router:
    """K31 ``select_experts`` with the rollout's kernels; returns FP32 weights and int32 ids."""

    def __init__(
        self,
        *,
        top_k: int,
        num_experts: int,
        e_score_correction_bias: Tensor,
        renormalize: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        scoring_func: str = "sigmoid",
        routed_scaling_factor: float = 1.0,
    ) -> None:
        if scoring_func != "sigmoid" or num_expert_group != 1 or topk_group != 1:
            raise NotImplementedError("K31 router: sigmoid, no expert groups")
        self.top_k = top_k
        self.num_experts = num_experts
        self.bias = e_score_correction_bias
        self.renormalize = renormalize
        self.scale = routed_scaling_factor

    def select_experts(self, hidden_states: Tensor, router_logits: Tensor) -> tuple[Tensor, Tensor]:
        # as the rollout's FusedTopKBiasRouter (AITER or _C top-k, per VLLM_ROCM_USE_AITER)
        FT = importlib.import_module("vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router")
        A = importlib.import_module("vllm._aiter_ops").rocm_aiter_ops
        n_exp = router_logits.shape[-1]
        if A.is_fused_moe_enabled():
            # AITER biased_grouped_topk, called directly as the rollout does
            g = FT._aiter_get_num_expert_group(n_exp)
            w = torch.empty(router_logits.shape[0], self.top_k, dtype=torch.float32, device=router_logits.device)
            i = torch.empty(router_logits.shape[0], self.top_k, dtype=torch.int32, device=router_logits.device)
            A.biased_grouped_topk(router_logits.float().contiguous(), self.bias, w, i, num_expert_group=g, topk_group=g,
                                  need_renorm=self.renormalize)
            if self.scale != 1.0:
                w *= self.scale
            return w, i
        w, i = FT.fused_topk_bias(hidden_states, router_logits, "sigmoid", self.bias, self.top_k, self.renormalize,
                                  routed_scaling_factor=self.scale)
        return w.float(), i.to(torch.int32)


def build_k31_router(**kwargs) -> TorchK31Router:
    return TorchK31Router(**kwargs)


def tree16_root(leaves: Tensor) -> Tensor:
    """FP32 adjacent-pair tree over ``leaves[16, n]`` (the CUDA ``local_subtree`` order)."""

    if leaves.dtype != torch.float32 or leaves.ndim != 2 or leaves.shape[0] != 16:
        raise ValueError("tree16_root expects fp32 [16, n] leaves")
    values = leaves
    while values.shape[0] > 1:
        values = values[0::2] + values[1::2]
    return values[0]


class MoELatentUpAdapter:
    """``torch.mm(normed_latent, routed_expert_up_proj.weight.t())`` (BF16, no BI pins)."""

    def __call__(self, latent: Tensor, weight: Tensor) -> Tensor:
        if latent.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError("latent up-projection is BF16 x BF16")
        if latent.ndim != 2 or weight.ndim != 2 or latent.shape[1] != weight.shape[1]:
            raise ValueError("latent [T, L] and weight [H, L] expected")
        if dense_bi():
            # BI: the BI linear, as a TP1 rollout (TP>1 rollouts are not mirrored here)
            return rocm_linear(latent, weight)
        return torch.mm(latent, weight.t())


__all__ = [
    "AttnResAdapter",
    "BILinearAdapter",
    "BIRouterAdapter",
    "LoRALeafAdapter",
    "MoELatentUpAdapter",
    "RMSNormAdapter",
    "SiTUAndMulAdapter",
    "TemperatureAdapter",
    "TokenLogprobAdapter",
    "TorchK31Router",
    "build_k31_router",
    "rocm_linear",
    "tree16_root",
]
