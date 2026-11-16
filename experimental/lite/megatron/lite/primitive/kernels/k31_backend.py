"""K31 kernel backend: ``MLITE_K31_BACKEND`` = ``vllm_cuda`` (:mod:`vllm_k31`) | ``rocm`` (:mod:`rocm_k31`).

Default: ``rocm`` on a HIP build of torch.  Adapter names imported from this module resolve to the active backend's
class; names a backend lacks fall back to the ``vllm_k31`` class.
"""

from __future__ import annotations

import functools
import os

import torch

BACKENDS = ("vllm_cuda", "rocm")

ADAPTERS = (
    "AttnResAdapter",
    "BILinearAdapter",
    "BIRouterAdapter",
    "LoRALeafAdapter",
    "PunicaLoRADeltaAdapter",
    "RMSNormAdapter",
    "SiTUAndMulAdapter",
    "TemperatureAdapter",
    "TokenLogprobAdapter",
    "DSAAbsorbBmmAdapter",
    "DSAFP8KVUpconvertAdapter",
    "DSAIndexerAdapter",
    "DSAMLAKVInsertAdapter",
    "DSASparseMLAAdapter",
    "KDACausalConvAdapter",
    "KDACFBIPrefillAdapter",
    "KDAGatedNormAdapter",
)


@functools.lru_cache(maxsize=1)
def backend() -> str:
    value = os.environ.get("MLITE_K31_BACKEND")
    if value is None:
        value = "rocm" if getattr(torch.version, "hip", None) else "vllm_cuda"
    if value not in BACKENDS:
        raise ValueError(f"MLITE_K31_BACKEND must be one of {BACKENDS}, got {value!r}")
    return value


def is_rocm() -> bool:
    return backend() == "rocm"


def bi() -> bool:
    """``VLLM_BATCH_INVARIANT=1`` (the trainer mirrors the BI rollout)."""

    return os.environ.get("VLLM_BATCH_INVARIANT", "0") not in ("", "0")


def moe_kernels():
    """MoE helper module of the active backend (``build_k31_router``, ``MoELatentUpAdapter``, ``tree16_root``)."""

    if is_rocm():
        from megatron.lite.primitive.kernels import rocm_k31

        return rocm_k31
    from megatron.lite.primitive.kernels import vllm_k31_moe

    return vllm_k31_moe


def __getattr__(name: str):
    if name not in ADAPTERS:
        raise AttributeError(name)
    from megatron.lite.primitive.kernels import vllm_k31

    if is_rocm():
        from megatron.lite.primitive.kernels import rocm_k31

        return getattr(rocm_k31, name, None) or getattr(vllm_k31, name)
    return getattr(vllm_k31, name)


__all__ = ["ADAPTERS", "BACKENDS", "backend", "bi", "is_rocm", "moe_kernels"]
