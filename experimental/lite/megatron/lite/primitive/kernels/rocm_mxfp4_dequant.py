"""Fused MXFP4 -> BF16 dequant of selected experts (Triton).

``dequant_experts(packed [E, N, K/2] u8, scale [E, N, K/32] u8, experts [c] i64) -> [c, N, K] bf16``, bitwise equal to
``vllm_fi_w4a16_moe.dequantize_k31_mxfp4_bf16(packed[experts], scale[experts])``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1(c):
    s = (c >> 3) & 1
    e = (c >> 1) & 3
    m = (c & 1).to(tl.float32)
    mag = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where(s == 1, -mag, mag)


@triton.jit
def _dequant_kernel(
    packed, scale, experts, out,
    N, KH,
    s_pe, s_pn, s_se, s_sn,
    BLOCK_N: tl.constexpr, BLOCK_KH: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    e = tl.load(experts + pid_e).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_KH + tl.arange(0, BLOCK_KH)
    mask = (rn[:, None] < N) & (rk[None, :] < KH)
    p = tl.load(packed + e * s_pe + rn[:, None].to(tl.int64) * s_pn + rk[None, :], mask=mask, other=0).to(tl.int32)
    sc = tl.load(scale + e * s_se + rn[:, None].to(tl.int64) * s_sn + rk[None, :] // 16, mask=mask, other=127)
    f = tl.exp2(sc.to(tl.float32) - 127.0)
    lo = (_e2m1(p & 15) * f).to(tl.bfloat16)
    hi = (_e2m1(p >> 4) * f).to(tl.bfloat16)
    base = out + pid_e.to(tl.int64) * N * KH * 2 + rn[:, None].to(tl.int64) * (KH * 2) + rk[None, :] * 2
    tl.store(base, lo, mask=mask)
    tl.store(base + 1, hi, mask=mask)


def dequant_experts(packed: torch.Tensor, scale: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8 or scale.dtype != torch.uint8 or packed.ndim != 3:
        raise TypeError("packed / scale must be uint8 [E, N, K/2] / [E, N, K/32]")
    if packed.stride(-1) != 1 or scale.stride(-1) != 1:
        raise ValueError("packed / scale need a contiguous last dim")
    E, N, KH = packed.shape
    if scale.shape != (E, N, KH * 2 // 32):
        raise ValueError(f"scale shape {tuple(scale.shape)} does not match packed {tuple(packed.shape)}")
    ex = experts.to(device=packed.device, dtype=torch.int64).contiguous()
    c = ex.numel()
    out = torch.empty((c, N, KH * 2), dtype=torch.bfloat16, device=packed.device)
    if c == 0:
        return out
    BLOCK_N, BLOCK_KH = 32, 128
    grid = (c, triton.cdiv(N, BLOCK_N), triton.cdiv(KH, BLOCK_KH))
    _dequant_kernel[grid](
        packed, scale, ex, out, N, KH,
        packed.stride(0), packed.stride(1), scale.stride(0), scale.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_KH=BLOCK_KH,
    )
    return out


__all__ = ["dequant_experts"]
