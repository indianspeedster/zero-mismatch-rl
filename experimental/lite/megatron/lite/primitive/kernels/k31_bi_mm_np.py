"""Non-persistent BF16 matmul with the same per-element arithmetic as vLLM's BI ``matmul_persistent`` (bitwise equal).

``install(min_m=...)`` routes large-M BF16 calls of ``batch_invariant.matmul_persistent`` to ``matmul_np``; other calls
use the original.  Trainer process only."""
import torch
import triton
import triton.language as tl

CFG = dict(BM=256, BN=128, BK=64, GROUP_M=8, num_warps=8, num_stages=3, matrix_instr_nonkdim=16, kpack=2)


@triton.jit
def _mm_np_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                  stride_cn, stride_bias, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
                  EVEN_K: tl.constexpr, HAS_BIAS: tl.constexpr, LARGE: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = pid_m * BM + tl.arange(0, BM)
    offs_bn = pid_n * BN + tl.arange(0, BN)
    if LARGE:
        offs_am = offs_am.to(tl.int64)
        offs_bn = offs_bn.to(tl.int64)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BM), BM)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BN), BN)
    offs_k = tl.arange(0, BK)
    if LARGE:
        offs_k = offs_k.to(tl.int64)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(0, tl.cdiv(K, BK)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - ki * BK, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - ki * BK, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    if LARGE:
        offs_cm = offs_cm.to(tl.int64)
        offs_cn = offs_cn.to(tl.int64)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_cn * stride_bias, mask=offs_cn < N, other=0.0).to(tl.float32)
        acc += bias
    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c,
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul_np(a, b, bias=None):
    """vLLM ``matmul_persistent`` contract: a [M, K], b [K, N] BF16, bias None or [N]; returns [M, N] in a.dtype."""
    assert a.shape[1] == b.shape[0] and a.dtype == b.dtype == torch.bfloat16
    assert bias is None or bias.dim() == 1
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0 or N == 0:
        return c
    cfg = dict(CFG)
    bm, bn, bk, g = cfg.pop("BM"), cfg.pop("BN"), cfg.pop("BK"), cfg.pop("GROUP_M")
    large = max(a.numel(), b.numel(), c.numel()) > 2 ** 31
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _mm_np_kernel[grid](a, b, c, bias if bias is not None else c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), bias.stride(0) if bias is not None else 1, BM=bm, BN=bn, BK=bk,
                        GROUP_M=g, EVEN_K=(K % bk == 0), HAS_BIAS=bias is not None, LARGE=large, **cfg)
    return c


_ORIG = {}


def install(min_m=1024, min_k=256, min_n=256):
    """Wrap ``batch_invariant.matmul_persistent``: BF16 calls with M >= min_m, K >= min_k, N >= min_n use ``matmul_np``.
    Idempotent; returns the wrapper."""
    from vllm.model_executor.determinism import batch_invariant as BI
    if "fn" in _ORIG:
        return BI.matmul_persistent
    orig = BI.matmul_persistent
    _ORIG["fn"] = orig

    def matmul_persistent(a, b, bias=None):
        if (a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16 and a.dim() == 2 and a.shape[0] >= min_m
                and a.shape[1] >= min_k and b.shape[1] >= min_n):
            return matmul_np(a, b, bias=bias)
        return orig(a, b, bias=bias)
    BI.matmul_persistent = matmul_persistent
    return matmul_persistent
