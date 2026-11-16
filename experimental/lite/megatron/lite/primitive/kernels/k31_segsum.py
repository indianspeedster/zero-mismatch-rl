"""Deterministic scatter-add: ``segsum_index_add_`` sums each destination row's contributions in index order (FP32)
and adds them once, instead of ``index_add_``'s atomics."""
import torch
import triton
import triton.language as tl


@triton.jit
def _segsum_add_kernel(DKV, VALS, PERM, DEST, STARTS, D, sd, sv, BD: tl.constexpr):
    sg = tl.program_id(0)
    cb = tl.program_id(1)
    row = tl.load(DEST + sg).to(tl.int64)
    lo = tl.load(STARTS + sg)
    hi = tl.load(STARTS + sg + 1)
    cols = cb * BD + tl.arange(0, BD)
    m = cols < D
    acc = tl.zeros([BD], dtype=tl.float32)
    for j in range(lo, hi):
        src = tl.load(PERM + j).to(tl.int64)
        acc += tl.load(VALS + src * sv + cols, mask=m, other=0.0)
    p = DKV + row * sd + cols
    tl.store(p, tl.load(p, mask=m, other=0.0) + acc, mask=m)


def segsum_index_add_(dkv: torch.Tensor, idx: torch.Tensor, vals: torch.Tensor, valid: torch.Tensor | None = None) -> None:
    """dkv[idx[i]] += vals[i] (fp32, 2-D, contiguous rows) in a fixed order: per destination row, the contributions in
    ascending i, summed first, then added once. Entries with valid == False are skipped."""
    assert dkv.dtype == torch.float32 and vals.dtype == torch.float32 and dkv.dim() == 2 and vals.dim() == 2
    vals = vals.contiguous()
    if valid is not None:
        sel = valid.reshape(-1).nonzero().squeeze(1)
        keys = idx.reshape(-1)[sel]
    else:
        sel = None
        keys = idx.reshape(-1)
    if keys.numel() == 0:
        return
    sk, order = torch.sort(keys, stable=True)
    perm = order if sel is None else sel[order]
    dest, counts = torch.unique_consecutive(sk, return_counts=True)
    starts = torch.zeros(dest.numel() + 1, dtype=torch.int64, device=dkv.device)
    starts[1:] = counts.cumsum(0)
    D = dkv.shape[1]
    BD = 128
    _segsum_add_kernel[(dest.numel(), triton.cdiv(D, BD))](dkv, vals, perm, dest, starts, D, dkv.stride(0), vals.stride(0),
                                                           BD=BD, num_warps=2)


__all__ = ["segsum_index_add_"]
