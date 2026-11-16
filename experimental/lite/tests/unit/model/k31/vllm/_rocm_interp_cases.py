"""CPU cases for the ROCm K31 Triton kernels, run in a subprocess under TRITON_INTERPRET=1 (set before import).

    TRITON_INTERPRET=1 python _rocm_interp_cases.py <case>     -> prints "OK <case>" or raises
"""
from __future__ import annotations

import sys

import torch


def _ref_segsum(dst, idx, vals, valid=None):
    """dst[idx[i]] += vals[i]: per destination row, contributions summed in ascending i (fp32), then added once."""
    out = dst.clone()
    idx = idx.reshape(-1)
    ok = torch.ones_like(idx, dtype=torch.bool) if valid is None else valid.reshape(-1)
    for row in torch.unique(idx[ok]).tolist():
        acc = torch.zeros(dst.shape[1], dtype=torch.float32)
        for i in range(idx.numel()):
            if ok[i] and int(idx[i]) == row:
                acc = acc + vals[i]
        out[row] = out[row] + acc
    return out


def case_segsum_bitwise() -> None:
    from megatron.lite.primitive.kernels.k31_segsum import segsum_index_add_

    g = torch.Generator().manual_seed(0)
    for rows, n, d in ((7, 30, 200), (1, 9, 3), (64, 257, 129)):
        dst = torch.randn(rows, d, generator=g)
        vals = torch.randn(n, d, generator=g)
        idx = torch.randint(0, rows, (n,), generator=g)
        valid = torch.rand(n, generator=g) > 0.3
        for v in (None, valid):
            got = dst.clone()
            segsum_index_add_(got, idx, vals, v)
            assert torch.equal(got, _ref_segsum(dst, idx, vals, v)), (rows, n, d, v is None)
            again = dst.clone()
            segsum_index_add_(again, idx, vals, v)
            assert torch.equal(got, again)


def case_segsum_edges() -> None:
    from megatron.lite.primitive.kernels.k31_segsum import segsum_index_add_

    dst = torch.randn(4, 8)
    got = dst.clone()
    segsum_index_add_(got, torch.tensor([1, 2]), torch.randn(2, 8), torch.tensor([False, False]))  # all masked
    assert torch.equal(got, dst)
    segsum_index_add_(got, torch.zeros(0, dtype=torch.long), torch.zeros(0, 8))  # empty
    assert torch.equal(got, dst)
    idx2d = torch.tensor([[0, 3], [3, 0]])
    vals = torch.randn(4, 8)
    segsum_index_add_(got, idx2d, vals)  # 2-D idx is flattened row-major
    assert torch.equal(got, _ref_segsum(dst, idx2d.reshape(-1), vals))
    for bad in ((dst.double(), vals.double()), (dst, vals.half()), (dst[0], vals)):
        try:
            segsum_index_add_(bad[0], torch.tensor([0]), bad[1])
        except AssertionError:
            continue
        raise AssertionError("segsum accepted non-fp32 / non-2-D operands")


def _sparse_mla_ref(q, kv, idx, scale, d_v):
    """Dense reference forward of the absorbed sparse MLA (fp64): o[t,h] = softmax_k(q.kv[idx]) @ kv[idx, :d_v]."""
    valid = idx >= 0
    kvg = kv[idx.clamp(min=0)]  # [T, K, D]
    s = torch.einsum("thd,tkd->thk", q, kvg) * scale
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    p = torch.softmax(s, dim=-1).nan_to_num(0.0)
    return torch.einsum("thk,tkd->thd", p, kvg[..., :d_v])


def case_dsa_bwd() -> None:
    from megatron.lite.model.k31.vllm.primitive.dsa_rocm import _sparse_mla_bwd_explicit

    g = torch.Generator().manual_seed(1)
    T, H, D, d_v, K = 37, 4, 24, 16, 6
    scale = D ** -0.5
    q = torch.randn(T, H, D, generator=g)
    kv = torch.randn(T, D, generator=g)
    # causal top-k with -1 padding (row t sees keys <= t; early rows have fewer than K valid, row 0 has one)
    idx = torch.full((T, K), -1, dtype=torch.long)
    for t in range(T):
        cand = torch.randperm(t + 1, generator=g)[:K]
        idx[t, : cand.numel()] = cand
    grad = torch.randn(T, H, d_v, generator=g)

    q64 = q.double().requires_grad_()
    kv64 = kv.double().requires_grad_()
    _sparse_mla_ref(q64, kv64, idx, scale, d_v).backward(grad.double())

    runs = []
    for chunk in (1 << 30, 4096):  # one chunk, and many small chunks
        dq, dkv = _sparse_mla_bwd_explicit(q, kv, idx, scale, d_v, grad, True, True, chunk_bytes=chunk)
        assert dq.dtype == torch.float32 and dkv.dtype == torch.float32
        torch.testing.assert_close(dq.double(), q64.grad, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(dkv.double(), kv64.grad, rtol=1e-4, atol=1e-5)
        dq2, dkv2 = _sparse_mla_bwd_explicit(q, kv, idx, scale, d_v, grad, True, True, chunk_bytes=chunk)
        assert torch.equal(dq, dq2) and torch.equal(dkv, dkv2), "explicit DSA backward is not run-to-run bitwise"
        runs.append((dq, dkv))
    # bitwise per chunk size only (not across chunk sizes)
    dq_only, none_kv = _sparse_mla_bwd_explicit(q, kv, idx, scale, d_v, grad, True, False)
    none_q, dkv_only = _sparse_mla_bwd_explicit(q, kv, idx, scale, d_v, grad, False, True)
    assert none_kv is None and none_q is None
    assert torch.equal(dq_only, runs[0][0]) and torch.equal(dkv_only, runs[0][1])


def ds_mla_write_rows(kv_c, k_pe):
    """Torch emulation of concat_and_cache_ds_mla_kernel: (T, 656) uint8 rows (512 e4m3 | 4 fp32 scales | 64 bf16)."""
    t = kv_c.shape[0]
    x = kv_c.float().view(t, 4, 128)
    amax = x.abs().amax(-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(torch.clamp(amax / 448.0, min=1e-4))))
    q = (x / scale).to(torch.float8_e4m3fn).view(t, 512).view(torch.uint8)
    return torch.cat((q, scale.view(t, 4).contiguous().view(torch.uint8), k_pe.contiguous().view(torch.uint8)), dim=1)


def ds_mla_round_ref(kv_c, k_pe):
    """The rows' read-back value from the writer formula directly (independent of dsa_rocm.ds_mla_dequant)."""
    t = kv_c.shape[0]
    x = kv_c.float().view(t, 4, 128)
    scale = torch.exp2(torch.ceil(torch.log2(torch.clamp(x.abs().amax(-1, keepdim=True) / 448.0, min=1e-4))))
    lat = ((x / scale).to(torch.float8_e4m3fn).float() * scale).view(t, 512).to(torch.bfloat16)
    return torch.cat((lat, k_pe), -1)


def case_dsa_core() -> None:
    """Base attention_core + ROCm adapters/backward (torch stand-ins for the kernels) vs fp64 dense autograd."""
    from megatron.lite.model.k31.vllm.primitive import attention as A
    from megatron.lite.model.k31.vllm.primitive import dsa_rocm as D

    g = torch.Generator().manual_seed(5)
    lens = [9, 21, 6]
    T, H, nope, rope, V, K = sum(lens), 4, 16, D.ROPE_DIM, D.LATENT_DIM, 8
    md = D.build_dsa_rocm_metadata(lens, device="cpu", topk=K)
    logical = torch.full((T, K), -1, dtype=torch.int32)
    for t in range(T):
        pos = int(md.positions[t])
        cand = torch.randperm(pos + 1, generator=g)[:K].sort().values
        logical[t, : cand.numel()] = cand.to(torch.int32)

    class Insert:
        def __call__(self, kv_c, k_pe, mla_cache, slot_mapping, k_scale):
            mla_cache.view(-1, D.DS_MLA_ROW_BYTES)[slot_mapping] = ds_mla_write_rows(kv_c, k_pe)

    class Sparse(D.DSASparseMLAAdapter):
        def __call__(self, q, mla_cache, rows, *, softmax_scale, padded_heads, cache_lens, dummy_block_table):
            kv = D.ds_mla_dequant(mla_cache.view(-1, D.DS_MLA_ROW_BYTES)[: q.shape[0]]).float()
            valid = rows >= 0
            s = torch.einsum("thd,tkd->thk", q.float(), kv[rows.clamp(min=0)]) * softmax_scale
            p = torch.softmax(s.masked_fill(~valid[:, None, :], float("-inf")), -1)
            return torch.einsum("thk,tkd->thd", p, kv[rows.clamp(min=0)][..., :V]).to(q.dtype), None

    ad = D.dsa_adapters()
    ad.kv_insert, ad.sparse = Insert(), Sparse()
    scale = (nope + rope) ** -0.5
    q = (torch.randn(T, H, nope + rope, generator=g)).to(torch.bfloat16).requires_grad_()
    kv_c = torch.randn(T, V, generator=g).to(torch.bfloat16).requires_grad_()
    k_pe = torch.randn(T, rope, generator=g).to(torch.bfloat16).requires_grad_()
    w_uk_t = (torch.randn(H, nope, V, generator=g) * 0.1).to(torch.bfloat16).requires_grad_()
    w_uv = (torch.randn(H, V, 16, generator=g) * 0.1).to(torch.bfloat16).requires_grad_()
    grad = torch.randn(T, H * 16, generator=g)

    out = A.attention_core(ad, q, kv_c, k_pe, logical, md, w_uk_t=w_uk_t, w_uv=w_uv, softmax_scale=scale,
                           qk_nope_head_dim=nope, padded_heads=64, backward_op=ad.backward)
    out.float().backward(grad)
    got = [t.grad.double() for t in (q, kv_c, k_pe, w_uk_t, w_uv)]

    # reference: fp64, KV = the rounded cache value with identity gradient
    leaves = [t.detach().double().requires_grad_() for t in (q, kv_c, k_pe, w_uk_t, w_uv)]
    rq, rkv, rpe, ruk, ruv = leaves
    rounded = ds_mla_round_ref(kv_c.detach(), k_pe.detach()).double()
    rows = torch.cat((rkv, rpe), -1)
    rows = rows + (rounded - rows).detach()
    packed = torch.where(logical >= 0, logical.long() + md.workspace_starts[md.req_id_per_token][:, None], -1)
    valid = packed >= 0
    def bf16(x):  # the real path's bf16 intermediates (absorb output, attention output), identity gradient
        return x + (x.to(torch.bfloat16).double() - x).detach()

    qc = torch.cat((bf16(torch.einsum("thp,hpl->thl", rq[..., :nope], ruk)), rq[..., nope:]), -1)
    kvg = rows[packed.clamp(min=0)]
    s = torch.einsum("thd,tkd->thk", qc, kvg) * scale
    p = torch.softmax(s.masked_fill(~valid[:, None, :], float("-inf")), -1)
    o = bf16(torch.einsum("thk,tkd->thd", p, kvg[..., :V]))
    torch.einsum("thl,hlv->thv", o, ruv).reshape(T, -1).backward(grad.double())
    for name, a, b in zip(("dq", "dkv_c", "dk_pe", "dw_uk", "dw_uv"), got, (t.grad for t in leaves)):
        torch.testing.assert_close(a, b, rtol=3e-2, atol=3e-2, msg=name)  # bf16 grads / bf16 forward intermediates
        assert float((a - b).norm() / b.norm()) < 1e-2, name
    # the forward the backward differentiated is the cache read back: dequant == the writer emulation round trip
    assert torch.equal(D.ds_mla_dequant(md.mla_cache.view(-1, D.DS_MLA_ROW_BYTES)[:T]), rounded.to(torch.bfloat16))


CASES = {name[5:]: fn for name, fn in globals().items() if name.startswith("case_")}

if __name__ == "__main__":
    CASES[sys.argv[1]]()
    print("OK", sys.argv[1])
