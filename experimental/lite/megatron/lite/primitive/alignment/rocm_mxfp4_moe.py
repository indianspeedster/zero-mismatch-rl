"""K31 routed experts on ROCm: MXFP4 checkpoint buffers, expert LoRA, FP32 leaves per routed slot.

Forward: the rollout's ``k31lora.bi_moe.bi_moe_forward`` (both BI arms), leaves captured via its ``leaf_reduce`` hook.
Per routed slot ``(t, k)`` with local expert ``e`` (rows ``[gate; up]`` = ``[w1; w3]``)::

    gu    = x_t @ W13_e^T            (BF16 GEMM)
    gate += (x_t A1^T) B1_e^T * s ;  up += (x_t A3^T) B3_e^T * s      (expert LoRA)
    act   = SiTU(gate, up)           (FP32 math) -> BF16
    y     = act @ W2_e^T             (BF16 GEMM)
    leaf  = w_tk * (y + (act A2_e^T) B2^T * s)                         FP32

Leaves are FP32 rows in ``rows_index`` order (the ``VLLMFiW4A16LeafRowsFunction`` contract).
Backward: chunked autograd recompute of the graph above on the dequantized weights (base frozen; BF16 GEMM outputs);
returns dX (fixed-order ``k31_segsum`` scatter), d(route weight) and the LoRA grads.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from megatron.lite.primitive.alignment.vllm_fi_w4a16_moe import dequantize_k31_mxfp4_bf16
from megatron.lite.primitive.kernels import k31_backend

# Rows (padded) per chunk and experts per chunk; bound the FP32 temporaries.
CHUNK_ROWS = 32768
CHUNK_EXPERTS = 64


@dataclass
class RocmMxfp4Experts:
    """EP-local checkpoint MXFP4 buffers (rows ``[gate; up]``) and the SiTU parameters."""

    ckpt_w13: Tensor  # uint8 [E, 2I, H/2]
    ckpt_w13_scale: Tensor  # uint8 [E, 2I, H/32]
    ckpt_w2: Tensor  # uint8 [E, H, I/2]
    ckpt_w2_scale: Tensor  # uint8 [E, H, I/32]
    num_experts: int
    local_offset: int
    situ_beta: float
    situ_linear_beta: float | None

    @property
    def num_local_experts(self) -> int:
        return int(self.ckpt_w13.shape[0])

    @property
    def intermediate(self) -> int:
        return int(self.ckpt_w13.shape[1]) // 2

    @property
    def hidden(self) -> int:
        return int(self.ckpt_w2.shape[1])

    def dequant(self, ex: Tensor) -> tuple[Tensor, Tensor]:
        """BF16 ``W13 [c, 2I, H]`` and ``W2 [c, H, I]`` of the local experts ``ex`` (exact)."""

        if self.ckpt_w13.is_cuda:
            from megatron.lite.primitive.kernels.rocm_mxfp4_dequant import dequant_experts

            return (
                dequant_experts(self.ckpt_w13, self.ckpt_w13_scale, ex),
                dequant_experts(self.ckpt_w2, self.ckpt_w2_scale, ex),
            )
        w13 = dequantize_k31_mxfp4_bf16(
            self.ckpt_w13.index_select(0, ex), self.ckpt_w13_scale.index_select(0, ex)
        )
        w2 = dequantize_k31_mxfp4_bf16(
            self.ckpt_w2.index_select(0, ex), self.ckpt_w2_scale.index_select(0, ex)
        )
        return w13, w2


def situ_fp32(gate: Tensor, up: Tensor, beta: float, linear_beta: float | None) -> Tensor:
    act = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return act * up


# --------------------------------------------------------------------------- chunk planning
def _plan(topk_ids: Tensor, rows_index: Tensor, experts: RocmMxfp4Experts):
    """Group the listed routed slots by local expert; returns chunk tables (host lists)."""

    k = topk_ids.shape[1]
    e_loc = experts.num_local_experts
    flat_ids = topk_ids.reshape(-1).long().index_select(0, rows_index) - experts.local_offset
    if flat_ids.numel() and (int(flat_ids.min()) < 0 or int(flat_ids.max()) >= e_loc):
        raise RuntimeError("ROCm MoE: a listed route is not on this rank")
    order = torch.argsort(flat_ids, stable=True)  # positions in rows_index, grouped by expert
    counts = torch.bincount(flat_ids, minlength=e_loc)
    starts = torch.cumsum(counts, 0) - counts
    both = torch.cat((counts, starts)).tolist()
    counts_l, starts_l = both[:e_loc], both[e_loc:]
    items = [(e, starts_l[e], counts_l[e]) for e in range(e_loc) if counts_l[e]]
    items.sort(key=lambda it: -it[2])
    chunks, cur = [], []
    for it in items:
        n_first = -(-(cur[0][2] if cur else it[2]) // 8) * 8
        if cur and (len(cur) == CHUNK_EXPERTS or (len(cur) + 1) * n_first > CHUNK_ROWS):
            chunks.append(cur)
            cur = []
        cur.append(it)
    if cur:
        chunks.append(cur)
    del k
    return order, chunks


def _chunk_tensors(chunk, order: Tensor, rows_index: Tensor, topk: int, dev):
    c = len(chunk)
    n = -(-chunk[0][2] // 8) * 8
    ex = torch.tensor([it[0] for it in chunk], dtype=torch.long, device=dev)
    cnt = torch.tensor([it[2] for it in chunk], dtype=torch.long, device=dev)
    first = torch.tensor([it[1] for it in chunk], dtype=torch.long, device=dev)
    col = torch.arange(n, device=dev)
    real = col.unsqueeze(0) < cnt.unsqueeze(1)  # [c, n]
    pos = first.unsqueeze(1) + torch.minimum(col.unsqueeze(0), cnt.unsqueeze(1) - 1)
    out_pos = order.index_select(0, pos.reshape(-1))  # position in rows_index (pads repeat)
    flat = rows_index.index_select(0, out_pos)  # flat route n * K + k
    tok = flat // topk
    return c, n, ex, real, out_pos, flat, tok


def _chunk_leaves(
    experts: RocmMxfp4Experts,
    ex: Tensor,
    xp: Tensor,  # [c, n, H] bf16
    wp: Tensor,  # [c, n, 1] fp32 (0 on pads)
    lora: tuple | None,
    scaling: float,
) -> Tensor:
    """Differentiable leaves ``[c, n, H]`` fp32 of one chunk (grad flows to xp, wp, LoRA)."""

    inter = experts.intermediate
    w13, w2 = experts.dequant(ex)
    gu = torch.bmm(xp, w13.transpose(1, 2)).float()  # [c, n, 2I]
    if lora is not None:
        a1, a3, b1, b3, a2, b2 = lora
        xa1 = torch.matmul(xp.float(), a1.to(torch.bfloat16).float().t())  # [c, n, r]
        xa3 = torch.matmul(xp.float(), a3.to(torch.bfloat16).float().t())
        b1e = b1.index_select(0, ex).float() * scaling  # [c, I, r]
        b3e = b3.index_select(0, ex).float() * scaling
        gate = gu[..., :inter] + torch.bmm(xa1, b1e.transpose(1, 2))
        up = gu[..., inter:] + torch.bmm(xa3, b3e.transpose(1, 2))
    else:
        gate, up = gu[..., :inter], gu[..., inter:]
    act = situ_fp32(gate, up, experts.situ_beta, experts.situ_linear_beta).to(torch.bfloat16)
    y = torch.bmm(act, w2.transpose(1, 2)).float()  # [c, n, H]
    if lora is not None:
        a2e = a2.index_select(0, ex).to(torch.bfloat16).float()  # [c, r, I]
        d2 = torch.matmul(torch.bmm(act.float(), a2e.transpose(1, 2)), b2.float().t() * scaling)
        y = y + d2
    return wp * y


def bi_moe_slot_leaves(experts: "RocmMxfp4Experts", x: Tensor, topk_ids: Tensor, topk_weights: Tensor,
                       scaling: float, lora: tuple | None) -> Tensor:
    """FP32 leaves ``[T * K, H]`` in flat slot order from ``k31lora.bi_moe.bi_moe_forward`` (via ``leaf_reduce``).

    Non-local slots (EP) are zero. Expert LoRA uses bi_moe's dict layout (A1/A3/B1/B3/A2/B2 + scaling)."""
    import importlib

    if lora is not None and k31_backend.bi() and not (scaling > 0 and math.frexp(scaling)[0] == 0.5):
        # the BI rollout keeps bf16(B * s); bf16(B) * s matches it exactly only for a power-of-two s
        raise ValueError(f"K31 BI expert LoRA needs a power-of-two lora_alpha / r, got scaling {scaling}")
    BM = importlib.import_module("k31lora.bi_moe")
    tc = os.environ.get("MLITE_K31_MOE_BI_TOK_CHUNK", "")
    if tc:
        # trainer-only bi_moe token block; rows are independent, so the bits do not change
        BM.TOK_CHUNK = int(tc)
    lora_dict = None
    if lora is not None:
        # shared A1 / A3 / B2, per-expert B1 / B3 / A2 (vLLM --enable-moe-shared-loras layout)
        a1, a3, b1, b3, a2, b2 = (t.detach().to(torch.bfloat16) for t in lora)
        lora_dict = {"A1": a1[None].contiguous(), "A3": a3[None].contiguous(), "B1": b1.contiguous(), "B3": b3.contiguous(),
                     "A2": a2.contiguous(), "B2": b2[None].contiguous(), "scaling": float(scaling),
                     "shared_A1": True, "shared_A3": True, "shared_B1": False, "shared_B3": False,
                     "shared_A2": False, "shared_B2": True}
    expert_map = None
    if experts.num_local_experts != experts.num_experts:
        expert_map = torch.full((experts.num_experts,), -1, dtype=torch.int64, device=x.device)
        expert_map[experts.local_offset: experts.local_offset + experts.num_local_experts] = torch.arange(
            experts.num_local_experts, device=x.device)
    got = []

    def capture(slots):
        got.append(slots)
        return slots

    BM.bi_moe_forward(x.contiguous(), experts.ckpt_w13, experts.ckpt_w13_scale, experts.ckpt_w2, experts.ckpt_w2_scale,
                      topk_weights.to(torch.float32), topk_ids, experts.situ_beta, experts.situ_linear_beta,
                      lora=lora_dict, expert_map=expert_map, leaf_reduce=capture)
    # one leaf_reduce call per token block, in token order
    return got[0] if len(got) == 1 else torch.cat(got, 0)


class RocmExpertLeafRowsFunction(torch.autograd.Function):
    """FP32 leaves of the listed routed slots (``rows_index``), forward no-grad, chunked recompute bwd."""

    @staticmethod
    def forward(ctx, experts, scaling, rows_index, x, topk_ids, topk_weights, *lora):
        has_lora = len(lora) > 0 and lora[0] is not None
        with torch.no_grad():
            slots = bi_moe_slot_leaves(experts, x.detach(), topk_ids, topk_weights.detach(), scaling,
                                       tuple(lora) if has_lora else None)
        out = slots.index_select(0, rows_index)
        ctx.experts = experts
        ctx.scaling = scaling
        ctx.has_lora = has_lora
        ctx.save_for_backward(rows_index, x, topk_ids, topk_weights, *(lora if has_lora else ()))
        ctx.mark_non_differentiable(topk_ids)
        return out

    @staticmethod
    def backward(ctx, grad_rows):
        rows_index, x, topk_ids, topk_weights, *lora = ctx.saved_tensors
        experts: RocmMxfp4Experts = ctx.experts
        dev = x.device
        h = x.shape[1]
        m, topk = topk_ids.shape
        order, chunks = _plan(topk_ids, rows_index, experts)
        need_x = ctx.needs_input_grad[3]
        need_w = ctx.needs_input_grad[5]
        dx = torch.zeros((m, h), dtype=torch.float32, device=dev)
        dw = torch.zeros((m * topk,), dtype=torch.float32, device=dev)
        lora_grads = [torch.zeros(t.shape, dtype=torch.float32, device=dev) for t in lora]
        we_flat = topk_weights.detach().reshape(-1).float()
        grad_rows = grad_rows.float()
        from megatron.lite.primitive.kernels.k31_segsum import segsum_index_add_

        for chunk in chunks:
            c, n, ex, real, out_pos, flat, tok = _chunk_tensors(chunk, order, rows_index, topk, dev)
            sel = real.reshape(-1)
            g = torch.where(sel.unsqueeze(1), grad_rows.index_select(0, out_pos), 0.0).view(c, n, h)
            with torch.enable_grad():
                xp = x.detach().index_select(0, tok).view(c, n, h).requires_grad_(need_x)
                wp = (
                    torch.where(sel, we_flat.index_select(0, flat), 0.0)
                    .view(c, n, 1)
                    .requires_grad_(need_w)
                )
                lp = tuple(t.detach().requires_grad_(True) for t in lora) if ctx.has_lora else None
                leaves = _chunk_leaves(experts, ex, xp, wp, lp, ctx.scaling)
                inputs = [t for t in (xp, wp) if t.requires_grad] + (list(lp) if lp else [])
                grads = torch.autograd.grad(leaves, inputs, g, allow_unused=True)
            it = iter(grads)
            if need_x:
                gx = next(it)
                # duplicate token indices: fixed-order sum
                segsum_index_add_(dx, tok, gx.reshape(-1, h).float(), sel)
            if need_w:
                gw = next(it)
                dw.index_add_(0, flat[sel], gw.reshape(-1)[sel].float())
            if lp:
                for acc, gl in zip(lora_grads, it):
                    if gl is not None:
                        acc += gl.float()
        out_lora = tuple(
            gl.to(t.dtype) for gl, t in zip(lora_grads, lora)
        ) if ctx.has_lora else ()
        return (
            None,
            None,
            None,
            dx.to(x.dtype) if need_x else None,
            None,
            dw.view(m, topk).to(topk_weights.dtype) if need_w else None,
            *out_lora,
        )


def rocm_expert_leaf_rows(
    experts: RocmMxfp4Experts,
    rows_index: Tensor,
    x: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    *,
    scaling: float = 1.0,
    lora: tuple[Tensor, ...] | None = None,
) -> Tensor:
    """``lora`` = ``(w1_a [r,H], w3_a [r,H], w1_b [E,I,r], w3_b [E,I,r], w2_a [E,r,I], w2_b [H,r])``."""

    return RocmExpertLeafRowsFunction.apply(
        experts, float(scaling), rows_index, x, topk_ids, topk_weights, *(lora or ())
    )


__all__ = [
    "RocmExpertLeafRowsFunction",
    "RocmMxfp4Experts",
    "bi_moe_slot_leaves",
    "rocm_expert_leaf_rows",
    "situ_fp32",
]
