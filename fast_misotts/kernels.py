"""Triton kernels for one utterance at a time (MisoTTS(kernels=True); experimental, so far slower than
compile="autotune"): fewer, fused kernels where a frame's time goes at one row (the depth decoder's 31 steps, each a few
small kernels whose fixed cost outweighs their work). The same arithmetic in another order, so no longer bit for bit
the original's in FP32: judged by what the audio says, like quantize.

- decode_attention: the depth decoder's attention (at most 32 positions) in one kernel instead of a product, a masked
  softmax and a product: each program one sequence, position and key/value head, the query heads that share it.
- sample_topk: the original's top-k sampling in one kernel instead of six (top k, masking, two softmaxes, the
  noise, the argmax): the same distribution (each code drawn with probability softmax(logits / temperature) over
  the top k), its random numbers from Triton's Philox, a seed tensor advanced in place each call so that a CUDA
  graph replays fresh draws.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # the CPU-only tests: these kernels need CUDA
    triton = tl = None


if triton is not None:
    @triton.jit
    def _decode_attention(q_ptr, k_ptr, v_ptr, pos_ptr, out_ptr, steps, slots, scale,
                          stride_qb, stride_qs, stride_qh, stride_kb, stride_kh, stride_kt,
                          stride_ob, stride_os, stride_oh,
                          GROUPS: tl.constexpr, BLOCK_G: tl.constexpr, HEAD_DIM: tl.constexpr,
                          BLOCK_T: tl.constexpr):
        row, kv_head, step = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        group, dim, slot = tl.arange(0, BLOCK_G), tl.arange(0, HEAD_DIM), tl.arange(0, BLOCK_T)
        head = kv_head * GROUPS + group
        query = tl.load(q_ptr + row * stride_qb + step * stride_qs + head[:, None] * stride_qh + dim[None, :],
                        mask=(group < GROUPS)[:, None], other=0.0).to(tl.float32)  # [G, D]
        kv = row * stride_kb + kv_head * stride_kh + slot[:, None] * stride_kt + dim[None, :]
        keys = tl.load(k_ptr + kv, mask=(slot < slots)[:, None], other=0.0).to(tl.float32)  # [T, D]
        values = tl.load(v_ptr + kv, mask=(slot < slots)[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(query[:, None, :] * keys[None, :, :], axis=2) * scale  # [G, T]
        position = tl.load(pos_ptr + row * steps + step)
        scores = tl.where((slot <= position)[None, :] & (slot < slots)[None, :], scores, float("-inf"))
        weights = tl.exp(scores - tl.max(scores, axis=1)[:, None])
        weights = weights / tl.sum(weights, axis=1)[:, None]
        out = tl.sum(weights[:, :, None] * values[None, :, :], axis=1)  # [G, D]
        tl.store(out_ptr + row * stride_ob + step * stride_os + head[:, None] * stride_oh + dim[None, :],
                 out.to(out_ptr.dtype.element_ty), mask=(group < GROUPS)[:, None])

    @triton.jit
    def _sample_topk(logits_ptr, out_ptr, seed_ptr, vocab, stride, inverse_temperature, TOPK: tl.constexpr,
                     BLOCK_V: tl.constexpr, GREEDY: tl.constexpr):
        row = tl.program_id(0)
        index = tl.arange(0, BLOCK_V)
        logits = tl.load(logits_ptr + row * stride + index, mask=index < vocab, other=float("-inf")).to(tl.float32)
        if GREEDY:
            choice = tl.argmax(logits, axis=0)
        else:
            scaled = logits * inverse_temperature
            threshold = tl.sort(scaled, descending=True)  # the k-th largest: everything below it is out
            kth = tl.sum(tl.where(index == TOPK - 1, threshold, 0.0), axis=0)
            kept = (scaled >= kth) & (index < vocab)
            weights = tl.where(kept, tl.exp(scaled - tl.max(scaled, axis=0)), 0.0)
            # argmax(p / e) with e exponential: an exact draw from p (p's normalisation does not move the argmax)
            uniform = tl.rand(tl.load(seed_ptr), row * BLOCK_V + index)
            noise = -tl.log(tl.maximum(uniform, 1e-20))
            choice = tl.argmax(tl.where(kept, weights / noise, -1.0), axis=0)
        tl.store(out_ptr + row, choice.to(tl.int64))


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


def decode_attention(q, keys, values, positions):
    """Attention of q [B, S, heads, head_dim] at positions [B, S] over keys, values [B, kv_heads, T, head_dim] (T at
    most 64) where slot <= position: [B, S, heads * head_dim], as model.grouped_attention gives it."""
    batch, steps, heads, head_dim = q.shape
    kv_heads, slots = keys.shape[1], keys.shape[2]
    groups = heads // kv_heads
    q = q.contiguous()
    out = torch.empty_like(q)
    _decode_attention[(batch, kv_heads, steps)](
        q, keys, values, positions.contiguous(), out, steps, slots, head_dim ** -0.5,
        q.stride(0), q.stride(1), q.stride(2), keys.stride(0), keys.stride(1), keys.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        GROUPS=groups, BLOCK_G=triton.next_power_of_2(groups), HEAD_DIM=head_dim,
        BLOCK_T=triton.next_power_of_2(slots))
    return out.view(batch, steps, heads * head_dim)


def sample_topk(logits, seed, temperature, topk, greedy=False):
    """One code a row [B, 1] from logits [B, vocab]: the argmax (greedy) or a top-k draw at temperature; `seed` a
    one-element int64 CUDA tensor, advanced in place."""
    batch, vocab = logits.shape
    logits = logits.contiguous()
    out = torch.empty(batch, dtype=torch.int64, device=logits.device)
    _sample_topk[(batch,)](logits, out, seed, vocab, logits.stride(0), 1.0 / temperature, TOPK=topk,
                           BLOCK_V=triton.next_power_of_2(vocab), GREEDY=greedy)
    if not greedy:
        seed += 1
    return out.unsqueeze(1)
