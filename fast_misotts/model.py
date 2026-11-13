"""MisoTTS 8B (MisoLabs/MisoTTS) in plain PyTorch, without torchtune.

The same parameters, names and arithmetic as the original (Sesame CSM's layout): a Llama backbone reads each position
as the sum of its text embedding and its 32 audio-code embeddings and predicts codebook 0; a small Llama depth decoder
reads the backbone's last state and codebook 0 and predicts codebooks 1-31, one at a time. What differs is the cache:
static key/value tensors indexed by each sequence's own position, so sequences of different lengths decode in one
batch, and every shape stays fixed from one frame to the next (CUDA graphs, fast_misotts/engine.py)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StackConfig:
    """One Llama stack (torchtune's llama3_2 builder with these arguments)."""
    layers: int
    dim: int
    heads: int
    kv_heads: int
    ffn: int
    max_seq_len: int = 2048
    rope_base: int = 500_000
    rope_scale: float = 32
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads


@dataclass(frozen=True)
class MisoConfig:
    backbone: StackConfig = field(default_factory=lambda: StackConfig(32, 4096, 32, 8, 14_336))
    decoder: StackConfig = field(default_factory=lambda: StackConfig(8, 1536, 24, 6, 6912))
    text_vocab: int = 128_256
    audio_vocab: int = 2051
    codebooks: int = 32


MISO_8B = MisoConfig()


def rope_table(spec: StackConfig) -> torch.Tensor:
    """cos and sin [max_seq_len, head_dim // 2, 2] in FP32, as torchtune's Llama3ScaledRoPE builds them (scale factor,
    low 1, high 4, old context 8192), op for op on the CPU (whatever the default device) so the values match."""
    dim = spec.head_dim
    freqs = 1.0 / (spec.rope_base ** (torch.arange(0, dim, 2, device="cpu")[: dim // 2].float() / dim))
    old_context, low_factor, high_factor = 8192, 1, 4
    low_wavelen, high_wavelen = old_context / low_factor, old_context / high_factor
    scaled = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_wavelen:
            scaled.append(freq)
        elif wavelen > low_wavelen:
            scaled.append(freq / spec.rope_scale)
        else:
            smooth = (old_context / wavelen - low_factor) / (high_factor - low_factor)
            scaled.append((1 - smooth) * freq / spec.rope_scale + smooth * freq)
    theta = torch.stack(scaled).to(freqs.dtype)
    angles = torch.einsum("i, j -> ij", torch.arange(spec.max_seq_len, dtype=theta.dtype, device="cpu"), theta).float()
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def apply_rope(x: torch.Tensor, table: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """x [B, S, heads, head_dim] rotated in adjacent pairs (torchtune's layout) at positions [B, S], in FP32."""
    rope = table[positions]  # [B, S, head_dim // 2, 2]
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    rope = rope.view(rope.shape[0], rope.shape[1], 1, rope.shape[2], 2)
    out = torch.stack([pairs[..., 0] * rope[..., 0] - pairs[..., 1] * rope[..., 1],
                       pairs[..., 1] * rope[..., 0] + pairs[..., 0] * rope[..., 1]], -1)
    return out.flatten(3).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.scale


def grouped_attention(q, keys, values, allowed):
    """Softmax attention of q [B, S, heads, head_dim] over keys, values [B, kv_heads, T, head_dim] where allowed
    [B, S, T] (bool) holds, each query head reading kv head (head // groups) without copying the cache per head.
    Scores and their softmax in FP32. Returns [B, S, heads * head_dim]."""
    batch, steps, heads, head_dim = q.shape
    kv_heads = keys.shape[1]
    groups = heads // kv_heads
    q = q.view(batch, steps, kv_heads, groups, head_dim).permute(0, 2, 3, 1, 4).reshape(
        batch, kv_heads, groups * steps, head_dim)
    scores = torch.matmul(q, keys.transpose(-1, -2)).float() * head_dim ** -0.5
    scores = scores.view(batch, kv_heads, groups, steps, -1).masked_fill(~allowed[:, None, None], -math.inf)
    probs = scores.softmax(-1).to(values.dtype).view(batch, kv_heads, groups * steps, -1)
    out = torch.matmul(probs, values).view(batch, kv_heads, groups, steps, head_dim)
    return out.permute(0, 3, 1, 2, 4).reshape(batch, steps, heads * head_dim)


class Attention(nn.Module):
    def __init__(self, spec: StackConfig):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = spec.heads, spec.kv_heads, spec.head_dim
        self.q_proj = nn.Linear(spec.dim, spec.heads * spec.head_dim, bias=False)
        self.k_proj = nn.Linear(spec.dim, spec.kv_heads * spec.head_dim, bias=False)
        self.v_proj = nn.Linear(spec.dim, spec.kv_heads * spec.head_dim, bias=False)
        self.output_proj = nn.Linear(spec.heads * spec.head_dim, spec.dim, bias=False)

    def forward(self, x, table, positions, cache, allowed=None, rows=None):
        """x [B, S, dim] at positions [B, S], written into cache = (keys, values) [B', kv_heads, T, head_dim] at
        those slots of its rows [B] (default: the first B). allowed [B, S, T]: attend over the cache where it holds
        (all its rows); None: a fresh prompt at positions 0..S-1 attends causally over itself (fused attention, the
        cache only written)."""
        batch, steps, _ = x.shape
        if getattr(self, "qkv_proj", None) is not None:  # fuse_projections(): one product for the three
            q, k, v = self.qkv_proj(x).split((self.heads * self.head_dim, self.kv_heads * self.head_dim,
                                              self.kv_heads * self.head_dim), dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = apply_rope(q.view(batch, steps, self.heads, self.head_dim), table, positions)
        k = apply_rope(k.view(batch, steps, self.kv_heads, self.head_dim), table, positions)
        v = v.view(batch, steps, self.kv_heads, self.head_dim)
        keys, values = cache
        if rows is None:
            rows = torch.arange(batch, device=x.device)
        rows = rows[:, None].expand(batch, steps)
        keys[rows, :, positions] = k
        values[rows, :, positions] = v
        if allowed is None:
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                 is_causal=True, enable_gqa=True)
            out = out.transpose(1, 2).reshape(batch, steps, -1)
        elif getattr(self, "kernel", None) is not None:  # MisoModel.use_kernels: allowed is slots <= positions
            out = self.kernel(q, keys, values, positions)
        else:
            out = grouped_attention(q, keys, values, allowed)
        return self.output_proj(out)


class FeedForward(nn.Module):
    def __init__(self, spec: StackConfig):
        super().__init__()
        self.w1 = nn.Linear(spec.dim, spec.ffn, bias=False)
        self.w2 = nn.Linear(spec.ffn, spec.dim, bias=False)
        self.w3 = nn.Linear(spec.dim, spec.ffn, bias=False)

    def forward(self, x):
        if getattr(self, "w13", None) is not None:  # fuse_projections(): one product for the two
            gate, up = self.w13(x).chunk(2, dim=-1)
            return self.w2(F.silu(gate) * up)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Layer(nn.Module):
    def __init__(self, spec: StackConfig):
        super().__init__()
        self.sa_norm = RMSNorm(spec.dim, spec.norm_eps)
        self.attn = Attention(spec)
        self.mlp_norm = RMSNorm(spec.dim, spec.norm_eps)
        self.mlp = FeedForward(spec)

    def forward(self, x, table, positions, cache, allowed, rows=None):
        h = self.attn(self.sa_norm(x), table, positions, cache, allowed, rows) + x
        return h + self.mlp(self.mlp_norm(h))


class Stack(nn.Module):
    """torchtune's TransformerDecoder without token embeddings or output projection: layers, then the final norm."""

    def __init__(self, spec: StackConfig):
        super().__init__()
        self.spec = spec
        self.layers = nn.ModuleList(Layer(spec) for _ in range(spec.layers))
        self.norm = RMSNorm(spec.dim, spec.norm_eps)
        self.register_buffer("rope", rope_table(spec), persistent=False)

    def _apply(self, fn, recurse=True):
        # The original's RoPE table is a buffer: model.to(dtype) rounds it like the weights (BF16 cos and sin, the
        # rotation itself in FP32). Rebuilt here from FP32 in the stack's dtype, so to_empty's blank buffer gets its
        # values and the original's rounding is kept.
        super()._apply(fn, recurse)
        self.rope = rope_table(self.spec).to(device=self.rope.device, dtype=self.norm.scale.dtype)
        return self

    def new_cache(self, batch, length, dtype, device):
        shape = (batch, self.spec.kv_heads, length, self.spec.head_dim)
        return [(torch.zeros(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device))
                for _ in self.layers]

    def forward(self, x, positions, cache, allowed=None, rows=None, call=None):
        """call(layer, x, rope, positions, cache, allowed, rows): how each layer runs (a compiled Layer.forward, say);
        by default the layer itself."""
        for layer, layer_cache in zip(self.layers, cache):
            if call is None:
                x = layer(x, self.rope, positions, layer_cache, allowed, rows)
            else:
                x = call(layer, x, self.rope, positions, layer_cache, allowed, rows)
        return self.norm(x)


class MisoModel(nn.Module):
    """The original Model's parameters under the same names: its model.safetensors loads as is."""

    def __init__(self, config: MisoConfig = MISO_8B):
        super().__init__()
        self.config = config
        self.backbone = Stack(config.backbone)
        self.decoder = Stack(config.decoder)
        self.text_embeddings = nn.Embedding(config.text_vocab, config.backbone.dim)
        self.audio_embeddings = nn.Embedding(config.audio_vocab * config.codebooks, config.backbone.dim)
        self.projection = nn.Linear(config.backbone.dim, config.decoder.dim, bias=False)
        self.codebook0_head = nn.Linear(config.backbone.dim, config.audio_vocab, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.codebooks - 1, config.decoder.dim, config.audio_vocab))
        self.register_buffer("offsets", torch.arange(config.codebooks) * config.audio_vocab, persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.offsets = torch.arange(self.config.codebooks, device=self.offsets.device) * self.config.audio_vocab
        return self

    @torch.no_grad()
    def fuse_projections(self):
        """In every layer of both stacks, q_proj, k_proj and v_proj become one qkv_proj and w1 and w3 one w13, their
        rows stacked (the separate ones dropped): three products a layer instead of seven, each output the same dot
        product. Done after loading (the original names load as they are)."""
        def stacked(*linears):
            weight = torch.cat([linear.weight for linear in linears])
            fused = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype)
            fused.weight.copy_(weight)
            return fused.requires_grad_(False)

        for stack in (self.backbone, self.decoder):
            for layer in stack.layers:
                attn, mlp = layer.attn, layer.mlp
                if getattr(attn, "qkv_proj", None) is None:
                    attn.qkv_proj = stacked(attn.q_proj, attn.k_proj, attn.v_proj)
                    del attn.q_proj, attn.k_proj, attn.v_proj
                if getattr(mlp, "w13", None) is None:
                    mlp.w13 = stacked(mlp.w1, mlp.w3)
                    del mlp.w1, mlp.w3
        return self

    def use_kernels(self, on: bool = True):
        """The depth decoder's attention through fast_misotts.kernels.decode_attention (one kernel a layer and step;
        its cache holds at most 32 positions), or (on=False) back to grouped_attention."""
        from .kernels import decode_attention
        for layer in self.decoder.layers:
            layer.attn.kernel = decode_attention if on else None
        return self

    def embed(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tokens, mask [B, S, codebooks + 1] (32 codes, then text) to the backbone's inputs [B, S, dim]: each
        position's masked embeddings summed, as Model._embed_tokens and generate_frame do."""
        text = self.text_embeddings(tokens[..., -1]).unsqueeze(-2)
        audio = self.audio_embeddings(tokens[..., :-1] + self.offsets)
        return (torch.cat([audio, text], dim=-2) * mask.unsqueeze(-1)).sum(dim=2)

    def embed_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """A generated frame [B, codebooks] as the next backbone input [B, 1, dim], as Generator.generate feeds it:
        the codes and a text token 0 that the mask leaves out, the same 33-way sum."""
        tokens = torch.cat([codes, torch.zeros_like(codes[:, :1])], dim=1).unsqueeze(1)
        mask = torch.ones_like(tokens, dtype=torch.bool)
        mask[..., -1] = False
        return self.embed(tokens, mask)

    def embed_code(self, codebook: int, codes: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(codes + codebook * self.config.audio_vocab)


def load_model(path, device="cuda", dtype=torch.bfloat16, config: MisoConfig = MISO_8B) -> MisoModel:
    """The original model.safetensors (MisoLabs/MisoTTS) tensor by tensor onto the device in dtype, so the host never
    holds the FP32 copy whole. Cache and mask buffers an original checkpoint may carry are ignored; every parameter
    must be present."""
    from safetensors import safe_open
    with torch.device("meta"):
        model = MisoModel(config)
    model = model.to(dtype=dtype).to_empty(device=device)  # parameters allocated once, in dtype
    wanted = dict(model.named_parameters())
    seen = set()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key in wanted:
                with torch.no_grad():
                    wanted[key].copy_(handle.get_tensor(key).to(device=device, dtype=dtype))
                seen.add(key)
    missing = sorted(set(wanted) - seen)
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} parameters, e.g. {missing[:3]}")
    return model.eval().requires_grad_(False)
