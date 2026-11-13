"""Low-bit weights for MisoTTS's two stacks: MisoTTS(quantize="fp8" | "int8" | "int4"). Faster, but no longer the
original's arithmetic, so judged by what the audio says (python -m fast_misotts.bench quality and score). Each Linear
in the layers of the backbone and the depth decoder is replaced; the embeddings, the projection between the stacks,
the heads and the norms keep the model's dtype. Use it with compile=True: the inputs' quantization is a few small
kernels a layer, which compiling fuses.

- "fp8": weights and inputs in FP8 (E4M3), each weight row (output channel) and each input row scaled to the format's
  range, multiplied by torch._scaled_mm and scaled back. Needs FP8 GPUs (compute capability 8.9 or more: L40S, RTX
  4090, H100, H200).
- "int8": the same in 8-bit integers through torch._int_mm (compute capability 8.0 or more).
- "int8w": weights alone in 8 bits (a scale a row), inputs in BF16, for few rows: compiled, the conversion folds into
  the product's kernel (eager converts the whole weight at each call).
- "int4": weights alone in 4 bits, in groups of 128 along each row with a scale and a zero point each, inputs in BF16,
  through PyTorch's tinygemm kernel (aten._weight_int4pack_mm; compute capability 8.0 or more): the fewest bytes a
  frame reads, for one utterance at a time. "int4/gN": groups of N (32, 64, 128 or 256).

"MODE:backbone" or "MODE:decoder" quantizes that stack alone; parts joined by "+" give each stack its own
("int4:backbone+int4/g32:decoder").
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

MODES = ("fp8", "int8", "int8w", "int4")
STACKS = ("backbone", "decoder")
INT4_GROUPS = (32, 64, 128, 256)  # tinygemm's
FP8_MAX = 448.0  # torch.float8_e4m3fn's largest value


def _rows(x):
    return x.reshape(-1, x.shape[-1])


def _matmul_output(out, x, count, x_scale, w_scale, features):
    """The low-precision product's first `count` rows, scaled back by each input row's and weight row's scale."""
    out = out[0] if isinstance(out, tuple) else out  # torch._scaled_mm returned (out, amax) before torch 2.4
    out = (out[:count].float() * (x_scale[:count] * w_scale)).to(x.dtype)
    return out.reshape(*x.shape[:-1], features)


class Fp8Linear(nn.Module):
    """y = x W^T in FP8: W row-scaled once, x row-scaled at each call; inputs padded to a multiple of 16 rows."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        w = weight.detach().float()
        scale = w.abs().amax(dim=1).clamp(min=1e-12) / FP8_MAX
        self.register_buffer("weight", (w / scale[:, None]).to(torch.float8_e4m3fn))
        self.register_buffer("scale", scale)
        self.register_buffer("one", torch.ones((), device=weight.device))
        self.out_features = weight.shape[0]

    def forward(self, x):
        rows = _rows(x)
        count = rows.shape[0]
        if count % 16:
            rows = F.pad(rows, (0, 0, 0, 16 - count % 16))
        x_scale = rows.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / FP8_MAX
        quantized = (rows.float() / x_scale).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(quantized, self.weight.t(), scale_a=self.one, scale_b=self.one,
                               out_dtype=torch.bfloat16)
        return _matmul_output(out, x, count, x_scale, self.scale, self.out_features)


class Int8Linear(nn.Module):
    """y = x W^T in int8 (symmetric, 127 levels each side): W row-scaled once, x row-scaled at each call; inputs padded
    to more than 16 rows and a multiple of 8, as torch._int_mm needs."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        w = weight.detach().float()
        scale = w.abs().amax(dim=1).clamp(min=1e-12) / 127
        self.register_buffer("weight", torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale)
        self.out_features = weight.shape[0]

    def forward(self, x):
        rows = _rows(x)
        count = rows.shape[0]
        padded = max(32, -(-count // 8) * 8)
        if padded != count:
            rows = F.pad(rows, (0, 0, 0, padded - count))
        x_scale = rows.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / 127
        quantized = torch.round(rows.float() / x_scale).clamp(-127, 127).to(torch.int8)
        return _matmul_output(torch._int_mm(quantized, self.weight.t()), x, count, x_scale, self.scale,
                              self.out_features)


class Int8WeightLinear(nn.Module):
    """y = (x W^T) * scale with W in int8 (symmetric, 127 levels each side, a scale a row) and x as it comes. Compiled
    with a few rows (one utterance's step), the product is written as a reduction over the inputs so that Inductor
    reads the int8 weight once in one kernel, converting as it goes (as a product, compile="layers" converted the
    whole weight to BF16 first: 118 ms a frame on an L40S against 25 ms this way)."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        w = weight.detach().float()
        scale = w.abs().amax(dim=1).clamp(min=1e-12) / 127
        self.register_buffer("weight", torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.to(weight.dtype))
        self.out_features = weight.shape[0]

    def forward(self, x):
        if torch.compiler.is_compiling() and x.numel() <= 16 * x.shape[-1]:
            return (x.unsqueeze(-2) * self.weight.to(x.dtype)).sum(-1) * self.scale
        return F.linear(x, self.weight.to(x.dtype)) * self.scale


def int4_groups(weight: torch.Tensor, groupsize: int = 128):
    """Asymmetric 4-bit codes of each row's groups of `groupsize`: codes [N, K] (0-15), scales and zero points
    [N, K / groupsize], with w ~ (code - 8) * scale + zero, as tinygemm reads them."""
    rows, columns = weight.shape
    groups = weight.float().reshape(rows, columns // groupsize, groupsize)
    top, bottom = groups.amax(dim=-1), groups.amin(dim=-1)
    scale = (top - bottom).clamp(min=1e-6) / 15
    codes = torch.round((groups - bottom[..., None]) / scale[..., None]).clamp(0, 15).to(torch.int32)
    return codes.reshape(rows, columns), scale, bottom + 8 * scale


class Int4Linear(nn.Module):
    """y = x W^T with W in 4-bit groups (int4_groups), packed for tinygemm; x in BF16."""

    def __init__(self, weight: torch.Tensor, groupsize: int = 128, inner_k_tiles: int = 8):
        super().__init__()
        if not hasattr(torch.ops.aten, "_weight_int4pack_mm"):
            raise NotImplementedError("quantize='int4' needs PyTorch's tinygemm kernel (aten._weight_int4pack_mm)")
        codes, scale, zero = int4_groups(weight, groupsize)
        packed = (codes[:, ::2] << 4 | codes[:, 1::2]).to(torch.uint8)  # two codes a byte (PyTorch 2.5 on)
        self.register_buffer("weight", torch.ops.aten._convert_weight_to_int4pack(packed, inner_k_tiles))
        self.register_buffer("scales_and_zeros", torch.stack([scale, zero], dim=-1).to(torch.bfloat16)
                             .transpose(0, 1).contiguous())  # [K / groupsize, N, 2]
        self.groupsize, self.out_features = groupsize, weight.shape[0]

    def forward(self, x):
        out = torch.ops.aten._weight_int4pack_mm(_rows(x).to(torch.bfloat16), self.weight, self.groupsize,
                                                 self.scales_and_zeros)
        return out.to(x.dtype).reshape(*x.shape[:-1], self.out_features)


def parse(spec: str, groupsize: int = 128) -> dict[str, tuple[str, int]]:
    """A quantize spec as {stack: (mode, int4 group size)}: "MODE[/gN][:STACK]", parts joined by "+"."""
    plan = {}
    for part in spec.split("+"):
        name, _, only = part.partition(":")
        mode, _, group = name.partition("/")
        size = int(group[1:]) if group[1:].isdigit() and group[:1] == "g" else groupsize
        if (mode not in MODES or only not in ("", *STACKS) or (group and (mode != "int4" or f"g{size}" != group))
                or (mode == "int4" and size not in INT4_GROUPS)):
            raise ValueError(f"quantize: MODE[/gN][:backbone|:decoder], parts joined by '+', MODE one of {MODES}, "
                             f"N one of {INT4_GROUPS}; not {spec!r}")
        for stack in [only] if only else STACKS:
            if stack in plan:
                raise ValueError(f"quantize: {stack} twice in {spec!r}")
            plan[stack] = (mode, size)
    return plan


def quantize(model, spec: str, groupsize: int = 128):
    """Every Linear in the layers of model.backbone and model.decoder replaced by its stack's mode's (parse), in
    place."""
    for stack, (mode, size) in parse(spec, groupsize).items():
        make = {"fp8": Fp8Linear, "int8": Int8Linear, "int8w": Int8WeightLinear,
                "int4": lambda weight: Int4Linear(weight, size)}[mode]
        for layer in getattr(model, stack).layers:
            for parent in (layer.attn, layer.mlp):  # whatever Linears it has, fused (fuse_projections) or not
                for name, child in list(parent.named_children()):
                    if isinstance(child, nn.Linear):
                        setattr(parent, name, make(child.weight))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model
