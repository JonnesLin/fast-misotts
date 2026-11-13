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

A pre-quantized checkpoint (int4 and int8w): export_quantized saves a fused model's codes and scales, AWQ in them if
applied; MisoTTS(model=...) recognizes one (saved_spec) and builds its quantized layers straight from the codes
(load_quantized), never holding the BF16 weights.
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


def int8_rows(weight: torch.Tensor):
    """Symmetric 8-bit codes [N, K] (int8, 127 levels each side) of weight [N, K] and each row's scale [N] (in the
    weight's dtype)."""
    w = weight.detach().float()
    scale = w.abs().amax(dim=1).clamp(min=1e-12) / 127
    return torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8), scale.to(weight.dtype)


class Int8WeightLinear(nn.Module):
    """y = (x W^T) * scale with W in int8 (int8_rows) and x as it comes; from a weight, or from its codes and scales
    (a pre-quantized checkpoint). Compiled with a few rows (one utterance's step), the product is written as a
    reduction over the inputs so that Inductor reads the int8 weight once in one kernel, converting as it goes (as a
    product, compile="layers" converted the whole weight to BF16 first: 118 ms a frame on an L40S against 25 ms this
    way)."""

    def __init__(self, weight: torch.Tensor | None = None, *, codes: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None):
        super().__init__()
        if weight is not None:
            codes, scale = int8_rows(weight)
        self.register_buffer("weight", codes)
        self.register_buffer("scale", scale)
        self.out_features = codes.shape[0]

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


def int4_parts(weight: torch.Tensor, groupsize: int = 128):
    """int4_groups' codes two a byte [N, K / 2] (uint8, the layout tinygemm's packer takes; PyTorch 2.5 on) and the
    scales and zero points [K / groupsize, N, 2] (BF16) of weight [N, K]: what Int4Linear packs."""
    codes, scale, zero = int4_groups(weight, groupsize)
    packed = (codes[:, ::2] << 4 | codes[:, 1::2]).to(torch.uint8)
    return packed, torch.stack([scale, zero], dim=-1).to(torch.bfloat16).transpose(0, 1).contiguous()


class Int4Linear(nn.Module):
    """y = x W^T with W in 4-bit groups (int4_groups), packed for tinygemm; x in BF16. From a weight, or from
    int4_parts' codes and scales (a pre-quantized checkpoint)."""

    def __init__(self, weight: torch.Tensor | None = None, groupsize: int = 128, inner_k_tiles: int = 8, *,
                 codes: torch.Tensor | None = None, scales_and_zeros: torch.Tensor | None = None):
        super().__init__()
        if not hasattr(torch.ops.aten, "_weight_int4pack_mm"):
            raise NotImplementedError("quantize='int4' needs PyTorch's tinygemm kernel (aten._weight_int4pack_mm)")
        if weight is not None:
            codes, scales_and_zeros = int4_parts(weight, groupsize)
        self.register_buffer("weight", torch.ops.aten._convert_weight_to_int4pack(codes, inner_k_tiles))
        self.register_buffer("scales_and_zeros", scales_and_zeros)  # [K / groupsize, N, 2]
        self.groupsize, self.out_features = groupsize, codes.shape[0]

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


FORMAT = "fast_misotts.quantized/1"  # a pre-quantized checkpoint's metadata "format"


def _linears(model, plan):
    """(key prefix, parent, name, mode, group size) of every Linear in the planned stacks' layers."""
    found = []
    for stack, (mode, size) in plan.items():
        for number, layer in enumerate(getattr(model, stack).layers):
            for part in ("attn", "mlp"):
                parent = getattr(layer, part)
                for name, child in parent.named_children():
                    if isinstance(child, nn.Linear):
                        found.append((f"{stack}.layers.{number}.{part}.{name}", parent, name, mode, size))
    return found


def export_quantized(model, spec: str, groupsize: int = 128):
    """A fused model's tensors as quantize(model, spec) would hold them, for load_quantized: each quantized Linear's
    codes (PREFIX.int4 and PREFIX.scales_and_zeros, or PREFIX.int8 and PREFIX.scale) and every other tensor as it is
    (AWQ, if applied, is in them), with the safetensors metadata. int4 and int8w only; the model is left as it is."""
    plan = parse(spec, groupsize)
    if any(mode not in ("int4", "int8w") for mode, _ in plan.values()):
        raise ValueError(f"export_quantized: int4 and int8w only, not {spec!r}")
    if getattr(model.backbone.layers[0].attn, "qkv_proj", None) is None:
        raise ValueError("export_quantized: a fused model (fuse_projections)")
    tensors, replaced = {}, set()
    for prefix, parent, name, mode, size in _linears(model, plan):
        weight = getattr(parent, name).weight
        if mode == "int4":
            tensors[f"{prefix}.int4"], tensors[f"{prefix}.scales_and_zeros"] = int4_parts(weight, size)
        else:
            tensors[f"{prefix}.int8"], tensors[f"{prefix}.scale"] = int8_rows(weight)
        replaced.add(f"{prefix}.weight")
    tensors.update({key: value for key, value in model.state_dict().items() if key not in replaced})
    return ({key: value.detach().contiguous().cpu() for key, value in tensors.items()},
            {"format": FORMAT, "quantize": spec, "groupsize": str(groupsize)})


def saved_spec(path) -> str | None:
    """The quantize spec a pre-quantized checkpoint was saved with (export_quantized), or None for another file."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    return metadata.get("quantize") if metadata.get("format") == FORMAT else None


def load_quantized(path, device="cuda", dtype=torch.bfloat16, config=None):
    """A pre-quantized checkpoint (export_quantized) onto the device: the model's skeleton fused, each quantized Linear
    built from its codes, every other tensor read in dtype. The BF16 weights of the quantized Linears are never held:
    a few GB where loading the original and quantizing it takes 16 first."""
    from safetensors import safe_open
    from .model import MISO_8B, MisoModel
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != FORMAT:
            raise ValueError(f"{path}: not a fast_misotts pre-quantized checkpoint")
        plan = parse(metadata["quantize"], int(metadata.get("groupsize", 128)))
        with torch.device("meta"):
            model = MisoModel(config or MISO_8B).fuse_projections()
        linears = _linears(model, plan)
        for _, parent, name, _, _ in linears:
            setattr(parent, name, nn.Module())  # a placeholder: no storage for the BF16 weight
        model = model.to(dtype=dtype).to_empty(device=device)  # the rest allocated once (RoPE and offsets rebuilt)
        with torch.no_grad():
            for key, value in model.named_parameters():
                value.copy_(handle.get_tensor(key).to(device=device, dtype=dtype))
        for prefix, parent, name, mode, size in linears:
            def get(suffix):
                return handle.get_tensor(f"{prefix}.{suffix}").to(device)
            if mode == "int4":
                module = Int4Linear(groupsize=size, codes=get("int4"), scales_and_zeros=get("scales_and_zeros"))
            else:
                module = Int8WeightLinear(codes=get("int8"), scale=get("scale"))
            setattr(parent, name, module)
    return model.eval().requires_grad_(False)


def main():
    """python -m fast_misotts.quant --quantize SPEC [--awq FILE] --out model.safetensors: a pre-quantized checkpoint."""
    import argparse
    import json
    from pathlib import Path
    from safetensors.torch import save_file
    from .api import DEFAULT_MODEL, model_file
    from .model import load_model
    parser = argparse.ArgumentParser(description="A pre-quantized fast_misotts checkpoint: the model (AWQ applied if "
                                                 "given, fused) as quantize=SPEC holds it, for MisoTTS(model=OUT).")
    parser.add_argument("--model", help="model.safetensors, its folder or a Hub repo (default: MisoLabs/MisoTTS)")
    parser.add_argument("--quantize", required=True, help="int4 and int8w parts, e.g. int4:backbone+int8w:decoder")
    parser.add_argument("--awq", help="a python -m fast_misotts.awq search, for the int4 stacks")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = load_model(model_file(args.model or DEFAULT_MODEL), device=args.device, dtype=torch.bfloat16)
    if args.awq:
        from .awq import apply, load
        apply(model, load(args.awq), {stack: size for stack, (mode, size) in parse(args.quantize).items()
                                      if mode == "int4"})
    tensors, metadata = export_quantized(model.fuse_projections(), args.quantize)
    if args.awq:
        metadata["awq"] = Path(args.awq).name
    save_file(tensors, args.out, metadata=metadata)
    print(json.dumps({"out": args.out, "tensors": len(tensors), **metadata,
                      "gb": round(sum(value.numel() * value.element_size() for value in tensors.values()) / 1e9, 2)}))


if __name__ == "__main__":
    main()
