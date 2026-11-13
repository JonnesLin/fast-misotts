"""AWQ (Lin et al., 2024: activation-aware weight quantization) for quantize="int4": the same groups of 128, kernel
and speed (fast_misotts.quant), the weights first transformed so that rounding them costs the outputs less. AutoAWQ's
recipe for Llama layers, on both stacks:

1. Scales. A Linear's input channels times s and its input divided by s, the division folded into what makes the
   input: sa_norm's scale before q, k and v; mlp_norm's before w1 and w3; w3's rows before w2 (which reads
   silu(w1 x) * w3 x). In full precision the model computes the same function, and the channels whose inputs are
   large get more of the 4-bit grid. s = mean|x|^a / mean|w|^(1 - a) (AutoAWQ's; |w| a weight's share of its group's
   largest) or mean|x|^a (the paper's; a = 0 is no scaling), a in 0, 0.05, ..., 0.95: of these 40, the s whose
   rounded weights bring the output closest to the unrounded (mean squared error on the sampled rows; the output of
   q, k and v, of the whole MLP for w1 and w3, of w2). output_proj keeps its inputs: its 1/s would go into v's rows,
   each shared by four query heads (AutoAWQ leaves o_proj too when a model has fewer key/value heads than query heads).
2. Clipping. Each group of 128 weights clamped to +-r times its largest magnitude, r in 1, 0.95, ..., 0.55: the r
   whose rounded group gives the least squared error in the group's share of the output on 512 sampled rows. Not in
   q_proj and k_proj, whose errors meet in the attention scores rather than in the output.
3. Rounding: quantize="int4" as without AWQ.

Calibration: the Linears' inputs while the unquantized model speaks calibration_texts.json (128 texts, sampled): each
channel's mean |x| over every row, and a uniform sample of 4,096 rows. Every layer is transformed from those (the
unquantized model's) inputs, as AutoAWQ does. The scales and clipping are saved (about 50 MB) and applied as the model
loads, before fuse_projections:

    python3 -m fast_misotts.awq --model M --tokenizer T --out awq.safetensors     # once
    MisoTTS(..., quantize="int4", awq="awq.safetensors")

Each stack is searched for its own int4 groups (--groupsize backbone=128,decoder=32 for
quantize="int4:backbone+int4/g32:decoder"); applying a search to other groups is refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .quant import int4_groups

ROLES = ("attn_in", "attn_out", "mlp_in", "mlp_out")  # the inputs of q, k and v; of output_proj; of w1 and w3; of w2
SCALED = ("attn_in", "mlp_in", "mlp_out")
CLIPPED = (("attn", "v_proj", "attn_in"), ("attn", "output_proj", "attn_out"), ("mlp", "w1", "mlp_in"),
           ("mlp", "w3", "mlp_in"), ("mlp", "w2", "mlp_out"))  # (part, Linear, the role of its input)


class Inputs:
    """A Linear's inputs while calibrating: each channel's mean |x| over every row, and a uniform sample of up to
    `size` rows (each row a random key, the `size` smallest kept)."""

    def __init__(self, size: int):
        self.size, self.count = size, 0
        self.total = self.keys = self.rows = None

    def add(self, rows: torch.Tensor):
        if self.rows is None:
            self.total = torch.zeros(rows.shape[1], device=rows.device)
            self.keys = torch.full((self.size,), 2.0, device=rows.device)  # above any key: a free slot
            self.rows = torch.zeros(self.size, rows.shape[1], dtype=rows.dtype, device=rows.device)
        self.total += rows.float().abs().sum(0)
        self.count += rows.shape[0]
        keys = torch.rand(rows.shape[0], device=rows.device)
        passing = keys < self.keys.max()
        if bool(passing.any()):
            keys, order = torch.sort(keys[passing])
            worst, slots = torch.topk(self.keys, min(len(keys), self.size))  # the largest kept keys, largest first
            keys, rows = keys[:len(slots)], rows[passing][order][:len(slots)]
            take = keys < worst  # the smallest new keys against the largest old: the `size` smallest of both
            self.keys[slots[take]] = keys[take]
            self.rows[slots[take]] = rows[take]

    def mean(self) -> torch.Tensor:
        return self.total / max(1, self.count)

    def sample(self) -> torch.Tensor:
        """The sampled rows in FP32, in the order of their keys: any first n of them are a uniform sample too (the
        slots are not: the first rows to come fill the first slots)."""
        order = torch.argsort(self.keys)
        return self.rows[order[:int((self.keys <= 1).sum())]].float()


@torch.inference_mode()
def calibrate(model, prompts, rows: int = 4096, batch: int = 32, positions: int = 512, frames: int = 250,
              sampling=None, seed: int = 0, stacks=("backbone", "decoder")) -> dict[str, Inputs]:
    """Inputs (keyed "backbone.3.mlp_in", ROLES) of every layer's Linears in `stacks` while `model` (unquantized)
    generates from `prompts` [(tokens, mask)], `batch` rows at a time, at most `frames` frames each, without CUDA
    graphs (the hooks see every call). Only rows of an utterance count: a prefill's prompt positions (not its padding),
    and in a frame the rows in progress (not free rows, nor the backbone step a fresh row throws away)."""
    from .engine import Engine, Sampling
    engine = Engine(model, batch, positions, frames, sampling or Sampling(), graphs=False)
    prefill = {}

    def live(x):
        if "mask" in prefill:
            return x[prefill["mask"]]
        return x[~engine.ended & ~engine.fresh].reshape(-1, x.shape[-1])  # fresh is cleared before the depth decoder

    inputs, hooks = {}, []
    for stack in stacks:
        for number, layer in enumerate(getattr(model, stack).layers):
            attn, mlp = layer.attn, layer.mlp
            readers = (getattr(attn, "qkv_proj", None) or attn.q_proj, attn.output_proj,
                       getattr(mlp, "w13", None) or mlp.w1, mlp.w2)
            for role, linear in zip(ROLES, readers):
                key = f"{stack}.{number}.{role}"
                inputs[key] = Inputs(rows)
                hooks.append(linear.register_forward_pre_hook(
                    lambda module, args, key=key: inputs[key].add(live(args[0]))))
    admit = engine._admit

    def admitted(free, taken, limit):
        lengths = torch.tensor([len(tokens) for tokens, _ in taken], device=engine.device)
        prefill["mask"] = torch.arange(int(lengths.max()), device=engine.device) < lengths[:, None]
        try:
            admit(free, taken, limit)
        finally:
            prefill.clear()

    engine._admit = admitted
    torch.manual_seed(seed)
    try:
        for _ in engine.generate(prompts, frames):
            pass
    finally:
        for handle in hooks:
            handle.remove()
    return inputs


def rounded(weight: torch.Tensor, groupsize: int = 128) -> torch.Tensor:
    """weight [N, K] as quantize="int4" will hold it: int4_groups' codes back through their scales and zero points,
    those rounded to BF16 as Int4Linear keeps them; in FP32."""
    codes, scale, zero = int4_groups(weight, groupsize)
    rows, columns = weight.shape
    scale, zero = scale.bfloat16().float()[..., None], zero.bfloat16().float()[..., None]
    return ((codes.view(rows, -1, groupsize) - 8) * scale + zero).view(rows, columns)


def search_scale(weights, x, mean, output, groupsize: int = 128, grid: int = 20):
    """AWQ's scale for Linears sharing an input: their weights [N_i, K] (FP32), the sampled input rows x [n, K] and
    each channel's mean |x| [K]; output(weights, x) is what is compared. Returns the best s [K] and the mean squared
    error of the output (relative to its mean square) unscaled ("nearest") and with s."""
    stacked = torch.cat(weights)
    share = stacked.abs().view(-1, groupsize)
    share = (share / (share.amax(dim=1, keepdim=True) + 1e-6)).view(stacked.shape).mean(0)
    want = output(weights, x)
    power = want.pow(2).mean()
    best, nearest = None, None
    for duo in (True, False):
        for step in range(grid):
            a = step / grid
            s = (mean.pow(a) / (share.pow(1 - a) + 1e-4) if duo else mean.pow(a)).clamp(min=1e-4)
            s = s / (s.max() * s.min()).sqrt()
            loss = float((output([rounded(w * s, groupsize) / s for w in weights], x) - want).pow(2).mean() / power)
            if not duo and step == 0:
                nearest = loss  # s = 1
            if best is None or loss < best[0]:
                best = (loss, s, a, duo)
    loss, s, a, duo = best
    return s, {"nearest": nearest, "awq": loss, "a": a, "duo": duo}


def search_clip(weight, x, groupsize: int = 128, grid: int = 20, shrink: float = 0.5, chunk: int = 2 ** 26):
    """AWQ's clipping for weight [N, K] (FP32, scaled) on input rows x [n, K]: for each group, the step i (bound
    1 - i / grid times the group's largest magnitude, i below shrink * grid) whose rounded group gives the least mean
    squared error in the group's share of the output. Returns the steps [N, K / groupsize] (uint8)."""
    rows, columns = weight.shape
    groups = columns // groupsize
    xg = x.view(-1, groups, groupsize).transpose(0, 1)  # [G, n, g]
    per = max(1, chunk // (groups * xg.shape[1]))  # weight rows at a time: [G, n, rows] products of `chunk` floats
    steps = torch.zeros(rows, groups, dtype=torch.uint8, device=weight.device)
    for start in range(0, rows, per):
        part = weight[start:start + per].view(-1, groups, groupsize)
        want = torch.bmm(xg, part.permute(1, 2, 0))  # [G, n, rows]
        biggest = part.abs().amax(dim=-1, keepdim=True)
        least = None
        for step in range(int(shrink * grid)):
            bound = biggest * (1 - step / grid)
            got = rounded(torch.clamp(part, -bound, bound).view(-1, columns), groupsize).view(-1, groups, groupsize)
            error = (torch.bmm(xg, got.permute(1, 2, 0)) - want).pow(2).mean(dim=1).t()  # [rows, G]
            if least is None:
                least = error
                continue
            better = error < least
            least = torch.where(better, error, least)
            steps[start:start + per][better] = step
    return steps


def _fold(layer, role, s):
    """1/s into what makes `role`'s input, s into the input channels of the Linears that read it: in place."""
    attn, mlp = layer.attn, layer.mlp
    if role == "attn_in":
        layer.sa_norm.scale.div_(s)
        for linear in (attn.q_proj, attn.k_proj, attn.v_proj):
            linear.weight.mul_(s)
    elif role == "mlp_in":
        layer.mlp_norm.scale.div_(s)
        for linear in (mlp.w1, mlp.w3):
            linear.weight.mul_(s)
    elif role == "mlp_out":
        mlp.w3.weight.div_(s[:, None])
        mlp.w2.weight.mul_(s)
    else:
        raise ValueError(f"awq: no scale for {role!r}")


def _clip(linear, steps, grid, groupsize):
    weight = linear.weight
    grouped = weight.float().view(weight.shape[0], -1, groupsize)
    bound = grouped.abs().amax(dim=-1, keepdim=True) * (1 - steps.float()[..., None] / grid)
    weight.copy_(torch.clamp(grouped, -bound, bound).view(weight.shape))


def _unfused(layer):
    if getattr(layer.attn, "qkv_proj", None) is not None or getattr(layer.mlp, "w13", None) is not None:
        raise ValueError("awq: before fuse_projections (q, k and v, and w1 and w3, apart)")


@torch.no_grad()
def search(model, inputs: dict[str, Inputs], groupsize: int | dict[str, int] = 128, grid: int = 20,
           shrink: float = 0.5, clip_rows: int = 512, only: str | None = None,
           log=lambda line: print(line, flush=True)):
    """AWQ on every layer of the stacks calibrated (or `only` one), in place and in AutoAWQ's order: a layer's three
    scales searched on its unscaled weights and folded in, then its clipping searched on the scaled weights and
    inputs, and applied. groupsize: the int4 groups to search for, or a stack's own ({"decoder": 32}, 128 for the
    other). The model must not be fused. Returns the state (save(); apply() redoes it on a freshly loaded model) and
    each layer's errors."""
    state = {"grid": torch.tensor([grid])}
    report = {}
    stacks = [only] if only else [stack for stack in ("backbone", "decoder") if f"{stack}.0.attn_in" in inputs]
    for stack in stacks:
        size = groupsize.get(stack, 128) if isinstance(groupsize, dict) else groupsize
        state[f"{stack}.groupsize"] = torch.tensor([size])
        for number, layer in enumerate(getattr(model, stack).layers):
            _unfused(layer)
            name, attn, mlp = f"{stack}.{number}", layer.attn, layer.mlp
            x = {role: inputs[f"{name}.{role}"].sample() for role in ROLES}
            mean = {role: inputs[f"{name}.{role}"].mean() for role in ROLES}
            w2 = mlp.w2.weight.float()
            searched = {"attn_in": ((attn.q_proj, attn.k_proj, attn.v_proj), lambda ws, rows: rows @ torch.cat(ws).t()),
                        "mlp_in": ((mlp.w1, mlp.w3),
                                   lambda ws, rows: (F.silu(rows @ ws[0].t()) * (rows @ ws[1].t())) @ w2.t()),
                        "mlp_out": ((mlp.w2,), lambda ws, rows: rows @ ws[0].t())}
            scales, errors = {}, {}
            for role, (linears, output) in searched.items():
                scales[role], errors[role] = search_scale([linear.weight.float() for linear in linears], x[role],
                                                          mean[role], output, size, grid)
            for role, s in scales.items():
                _fold(layer, role, s)
                x[role] = x[role] / s
                state[f"{name}.{role}.scale"] = s
            for part, linear_name, role in CLIPPED:
                linear = getattr(getattr(layer, part), linear_name)
                steps = search_clip(linear.weight.float(), x[role][:clip_rows], size, grid, shrink)
                _clip(linear, steps, grid, size)
                state[f"{name}.{linear_name}.clip"] = steps
                errors[f"{linear_name}_clipped"] = round(float((steps > 0).float().mean()), 4)
            report[name] = errors
            log(json.dumps({name: errors}))
    return state, report


@torch.no_grad()
def apply(model, state: dict[str, torch.Tensor], groupsizes: dict[str, int] | None = None):
    """A search's scales and clipping (load()) on an unfused model, in place: each layer as search() left it.
    groupsizes: the int4 groups of the stacks being quantized ({"decoder": 32}), each the one searched for; the other
    stacks are left as they are. None: every stack the state has."""
    grid = int(state["grid"])
    for stack in ("backbone", "decoder"):
        if f"{stack}.0.attn_in.scale" not in state or (groupsizes is not None and stack not in groupsizes):
            continue
        groupsize = int(state[f"{stack}.groupsize"] if f"{stack}.groupsize" in state else state["groupsize"])
        if groupsizes is not None and groupsizes[stack] != groupsize:
            raise ValueError(f"awq: the {stack} was searched for groups of {groupsize}, not {groupsizes[stack]}")
        for number, layer in enumerate(getattr(model, stack).layers):
            name = f"{stack}.{number}"
            if f"{name}.attn_in.scale" not in state:
                continue
            _unfused(layer)
            device = layer.mlp.w2.weight.device
            for role in SCALED:
                _fold(layer, role, state[f"{name}.{role}.scale"].to(device=device, dtype=torch.float32))
            for part, linear_name, _ in CLIPPED:
                _clip(getattr(getattr(layer, part), linear_name), state[f"{name}.{linear_name}.clip"].to(device),
                      grid, groupsize)
    return model


def save(path, state: dict[str, torch.Tensor]):
    from safetensors.torch import save_file
    save_file({key: value.detach().contiguous().cpu() for key, value in state.items()}, str(path))


def load(path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="model.safetensors, its folder or a Hub repo (default: MisoLabs/MisoTTS)")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--texts", help="a JSON list of texts (default: fast_misotts/calibration_texts.json)")
    parser.add_argument("--rows", type=int, default=4096, help="input rows sampled for each Linear")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--frames", type=int, default=250, help="at most this many frames (80 ms each) a text")
    parser.add_argument("--positions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", choices=("backbone", "decoder"), help="one stack (default: both)")
    parser.add_argument("--groupsize", default="128",
                        help="the int4 groups searched for: N, or each stack's (backbone=128,decoder=32)")
    parser.add_argument("--out", required=True, help="the scales and clipping (.safetensors)")
    parser.add_argument("--report", help="each layer's errors and the times (JSON)")
    args = parser.parse_args()
    from .api import DEFAULT_MODEL, model_file
    from .model import load_model
    from .text import TextTokenizer, text_frames
    start = time.perf_counter()
    model = load_model(model_file(args.model or DEFAULT_MODEL), device="cuda", dtype=torch.bfloat16)
    tokenizer = TextTokenizer(args.tokenizer)
    source = Path(args.texts) if args.texts else Path(__file__).with_name("calibration_texts.json")
    texts = json.loads(source.read_text(encoding="utf-8"))
    prompts = [text_frames(tokenizer, text, number % 2) for number, text in enumerate(texts)]
    loaded = time.perf_counter()
    stacks = [args.only] if args.only else ["backbone", "decoder"]
    inputs = calibrate(model, prompts, args.rows, args.batch, args.positions, args.frames, seed=args.seed,
                       stacks=stacks)
    calibrated = time.perf_counter()
    rows = {stack: inputs[f"{stack}.0.attn_in"].count for stack in stacks}
    print(json.dumps({"texts": len(texts), "rows": rows, "calibration_s": round(calibrated - loaded, 1)}), flush=True)
    groupsize = int(args.groupsize) if args.groupsize.isdigit() else \
        {stack: int(size) for stack, _, size in (item.partition("=") for item in args.groupsize.split(","))}
    state, report = search(model, inputs, groupsize, only=args.only)
    del inputs
    save(args.out, state)
    searched = time.perf_counter()
    summary = {}
    for stack in ("backbone", "decoder"):
        layers = [errors for name, errors in report.items() if name.startswith(stack + ".")]
        if layers:
            summary[stack] = {role: {kind: round(sum(errors[role][kind] for errors in layers) / len(layers), 5)
                                     for kind in ("nearest", "awq")} for role in SCALED}
            summary[stack]["clipped"] = {linear_name: round(sum(errors[f"{linear_name}_clipped"] for errors in layers)
                                                            / len(layers), 4) for _, linear_name, _ in CLIPPED}
    times = {"load_s": round(loaded - start, 1), "calibration_s": round(calibrated - loaded, 1),
             "search_s": round(searched - calibrated, 1)}
    print(json.dumps({"summary": summary, **times}), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                                                 "args": vars(args), "texts": len(texts), "rows": rows, **times,
                                                 "summary": summary, "layers": report}, indent=1) + "\n",
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
