"""Batched MisoTTS generation with static caches, CUDA graphs and continuous batching.

The engine has `batch` rows. Prompts are taken in order as rows come free: those admitted together are prefilled at
once (right-padded, so each prompt sits at positions 0..L-1 as the original puts it) into their rows' caches, while
the other rows keep their state. Then every frame is one replay of a captured graph for all rows: the backbone's step
on each row's last frame, the depth decoder's 31 steps and their sampling, the frame kept and each row's end-of-audio
or length limit noted. The host looks every few frames, hands back the utterances that ended and refills their rows,
so a long utterance holds up only its own row. Arithmetic and sampling follow the original Generator and
Model.generate_frame; rows without an utterance in progress keep stepping in place and keep nothing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch

from .model import Layer, MisoModel


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.9
    topk: int = 50
    greedy: bool = False  # argmax instead of sampling (the parity checks)


def sample(logits: torch.Tensor, how: Sampling) -> torch.Tensor:
    """The original sample_topk, op for op ([B, 1] codes), or the argmax."""
    if how.greedy:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / how.temperature
    removed = logits < torch.topk(logits, how.topk)[0][..., -1, None]
    scores = torch.nn.functional.log_softmax(logits.masked_fill(removed, -float("inf")), dim=-1)
    probs = torch.nn.functional.softmax(scores, dim=-1)
    noise = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / noise, dim=-1, keepdim=True)


class Engine:
    """Generation for any number of prompts, `batch` rows at a time, positions (prompt and frames) below `positions`,
    at most `max_frames` frames an utterance. compile, on first use: "layers" (or True) runs each layer of the frame
    step through torch.compile (one compiled layer serves every layer of the same shape, so it compiles in about a
    minute); "step" compiles the whole frame step, sampling included (random numbers still drawn by the eager ops;
    about 4.5 minutes a batch size). Prompts are prefilled without it."""

    def __init__(self, model: MisoModel, batch: int, positions: int = 2048, max_frames: int = 1125,
                 sampling: Sampling = Sampling(), graphs: bool = True, compile: bool | str = False,
                 kernels: bool = False):
        config = model.config
        if positions > config.backbone.max_seq_len:
            raise ValueError(f"positions above the model's {config.backbone.max_seq_len}")
        self.model, self.batch, self.positions, self.max_frames, self.sampling = model, batch, positions, max_frames, \
            sampling
        weight = model.codebook0_head.weight
        self.device, self.dtype = weight.device, weight.dtype
        self.cache = model.backbone.new_cache(batch, positions, self.dtype, self.device)
        self.depth_cache = model.decoder.new_cache(batch, config.codebooks, self.dtype, self.device)
        self.slots = torch.arange(positions, device=self.device)
        self.rows = torch.arange(batch, device=self.device)
        steps = [torch.arange(2, device=self.device)] + [torch.tensor([i], device=self.device)
                                                        for i in range(2, config.codebooks)]
        self.depth_positions = [step.expand(batch, -1) for step in steps]
        depth_slots = torch.arange(config.codebooks, device=self.device)
        self.depth_allowed = [(depth_slots <= step[:, None]).expand(batch, -1, -1) for step in steps]
        # The graph's inputs and outputs: fixed tensors, written in place.
        self.hidden = torch.zeros(batch, config.backbone.dim, dtype=self.dtype, device=self.device)
        self.codes = torch.zeros(batch, config.codebooks, dtype=torch.long, device=self.device)
        self.position = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.fresh = torch.zeros(batch, dtype=torch.bool, device=self.device)  # admitted, its first frame not made
        self.ended = torch.ones(batch, dtype=torch.bool, device=self.device)  # no utterance in progress
        self.count = torch.zeros(batch, dtype=torch.long, device=self.device)  # frames kept
        self.limit = torch.zeros(batch, dtype=torch.long, device=self.device)
        # A row's frames; the extra slot takes the writes of rows that keep nothing.
        self.frames = torch.zeros(batch, max_frames + 1, config.codebooks, dtype=torch.long, device=self.device)
        self.graphs = graphs and self.device.type == "cuda"
        self.graph = None
        # kernels: fast_misotts.kernels' depth-decoder attention and top-k sampling (the model's attention switched
        # for every engine on it); the sampling's seed drawn from torch's generator, advanced in place each draw
        self.kernels = kernels and self.device.type == "cuda"
        model.use_kernels(self.kernels)
        self.seed = torch.randint(0, 2 ** 31 - 1, (1,), device="cpu").to(self.device) if self.kernels else None
        self.compiled = "layers" if compile is True else (compile or None)
        if self.compiled not in (None, "layers", "step", "autotune"):
            raise ValueError(f"compile: False, True, 'layers', 'autotune' or 'step', not {compile!r}")
        if self.compiled:  # each batch size compiles its own graphs: more engines than dynamo's default 8 allows
            from torch import _dynamo
            for limit in ("recompile_limit", "cache_size_limit"):
                if hasattr(_dynamo.config, limit):
                    setattr(_dynamo.config, limit, max(getattr(_dynamo.config, limit), 64))
        # autotune: as layers, Inductor also timing its own kernels against cuBLAS's (slower to compile)
        tuned = {"max_autotune": True, "coordinate_descent_tuning": True} if self.compiled == "autotune" else None
        self._layer = torch.compile(Layer.forward, dynamic=False, options=tuned) \
            if self.compiled in ("layers", "autotune") else None
        # fallback_random: sampling's noise from the same eager ops
        self._frame = torch.compile(self._step, dynamic=False, options={"fallback_random": True}) \
            if self.compiled == "step" else self._step

    # One frame from the backbone's last state: codebook 0 from the backbone's head, 1-31 from the depth decoder.
    def _sample(self, logits):
        if self.kernels:
            from .kernels import sample_topk
            how = self.sampling
            return sample_topk(logits, self.seed, how.temperature, how.topk, how.greedy)
        return sample(logits, self.sampling)

    def _depth(self, hidden):
        model = self.model
        code = self._sample(model.codebook0_head(hidden))
        codes = [code]
        x = torch.cat([hidden.unsqueeze(1), model.embed_code(0, code)], dim=1)
        for i in range(1, model.config.codebooks):
            h = model.decoder(model.projection(x), self.depth_positions[i - 1], self.depth_cache,
                              self.depth_allowed[i - 1], call=self._layer)
            code = self._sample(torch.mm(h[:, -1], model.audio_head[i - 1]))
            codes.append(code)
            x = model.embed_code(i, code)
        return torch.cat(codes, dim=1)

    def _step(self):
        """One frame for every row. The backbone reads each row's last frame at its position; a fresh row keeps its
        prompt's last state instead and stays at its position (the step's cache write there is overwritten by its
        first frame next time). Rows in progress then advance; ended rows stay put."""
        position = self.position.unsqueeze(1)
        allowed = self.slots <= position.unsqueeze(-1)  # [B, 1, T]
        h = self.model.backbone(self.model.embed_frame(self.codes), position, self.cache, allowed, call=self._layer)
        self.hidden.copy_(torch.where(self.fresh.unsqueeze(1), self.hidden, h[:, -1]))
        self.position += (~self.fresh & ~self.ended).long()
        self.fresh.zero_()
        self.codes.copy_(self._depth(self.hidden))
        self._record()

    def _record(self):
        """Each row in progress keeps its frame, unless it is all zeros: that ends the row unkept (the original's
        stop), as reaching the row's limit ends it after the frame."""
        live = ~self.ended
        keep = live & ~(self.codes == 0).all(dim=1)
        self.frames[self.rows, torch.where(keep, self.count, self.max_frames)] = self.codes
        self.count += keep.long()
        self.ended |= live & (~keep | (self.count >= self.limit))

    def _capture(self):
        """Record _step as a CUDA graph, after warm-up runs on a side stream. Every row is free then, so the warm-ups
        only write position 0 of each row's cache, which a prompt rewrites."""
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._frame()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._frame()

    def _admit(self, rows: list[int], prompts: Sequence[tuple[torch.Tensor, torch.Tensor]], frames: int):
        """Prompts (tokens, mask [L, codebooks + 1]) into free rows: prefilled together, right-padded, into those
        rows' caches at positions 0..L-1; each row's first frame comes from its prompt's last state."""
        lengths = [len(tokens) for tokens, _ in prompts]
        longest = max(lengths)
        if longest >= self.positions:
            raise ValueError(f"A prompt of {longest} positions leaves no room below {self.positions}")
        width = prompts[0][0].shape[-1]
        tokens = torch.zeros(len(rows), longest, width, dtype=torch.long, device=self.device)
        masks = torch.zeros(len(rows), longest, width, dtype=torch.bool, device=self.device)
        for row, (prompt_tokens, prompt_mask) in enumerate(prompts):
            tokens[row, :len(prompt_tokens)] = prompt_tokens.to(self.device)
            masks[row, :len(prompt_tokens)] = prompt_mask.to(self.device)
        positions = torch.arange(longest, device=self.device).expand(len(rows), -1)
        # 128 positions at a time: embed's [B, L, 33, dim] stays small (8.6 GB for 32 prompts of 1,000), same values.
        inputs = torch.cat([self.model.embed(tokens[:, start:start + 128], masks[:, start:start + 128])
                            for start in range(0, longest, 128)], dim=1)
        index = torch.tensor(rows, device=self.device)
        h = self.model.backbone(inputs, positions, self.cache, rows=index)
        last = torch.tensor(lengths, device=self.device) - 1
        self.hidden[index] = h[torch.arange(len(rows), device=self.device), last]
        self.position[index] = last + 1
        self.count[index] = 0
        self.limit[index] = torch.tensor([min(frames, self.max_frames, self.positions - length) for length in lengths],
                                         device=self.device)
        self.fresh[index] = True
        self.ended[index] = False

    @torch.inference_mode()
    def stream(self, prompt: tuple[torch.Tensor, torch.Tensor], frames: int,
               chunk: int = 4) -> Iterator[torch.Tensor]:
        """One prompt's codes in pieces [n, codebooks] as they are made, `chunk` frames at a time (the last piece may
        be shorter), until its end-of-audio frame (not included) or `frames` (and the positions left): the codes
        generate() gives, in order. It takes row 0; for the lowest latency, an engine of one row."""
        self.ended.fill_(True)
        self.fresh.zero_()
        if self.graphs and self.graph is None:
            self._capture()
        step = self.graph.replay if self.graphs else self._frame
        self._admit([0], [prompt], frames)
        sent = 0
        while True:
            for _ in range(chunk):
                step()
            count, ended = torch.stack([self.count[0], self.ended[0].long()]).tolist()
            if count > sent:
                yield self.frames[0, sent:count].clone()
                sent = count
            if ended:
                return

    @torch.inference_mode()
    def generate(self, prompts: Iterable[tuple[torch.Tensor, torch.Tensor]], frames: int,
                 check_every: int = 8) -> Iterator[tuple[int, torch.Tensor]]:
        """Each prompt's codes [n, codebooks], at most `frames` frames (and the positions left after its prompt),
        yielded as (its index, codes) once its utterance ends: up to its end-of-audio frame, which is not included, as
        the original drops it. Prompts are read in order as rows come free (an iterator is read no further ahead)."""
        self.ended.fill_(True)  # every row free, whatever an earlier call left
        self.fresh.zero_()
        if self.graphs and self.graph is None:
            self._capture()
        step = self.graph.replay if self.graphs else self._frame
        pending, read, exhausted = iter(prompts), 0, False
        owner: list[int | None] = [None] * self.batch
        while True:
            free = [row for row, index in enumerate(owner) if index is None]
            taken = []
            while not exhausted and len(taken) < len(free):
                try:
                    taken.append(next(pending))
                except StopIteration:
                    exhausted = True
            if taken:
                self._admit(free[:len(taken)], taken, frames)
                for row in free[:len(taken)]:
                    owner[row], read = read, read + 1
            if all(index is None for index in owner):
                return
            for _ in range(check_every):
                step()
            ended, counts = self.ended.tolist(), self.count.tolist()
            for row, index in enumerate(owner):
                if index is not None and ended[row]:
                    yield index, self.frames[row, :counts[row]].clone()
                    owner[row] = None
