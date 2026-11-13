"""MisoTTS, fast: the original Generator's interface (text, speaker, context) for one utterance or many at once."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterator, Sequence

import torch

from .codec import Mimi
from .engine import Engine, Sampling
from .model import load_model
from .quant import load_quantized, parse as parse_quantize, quantize as quantize_model, saved_spec
from .text import TextTokenizer, audio_frames, text_frames
from .watermark import MISO_TTS_WATERMARK, Watermarker

DEFAULT_MODEL = "MisoLabs/MisoTTS"


@dataclass
class Segment:
    """A turn of context: its speaker and text, and its audio (24 kHz mono) or, faster, its codes [frames, 32]."""
    speaker: int
    text: str
    audio: torch.Tensor | None = None
    codes: torch.Tensor | None = None


@dataclass
class Request:
    text: str
    speaker: int = 0
    context: Sequence[Segment] = ()


def model_file(source: str | Path) -> Path:
    path = Path(source)
    if path.is_file():
        return path
    if path.is_dir():
        return path / "model.safetensors"
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(str(source), "model.safetensors"))


class MisoTTS:
    """batch: utterances generated together (a CUDA graph is captured for it on first use; more requests than that
    queue for rows as utterances end); positions: the longest prompt plus audio frames any of them may take (at most
    2048, the model's). The key/value cache takes 268 MB a row at 2048 positions in BF16 (2.1 GB at the default 8
    rows, beside the weights' 16.4 GB); max_frames: the longest audio (80 ms a frame, 1125 = the original's 90 s default); compile:
    the frame step's layers (True) or the whole step ("step") through torch.compile, on first use (Engine);
    watermark: the original's SilentCipher watermark on the audio (off by default; fast_misotts.watermark);
    quantize: "fp8", "int8", "int8w" or "int4" weights in the two stacks' layers, faster and no longer exact
    (fast_misotts.quant; a stack each: "int4:backbone+int8w:decoder"; use with compile=True); awq: with int4, a
    fast_misotts.awq search's scales and clipping (.safetensors; python -m fast_misotts.awq makes it once), applied
    before rounding (AWQ: the same kernel and speed); a pre-quantized model (fast_misotts.quant.export_quantized) is
    loaded as saved, without quantize or awq; fuse: q, k and v, and w1 and w3, as one product each
    (MisoModel.fuse_projections; on by default: 17% faster a frame at one row on an H200, and its FP32 greedy
    codes still equal the original's); kernels: fast_misotts.kernels' Triton depth-decoder attention and top-k
    sampling (no longer exact: judged as quantize is)."""

    def __init__(self, model: str | Path | None = None, *, device: str = "cuda", dtype=torch.bfloat16,
                 tokenizer: str | Path = "meta-llama/Llama-3.2-1B", mimi: str | Path | None = None, batch: int = 8,
                 positions: int = 2048, max_frames: int = 1125, temperature: float = 0.9, topk: int = 50,
                 greedy: bool = False, graphs: bool = True, compile: bool | str = False, watermark: bool = False,
                 watermark_key: Sequence[int] = MISO_TTS_WATERMARK, quantize: str | None = None, fuse: bool = True,
                 kernels: bool = False, awq: str | Path | None = None):
        source = model or os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MODEL)
        path = model_file(source)
        saved = saved_spec(path)
        if saved:  # pre-quantized (fast_misotts.quant.export_quantized): fused, its low-bit weights and AWQ in it
            if (quantize and quantize != saved) or awq:
                raise ValueError(f"{source} is saved quantized ({saved!r}, AWQ in it): give neither quantize nor awq")
            self.model = load_quantized(path, device=device, dtype=dtype)
        else:
            self.model = load_model(path, device=device, dtype=dtype)
            if awq:
                from .awq import apply as apply_awq, load as load_awq
                plan = parse_quantize(quantize) if quantize else {}
                groupsizes = {stack: size for stack, (mode, size) in plan.items() if mode == "int4"}
                if not groupsizes:
                    raise ValueError("awq: with int4 on a stack (quantize='int4...'): it prepares 4-bit rounding")
                apply_awq(self.model, load_awq(awq), groupsizes)  # before fusing: q, k, v and w1, w3 apart
            if fuse:
                self.model.fuse_projections()
            if quantize:
                quantize_model(self.model, quantize)
        self.tokenizer = TextTokenizer(tokenizer)
        self.codec = Mimi(mimi, device=device)
        self.engine = Engine(self.model, batch, positions, max_frames, Sampling(temperature, topk, greedy), graphs,
                             compile, kernels)
        self.sample_rate = self.codec.sample_rate
        self.watermarker = Watermarker(device, watermark_key) if watermark else None

    def prompt(self, request: Request):
        pieces = []
        for segment in request.context:
            pieces.append(text_frames(self.tokenizer, segment.text, segment.speaker))
            codes = segment.codes if segment.codes is not None else self.codec.encode(segment.audio)
            pieces.append(audio_frames(codes))
        pieces.append(text_frames(self.tokenizer, request.text, request.speaker))
        return torch.cat([piece[0] for piece in pieces]), torch.cat([piece[1] for piece in pieces])

    def generate_codes(self, requests: Sequence[Request], max_audio_length_ms: float = 90_000) -> list[torch.Tensor]:
        """Codes [frames, 32] for each request, `batch` generated at a time: a row takes the next request as soon as
        its utterance ends, the longest texts first (they tend to take longest)."""
        frames = int(max_audio_length_ms / 80)
        order = sorted(range(len(requests)), key=lambda index: -len(requests[index].text))
        out: list[torch.Tensor | None] = [None] * len(requests)
        for place, codes in self.engine.generate((self.prompt(requests[index]) for index in order), frames):
            out[order[place]] = codes
        return out

    def generate_batch(self, requests: Sequence[Request], max_audio_length_ms: float = 90_000) -> list[torch.Tensor]:
        """Audio [samples] at 24 kHz for each request (watermarked as the original does, with watermark=True)."""
        audio = self.codec.decode(self.generate_codes(requests, max_audio_length_ms))
        if self.watermarker is not None:
            audio = [self.watermarker(wave, self.sample_rate) for wave in audio]
        return audio

    def generate(self, text: str, speaker: int = 0, context: Sequence[Segment] = (),
                 max_audio_length_ms: float = 90_000) -> torch.Tensor:
        """One utterance, as the original Generator.generate (without its watermark)."""
        return self.generate_batch([Request(text, speaker, context)], max_audio_length_ms)[0]

    def stream(self, text: str, speaker: int = 0, context: Sequence[Segment] = (),
               max_audio_length_ms: float = 90_000, chunk_frames: int = 4) -> Iterator[torch.Tensor]:
        """One utterance as audio pieces [samples] at 24 kHz while it is made, chunk_frames frames (80 ms each) at a
        time: the first after the prompt and those frames, joined the same audio as generate()'s for the same codes.
        Each piece waits for the caller to take the last, so hand them to a player's queue rather than play them
        here. Not with watermark=True (the original marks a whole utterance at once)."""
        if self.watermarker is not None:
            raise NotImplementedError("stream() with watermark=True: the watermark is put on a whole utterance")
        prompt = self.prompt(Request(text, speaker, context))  # context audio encoded before Mimi streams
        with self.codec.streaming():
            for codes in self.engine.stream(prompt, int(max_audio_length_ms / 80), chunk_frames):
                yield self.codec.decode_stream(codes, chunk_frames)
