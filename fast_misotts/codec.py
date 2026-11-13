"""Mimi, the codec MisoTTS speaks in (kyutai's, through the moshi package), with all 32 codebooks."""
from __future__ import annotations

from contextlib import contextmanager

import torch

FRAME_SAMPLES = 1920  # 80 ms at 24 kHz


class Mimi:
    def __init__(self, weights=None, device="cuda", codebooks: int = 32):
        from moshi.models import loaders
        if weights is None:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.model = loaders.get_mimi(str(weights), device=device)
        self.model.set_num_codebooks(codebooks)
        self.device, self.sample_rate = device, self.model.sample_rate

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Mono audio [samples] at 24 kHz to codes [frames, codebooks]."""
        return self.model.encode(audio.to(self.device).view(1, 1, -1))[0].transpose(0, 1)

    @torch.inference_mode()
    def decode(self, codes: list[torch.Tensor], batch: int = 16) -> list[torch.Tensor]:
        """Codes [frames, codebooks] each to audio [frames * 1920] at 24 kHz, `batch` at a time, similar lengths
        together: the decoder's activations at 24 kHz are large (64 utterances of up to 20 s at once peaked near 24
        GB on top of the model, and 1,250 at once ran out of memory)."""
        out: list[torch.Tensor | None] = [None] * len(codes)
        order = sorted(range(len(codes)), key=lambda index: len(codes[index]))
        for start in range(0, len(order), batch):
            chunk = order[start:start + batch]
            for index, audio in zip(chunk, self._decode([codes[index] for index in chunk])):
                out[index] = audio
        return out

    @contextmanager
    def streaming(self):
        """Within it, decode_stream decodes one utterance piece by piece, Mimi carrying its state from piece to piece
        (its decoder is causal: the same audio as decoding the whole at once). Encoding does not belong inside."""
        with self.model.streaming(1):
            yield self

    @torch.inference_mode()
    def decode_stream(self, codes: torch.Tensor, frames: int | None = None) -> torch.Tensor:
        """The next piece of codes [n, codebooks] to audio [n * 1920] at 24 kHz (inside streaming()). Mimi's
        streaming decoder replays a CUDA graph that takes one shape: a shorter last piece is padded to `frames` with
        its last frame repeated (the decoder is causal, so the padding changes nothing before it) and its audio
        trimmed."""
        count = len(codes)
        if frames and count < frames:
            codes = torch.cat([codes, codes[-1:].expand(frames - count, -1)])
        audio = self.model.decode(codes.transpose(0, 1).unsqueeze(0).to(self.device))[0, 0].float().cpu()
        return audio[:count * FRAME_SAMPLES]

    def _decode(self, codes: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decoded together: Mimi's decoder is causal, so the zeros that pad the shorter ones change nothing before
        their ends."""
        longest = max(len(item) for item in codes)
        if longest == 0:
            return [torch.zeros(0) for _ in codes]
        batch = torch.zeros(len(codes), codes[0].shape[-1], longest, dtype=torch.long, device=self.device)
        for row, item in enumerate(codes):
            batch[row, :, :len(item)] = item.transpose(0, 1).to(self.device)
        audio = self.model.decode(batch)[:, 0].float().cpu()
        return [audio[row, :len(item) * FRAME_SAMPLES] for row, item in enumerate(codes)]
