"""Prompts as the original Generator builds them: each position a row of 32 audio codes and a text token, with a mask
of which are present."""
from __future__ import annotations

from pathlib import Path

import torch

BOS, EOS = "<|begin_of_text|>", "<|end_of_text|>"


class TextTokenizer:
    """Llama 3.2's tokenizer with each text wrapped in the base model's BOS and EOS, as the original sets it up. The
    source is a folder or file with tokenizer.json (Llama 3.2 1B, base or instruct: the same vocabulary), or a Hub
    repo (meta-llama/Llama-3.2-1B, gated). The instruct tokenizer's own EOS (<|eot_id|>) is not used."""

    def __init__(self, source: str | Path = "meta-llama/Llama-3.2-1B"):
        from tokenizers import Tokenizer
        from tokenizers.processors import TemplateProcessing
        path = Path(source)
        if path.is_dir():
            path = path / "tokenizer.json"
        if not path.is_file():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(str(source), "tokenizer.json"))
        self.tokenizer = Tokenizer.from_file(str(path))
        bos, eos = self.tokenizer.token_to_id(BOS), self.tokenizer.token_to_id(EOS)
        if bos is None or eos is None:
            raise ValueError(f"{path} is not a Llama 3 tokenizer")
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS}:0 $A:0 {EOS}:0", pair=f"{BOS}:0 $A:0 {EOS}:0 {BOS}:1 $B:1 {EOS}:1",
            special_tokens=[(BOS, bos), (EOS, eos)])

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids


def text_frames(tokenizer: TextTokenizer, text: str, speaker: int, codebooks: int = 32):
    """A speaker's text as the original tokenizes it ("[speaker] text"): one row a token, only the text column on."""
    ids = tokenizer.encode(f"[{speaker}] {text.lstrip()}")
    tokens = torch.zeros(len(ids), codebooks + 1, dtype=torch.long)
    tokens[:, -1] = torch.tensor(ids)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, -1] = True
    return tokens, mask


def audio_frames(codes: torch.Tensor):
    """Audio codes [frames, codebooks] as context rows, then the end-of-audio row of zeros, only the code columns on."""
    frames, codebooks = codes.shape
    tokens = torch.zeros(frames + 1, codebooks + 1, dtype=torch.long)
    tokens[:frames, :codebooks] = codes.to(torch.long).cpu()
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, :codebooks] = True
    return tokens, mask
