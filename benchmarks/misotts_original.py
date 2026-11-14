"""The original MisoTTS (github.com/MisoLabsAI/MisoTTS at cd7dde1: a checkout in $MISOTTS_DIR, else
third_party/misotts) on the texts fast_misotts is timed and checked on, in the original's own stack (torch 2.4,
torchtune 0.4: requirements-misotts.txt, in an environment of its own). Its Hub downloads are pointed at local files
(Llama 3.2's tokenizer from a folder, with the base model's BOS and EOS as load_llama3_tokenizer sets them; Mimi's
weights) and its watermark is left out, as fast_misotts leaves it out by default.

    python3 misotts_original.py speed  --model M --tokenizer T --mimi W --out FILE [--frames 250]
    python3 misotts_original.py parity --model M --tokenizer T --mimi W --dtype float32 --out FILE

speed: Generator.generate on each of fast_misotts/bench_texts.json's 64 texts (speakers alternating, no context, its
default sampling, at most --frames frames), one at a time as it runs, after one untimed warm-up utterance: the wall
time of each call (its frames and Mimi's decoding), audio seconds, times real time, peak memory.
parity: the first --texts texts greedy (models.sample_topk replaced by the argmax) in --dtype: the codes each call
decoded, for fast_misotts.bench compare.
quality: the 64 texts sampled once for each of --seeds, the audio saved (.npz) for fast_misotts.bench quality.
demo: the texts at --indices one at a time, each call's time and audio (the demo videos, fast_misotts.bench demo).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
# A MisoTTS checkout at cd7dde1 (generator.py, models.py, moshi_compat.py): $MISOTTS_DIR, else third_party/misotts.
sys.path.insert(0, os.environ.get("MISOTTS_DIR", str(ROOT / "third_party" / "misotts")))
BOS, EOS = "<|begin_of_text|>", "<|end_of_text|>"


def original(args, dtype):
    """The original Generator, its downloads pointed at the staged files and its watermark a no-op. Returns it and a
    list that collects the codes [frames, 32] of each utterance it decodes."""
    import torch
    stub = types.ModuleType("watermarking")
    stub.MISO_TTS_WATERMARK = [0, 0, 0, 0, 0]
    stub.load_watermarker = lambda device="cuda": None
    stub.watermark = lambda model, audio, sample_rate, key: (audio, sample_rate)
    sys.modules["watermarking"] = stub
    import generator
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer

    def tokenizer():
        loaded = AutoTokenizer.from_pretrained(args.tokenizer)
        loaded._tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS}:0 $A:0 {EOS}:0", pair=f"{BOS}:0 $A:0 {EOS}:0 {BOS}:1 $B:1 {EOS}:1",
            special_tokens=[(BOS, loaded.convert_tokens_to_ids(BOS)), (EOS, loaded.convert_tokens_to_ids(EOS))])
        return loaded

    hub = generator.hf_hub_download
    generator.load_llama3_tokenizer = tokenizer
    generator.hf_hub_download = lambda repo, name, **kw: args.mimi if name == generator.loaders.MIMI_NAME else \
        hub(repo, name, **kw)
    made = generator.load_miso_8b("cuda", model_path_or_repo_id=args.model, dtype=getattr(torch, dtype))
    codes = []
    decode = made._audio_tokenizer.decode

    def recording(frames):
        codes.append(frames[0].transpose(0, 1).cpu())
        return decode(frames)

    made._audio_tokenizer.decode = recording
    return made, codes


def texts():
    return json.loads((ROOT / "fast_misotts" / "bench_texts.json").read_text(encoding="utf-8"))


def generate(made, **kwargs):
    """The original's generate, or None where it fails because the first frame already ends the audio: it then
    stacks no frames and raises (seen once on an H200); fast_misotts returns no audio there."""
    try:
        return made.generate(**kwargs)
    except RuntimeError as error:
        if "non-empty TensorList" not in str(error):
            raise
        return None


def speed(args):
    import torch
    made, codes = original(args, "bfloat16")
    items = texts()
    generate(made, text=items[0], speaker=0, context=[], max_audio_length_ms=args.frames * 80)  # warm-up
    codes.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    calls, failed = [], 0
    for number, text in enumerate(items):
        start = time.perf_counter()
        failed += generate(made, text=text, speaker=number % 2, context=[],
                           max_audio_length_ms=args.frames * 80) is None
        torch.cuda.synchronize()
        calls.append(time.perf_counter() - start)
    seconds = sum(len(item) for item in codes) * 0.08
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "dtype": "bfloat16", "max_frames": args.frames, "sampling": {"temperature": 0.9, "topk": 50},
              "utterances": len(calls), "audio_seconds": round(seconds, 2), "seconds": round(sum(calls), 3),
              "times_real_time": round(seconds / sum(calls), 3),
              "unfinished": sum(len(item) >= args.frames for item in codes), "failed_empty": failed,
              "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
              "per_call_seconds": [round(value, 3) for value in calls]}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("device", "audio_seconds", "seconds", "times_real_time")}),
          flush=True)


def parity(args):
    import torch
    import models
    models.sample_topk = lambda logits, topk, temperature: logits.argmax(-1, keepdim=True).to(dtype=torch.int)
    made, codes = original(args, args.dtype)
    items = texts()[:args.texts]
    for number, text in enumerate(items):
        if generate(made, text=text, speaker=number % 2, context=[], max_audio_length_ms=args.frames * 80) is None:
            codes.append(torch.zeros(0, 32, dtype=torch.long))  # no frame before the end of audio
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "dtype": args.dtype, "texts": items,
              "max_frames": args.frames, "greedy": True, "codes": [item.tolist() for item in codes]}
    Path(args.out).write_text(json.dumps(report) + "\n", encoding="utf-8")
    print(json.dumps({"frames": [len(item) for item in codes]}), flush=True)


def quality(args):
    """The 64 texts sampled once a seed (one at a time, as it runs), the audio of each kept for fast_misotts.bench
    quality to transcribe beside fast_misotts's own (the same recognizer for both). Writes an .npz: for seed s and
    text n, array s-n (int16, 24 kHz)."""
    import numpy as np
    import torch
    made, codes = original(args, "bfloat16")
    items = texts()
    audio = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        for number, text in enumerate(items):
            wave = generate(made, text=text, speaker=number % 2, context=[], max_audio_length_ms=args.frames * 80)
            wave = np.zeros(0, dtype=np.float32) if wave is None else wave.float().cpu().numpy()
            audio[f"{seed}-{number}"] = np.clip(np.rint(wave * 32768), -32768, 32767).astype("<i2")
    np.savez(args.out, **audio)
    print(json.dumps({"utterances": len(audio), "seconds": round(sum(len(item) for item in audio.values()) / 24000, 1)}),
          flush=True)


def demo(args):
    """The demo videos' original lane: the texts at --indices of bench_texts.json, sampled (speakers alternating),
    one at a time as it runs, after one untimed warm-up: each call's wall time (frames and Mimi) in --times, and
    the audio in an .npz like quality's (array s-n: seed s, text n). --seeds: the texts again for each seed (the
    seed set before each pass); "seconds" holds the first seed's times, "by_key" every one's."""
    import numpy as np
    import torch
    made, codes = original(args, "bfloat16")
    items = texts()
    generate(made, text=items[0], speaker=0, context=[], max_audio_length_ms=args.frames * 80)  # warm-up
    audio, by_key = {}, {}
    for seed in args.seeds or [args.seed]:
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        for place, number in enumerate(args.indices):
            start = time.perf_counter()
            wave = generate(made, text=items[number], speaker=place % 2, context=[],
                            max_audio_length_ms=args.frames * 80)
            torch.cuda.synchronize()
            by_key[f"{seed}-{number}"] = round(time.perf_counter() - start, 3)
            wave = np.zeros(0, dtype=np.float32) if wave is None else wave.float().cpu().numpy()
            audio[f"{seed}-{number}"] = np.clip(np.rint(wave * 32768), -32768, 32767).astype("<i2")
    np.savez(args.out, **audio)
    first = (args.seeds or [args.seed])[0]
    seconds = [by_key[f"{first}-{number}"] for number in args.indices]
    Path(args.times).write_text(json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                                            "indices": args.indices, "texts": [items[n] for n in args.indices],
                                            "seconds": seconds, "total": round(sum(seconds), 3), "by_key": by_key},
                                           indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"total_seconds": round(sum(by_key.values()), 1)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("speed", "parity", "quality", "demo"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True, help="the original model.safetensors")
        command.add_argument("--tokenizer", required=True, help="a folder with Llama 3.2's tokenizer")
        command.add_argument("--mimi", required=True, help="Mimi's weights file")
        command.add_argument("--frames", type=int, default=250)
        command.add_argument("--out", required=True)
        if name == "parity":
            command.add_argument("--dtype", default="float32", choices=("bfloat16", "float32"))
            command.add_argument("--texts", type=int, default=8)
        if name == "quality":
            command.add_argument("--seeds", type=lambda value: [int(item) for item in value.split(",")],
                                 default=[0, 1, 2])
        if name == "demo":
            command.add_argument("--indices", type=lambda value: [int(item) for item in value.split(",")],
                                 required=True)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--seeds", type=lambda value: [int(item) for item in value.split(",")],
                                 help="several seeds, the texts again for each (instead of --seed)")
            command.add_argument("--times", required=True)
    args = parser.parse_args()
    {"speed": speed, "parity": parity, "quality": quality, "demo": demo}[args.command](args)


if __name__ == "__main__":
    main()
