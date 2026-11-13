"""Command line (`fast-misotts` once installed):

    python -m fast_misotts "Hello there, how was the trip?" -o hello.wav [--speaker 0]
    python -m fast_misotts --jsonl requests.jsonl --out-dir wavs [--batch 32]

A requests line is {"text": ..., "speaker": 0, "name": "file stem"}; each is written to OUT_DIR/NAME.wav (its line
number without a name), all generated together, --batch at a time. Weights: --model (a model.safetensors, its folder
or a Hub repo; default MisoLabs/MisoTTS, or $MISO_TTS_8B_MODEL), cast to --dtype as they load. Tokenizer: Llama 3.2's
(--tokenizer; default the gated meta-llama/Llama-3.2-1B, as the original)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fast-misotts", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="one utterance")
    parser.add_argument("-o", "--out", default="out.wav", help="its WAV file")
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--jsonl", help="many requests, one JSON object a line")
    parser.add_argument("--out-dir", help="their WAV files")
    parser.add_argument("--model", help="model.safetensors, its folder or a Hub repo")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--mimi", help="Mimi's weights (default: moshi's, from the Hub)")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--batch", type=int, default=16, help="utterances generated together")
    parser.add_argument("--max-seconds", type=float, default=90.0, help="the longest utterance")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--compile", nargs="?", const=True, default=False, choices=(True, "layers", "autotune", "step"),
                        help="the frame step through torch.compile: its layers (no value), or 'autotune' (fastest for "
                             "one utterance, compiles longer); after TEXT")
    parser.add_argument("--quantize", help="low-bit weights, e.g. int4:backbone+int8w:decoder (fast_misotts.quant)")
    parser.add_argument("--awq", help="with int4: a python -m fast_misotts.awq search (.safetensors)")
    parser.add_argument("--stream", action="store_true",
                        help="one TEXT: write its audio as it is made, and say when the first piece came")
    parser.add_argument("--watermark", action="store_true",
                        help="the original's SilentCipher watermark on the audio (needs silentcipher)")
    args = parser.parse_args(argv)
    if (args.text is None) == (args.jsonl is None):
        parser.error("give a TEXT or --jsonl")
    if args.jsonl and not args.out_dir:
        parser.error("--jsonl needs --out-dir")
    if args.text is not None:
        rows, paths = [{"text": args.text, "speaker": args.speaker}], [Path(args.out)]
    else:
        lines = Path(args.jsonl).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        paths = [Path(args.out_dir) / f"{row.get('name', number)}.wav" for number, row in enumerate(rows)]
    import soundfile
    import torch
    from .api import MisoTTS, Request
    frames = int(args.max_seconds / 0.08)
    tts = MisoTTS(args.model, tokenizer=args.tokenizer, mimi=args.mimi, dtype=getattr(torch, args.dtype),
                  batch=max(1, min(args.batch, len(rows))), max_frames=frames, temperature=args.temperature,
                  topk=args.topk, compile=args.compile, watermark=args.watermark, quantize=args.quantize,
                  awq=args.awq)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.stream:
        if args.text is None:
            parser.error("--stream takes one TEXT")
        import time
        start, first, total = time.perf_counter(), None, 0
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        with soundfile.SoundFile(str(paths[0]), "w", tts.sample_rate, 1, subtype="PCM_16") as handle:
            for piece in tts.stream(args.text, args.speaker, max_audio_length_ms=frames * 80):
                first = first if first is not None else time.perf_counter() - start
                handle.write(piece.numpy())
                total += len(piece)
        print(f"first audio after {first or 0:.2f} s; {total / tts.sample_rate:.1f} s of audio in "
              f"{time.perf_counter() - start:.1f} s: {paths[0]}")
        return
    audio = tts.generate_batch([Request(row["text"], int(row.get("speaker", 0))) for row in rows], frames * 80)
    for path, wave in zip(paths, audio):
        path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(path), wave.numpy(), tts.sample_rate, subtype="PCM_16")
    print(f"{len(paths)} file{'s' if len(paths) != 1 else ''}, {sum(len(wave) for wave in audio) / tts.sample_rate:.1f} s "
          f"of audio: {paths[0] if len(paths) == 1 else Path(args.out_dir)}")


if __name__ == "__main__":
    main()
