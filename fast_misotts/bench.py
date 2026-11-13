"""Speed, parity and quality runs for fast_misotts, against the original's (misotts_original.py, its own environment).

    python3 -m fast_misotts.bench speed  --model M --tokenizer T --mimi W [--batches 1,8,32,64] --out FILE
    python3 -m fast_misotts.bench parity --model M --tokenizer T --mimi W --dtype float32 [--texts 8] --out FILE
    python3 -m fast_misotts.bench listen --model M --tokenizer T --mimi W --folder DIR
    python3 -m fast_misotts.bench quality --model M --tokenizer T --mimi W [--batch 64] [--seeds 0,1,2] --out FILE.npz
    python3 -m fast_misotts.bench score --audio NAME=FILE.npz ... --against NAME --out FILE
    python3 -m fast_misotts.bench stream --model M --tokenizer T --mimi W [--chunk 4] [--compile] --out FILE
    python3 -m fast_misotts.bench kl --model M --tokenizer T --mimi W --fuse --variants NAME=OPTIONS ... --out FILE

speed: the 64 texts of bench_texts.json (--repeat times over; speakers alternating, no context, sampled at temperature
0.9 and top 50, at most --frames frames each) through `batch` rows for each batch size, a row taking the next text as
its utterance ends ("continuous"), and as the engine ran before that ("static": the longest texts together, each batch
until its longest ends); with --eager, also at batch 1 without CUDA graphs. Wall time of the codes (prefill and frames)
and of Mimi's decoding, audio seconds, times real time, peak memory. Each size's first batch is a warm-up (its graph
captured there) and is not timed. misotts_original.py speed times the original the same way.
parity: the first --texts texts greedy (argmax for sampling) in --dtype, all in one batch, at most --frames frames:
their codes, to compare with the original's on the same texts (misotts_original.py parity; compare: the
leading frames that agree).
listen: eight sampled utterances and the original's example dialogue (each turn with the turns before as context, as
codes) as 24 kHz WAV files.
quality: the 64 texts sampled once for each of --seeds, --batch at a time: their audio (an .npz, array s-n the 24 kHz
PCM16 of seed s, text n), as misotts_original.py quality keeps the original's, one at a time.
score: Whisper on each set of audio (the same recognizer for all; its English normalizer on the text and the
transcript): word error rate, length, runaways (no end of audio within --frames), speaking rate; and each set against
--against, text by text (a text's mean over the seeds): the mean difference, its 95% bootstrap interval over the
texts and a sign-flip p value. In BF16 the two round differently and their codes part within a frame or two, so
this is the check that they say the same things.
kl: how far each variant (low-bit weights, AWQ, ...) moves the model's next-code distributions from the reference's,
teacher-forced on the codes the reference samples for the texts: KL(reference || variant) for codebook 0 and codebooks
1-31, top-1 agreement, the sampled codes' log-probability. Finer than score's word error rates, whose spread between
runs of one model is about 5 points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

EXAMPLE_DIALOGUE = [(0, "I'm just honestly not that into him, you know?"), (1, "Yeah, I get it."),
                    (0, "And it's just like, I know I said I'd go out with you and stuff, but it's just like, I can't "
                        "you know."), (1, "Yeah, honestly that's totally fair.")]


def texts():
    return json.loads(Path(__file__).with_name("bench_texts.json").read_text(encoding="utf-8"))


def load(args, **options):
    from .api import MisoTTS
    return MisoTTS(args.model, tokenizer=args.tokenizer, mimi=args.mimi, dtype=getattr(torch, args.dtype),
                   compile=args.compile, quantize=getattr(args, "quantize", None), fuse=getattr(args, "fuse", False),
                   kernels=getattr(args, "kernels", False), awq=getattr(args, "awq", None), **options)


def environment():
    return {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda}


def static(tts, requests, frames):
    """As the engine ran before continuous batching: the longest texts together, `batch` at a time, each batch until
    its longest utterance ends."""
    order = sorted(range(len(requests)), key=lambda index: -len(requests[index].text))
    codes = [None] * len(requests)
    for start in range(0, len(order), tts.engine.batch):
        chunk = order[start:start + tts.engine.batch]
        for index, item in zip(chunk, tts.generate_codes([requests[index] for index in chunk], frames * 80)):
            codes[index] = item
    return codes


def speed(args):
    from .api import Request
    from .engine import Engine
    tts = load(args, batch=1, positions=args.positions, max_frames=args.frames, graphs=False)
    sampling = tts.engine.sampling
    requests = [Request(text, speaker=number % 2) for number, text in enumerate(texts())] * args.repeat
    runs = [(1, False, "continuous")] if args.eager else []
    for size in args.batches:  # with fewer requests than rows, or one row, the two are the same
        runs += [(size, True, "continuous")] + ([(size, True, "static")] if 1 < size < len(requests) else [])
    results = []
    for batch, graphs, mode in runs:
        if (tts.engine.batch, tts.engine.graphs) != (batch, graphs) or not results:
            tts.engine = None  # the last size's caches and graph freed before the next is built
            torch.cuda.empty_cache()
            tts.engine = Engine(tts.model, batch, args.positions, args.frames, sampling, graphs, args.compile,
                                args.kernels)
            tts.generate_codes(requests[:batch], args.frames * 80)  # warm-up: the graph is captured here
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        codes = static(tts, requests, args.frames) if mode == "static" else tts.generate_codes(requests,
                                                                                                args.frames * 80)
        torch.cuda.synchronize()
        generated = time.perf_counter() - start
        start = time.perf_counter()
        tts.codec.decode(codes)
        torch.cuda.synchronize()
        decoded = time.perf_counter() - start
        seconds = sum(len(item) for item in codes) * 0.08
        result = {"batch": batch, "cuda_graphs": graphs, "batching": mode, "utterances": len(codes),
                  "audio_seconds": round(seconds, 2),
                  "generate_seconds": round(generated, 3), "decode_seconds": round(decoded, 3),
                  "times_real_time": round(seconds / (generated + decoded), 2),
                  "unfinished": sum(len(item) >= args.frames for item in codes),
                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {**environment(), "dtype": args.dtype, "positions": args.positions, "max_frames": args.frames,
              "compile": args.compile, "sampling": {"temperature": 0.9, "topk": 50}, "results": results}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


def frame_ms(engine, prompts, steps, graph=True):
    """Milliseconds a frame for all rows, every row given a prompt (rows that end keep stepping: the same work), the
    graph replayed (or the step called) `steps` times after 4 untimed. The first call captures the graph."""
    with torch.inference_mode():
        engine.ended.fill_(True)
        engine.fresh.zero_()
        if engine.graphs and engine.graph is None:
            engine._capture()
        engine._admit(list(range(engine.batch)), [prompts[row % len(prompts)] for row in range(engine.batch)],
                      engine.max_frames)
        run = engine.graph.replay if graph else engine._frame
        for _ in range(4):
            run()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            run()
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / steps


KINDS = (("gemm", r"gemm|gemv|cutlass|cublas|matmul|xmma|nvjet|splitk|sm\d\d_"), ("softmax", r"softmax"),
         ("sampling", r"topk|sort|radix|exponential|distribution|argmax|bitonic"),
         ("indexing", r"index|scatter|gather"), ("triton", r"triton"))


def kernels(engine, prompts, steps=3, top=25):
    """CUDA time of `steps` frames called without the graph, by kernel (torch.profiler): the top kernels and the
    totals by kind (by name; the rest elementwise and reductions)."""
    import re
    from torch.profiler import ProfilerActivity, profile as profiler
    frame_ms(engine, prompts, 1, graph=False)  # rows given prompts, warmed up
    with torch.inference_mode(), profiler(activities=[ProfilerActivity.CUDA]) as trace:
        for _ in range(steps):
            engine._frame()
        torch.cuda.synchronize()
    rows = []
    for event in trace.key_averages():
        total = getattr(event, "device_time_total", None) or getattr(event, "cuda_time_total", 0)
        if total > 0:
            rows.append({"kernel": event.key[:120], "calls": event.count // steps,
                         "ms_per_frame": round(total / 1000 / steps, 3)})
    rows.sort(key=lambda row: -row["ms_per_frame"])
    kinds = {}
    for row in rows:
        kind = next((name for name, pattern in KINDS if re.search(pattern, row["kernel"], re.I)), "elementwise")
        kinds[kind] = round(kinds.get(kind, 0) + row["ms_per_frame"], 3)
    return {"ms_per_frame": round(sum(row["ms_per_frame"] for row in rows), 3), "kinds": kinds,
            "launches_per_frame": sum(row["calls"] for row in rows), "top": rows[:top]}


def profile(args):
    """Per-frame time of the frame step at each batch size, for each of --modes (eager, or a torch.compile mode):
    replayed as the CUDA graph and called without it (the host's share), the first call's seconds (capture and
    compilation) and graphs compiled, and where the step's GPU time goes (kernels, at the smallest and largest
    batch)."""
    from torch._dynamo.utils import counters
    from .api import Request
    from .engine import Engine
    tts = load(args, batch=1, positions=args.positions, max_frames=args.frames, graphs=False)
    sampling = tts.engine.sampling
    prompts = [tts.prompt(Request(text, speaker=number % 2)) for number, text in enumerate(texts())]
    report = {**environment(), "dtype": args.dtype, "positions": args.positions,
              "sampling": {"temperature": 0.9, "topk": 50}, "frames": [], "kernels": {}}
    for mode in args.modes:
        compiled = False if mode == "eager" else mode
        for batch in args.batches:
            tts.engine = engine = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            graphs_before = counters["stats"]["unique_graphs"]
            try:
                engine = tts.engine = Engine(tts.model, batch, args.positions, args.frames, sampling, True, compiled,
                                             args.kernels)
                start = time.perf_counter()
                graph_ms = frame_ms(engine, prompts, args.steps)
                setup = time.perf_counter() - start - graph_ms * args.steps / 1000
                row = {"compile": compiled, "batch": batch, "graph_ms": round(graph_ms, 2),
                       "call_ms": round(frame_ms(engine, prompts, max(4, args.steps // 4), graph=False), 2),
                       "first_call_seconds": round(setup, 1),
                       "graphs_compiled": counters["stats"]["unique_graphs"] - graphs_before,
                       "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
                if batch in (min(args.batches), max(args.batches)):
                    report["kernels"][f"{mode}-{batch}"] = kernels(engine, prompts)
            except Exception as error:  # a failed compilation is a result too; the eager rows are kept
                if not compiled:
                    raise
                row = {"compile": compiled, "batch": batch, "error": f"{type(error).__name__}: {error}"[:2000]}
            report["frames"].append(row)
            print(json.dumps(row), flush=True)
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


def parity(args):
    from .api import Request
    items = texts()[:args.texts]
    tts = load(args, batch=len(items), positions=args.positions, max_frames=args.frames, greedy=True,
               graphs=not args.eager)
    codes = tts.generate_codes([Request(text, speaker=number % 2) for number, text in enumerate(items)],
                               args.frames * 80)
    report = {**environment(), "dtype": args.dtype, "texts": items, "max_frames": args.frames, "greedy": True,
              "batch": len(items), "cuda_graphs": not args.eager, "codes": [item.tolist() for item in codes]}
    Path(args.out).write_text(json.dumps(report) + "\n", encoding="utf-8")
    print(json.dumps({"frames": [len(item) for item in codes]}), flush=True)


def compare(args):
    """Leading frames on which two parity runs agree, text by text (pure JSON, no model)."""
    ours, theirs = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (args.fast, args.original))
    rows = []
    for text, mine, other in zip(ours["texts"], ours["codes"], theirs["codes"]):
        agree = next((n for n, (a, b) in enumerate(zip(mine, other)) if a != b), min(len(mine), len(other)))
        rows.append({"text": text[:60], "agree": agree, "fast_frames": len(mine), "original_frames": len(other),
                     "identical": mine == other})
    report = {"fast": args.fast, "original": args.original, "identical": sum(row["identical"] for row in rows),
              "texts": len(rows), "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("identical", "texts")}), flush=True)


def listen(args):
    """WAV files in --folder and, with --json, the same files base64-encoded in one JSON (a report fetch can carry)."""
    import base64
    import io
    import soundfile
    from .api import Request, Segment
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    tts = load(args, batch=8, positions=args.positions, max_frames=args.frames)
    torch.manual_seed(args.seed)
    samples = texts()[:8]
    outputs = {}
    for number, audio in enumerate(tts.generate_batch([Request(text, speaker=n % 2) for n, text in
                                                       enumerate(samples)], args.frames * 80)):
        outputs[f"utterance-{number}"] = (samples[number], audio)
    context, turns = [], []
    for speaker, text in EXAMPLE_DIALOGUE:
        codes = tts.generate_codes([Request(text, speaker, list(context))], args.frames * 80)[0]
        context.append(Segment(speaker, text, codes=codes))
        turns.append(codes)
    outputs["dialogue"] = (" / ".join(text for _, text in EXAMPLE_DIALOGUE), torch.cat(tts.codec.decode(turns)))
    carried = {}
    for name, (text, audio) in outputs.items():
        buffer = io.BytesIO()
        soundfile.write(buffer, audio.numpy(), tts.sample_rate, format="WAV", subtype="PCM_16")
        (folder / f"{name}.wav").write_bytes(buffer.getvalue())
        carried[name] = {"text": text, "seconds": round(len(audio) / tts.sample_rate, 2),
                         "wav_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}
    if args.json:
        Path(args.json).write_text(json.dumps(carried) + "\n", encoding="utf-8")
    print(json.dumps({name: item["seconds"] for name, item in carried.items()}), flush=True)


def pcm16(wave):
    import numpy as np
    return np.clip(np.rint(np.asarray(wave, dtype=np.float32) * 32768), -32768, 32767).astype("<i2")


def quality(args):
    import numpy as np
    from .api import Request
    tts = load(args, batch=args.batch, positions=args.positions, max_frames=args.frames)
    requests = [Request(text, speaker=number % 2) for number, text in enumerate(texts())]
    audio = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        for number, wave in enumerate(tts.generate_batch(requests, args.frames * 80)):
            audio[f"{seed}-{number}"] = pcm16(wave.numpy())
    np.savez(args.out, **audio)
    print(json.dumps({"batch": args.batch, "utterances": len(audio),
                      "seconds": round(sum(len(item) for item in audio.values()) / 24000, 1)}), flush=True)


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, heard in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != heard)))
        previous = current
    return previous[-1]


def speech_seconds(wave, window=0.02, floor_db=-45.0):
    """From the first to the last 20 ms window within floor_db of the loudest (0 for silence)."""
    import numpy as np
    size = int(window * 24000)
    count = len(wave) // size
    if count == 0:
        return 0.0
    level = 10 * np.log10((np.asarray(wave[:count * size], dtype=np.float64).reshape(count, size) ** 2).mean(1) + 1e-12)
    loud = np.nonzero(level >= level.max() + floor_db)[0]
    return 0.0 if level.max() < -70 else float((loud[-1] + 1 - loud[0]) * window)


def paired(first, second, draws=10000):
    """first - second over the texts: mean, 95% bootstrap interval (texts resampled), two-sided sign-flip p."""
    import numpy as np
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    generator = np.random.default_rng(0)
    means = difference[generator.integers(0, len(difference), (draws, len(difference)))].mean(1)
    flipped = (difference * generator.choice((-1.0, 1.0), (draws, len(difference)))).mean(1)
    p = (1 + int((np.abs(flipped) >= abs(difference.mean()) - 1e-12).sum())) / (draws + 1)
    low, high = np.quantile(means, (0.025, 0.975))
    return {"mean": round(float(difference.mean()), 4), "ci95": [round(float(low), 4), round(float(high), 4)],
            "p": round(p, 4)}


def score(args):
    import numpy as np
    from scipy.signal import resample_poly
    from transformers import pipeline
    repo, _, revision = args.recognizer.partition("@")
    recognizer = pipeline("automatic-speech-recognition", model=repo, revision=revision or None, dtype=torch.float16,
                          device="cuda")
    try:  # the tokenizer's normalize builds a new normalizer on every call
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        normalize = EnglishTextNormalizer(recognizer.tokenizer.english_spelling_normalizer)
    except (ImportError, AttributeError):
        normalize = recognizer.tokenizer.normalize
    items = texts()
    rows = []
    for name, path in (item.split("=", 1) for item in args.audio):
        audio = np.load(path)
        keys = sorted(audio.files, key=lambda key: tuple(int(part) for part in key.split("-")))
        waves = [audio[key].astype(np.float32) / 32768 for key in keys]
        heard = recognizer([{"raw": resample_poly(wave, 2, 3).astype(np.float32), "sampling_rate": 16000}
                            for wave in waves], batch_size=args.batch,
                           generate_kwargs={"language": "en", "task": "transcribe"})  # Whisper's 16 kHz
        for key, wave, output in zip(keys, waves, heard):
            seed, number = (int(part) for part in key.split("-"))
            reference, hypothesis = normalize(items[number]).split(), normalize(output["text"]).split()
            rows.append({"set": name, "seed": seed, "text": number, "seconds": round(len(wave) / 24000, 3),
                         "speech_seconds": round(speech_seconds(wave), 3), "runaway": len(wave) >= args.frames * 1920,
                         "words": len(reference), "errors": edit_distance(reference, hypothesis),
                         "transcript": output["text"].strip()})
        print(json.dumps({"set": name, "utterances": len(keys)}), flush=True)
    sets, by_text = {}, {}
    for name in dict.fromkeys(row["set"] for row in rows):
        mine = [row for row in rows if row["set"] == name]
        ended = [row for row in mine if not row["runaway"]]
        sets[name] = {"utterances": len(mine), "seeds": sorted({row["seed"] for row in mine}),
                      "wer": round(sum(row["errors"] for row in mine) / sum(row["words"] for row in mine), 4),
                      "wer_ended": round(sum(row["errors"] for row in ended) / max(1, sum(row["words"] for row in ended)),
                                         4),
                      "over_0.25": sum(row["errors"] > 0.25 * row["words"] for row in mine),
                      "runaways": len(mine) - len(ended),
                      "mean_seconds": round(float(np.mean([row["seconds"] for row in mine])), 3),
                      "mean_seconds_ended": round(float(np.mean([row["seconds"] for row in ended] or [0])), 3),
                      "words_per_speech_second": round(sum(row["words"] for row in ended) /
                                                       max(1e-9, sum(row["speech_seconds"] for row in ended)), 3)}
        by_text[name] = {measure: [float(np.mean([pick(row) for row in mine if row["text"] == number]))
                                   for number in range(len(items))]
                         for measure, pick in (("seconds", lambda row: row["seconds"]),
                                               ("speech_seconds", lambda row: row["speech_seconds"]),
                                               ("wer", lambda row: row["errors"] / max(1, row["words"])),
                                               ("runaway", lambda row: float(row["runaway"])))}
    against = {name: {measure: paired(values, by_text[args.against][measure]) for measure, values in by_text[name].items()}
               for name in sets if name != args.against}
    report = {"recognizer": args.recognizer, "normalizer": "Whisper's English normalizer", "max_frames": args.frames,
              "texts": len(items), "sets": sets, "against": args.against, "differences": against, "utterances": rows}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"sets": sets, "differences": against}), flush=True)


def spec(args):
    """Self-speculative decoding in the depth decoder, measured before it is built: the depth decoder's own first L
    layers (then its final norm and the codebook's head) would draft the next codebooks, and its remaining layers
    verify several at once. For the first --texts texts, generated as usual (sampled, --batch at a time, no CUDA
    graphs), at every frame and codebook 1-31: the target's sampling distribution p (top 50 at temperature 0.9 of the
    full decoder's logits) and the same transform of the logits from layer L's state, q_L, for L = 1-7. Exact
    speculative sampling accepts a drafted code with probability min(1, p/q); averaged, sum_c min(p, q_L). Reports
    that acceptance and the argmax agreement per codebook and L, and the expected depth-decoder speedup for drafts of
    k codebooks, a round costing k drafts of L layers and head ((L + 1) / 9 of a step each) and one verifying step."""
    from .api import Request
    from .engine import sample
    tts = load(args, batch=args.batch, positions=args.positions, max_frames=args.frames, graphs=False)
    engine, model = tts.engine, tts.model
    decoder, how = model.decoder, engine.sampling
    layers, codebooks = len(decoder.layers), model.config.codebooks
    device = engine.device
    accept = torch.zeros(codebooks, layers - 1, dtype=torch.float64, device=device)
    agree = torch.zeros_like(accept)
    counts = torch.zeros(codebooks, dtype=torch.float64, device=device)

    def distribution(logits):
        logits = logits.float() / how.temperature
        removed = logits < torch.topk(logits, how.topk)[0][..., -1, None]
        return torch.softmax(logits.masked_fill(removed, -float("inf")), dim=-1)

    torch.manual_seed(args.seed)
    prompts = [tts.prompt(Request(text, speaker=number % 2)) for number, text in enumerate(texts()[:args.texts])]
    frames = 0
    with torch.inference_mode():
        for start in range(0, len(prompts), engine.batch):
            chunk = prompts[start:start + engine.batch]
            engine.ended.fill_(True)
            engine.fresh.zero_()
            engine._admit(list(range(len(chunk))), chunk, args.frames)
            for _ in range(args.frames):  # Engine._step, its depth decoder layer by layer
                position = engine.position.unsqueeze(1)
                allowed = engine.slots <= position.unsqueeze(-1)
                h = model.backbone(model.embed_frame(engine.codes), position, engine.cache, allowed)
                engine.hidden.copy_(torch.where(engine.fresh.unsqueeze(1), engine.hidden, h[:, -1]))
                engine.position += (~engine.fresh & ~engine.ended).long()
                engine.fresh.zero_()
                live = (~engine.ended).double()
                code = sample(model.codebook0_head(engine.hidden), how)
                codes = [code]
                x = torch.cat([engine.hidden.unsqueeze(1), model.embed_code(0, code)], dim=1)
                for i in range(1, codebooks):
                    state, exits = model.projection(x), []
                    for layer, cache in zip(decoder.layers, engine.depth_cache):
                        state = layer(state, decoder.rope, engine.depth_positions[i - 1], cache,
                                      engine.depth_allowed[i - 1])
                        exits.append(state[:, -1])
                    head = model.audio_head[i - 1]
                    logits = torch.mm(decoder.norm(exits[-1]), head)
                    target = distribution(logits)
                    for depth in range(layers - 1):
                        draft = distribution(torch.mm(decoder.norm(exits[depth]), head))
                        accept[i, depth] += (torch.minimum(target, draft).sum(-1).double() * live).sum()
                        agree[i, depth] += ((draft.argmax(-1) == target.argmax(-1)).double() * live).sum()
                    counts[i] += live.sum()
                    code = sample(logits, how)
                    codes.append(code)
                    x = model.embed_code(i, code)
                engine.codes.copy_(torch.cat(codes, dim=1))
                engine._record()
                frames += int(live.sum())
                if bool(engine.ended.all()):
                    break
    alpha = (accept[1:] / counts[1:, None]).cpu()  # [31, layers - 1]: codebooks 1-31, L = 1..layers-1

    def speedup(depth, k):
        """Expected steps for codebooks 1-31 without drafts (31) over those with: drafts of k from the first `depth`
        layers, each round's draft stopping at the first rejection, a bonus code after k accepted ones."""
        rates = alpha[:, depth - 1].tolist()
        draft_cost, cost = (depth + 1) / (layers + 1), [0.0] * (len(rates) + 2)
        for i in range(len(rates) - 1, -1, -1):  # cost[i]: codebooks i+1 .. 31 left
            drafted = min(k, len(rates) - i)
            total, reach = drafted * draft_cost + 1.0, 1.0
            for j in range(drafted):
                total += reach * (1 - rates[i + j]) * cost[i + j + 1]
                reach *= rates[i + j]
            total += reach * cost[min(len(rates), i + drafted + (1 if i + drafted < len(rates) else 0))]
            cost[i] = total
        return len(rates) / cost[0]

    projections = {f"L{depth}": {f"k{k}": round(speedup(depth, k), 3) for k in (1, 2, 3, 4, 6, 8)}
                   for depth in range(1, layers)}
    best = max(((depth, k) for depth in range(1, layers) for k in (1, 2, 3, 4, 6, 8)),
               key=lambda pair: speedup(*pair))
    report = {**environment(), "dtype": args.dtype, "texts": args.texts, "frames": frames,
              "samples_per_codebook": int(counts[1].item()),
              "acceptance_mean_by_layer": {f"L{depth}": round(float(alpha[:, depth - 1].mean()), 4)
                                           for depth in range(1, layers)},
              "greedy_agreement_mean_by_layer": {
                  f"L{depth}": round(float((agree[1:, depth - 1] / counts[1:]).mean()), 4)
                  for depth in range(1, layers)},
              "acceptance_by_codebook": {f"L{depth}": [round(value, 4) for value in alpha[:, depth - 1].tolist()]
                                         for depth in range(1, layers)},
              "depth_speedup_projection": projections,
              "best": {"layers": best[0], "draft": best[1], "speedup": round(speedup(*best), 3)}}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("frames", "acceptance_mean_by_layer", "best")}), flush=True)


def quantcheck(args):
    """Each quantize mode's Linear against the model's own on a few of its real weights (the first layer's q_proj,
    w1 and w2 in the backbone and the depth decoder): relative error of the output on random inputs, and the time of
    one call at 1 and 64 rows against the BF16 Linear's (CUDA events, the median of 50 after 10 untimed)."""
    from .model import load_model
    from .quant import Fp8Linear, Int4Linear, Int8Linear
    makers = {"fp8": Fp8Linear, "int8": Int8Linear, "int4": Int4Linear}
    model = load_model(args.model, device="cuda", dtype=torch.bfloat16)
    picks = {f"{stack}.{name}": getattr(getattr(getattr(model, stack).layers[0], part), name)
             for stack in ("backbone", "decoder") for part, name in (("attn", "q_proj"), ("mlp", "w1"), ("mlp", "w2"))}

    def timed(module, x):
        for _ in range(10):
            module(x)
        times = []
        for _ in range(50):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            module(x)
            end.record()
            torch.cuda.synchronize()
            times.append(begin.elapsed_time(end))
        return sorted(times)[len(times) // 2]

    report = {**environment(), "results": {}}
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.inference_mode():
        for mode, make in makers.items():
            rows = {}
            for name, linear in picks.items():
                try:
                    quantized, row = make(linear.weight), {}
                    for count in (1, 64):
                        x = torch.randn(count, linear.in_features, device="cuda", dtype=torch.bfloat16,
                                        generator=generator)
                        want, got = linear(x).float(), quantized(x).float()
                        row[f"rows{count}"] = {"relative_error": round(float((got - want).norm() / want.norm()), 5),
                                               "ms": round(timed(quantized, x), 4),
                                               "bf16_ms": round(timed(linear, x), 4)}
                    rows[name] = row
                except Exception as error:  # an unsupported mode on this GPU or PyTorch is a result too
                    rows[name] = {"error": f"{type(error).__name__}: {error}"[:500]}
            report["results"][mode] = rows
            print(json.dumps({mode: rows}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


@torch.inference_mode()
def teacher_forced(model, prompt, codes):
    """Log-probabilities [frames, codebooks, vocab] (FP32) of an utterance's codes [frames, codebooks] after its prompt
    (tokens, mask), each as generation computes it: codebook 0 from the backbone's state after the prompt and the
    frames before, codebook i from the depth decoder over that state and the frame's codes 0..i-1."""
    weight = model.codebook0_head.weight
    device, dtype = weight.device, weight.dtype
    tokens, mask = prompt
    frames, codebooks = codes.shape
    codes = codes.to(device)
    rows = torch.cat([codes, torch.zeros_like(codes[:, :1])], dim=1)[:-1]  # the inputs: every frame but the last
    row_mask = torch.ones_like(rows, dtype=torch.bool)
    row_mask[:, -1] = False  # as embed_frame: the codes, not the text column
    tokens, mask = torch.cat([tokens.to(device), rows]), torch.cat([mask.to(device), row_mask])
    length = len(tokens)
    cache = model.backbone.new_cache(1, length, dtype, device)
    hidden = model.backbone(model.embed(tokens[None], mask[None]), torch.arange(length, device=device)[None],
                            cache)[0, length - frames:]  # the states frames 1..T are made from
    depth = torch.cat([hidden[:, None], torch.stack([model.embed_code(i, codes[:, i]) for i in range(codebooks - 1)],
                                                    dim=1)], dim=1)  # [T, codebooks, dim]
    cache = model.decoder.new_cache(frames, codebooks, dtype, device)
    out = model.decoder(model.projection(depth), torch.arange(codebooks, device=device).expand(frames, -1), cache)
    logits = torch.cat([model.codebook0_head(hidden)[:, None],
                        torch.einsum("tid,idv->tiv", out[:, 1:], model.audio_head)], dim=1)
    return logits.float().log_softmax(-1)


def divergence(wanted, got, codes):
    """KL(wanted || got) in nats, the share of positions where both rank the same code first, and the mean
    log-probability `got` gives the sampled codes: for codebook 0 and over codebooks 1-31; and KL per codebook."""
    kl = torch.cat([(w.exp() * (w - g)).sum(-1) for w, g in zip(wanted, got)])  # [frames, codebooks]
    agree = torch.cat([(w.argmax(-1) == g.argmax(-1)).float() for w, g in zip(wanted, got)])
    sampled = torch.cat([g.gather(-1, c.to(g.device)[..., None])[..., 0] for g, c in zip(got, codes)])
    report = {}
    for key, values in (("kl", kl), ("top1", agree), ("logp", sampled)):
        report[f"{key}_0"], report[f"{key}_1_31"] = round(float(values[:, 0].mean()), 5), \
            round(float(values[:, 1:].mean()), 5)
    report["kl_per_codebook"] = [round(float(value), 5) for value in kl.mean(0)]
    return report


def kl(args):
    """Each --variants model's next-code distributions against the reference's (the model as the common flags load
    it), teacher-forced on the codes the reference samples for the first --texts texts (seed --seed). A variant is
    NAME=OPTIONS, OPTIONS comma-separated of fuse, quantize=MODE, awq=PATH and model=PATH (another checkpoint, a
    pre-quantized one say; none: BF16, not fused)."""
    from .api import MisoTTS, Request
    requests = [Request(text, speaker=number % 2) for number, text in enumerate(texts()[:args.texts])]
    reference = load(args, batch=16, positions=args.positions, max_frames=args.frames)
    torch.manual_seed(args.seed)
    codes = reference.generate_codes(requests, args.frames * 80)
    kept = [number for number, item in enumerate(codes) if len(item)]
    prompts, codes = [reference.prompt(requests[number]) for number in kept], [codes[number] for number in kept]
    wanted = [teacher_forced(reference.model, prompt, item) for prompt, item in zip(prompts, codes)]
    del reference
    torch.cuda.empty_cache()
    report = {**environment(), "texts": len(kept), "frames": sum(len(item) for item in codes), "seed": args.seed,
              "reference": {"fuse": args.fuse, "quantize": args.quantize, "awq": args.awq,
                            **divergence(wanted, wanted, codes)}, "variants": {}}
    for item in args.variants:
        name, _, spec = item.partition("=")
        options = dict(option.partition("=")[::2] for option in spec.split(",") if option)
        tts = MisoTTS(options.get("model") or args.model, tokenizer=args.tokenizer, mimi=args.mimi,
                      dtype=getattr(torch, args.dtype), batch=1, positions=args.positions, max_frames=args.frames,
                      graphs=False, fuse="fuse" in options, quantize=options.get("quantize") or None,
                      awq=options.get("awq") or None)
        got = [teacher_forced(tts.model, prompt, item) for prompt, item in zip(prompts, codes)]
        report["variants"][name] = {"options": options, **divergence(wanted, got, codes)}
        print(json.dumps({name: {key: value for key, value in report["variants"][name].items()
                                 if key != "kl_per_codebook"}}), flush=True)
        del tts, got
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


def stream(args):
    """Streaming one utterance at a time (one row): for each of the first --texts texts, sampled after an untimed
    warm-up, the time from the call to its first audio piece (--chunk frames) and to its last, the audio's length,
    and the largest difference between its pieces joined and its codes decoded at once."""
    from .api import Request
    tts = load(args, batch=1, positions=args.positions, max_frames=args.frames)
    items = texts()[:args.texts]
    tts.generate_codes([Request(items[0])], args.frames * 80)  # warm-up: compiling and the graph's capture
    torch.manual_seed(args.seed)
    rows = []
    for number, text in enumerate(items):
        prompt = tts.prompt(Request(text, speaker=number % 2))
        torch.cuda.synchronize()
        start, first, pieces, codes = time.perf_counter(), None, [], []
        with tts.codec.streaming():
            for chunk in tts.engine.stream(prompt, args.frames, args.chunk):
                pieces.append(tts.codec.decode_stream(chunk, args.chunk))  # on the CPU: the piece is ready
                first = first if first is not None else time.perf_counter() - start
                codes.append(chunk)
        total = time.perf_counter() - start
        joined = torch.cat(pieces) if pieces else torch.zeros(0)
        whole = tts.codec.decode([torch.cat(codes)])[0] if codes else torch.zeros(0)
        seconds = len(joined) / 24000
        rows.append({"text": number, "first_audio_seconds": round(first or total, 4), "seconds": round(total, 3),
                     "audio_seconds": round(seconds, 2), "times_real_time": round(seconds / total, 2),
                     "max_difference_from_whole": round(float((joined - whole).abs().max()) if len(joined) else 0.0,
                                                        6)})
    firsts = sorted(row["first_audio_seconds"] for row in rows)
    report = {**environment(), "dtype": args.dtype, "compile": args.compile, "quantize": args.quantize,
              "chunk_frames": args.chunk, "texts": len(rows),
              "median_first_audio_seconds": firsts[len(firsts) // 2],
              "times_real_time": round(sum(row["audio_seconds"] for row in rows) / sum(row["seconds"] for row in rows),
                                       2),
              "max_difference_from_whole": max(row["max_difference_from_whole"] for row in rows), "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("device", "compile", "chunk_frames", "median_first_audio_seconds",
                                                   "times_real_time", "max_difference_from_whole")}), flush=True)


def kernelcheck(args):
    """fast_misotts.kernels against what they replace (no model): decode_attention against grouped_attention on random
    queries, caches and positions (largest difference, BF16 and FP32); sample_topk greedy against the argmax; and
    sampled, 20,000 draws from fixed logits by it and by engine.sample, each draw's distribution against the exact
    top-k softmax (total variation)."""
    from .engine import Sampling, sample
    from .kernels import decode_attention, sample_topk
    from .model import grouped_attention
    report = {**environment()}
    generator = torch.Generator(device="cuda").manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        q = torch.randn(4, 2, 24, 64, device="cuda", dtype=dtype, generator=generator)
        keys, values = (torch.randn(4, 6, 32, 64, device="cuda", dtype=dtype, generator=generator) for _ in range(2))
        positions = torch.tensor([[0, 1], [5, 6], [20, 21], [30, 31]], device="cuda")
        allowed = torch.arange(32, device="cuda") <= positions.unsqueeze(-1)
        want = grouped_attention(q, keys, values, allowed).float()
        got = decode_attention(q, keys, values, positions).float()
        report[f"attention_max_difference_{str(dtype)[6:]}"] = float((got - want).abs().max())
        report[f"attention_scale_{str(dtype)[6:]}"] = float(want.abs().max())
    logits = torch.randn(16, 2051, device="cuda", generator=generator) * 3
    seed = torch.zeros(1, dtype=torch.int64, device="cuda")
    report["greedy_equal"] = bool((sample_topk(logits.bfloat16(), seed, 0.9, 50, greedy=True).squeeze(1) ==
                                   logits.bfloat16().argmax(-1)).all())
    how = Sampling(0.9, 50)
    row = logits[:1].float()
    scaled = row / how.temperature
    kept = scaled >= torch.topk(scaled, how.topk)[0][..., -1:]
    exact = torch.where(kept, scaled, torch.full_like(scaled, -float("inf"))).softmax(-1)[0]
    draws = 20000
    for name, codes in (("kernel", sample_topk(row.expand(draws, -1).contiguous(), seed, 0.9, 50)),
                        ("torch", sample(row.expand(draws, -1), how))):
        counts = torch.bincount(codes.view(-1), minlength=row.shape[-1]).float() / draws
        report[f"{name}_total_variation"] = round(float((counts - exact).abs().sum() / 2), 4)
        report[f"{name}_outside_top_k"] = int((~kept[0][codes.view(-1)]).sum())
    report["seed_advanced"] = int(seed.item())
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


def demo(args):
    """The demo videos' fast lane: the texts at --indices (speakers alternating), sampled, all at once, after one
    untimed run of them (compiling and the graph's capture): each utterance's time from the start to its audio (its
    codes as their row ends, then Mimi on them at once) in --times, the audio in an .npz like quality's."""
    import numpy as np
    from .api import Request
    items = texts()
    requests = [Request(items[number], speaker=place % 2) for place, number in enumerate(args.indices)]
    tts = load(args, batch=len(requests), positions=args.positions, max_frames=args.frames)
    tts.generate_codes(requests, args.frames * 80)  # warm-up
    torch.manual_seed(args.seed)
    torch.cuda.synchronize()
    done, audio = {}, {}
    start = time.perf_counter()
    for place, codes in tts.engine.generate((tts.prompt(request) for request in requests), args.frames):
        wave = tts.codec.decode([codes])[0]
        torch.cuda.synchronize()
        done[place] = round(time.perf_counter() - start, 3)
        audio[f"{args.seed}-{args.indices[place]}"] = pcm16(wave.numpy())
    np.savez(args.out, **audio)
    seconds = [done[place] for place in range(len(requests))]
    Path(args.times).write_text(json.dumps({**environment(), "compile": args.compile, "quantize": args.quantize,
                                            "indices": args.indices, "texts": [items[n] for n in args.indices],
                                            "seconds": seconds, "total": max(seconds)}, indent=1) + "\n",
                                encoding="utf-8")
    print(json.dumps({"total_seconds": max(seconds)}), flush=True)


def decode(args):
    """A parity run's codes (its JSON) as audio, through Mimi one utterance at a time: an .npz, array 0-n the 24 kHz
    PCM16 of its text n."""
    import numpy as np
    from .codec import Mimi
    report = json.loads(Path(args.codes).read_text(encoding="utf-8"))
    mimi = Mimi(args.mimi)
    numbers = args.texts if args.texts is not None else range(len(report["codes"]))
    audio = {f"0-{number}": pcm16(mimi.decode([torch.tensor(report["codes"][number], dtype=torch.long)])[0].numpy())
             for number in numbers}
    np.savez(args.out, **audio)
    print(json.dumps({key: len(value) for key, value in audio.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("speed", "parity", "listen", "quality", "profile", "spec", "demo", "stream", "kl"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True, help="model.safetensors, its folder or a Hub repo")
        command.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B")
        command.add_argument("--mimi", help="Mimi's weights (default: moshi's, from the Hub)")
        command.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
        command.add_argument("--positions", type=int, default=512)
        command.add_argument("--frames", type=int, default=250, help="at most this many frames (80 ms each)")
        command.add_argument("--fuse", action="store_true", help="q, k, v and w1, w3 as one product each")
        command.add_argument("--kernels", action="store_true", help="fast_misotts.kernels' attention and sampling")
        command.add_argument("--compile", nargs="?", const="layers", default=False,
                             choices=("layers", "step", "autotune"),
                             help="the frame step's layers (or with 'step', the whole step) through torch.compile")
        command.add_argument("--quantize", help="low-bit weights (fast_misotts.quant): fp8, int8, int8w or int4[/gN], "
                                                "optionally :backbone or :decoder, parts joined by +")
        command.add_argument("--awq", help="with --quantize int4: a fast_misotts.awq search (.safetensors)")
        if name == "kl":
            command.add_argument("--texts", type=int, default=64)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--variants", nargs="+", required=True,
                                 help="NAME=OPTIONS: fuse, quantize=MODE, awq=PATH, comma-separated")
            command.add_argument("--out", required=True)
        if name == "stream":
            command.add_argument("--texts", type=int, default=16)
            command.add_argument("--chunk", type=int, default=4, help="frames a piece (80 ms each)")
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--out", required=True)
        if name == "demo":
            command.add_argument("--indices", type=lambda value: [int(item) for item in value.split(",")],
                                 required=True)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--times", required=True)
            command.add_argument("--out", required=True)
        if name == "spec":
            command.add_argument("--texts", type=int, default=16)
            command.add_argument("--batch", type=int, default=16)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--out", required=True)
        if name == "profile":
            command.add_argument("--batches", type=lambda value: [int(item) for item in value.split(",")],
                                 default=[1, 8, 64])
            command.add_argument("--modes", type=lambda value: value.split(","), default=["eager", "layers", "step"])
            command.add_argument("--steps", type=int, default=40)
            command.add_argument("--out", required=True)
        if name == "speed":
            command.add_argument("--batches", type=lambda value: [int(item) for item in value.split(",")],
                                 default=[1, 8, 32, 64])
            command.add_argument("--repeat", type=int, default=1, help="the 64 texts this many times over")
            command.add_argument("--eager", action="store_true", help="also batch 1 without CUDA graphs")
            command.add_argument("--out", required=True)
        if name == "parity":
            command.add_argument("--texts", type=int, default=8)
            command.add_argument("--eager", action="store_true", help="without CUDA graphs")
            command.add_argument("--out", required=True)
        if name == "listen":
            command.add_argument("--folder", required=True)
            command.add_argument("--json", help="also the WAVs base64-encoded in this JSON")
            command.add_argument("--seed", type=int, default=7)
        if name == "quality":
            command.add_argument("--batch", type=int, default=64)
            command.add_argument("--seeds", type=lambda value: [int(item) for item in value.split(",")],
                                 default=[0, 1, 2])
            command.add_argument("--out", required=True)
    joined = sub.add_parser("compare")
    joined.add_argument("--fast", required=True)
    joined.add_argument("--original", required=True)
    joined.add_argument("--out", required=True)
    decoded = sub.add_parser("decode")
    decoded.add_argument("--codes", required=True, help="a parity run's JSON")
    decoded.add_argument("--mimi", help="Mimi's weights (default: moshi's, from the Hub)")
    decoded.add_argument("--texts", type=lambda value: [int(item) for item in value.split(",")])
    decoded.add_argument("--out", required=True)
    kerneled = sub.add_parser("kernel-check")
    kerneled.add_argument("--out", required=True)
    checked = sub.add_parser("quant-check")
    checked.add_argument("--model", required=True)
    checked.add_argument("--out", required=True)
    scored = sub.add_parser("score")
    scored.add_argument("--audio", nargs="+", required=True, help="NAME=FILE.npz from quality")
    scored.add_argument("--against", required=True, help="the NAME the others are compared with")
    scored.add_argument("--recognizer", default="openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1")
    scored.add_argument("--frames", type=int, default=250, help="the runs' --frames: a runaway reached it")
    scored.add_argument("--batch", type=int, default=16)
    scored.add_argument("--out", required=True)
    args = parser.parse_args()
    {"speed": speed, "parity": parity, "compare": compare, "listen": listen, "quality": quality,
     "score": score, "profile": profile, "spec": spec, "quant-check": quantcheck, "demo": demo,
     "decode": decode, "stream": stream, "kernel-check": kernelcheck, "kl": kl}[args.command](args)


if __name__ == "__main__":
    main()
