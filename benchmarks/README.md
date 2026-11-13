# Benchmarks

How the README's numbers were made, on one GPU each (an NVIDIA L40S and an H200), BF16 weights cast from the Hub's
FP32 `model.safetensors`.

The original runs in its own environment, with its own pinned stack:

```bash
git clone https://github.com/MisoLabsAI/MisoTTS && git -C MisoTTS checkout cd7dde13dbfec33fd8321e6c34a43f86724a8bad
python3 -m venv original && original/bin/pip install -r benchmarks/requirements-misotts.txt
export MISOTTS_DIR=$PWD/MisoTTS
```

fast-misotts runs in another (`pip install -e ".[bench]"`). With `M` the weights, `T` a folder with Llama 3.2's
tokenizer and `W` Mimi's weights:

```bash
# FP32 greedy parity: the original's codes and fast-misotts's on the first 8 sentences.
original/bin/python benchmarks/misotts_original.py parity --model M --tokenizer T --mimi W --dtype float32 --frames 125 --out original.json
python -m fast_misotts.bench parity --model M --tokenizer T --mimi W --dtype float32 --frames 125 [--compile] --out fast.json
python -m fast_misotts.bench compare --fast fast.json --original original.json --out compare.json

# Speed: the 64 sentences one at a time, and four times over through 64 rows.
original/bin/python benchmarks/misotts_original.py speed --model M --tokenizer T --mimi W --frames 250 --out original-speed.json
python -m fast_misotts.bench speed --model M --tokenizer T --mimi W --frames 250 --batches 1 [--compile] --out b1.json
python -m fast_misotts.bench speed --model M --tokenizer T --mimi W --frames 250 --batches 64 --repeat 4 [--compile] --out many.json

# Quality: the 64 sentences with seeds 0-2 on each side, all transcribed by the same Whisper.
original/bin/python benchmarks/misotts_original.py quality --model M --tokenizer T --mimi W --frames 250 --out original.npz
python -m fast_misotts.bench quality --model M --tokenizer T --mimi W --frames 250 --batch 64 [--compile] --out fast.npz
python -m fast_misotts.bench score --audio original=original.npz fast=fast.npz --against original --out quality.json

# Where a frame's time goes: eager and compiled, as the CUDA graph and without it, and the kernels.
python -m fast_misotts.bench profile --model M --tokenizer T --mimi W --modes eager,layers,autotune --batches 1,64 --fuse --out profile.json

# One utterance at a time with compile="autotune", and streaming in pieces of 2 frames.
python -m fast_misotts.bench speed --model M --tokenizer T --mimi W --frames 250 --batches 1 --fuse --compile autotune --out b1-autotune.json
python -m fast_misotts.bench stream --model M --tokenizer T --mimi W --frames 250 --chunk 2 --fuse --compile autotune --out stream.json

# Low-bit weights: AWQ's search, then each mode's next-code distributions against BF16's, its speed and its quality.
python -m fast_misotts.awq --model M --tokenizer T --out awq.safetensors --report awq.json
python -m fast_misotts.bench kl --model M --tokenizer T --mimi W --fuse --out kl.json --variants \
  "int4-int8w=fuse,quantize=int4:backbone+int8w:decoder,awq=awq.safetensors" int8=fuse,quantize=int8 int4=fuse,quantize=int4
python -m fast_misotts.bench speed --model M --tokenizer T --mimi W --frames 250 --batches 1 --fuse --compile autotune \
  --quantize int4:backbone+int8w:decoder --awq awq.safetensors --out b1-lowbit.json
python -m fast_misotts.bench quality --model M --tokenizer T --mimi W --frames 250 --batch 64 --fuse --compile \
  --quantize int4:backbone+int8w:decoder --awq awq.safetensors --out lowbit.npz
```

`bench kl` reports, for codebook 0 and codebooks 1-31, KL(BF16 || variant) in nats, the share of positions where
both rank the same code first, and the sampled codes' log-probability. A variant's options: `fuse`, `quantize=...`,
`awq=...`, comma-separated (none: BF16 without `fuse`, the rounding alone).
