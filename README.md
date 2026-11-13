# fast-misotts

[MisoTTS 8B](https://huggingface.co/MisoLabs/MisoTTS) by Miso Labs, fast. The same weights and the same arithmetic as
the original `Generator`, run with static key/value caches, a CUDA graph for each audio frame and continuous batching:
many utterances generate together, and a long one holds up only its own row.

- **Fast.** On an H200 the original speaks at 0.44x real time; fast-misotts speaks one utterance at 4.2x (9.5 times
  the original) and many at up to 191x. On an L40S (the original 0.50x): 1.3x for one, 2.9x with low-bit weights
  that lose about what 8-bit weights do, and up to 105x for many.
- **The same outputs.** In FP32 with greedy decoding, its audio codes equal the original's on every frame of all 32
  codebooks. In BF16 (the default) the rounding differs from the original's, so the two are compared by what they say:
  Whisper's error rates, the lengths and the runaways do not differ (below).
- **Streaming.** The first audio after 0.1 s on an H200 and 0.08-0.15 s on an L40S. The original speaks only once it
  has the whole utterance: about 7 s for a 3.5 s sentence.
- **Easy.** `pip install git+https://github.com/JonnesLin/fast-misotts`; the weights come from the original
  repository on first use.

## Install

```bash
pip install git+https://github.com/JonnesLin/fast-misotts
```

You need a CUDA GPU with room for the weights (16.4 GB in BF16; about 6 GB once in low-bit weights, which are made
from the BF16 ones as they load) and the cache (268 MB per row at the full 2,048 positions; 2.1 GB at the default 8
rows), and PyTorch 2.5 or later. The first run downloads [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS)
(33 GB in FP32, cast to BF16 as it loads).

The text tokenizer is Llama 3.2's, as in the original, from the gated
[meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B). Accept its license on the Hub and run
`huggingface-cli login` once, or pass `tokenizer=` a folder holding its `tokenizer.json`.

## Use

```python
import soundfile as sf
from fast_misotts import MisoTTS, Request, Segment

tts = MisoTTS()
audio = tts.generate("Hey, did you catch the game last night?", speaker=0)
sf.write("hey.wav", audio.numpy(), tts.sample_rate)  # 24 kHz

# Many at once: `batch` rows, each taking the next request as soon as its utterance ends.
tts = MisoTTS(batch=64, compile=True)
waves = tts.generate_batch([Request("First line.", speaker=0), Request("And the second one.", speaker=1)])

# A conversation: each turn with the turns before it as context. Generated turns go back in as codes, not audio.
context = []
for speaker, text in [(0, "So, how did it go?"), (1, "Honestly? Better than I expected.")]:
    codes = tts.generate_codes([Request(text, speaker, context)])[0]
    context.append(Segment(speaker, text, codes=codes))

# A voice from a recording: its transcript and its 24 kHz audio as context.
reference = Segment(speaker=0, text="Transcript of the clip.", audio=clip)
audio = tts.generate("Now say this in that voice.", speaker=0, context=[reference])

# One utterance at a time, as fast as it goes: Inductor tunes its own kernels the first time (about a minute).
tts = MisoTTS(batch=1, compile="autotune")
# ... and on GPUs that read their weights slowly (an L40S, an RTX 4090), low-bit weights (below).
tts = MisoTTS(batch=1, compile="autotune", quantize="int4:backbone+int8w:decoder")
```

Streaming: audio while it is made, 80 ms frames a few at a time. The first call in a process also captures its CUDA
graphs (about 2 s, and with `compile` its first compile on a machine); later calls start in about 0.1 s:

```python
tts = MisoTTS(batch=1, compile="autotune")
for piece in tts.stream("Hey, did you catch the game last night?", chunk_frames=2):
    player.write(piece.numpy())  # hand pieces to an audio queue; each waits for the last to be taken
```

From the command line:

```bash
fast-misotts "Hey, did you catch the game last night?" -o hey.wav --compile autotune
fast-misotts --jsonl requests.jsonl --out-dir wavs --batch 64 --compile  # {"text": ..., "speaker": 0, "name": ...}
```

`MisoTTS(model=None, dtype=torch.bfloat16, batch=8, positions=2048, max_frames=1125, temperature=0.9, topk=50,
compile=False, fuse=True, quantize=None, awq=None, watermark=False)`:

- `model`: a `model.safetensors`, its folder or a Hub repository (default MisoLabs/MisoTTS, or `$MISO_TTS_8B_MODEL`).
- `batch`: utterances generated together. More requests than that wait for a row.
- `positions`: context plus audio, at most 2,048 (the model's limit). `max_frames`: the longest utterance, 80 ms a
  frame (1,125 is the original's 90 s).
- `compile=True`: the frame step's layers through `torch.compile`, about 20 s the first time on a machine and 2 s
  once its compile cache has them. `compile="autotune"`: Inductor also times its own kernels against cuBLAS's (about
  a minute the first time): the fastest for one utterance at a time (4.2x real time on an H200 against 3.5x).
- `fuse=True`: q, k and v, and w1 and w3, as one product each. The same FP32 codes; a frame 18% shorter on an H200.
- `quantize`, `awq`: low-bit weights (below).
- `watermark=True`: the original's SilentCipher watermark on the audio (below).

MisoTTS is made for conversation. Read alone and without context, short sentences often come out wrong: a late start,
a babbled first word. The original does this as often as fast-misotts (below). Give it the turns before as context.

## Speed

Times real time (audio seconds per wall second, Mimi's decoding included), BF16 unless low-bit, sampled at the
original's defaults (temperature 0.9, top 50), at most 20 s per utterance. One at a time: the 64 sentences of
`fast_misotts/bench_texts.json` (about 11 words each) in turn. Many: the same sentences 16 times over (1,024
requests) through 128 rows.

| | L40S | H200 |
| --- | ---: | ---: |
| Original `Generator`, one at a time | 0.50 | 0.44 |
| fast-misotts, one at a time | 0.98 | 1.44 |
| fast-misotts, one at a time, `compile=True` | 1.33 | 3.50 |
| fast-misotts, one at a time, `compile="autotune"` | | 4.17 |
| fast-misotts, one at a time, `compile="autotune"`, `quantize="int4:backbone+int8w:decoder"` | 2.94 | 4.35 |
| fast-misotts, many | 51.0 | 102 |
| fast-misotts, many, `compile=True` | 69.9 | 191 |
| fast-misotts, many, `compile=True`, `quantize="int8"` | 105 | 170 |

The rows without `compile` and the "many" rows were measured before `fuse` (now the default). Through 64 rows instead
of 128: 41.9x and 45.4x on the L40S, 59.8x and 102x on the H200. Rows that take the next request as soon as their
utterance ends beat static batches (each batch until its longest utterance ends) by 2 to 2.7 times on this queue.
Peak memory at 128 rows (positions 512): 34 GB.

Streaming in pieces of 2 frames, the first piece (median): 0.15 s on an L40S with `compile=True` (0.076 s with
`compile="autotune"` and `quantize="int4:backbone+int8w:decoder"`), 0.099 s on an H200 with `compile="autotune"`.

One utterance at a time is bound by memory and by kernel launches: each frame the backbone reads its 14 GB of layer
weights once and the depth decoder its 0.6 GB 31 times, in thousands of small kernels. Rows share those reads.
`torch.compile` fuses the kernels, which pays most where bandwidth is high (an H200); fewer bytes pay most where it is
low (an L40S).

## Low-bit weights

`quantize=` gives the backbone and the depth decoder each a mode, `"MODE[:backbone|:decoder]"` joined by `+`:
`int4[/gN]` (4-bit weights in groups of N, 128 by default, PyTorch's tinygemm kernel), `int8w` (8-bit weights, BF16
inputs), `int8` (8-bit weights and inputs) and `fp8` (both in FP8). None is the original's arithmetic, so they are
judged by how far they move the model's next-code distributions from BF16's: KL(BF16 || mode) in nats,
teacher-forced on the codes BF16 samples for the 64 sentences (`python -m fast_misotts.bench kl`), for codebook 0 (the
backbone's) and codebooks 1-31 (the depth decoder's). One utterance at a time, times real time:

| `quantize` | KL, codebook 0 / 1-31 | L40S | H200 |
| --- | ---: | ---: | ---: |
| none (BF16) | 0 / 0 | 1.33 | 4.17 |
| `int4:backbone+int8w:decoder`, with AWQ | 0.036 / 0.025 | 2.94 | 4.35 |
| `int4:backbone+int4/g32:decoder`, with AWQ | 0.036 / 0.216 | 3.50 | 4.18 |
| `int8` | 0.013 / 0.027 | 1.62 | |
| `int4` | 0.044 / 0.491 | 3.66 | |

(BF16 on the L40S, `int8` and `int4` with `compile=True`; the rest with `compile="autotune"`.)

- The backbone takes 4-bit weights well; the depth decoder (300M parameters) does not, and its later codebooks, which
  carry fine acoustic detail, move most. `int4:backbone+int8w:decoder` loses about what `int8` loses, at 1.8 times
  its speed.
- Whisper's word error rates do not tell these apart: on 192 utterances each differs from BF16's by about as much as
  the original's does (5 points, with each utterance's WER capped at 1).
- Low-bit weights pay where a GPU reads its weights slowly for its compute. An H200 at one row waits on its kernels
  more than on its memory, so they gain it 4%.
- `int4` and `int8w` are for a few rows at a time, with `compile`. For many rows, `int8`.

AWQ (activation-aware weight quantization, AutoAWQ's recipe for Llama layers) prepares the 4-bit stacks: input
channel scales folded into the norms and Linears before them, then each group of weights clipped, both searched on
the model's own inputs while it speaks 128 sentences of `fast_misotts/calibration_texts.json`. It cuts a 4-bit
stack's KL by a fifth (backbone) to a third (depth decoder). Without it, `quantize` rounds to nearest. Once, about 5
minutes on an L40S, with room for the BF16 model and about 10 GB more (`--rows 2048 --batch 16` about halves that):

```bash
python -m fast_misotts.awq --out awq.safetensors                                      # groups of 128
python -m fast_misotts.awq --out awq-g32.safetensors --groupsize backbone=128,decoder=32  # int4/g32 on the decoder
```

```python
tts = MisoTTS(batch=1, compile="autotune", quantize="int4:backbone+int8w:decoder", awq="awq.safetensors")
```

A search is applied only to the stacks in `int4`, and only for the groups it was made for.

## The same outputs

The original `Generator` (MisoTTS at
[cd7dde1](https://github.com/MisoLabsAI/MisoTTS/tree/cd7dde13dbfec33fd8321e6c34a43f86724a8bad), torch 2.4, torchtune
0.4) and fast-misotts (torch 2.14) on the same GPU and weights:

- **FP32, greedy**, the first 8 sentences, fast-misotts all 8 together: identical codes on every frame of all 32
  codebooks, ending on the same frame. This holds without compiling and with `compile=True`, with and without
  `fuse`, on an L40S and an H200, and with `compile="autotune"` on an H200.
- **BF16, sampled**: the 64 sentences with seeds 0-2 (192 utterances per row of the table), transcribed by Whisper
  large-v3 (its English normalizer on both sides), on an L40S:

| | Median WER | WER over 25% | Mean length | Runaways (20 s) |
| --- | ---: | ---: | ---: | ---: |
| Original `Generator` | 8.3% | 57 | 3.50 s | 0 |
| fast-misotts, one at a time | 6.2% | 46 | 3.31 s | 0 |
| fast-misotts, 64 together | 8.3% | 49 | 3.27 s | 1 |
| fast-misotts, 64 together, `compile=True` | 6.1% | 46 | 3.45 s | 1 |
| fast-misotts, 64 together, `compile="autotune"` (H200) | 9.3% | 56 | 3.33 s | 0 |

Sentence by sentence (each one's mean over the seeds), no difference from the original is significant: in failed
utterances, length (for example -0.05 s, 95% interval -0.39 to 0.30, with `compile=True`) or runaways. A sentence
has 11 words on average, so the median WER moves in steps of about 9 points.

## How

- Plain PyTorch, no torchtune: the original `model.safetensors` loads as is, and each step does the original's
  arithmetic: Llama 3's scaled RoPE in adjacent pairs, its table rounded to the model's dtype as the original rounds
  it; RMSNorm in FP32; the 33-way embedding sum; `sample_topk` op for op.
- Static key/value caches indexed by each sequence's own position: prompts of any lengths prefill together, and each
  row decodes at its own position.
- One CUDA graph per frame: the backbone's step, the depth decoder's 31 steps and their sampling, and each row's
  end-of-audio check. The host looks in every 8 frames, hands back the utterances that ended and gives their rows new
  requests.
- `compile=True` compiles one backbone layer and one depth-decoder layer (at 2 and at 1 position) and uses them for
  every layer, inside the same CUDA graph. With `"autotune"`, Inductor's own reductions take the depth decoder's
  products over from cuBLAS at one row and fold RMSNorm and SwiGLU into them.
- `int8w`, compiled, writes a few rows' product as one reduction over the inputs, so the int8 weight is read once and
  converted as it is read.

## Differences from the original

- The SilentCipher watermark the original puts on every utterance is off by default. `watermark=True` (or
  `--watermark`) applies it the original's way, with its public key unless you pass `watermark_key=`. It needs
  silentcipher as the original pins it:
  `pip install "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@d46d7d0893a583d8968ab3a6626e2289faec9152"`.
- When the first frame is already the end of audio, the original raises (`torch.stack` of no frames); fast-misotts
  returns empty audio.
- A conversation's generated turns go back in as codes, not as decoded and re-encoded audio. Audio context (a
  `Segment` with `audio=`) is encoded with Mimi as the original does.

## License

fast-misotts is under MisoTTS's Modified MIT License (see `LICENSE`): MIT, plus Miso Labs' condition that a
commercial product or service with more than 50 million monthly active users or more than 10 million US dollars of
monthly revenue display "Miso Labs" on its user interface. The weights are Miso Labs' (Kamino Learning, Inc.),
downloaded from their repository under the same license. MisoTTS follows Sesame's CSM architecture; Mimi is Kyutai's
codec (moshi); the tokenizer is Meta's Llama 3.2, under the Llama 3.2 Community License.
