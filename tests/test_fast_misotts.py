"""fast_misotts on a tiny random model, on the CPU: the batched engine (static caches, rows of
different lengths, padding) against a reference that follows the original algorithm without any cache (the whole
sequence recomputed each frame, the depth decoder over all its inputs each codebook, one prompt at a time, heads
expanded as torchtune does). Greedy, so the codes must agree exactly."""
from pathlib import Path
import unittest
from uuid import uuid4

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def tiny():
    from fast_misotts.model import MisoConfig, MisoModel, StackConfig
    config = MisoConfig(backbone=StackConfig(2, 64, 4, 2, 128, max_seq_len=128),
                        decoder=StackConfig(2, 32, 4, 2, 64, max_seq_len=128),
                        text_vocab=300, audio_vocab=17, codebooks=4)
    torch.manual_seed(0)
    model = MisoModel(config)
    torch.nn.init.normal_(model.audio_head, std=0.5)
    for module in model.modules():  # larger weights than the defaults: sharper logits, no near-ties in argmax
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.3)
    return model.eval().requires_grad_(False)


def reference_stack(stack, x):
    """A stack over a whole sequence [1, S, dim] at positions 0..S-1, causal, with k and v repeated per head."""
    from fast_misotts.model import apply_rope
    steps = x.shape[1]
    positions = torch.arange(steps)[None]
    spec = stack.spec
    for layer in stack.layers:
        attn = layer.attn
        h = layer.sa_norm(x)
        q = apply_rope(attn.q_proj(h).view(1, steps, spec.heads, spec.head_dim), stack.rope, positions)
        k = apply_rope(attn.k_proj(h).view(1, steps, spec.kv_heads, spec.head_dim), stack.rope, positions)
        v = attn.v_proj(h).view(1, steps, spec.kv_heads, spec.head_dim)
        groups = spec.heads // spec.kv_heads
        k, v = k.repeat_interleave(groups, dim=2), v.repeat_interleave(groups, dim=2)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + attn.output_proj(out.transpose(1, 2).reshape(1, steps, -1))
        x = x + layer.mlp(layer.mlp_norm(x))
    return stack.norm(x)


def reference_generate(model, tokens, mask, frames):
    """The original Generator.generate's loop, greedy, recomputing everything each step."""
    codebooks = model.config.codebooks
    out = []
    for _ in range(frames):
        last = reference_stack(model.backbone, model.embed(tokens[None], mask[None]))[:, -1]
        code = model.codebook0_head(last).argmax(-1, keepdim=True)
        codes = [code]
        x = torch.cat([last[:, None], model.embed_code(0, code)], dim=1)
        for i in range(1, codebooks):
            h = reference_stack(model.decoder, model.projection(x))
            code = torch.mm(h[:, -1], model.audio_head[i - 1]).argmax(-1, keepdim=True)
            codes.append(code)
            x = torch.cat([x, model.embed_code(i, code)], dim=1)
        frame = torch.cat(codes, dim=1)
        if bool((frame == 0).all()):
            break
        out.append(frame[0])
        row = torch.cat([frame[0], torch.zeros(1, dtype=torch.long)])[None]
        row_mask = torch.ones_like(row, dtype=torch.bool)
        row_mask[0, -1] = False
        tokens, mask = torch.cat([tokens, row]), torch.cat([mask, row_mask])
    return torch.stack(out) if out else torch.zeros(0, codebooks, dtype=torch.long)


def prompt(seed, length, codebooks, context=0):
    """Text rows (and, with context, audio rows before them and their end-of-audio row) as the original lays out."""
    from fast_misotts.text import audio_frames
    generator = torch.Generator().manual_seed(seed)
    pieces = []
    if context:
        pieces.append(audio_frames(torch.randint(1, 17, (context, codebooks), generator=generator)))
    text = torch.zeros(length, codebooks + 1, dtype=torch.long)
    text[:, -1] = torch.randint(0, 300, (length,), generator=generator)
    text_mask = torch.zeros_like(text, dtype=torch.bool)
    text_mask[:, -1] = True
    pieces.append((text, text_mask))
    return torch.cat([piece[0] for piece in pieces]), torch.cat([piece[1] for piece in pieces])


@unittest.skipIf(torch is None, "Requires torch")
class EngineTests(unittest.TestCase):
    def test_batched_engine_matches_the_uncached_original_loop(self):
        from fast_misotts.engine import Engine, Sampling
        model = tiny()
        codebooks = model.config.codebooks
        prompts = [prompt(1, 5, codebooks), prompt(2, 9, codebooks, context=3), prompt(3, 2, codebooks)]
        self.assertEqual([len(tokens) for tokens, _ in prompts], [5, 13, 2])
        engine = Engine(model, batch=4, positions=64, max_frames=12, sampling=Sampling(greedy=True), graphs=False)
        got = dict(engine.generate(prompts, 10, check_every=2))
        self.assertEqual(sorted(got), [0, 1, 2])
        for index, (tokens, mask) in enumerate(prompts):
            self.assertEqual(got[index].tolist(), reference_generate(model, tokens, mask, 10).tolist())
            self.assertEqual(len(got[index]), 10)

    def test_rows_taken_over_as_utterances_end(self):
        """Six prompts through two rows: limits from the positions left (20 - length) end them at different frames,
        so most are prefilled while the other row is mid-utterance. Each must match its own run, and the engine must
        come back clean for a second call."""
        from fast_misotts.engine import Engine, Sampling
        model = tiny()
        codebooks = model.config.codebooks
        prompts = [prompt(1, 5, codebooks), prompt(2, 9, codebooks, context=3), prompt(3, 2, codebooks),
                   prompt(4, 12, codebooks, context=3), prompt(5, 11, codebooks), prompt(6, 7, codebooks)]
        engine = Engine(model, batch=2, positions=20, max_frames=12, sampling=Sampling(greedy=True), graphs=False)
        for _ in range(2):
            got = dict(engine.generate(iter(prompts), 10, check_every=3))
            self.assertEqual(sorted(got), list(range(6)))
            for index, (tokens, mask) in enumerate(prompts):
                frames = min(10, 20 - len(tokens))
                self.assertEqual(len(got[index]), frames)
                self.assertEqual(got[index].tolist(), reference_generate(model, tokens, mask, frames).tolist())

    def test_fused_projections_give_the_same_codes(self):
        from fast_misotts.engine import Engine, Sampling
        from fast_misotts.quant import Int8Linear, quantize
        codebooks = tiny().config.codebooks
        prompts = [prompt(1, 5, codebooks), prompt(2, 9, codebooks, context=3)]
        runs = []
        for fused in (False, True):
            model = tiny().fuse_projections() if fused else tiny()
            engine = Engine(model, batch=2, positions=64, max_frames=12, sampling=Sampling(greedy=True),
                            graphs=False)
            runs.append({index: codes.tolist() for index, codes in engine.generate(prompts, 8)})
        self.assertEqual(runs[0], runs[1])
        model = quantize(tiny().fuse_projections(), "int8")  # 4 Linears a layer once fused
        self.assertEqual(sum(isinstance(module, Int8Linear) for module in model.modules()),
                         4 * (len(model.backbone.layers) + len(model.decoder.layers)))
        self.assertFalse(hasattr(model.backbone.layers[0].attn, "q_proj"))

    def test_streamed_pieces_join_to_the_generated_codes(self):
        from fast_misotts.engine import Engine, Sampling
        model = tiny()
        codebooks = model.config.codebooks
        target = prompt(7, 6, codebooks, context=2)
        for batch, chunk, frames in ((1, 1, 7), (3, 3, 10), (1, 4, 10)):
            engine = Engine(model, batch=batch, positions=64, max_frames=12, sampling=Sampling(greedy=True),
                            graphs=False)
            pieces = list(engine.stream(target, frames, chunk=chunk))
            self.assertTrue(all(len(piece) <= chunk for piece in pieces))
            self.assertEqual(torch.cat(pieces).tolist(), reference_generate(model, *target, frames).tolist())
            self.assertEqual(torch.cat(pieces).tolist(), dict(engine.generate([target], frames))[0].tolist())

    def test_an_end_of_audio_frame_ends_the_row_unkept(self):
        from fast_misotts.engine import Engine, Sampling
        model = tiny()
        model.codebook0_head.weight.zero_()  # all codes tie: the argmax is code 0, every frame all zeros
        model.audio_head.zero_()
        codebooks = model.config.codebooks
        prompts = [prompt(1, 5, codebooks), prompt(2, 3, codebooks), prompt(3, 4, codebooks)]
        engine = Engine(model, batch=2, positions=64, max_frames=12, sampling=Sampling(greedy=True), graphs=False)
        got = dict(engine.generate(prompts, 10))
        self.assertEqual({index: codes.shape for index, codes in got.items()},
                         {index: (0, codebooks) for index in range(3)})
        self.assertEqual(reference_generate(model, *prompts[0], 10).tolist(), [])

    def test_a_prompt_alone_or_with_others(self):
        from fast_misotts.engine import Engine, Sampling
        model = tiny()
        codebooks = model.config.codebooks
        target = prompt(7, 6, codebooks, context=2)
        alone = Engine(model, batch=1, positions=64, max_frames=8, sampling=Sampling(greedy=True), graphs=False)
        together = Engine(model, batch=3, positions=64, max_frames=8, sampling=Sampling(greedy=True), graphs=False)
        others = [prompt(8, 11, codebooks), prompt(9, 3, codebooks)]
        self.assertEqual(dict(alone.generate([target], 8))[0].tolist(),
                         dict(together.generate([others[0], target, others[1]], 8))[1].tolist())

    def test_sampling_keeps_codes_in_the_top_k(self):
        from fast_misotts.engine import Sampling, sample
        torch.manual_seed(3)
        logits = torch.randn(64, 17)
        picked = sample(logits, Sampling(temperature=0.9, topk=3))
        top = torch.topk(logits, 3).indices
        self.assertTrue(bool((picked == top).any(dim=1).all()))
        self.assertEqual(sample(logits, Sampling(greedy=True)).squeeze(1).tolist(), logits.argmax(1).tolist())


@unittest.skipIf(torch is None, "Requires torch")
class WatermarkTests(unittest.TestCase):
    def test_the_original_steps_with_stand_ins(self):
        """silentcipher and torchaudio stood in for: the audio goes to 44.1 kHz, the key is encoded at a message SDR
        of 36 dB without SDR, and comes back at 24 kHz on the CPU; no audio stays none."""
        import sys
        import types
        from unittest import mock
        calls = []

        def resample(audio, orig_freq, new_freq):
            calls.append(("resample", orig_freq, new_freq))
            return audio * 2

        class Model:
            def encode_wav(self, audio, rate, key, calc_sdr=True, message_sdr=None):
                calls.append(("encode", rate, key, calc_sdr, message_sdr))
                return audio + 1, None

        cipher = types.ModuleType("silentcipher")
        cipher.get_model = lambda model_type, device: calls.append(("model", model_type, device)) or Model()
        audio_module = types.ModuleType("torchaudio")
        audio_module.functional = types.SimpleNamespace(resample=resample)
        with mock.patch.dict(sys.modules, {"silentcipher": cipher, "torchaudio": audio_module}):
            from fast_misotts.watermark import Watermarker
            marker = Watermarker(device="cpu", key=(1, 2, 3, 4, 5))
            out = marker(torch.ones(4), 24_000)
            self.assertEqual(marker(torch.zeros(0), 24_000).shape, (0,))
        self.assertEqual(calls, [("model", "44.1k", "cpu"), ("resample", 24_000, 44_100),
                                 ("encode", 44_100, [1, 2, 3, 4, 5], False, 36), ("resample", 44_100, 24_000)])
        self.assertEqual(out.tolist(), [6.0] * 4)
        self.assertEqual((out.dtype, out.device.type), (torch.float32, "cpu"))

    def test_a_clear_error_without_silentcipher(self):
        import sys
        from unittest import mock
        with mock.patch.dict(sys.modules, {"silentcipher": None}):
            from fast_misotts.watermark import Watermarker
            with self.assertRaisesRegex(ImportError, "SesameAILabs/silentcipher"):
                Watermarker(device="cpu")


@unittest.skipIf(torch is None, "Requires torch")
class QuantTests(unittest.TestCase):
    """The weights' side of each quantize mode, on the CPU (the products run on the GPU: bench quant-check)."""

    def test_int4_groups_round_to_half_a_step(self):
        from fast_misotts.quant import int4_groups
        torch.manual_seed(0)
        weight = torch.randn(8, 256)
        codes, scale, zero = int4_groups(weight, 128)
        self.assertEqual((codes.shape, scale.shape, zero.shape), ((8, 256), (8, 2), (8, 2)))
        self.assertTrue(bool(((codes >= 0) & (codes <= 15)).all()))
        back = ((codes.reshape(8, 2, 128) - 8) * scale[..., None] + zero[..., None]).reshape(8, 256)
        self.assertTrue(bool(((back - weight).abs() <= scale.repeat_interleave(128, dim=1) / 2 + 1e-5).all()))

    def test_eight_bit_rows_keep_to_their_scales(self):
        from fast_misotts.quant import Fp8Linear, Int8Linear
        torch.manual_seed(0)
        weight = torch.randn(16, 64)
        integer = Int8Linear(weight)
        back = integer.weight.float() * integer.scale[:, None]
        self.assertTrue(bool(((back - weight).abs() <= integer.scale[:, None] / 2 + 1e-6).all()))
        floating = Fp8Linear(weight)
        back = floating.weight.float() * floating.scale[:, None]
        # E4M3 keeps 3 mantissa bits: within 1/16 of each value (above its subnormals)
        large = weight.abs() > floating.scale[:, None] * 2 ** -6
        self.assertTrue(bool(((back - weight).abs()[large] <= weight.abs()[large] / 16 + 1e-6).all()))

    def test_quantize_replaces_every_layer_linear_and_nothing_else(self):
        from fast_misotts.quant import Int8Linear, quantize
        model = quantize(tiny(), "int8")
        replaced = [module for module in model.modules() if isinstance(module, Int8Linear)]
        self.assertEqual(len(replaced), 7 * (len(model.backbone.layers) + len(model.decoder.layers)))
        self.assertIsInstance(model.projection, torch.nn.Linear)
        self.assertIsInstance(model.codebook0_head, torch.nn.Linear)
        with self.assertRaisesRegex(ValueError, "quantize"):
            quantize(tiny(), "int3")

    def test_specs_give_each_stack_its_own(self):
        from fast_misotts.quant import Int8WeightLinear, parse, quantize
        self.assertEqual(parse("int4"), {"backbone": ("int4", 128), "decoder": ("int4", 128)})
        self.assertEqual(parse("int4:backbone+int4/g32:decoder"),
                         {"backbone": ("int4", 128), "decoder": ("int4", 32)})
        self.assertEqual(parse("int8w:decoder"), {"decoder": ("int8w", 128)})
        for bad in ("int4/g48", "int8/g32", "int4/32", "int4:backbone+int8:backbone", "int4:middle"):
            with self.assertRaisesRegex(ValueError, "quantize"):
                parse(bad)
        model = tiny()
        x = torch.randn(3, model.config.decoder.dim)
        want = model.decoder.layers[0].mlp.w1(x)
        quantize(model, "int8w:decoder")
        self.assertIsInstance(model.decoder.layers[0].mlp.w1, Int8WeightLinear)
        self.assertIsInstance(model.backbone.layers[0].mlp.w1, torch.nn.Linear)
        got = model.decoder.layers[0].mlp.w1(x)
        self.assertLess(float((got - want).norm() / want.norm()), 0.02)


@unittest.skipIf(torch is None, "Requires torch")
class AwqTests(unittest.TestCase):
    """fast_misotts.awq on the tiny model, in groups of 32 (its widths are 32 to 128)."""

    def calibrated(self):
        from fast_misotts.awq import calibrate
        from fast_misotts.engine import Sampling
        model = tiny()
        codebooks = model.config.codebooks
        prompts = [prompt(1, 5, codebooks), prompt(2, 9, codebooks, context=3), prompt(3, 2, codebooks)]
        inputs = calibrate(model, prompts, rows=64, batch=2, positions=64, frames=10, sampling=Sampling(greedy=True))
        return model, prompts, inputs

    def test_inputs_keep_a_uniform_sample_and_every_rows_mean(self):
        from fast_misotts.awq import Inputs
        torch.manual_seed(0)
        inputs = Inputs(50)
        rows = torch.arange(1000, dtype=torch.float32)[:, None].repeat(1, 3)
        for start in range(0, 1000, 7):
            inputs.add(rows[start:start + 7])
        sample = inputs.sample()
        self.assertEqual(tuple(sample.shape), (50, 3))
        self.assertEqual(len(set(sample[:, 0].tolist())), 50)  # 50 of the rows added, none twice
        self.assertTrue(bool((sample[:, 0] == sample[:, 2]).all()))
        self.assertEqual(inputs.count, 1000)
        self.assertAlmostEqual(float(inputs.mean()[0]), 499.5, places=3)
        self.assertTrue(300 < float(sample[:, 0].mean()) < 700)  # from the whole stream, not its start

    def test_calibration_counts_the_rows_of_utterances_alone(self):
        model, prompts, inputs = self.calibrated()
        codebooks, frames = model.config.codebooks, 10  # each of these prompts runs to the limit (greedy)
        prompted = sum(len(tokens) for tokens, _ in prompts)
        for number in range(2):
            for role in ("attn_in", "attn_out", "mlp_in", "mlp_out"):
                # the prompts' positions (not their padding), then every frame but the first (made from the prompt)
                self.assertEqual(inputs[f"backbone.{number}.{role}"].count, prompted + 3 * (frames - 1))
                # every frame: the depth decoder's first step reads two positions, each later step one
                self.assertEqual(inputs[f"decoder.{number}.{role}"].count, 3 * frames * codebooks)
        kept = inputs["backbone.0.mlp_out"]  # fewer rows than the 64 kept: all of them
        self.assertEqual(tuple(kept.sample().shape), (prompted + 3 * (frames - 1), model.config.backbone.ffn))
        torch.testing.assert_close(kept.mean(), kept.sample().abs().mean(0), rtol=1e-5, atol=1e-6)
        self.assertEqual(tuple(inputs["decoder.0.attn_in"].sample().shape), (64, model.config.decoder.dim))

    def test_folded_scales_keep_the_function(self):
        """Scales without clipping change no output in full precision."""
        import copy
        from fast_misotts.awq import CLIPPED, apply
        model = tiny()
        before = copy.deepcopy(model)
        generator = torch.Generator().manual_seed(1)
        state = {"groupsize": torch.tensor([32]), "grid": torch.tensor([20])}
        for stack_name in ("backbone", "decoder"):
            stack = getattr(model, stack_name)
            for number, layer in enumerate(stack.layers):
                for role, width in (("attn_in", stack.spec.dim), ("mlp_in", stack.spec.dim),
                                    ("mlp_out", stack.spec.ffn)):
                    state[f"{stack_name}.{number}.{role}.scale"] = torch.exp(torch.randn(width, generator=generator))
                for part, name, _ in CLIPPED:
                    weight = getattr(getattr(layer, part), name).weight
                    state[f"{stack_name}.{number}.{name}.clip"] = torch.zeros(weight.shape[0], weight.shape[1] // 32,
                                                                              dtype=torch.uint8)
        apply(model, state)
        self.assertFalse(torch.equal(model.backbone.layers[0].attn.q_proj.weight,
                                     before.backbone.layers[0].attn.q_proj.weight))
        for stack_name in ("backbone", "decoder"):
            x = torch.randn(1, 7, getattr(model, stack_name).spec.dim, generator=generator)
            torch.testing.assert_close(reference_stack(getattr(model, stack_name), x),
                                       reference_stack(getattr(before, stack_name), x), rtol=1e-4, atol=1e-4)

    def test_teacher_forcing_ranks_the_greedy_codes_first(self):
        """bench kl's teacher-forced log-probabilities: the codes greedy generation chose come first everywhere."""
        from fast_misotts.bench import teacher_forced
        from fast_misotts.engine import Engine, Sampling
        model = tiny()
        codebooks = model.config.codebooks
        prompts = [prompt(1, 5, codebooks), prompt(2, 9, codebooks, context=3)]
        engine = Engine(model, batch=2, positions=64, max_frames=10, sampling=Sampling(greedy=True), graphs=False)
        for index, codes in engine.generate(prompts, 10):
            logp = teacher_forced(model, prompts[index], codes)
            self.assertEqual(tuple(logp.shape), (len(codes), codebooks, model.config.audio_vocab))
            self.assertEqual(logp.argmax(-1).tolist(), codes.tolist())

    def test_search_beats_rounding_and_apply_redoes_it(self):
        from fast_misotts.awq import apply, load, save, search
        model, _, inputs = self.calibrated()
        state, report = search(model, inputs, groupsize=32, grid=4, clip_rows=16, log=lambda line: None)
        self.assertEqual(len(report), len(model.backbone.layers) + len(model.decoder.layers))
        for errors in report.values():
            for role in ("attn_in", "mlp_in", "mlp_out"):
                self.assertLessEqual(errors[role]["awq"], errors[role]["nearest"])
        folder = ROOT / ".cache" / "test-artifacts" / uuid4().hex
        folder.mkdir(parents=True)
        save(folder / "awq.safetensors", state)
        fresh = apply(tiny(), load(folder / "awq.safetensors"))  # the search's model, from the saved state alone
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(fresh.state_dict()[key], value), key)
        self.assertFalse(torch.equal(model.backbone.layers[0].mlp_norm.scale,
                                     tiny().backbone.layers[0].mlp_norm.scale))
        partial = apply(tiny(), state, {"decoder": 32})  # the decoder alone quantized: the backbone left as it is
        self.assertTrue(torch.equal(partial.backbone.layers[0].mlp_norm.scale,
                                    tiny().backbone.layers[0].mlp_norm.scale))
        self.assertTrue(torch.equal(partial.decoder.layers[0].mlp_norm.scale, model.decoder.layers[0].mlp_norm.scale))
        with self.assertRaisesRegex(ValueError, "groups of 32"):
            apply(tiny(), state, {"decoder": 64})
        with self.assertRaisesRegex(ValueError, "fuse_projections"):
            apply(tiny().fuse_projections(), state)


@unittest.skipIf(torch is None, "Requires torch")
class FormatTests(unittest.TestCase):
    def test_rows_and_masks(self):
        from fast_misotts.text import audio_frames
        tokens, mask = audio_frames(torch.tensor([[3, 4], [5, 6]]))
        self.assertEqual(tokens.tolist(), [[3, 4, 0], [5, 6, 0], [0, 0, 0]])  # the end-of-audio row of zeros
        self.assertEqual(mask.tolist(), [[True, True, False]] * 3)

    def test_rope_is_rounded_as_the_original_rounds_it(self):
        from fast_misotts.model import rope_table
        model = tiny()
        self.assertEqual(model.backbone.rope.dtype, torch.float32)
        model = model.to(torch.bfloat16)  # the original's model.to(dtype) rounds its RoPE buffer from FP32
        self.assertTrue(torch.equal(model.backbone.rope, rope_table(model.backbone.spec).to(torch.bfloat16)))
        self.assertEqual(model.decoder.layers[0].attn.q_proj.weight.dtype, torch.bfloat16)

    def test_load_model_reads_the_original_names_and_skips_extra_buffers(self):
        from safetensors.torch import save_file
        from fast_misotts.model import load_model
        model = tiny()
        state = {key: value.contiguous() for key, value in model.state_dict().items()}
        state["backbone_causal_mask"] = torch.ones(4, 4, dtype=torch.bool)  # an original checkpoint's extra buffer
        folder = ROOT / ".cache" / "test-artifacts" / uuid4().hex
        folder.mkdir(parents=True)
        save_file(state, str(folder / "model.safetensors"))
        loaded = load_model(folder / "model.safetensors", device="cpu", dtype=torch.float32, config=model.config)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[key], value), key)
        del state["projection.weight"]
        save_file(state, str(folder / "partial.safetensors"))
        with self.assertRaisesRegex(ValueError, "lacks 1 parameters"):
            load_model(folder / "partial.safetensors", device="cpu", dtype=torch.float32, config=model.config)


if __name__ == "__main__":
    unittest.main()
