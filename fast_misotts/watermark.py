"""The SilentCipher watermark the original MisoTTS puts on every utterance (its watermarking.py), here optional and
off by default: MisoTTS(watermark=True). It needs silentcipher as the original pins it (Sesame's fork):

    pip install "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@d46d7d0893a583d8968ab3a6626e2289faec9152"

and fetches its weights from the Hub (sony/silentcipher) on first use. The steps are the original's: the audio
resampled to 44.1 kHz, the key encoded at a message SDR of 36 dB, and the result resampled back."""
from __future__ import annotations

from typing import Sequence

import torch

# The original's key. Its watermarking.py: when using MisoTTS in another application, set your own and keep it secret.
MISO_TTS_WATERMARK = (0, 0, 0, 0, 0)
INSTALL = 'pip install "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@d46d7d0893a583d8968ab3a6626e2289faec9152"'


class Watermarker:
    def __init__(self, device="cuda", key: Sequence[int] = MISO_TTS_WATERMARK):
        try:
            import silentcipher
            import torchaudio  # noqa: F401  (silentcipher needs it too)
        except ImportError as error:
            raise ImportError(f"watermark=True needs silentcipher and torchaudio: {INSTALL}") from error
        self.model = silentcipher.get_model(model_type="44.1k", device=device)
        self.device, self.key = device, list(key)

    @torch.inference_mode()
    def __call__(self, audio: torch.Tensor, sample_rate: int = 24_000) -> torch.Tensor:
        """Mono audio [samples] at sample_rate, watermarked, at the same rate (on the CPU)."""
        import torchaudio
        if len(audio) == 0:
            return audio
        high = torchaudio.functional.resample(audio.to(self.device), orig_freq=sample_rate, new_freq=44_100)
        encoded, _ = self.model.encode_wav(high, 44_100, self.key, calc_sdr=False, message_sdr=36)
        rate = min(44_100, sample_rate)
        return torchaudio.functional.resample(encoded, orig_freq=44_100, new_freq=rate).float().cpu()
