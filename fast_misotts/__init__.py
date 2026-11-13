"""fast_misotts: MisoTTS 8B inference in plain PyTorch: the original's weights and arithmetic, batched with static
caches, CUDA graphs and continuous batching."""
from .api import MisoTTS, Request, Segment
from .engine import Engine, Sampling
from .model import MISO_8B, MisoConfig, MisoModel, StackConfig, load_model
from .watermark import MISO_TTS_WATERMARK, Watermarker

__version__ = "0.1.0"
__all__ = ["MisoTTS", "Request", "Segment", "Engine", "Sampling", "MISO_8B", "MisoConfig", "MisoModel",
           "StackConfig", "load_model", "MISO_TTS_WATERMARK", "Watermarker"]
