"""Qwen3-TTS with vLLM-style optimizations."""

from nano_qwen3tts_vllm.config import (
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from nano_qwen3tts_vllm.sampling_params import SamplingParams

__version__ = "0.1.0"

__all__ = [
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
    "SamplingParams",
]
