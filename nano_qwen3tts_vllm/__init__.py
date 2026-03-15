from __future__ import annotations

from pathlib import Path

_SOURCE_DIR = Path(__file__).resolve().parent.parent / "nano-qwen3tts-vllm"

# Allow `python api_server.py` without requiring `pip install -e .`.
__path__ = [str(_SOURCE_DIR)]

