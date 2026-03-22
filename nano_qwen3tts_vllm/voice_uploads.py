from __future__ import annotations

import asyncio
import io
import os
import re
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import UploadFile

FILENAME_PART_RE = re.compile(r"[^0-9A-Za-z\u0400-\u04FF_-]+")
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".au"}

_whisper_model = None
_whisper_lock = asyncio.Lock()


def _build_whisper_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        "tiny",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8",
    )


def sanitize_voice_name(raw: str) -> str:
    cleaned = FILENAME_PART_RE.sub("_", (raw or "").strip()).strip("_")
    cleaned = cleaned[:64]
    if not cleaned:
        raise ValueError("Voice name is required")
    return cleaned


def _safe_filename_part(raw: str) -> str:
    normalized = FILENAME_PART_RE.sub("_", (raw or "").strip()).strip("_")
    return normalized[:64] or "voice"


def _resample_to_24k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    target_sr = 24000
    if orig_sr == target_sr:
        return wav.astype(np.float32)
    n_orig = len(wav)
    n_new = int(round(n_orig * target_sr / orig_sr))
    if n_new <= 0:
        return wav.astype(np.float32)
    indices = np.linspace(0, n_orig - 1, n_new, dtype=np.float64)
    return np.interp(indices, np.arange(n_orig), wav).astype(np.float32)


async def get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    async with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        _whisper_model = await asyncio.to_thread(_build_whisper_model)
        return _whisper_model


async def transcribe_voice_file(voice_path: Path) -> str:
    model = await get_whisper_model()
    segments, _info = await asyncio.to_thread(model.transcribe, str(voice_path), beam_size=5)
    text = " ".join(segment.text for segment in segments).strip()
    if not text:
        raise ValueError("Whisper transcribed an empty string")
    return text


async def prepare_uploaded_voice_file(
    *,
    voices_dir: Path,
    upload: UploadFile,
    filename_prefix: str,
    reference_text: str | None = None,
) -> tuple[Path, str]:
    voices_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported audio format")

    audio_bytes = await upload.read()
    await upload.close()
    if not audio_bytes:
        raise ValueError("Uploaded audio file is empty")

    try:
        wav, sample_rate = await asyncio.to_thread(sf.read, io.BytesIO(audio_bytes), dtype="float32")
    except Exception as error:
        raise ValueError(f"Invalid audio file: {error}") from error

    if wav.ndim > 1:
        wav = np.mean(wav, axis=1).astype(np.float32)
    if len(wav) == 0:
        raise ValueError("Uploaded audio file is empty")

    duration_sec = len(wav) / float(sample_rate or 1)
    if duration_sec < 0.5:
        raise ValueError("Voice sample is too short. Minimum 0.5s")
    if duration_sec > 90.0:
        raise ValueError("Voice sample is too long. Maximum 90s")

    normalized_wav = _resample_to_24k(wav, int(sample_rate))
    safe_prefix = _safe_filename_part(filename_prefix)
    target_path = voices_dir / f"{safe_prefix}_{uuid.uuid4().hex}.wav"
    await asyncio.to_thread(sf.write, str(target_path), normalized_wav, 24000)

    resolved_text = (reference_text or "").strip()
    if not resolved_text:
        resolved_text = await transcribe_voice_file(target_path)
    return target_path, resolved_text


async def prepare_reference_audio_from_path(voice_path: Path) -> tuple[np.ndarray, int]:
    try:
        wav, sample_rate = await asyncio.to_thread(sf.read, str(voice_path), dtype="float32")
    except Exception as error:
        raise ValueError(f"Failed to read stored voice sample: {error}") from error

    if wav.ndim > 1:
        wav = np.mean(wav, axis=1).astype(np.float32)
    return wav, int(sample_rate)
