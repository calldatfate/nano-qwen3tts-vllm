from __future__ import annotations

import mimetypes
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from .voice_uploads import prepare_reference_audio_from_path, transcribe_voice_file


def parse_optional_user_id(raw_value: object) -> int | None:
    try:
        value = int(str(raw_value or "").strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


async def resolve_base_model_reference_voice(
    app: FastAPI,
    *,
    user_id_raw: object,
    requested_voice: str | None,
) -> tuple[dict, np.ndarray, int]:
    user_id = parse_optional_user_id(user_id_raw)
    voice_record = await app.state.voice_store.resolve_voice_record_for_user(user_id, requested_voice)
    if voice_record is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Base model requires either ref_audio upload or a stored Qwen clone voice. "
                "Upload a voice sample in Voice Management and select it before synthesis."
            ),
        )

    file_path_raw = str(voice_record.get("file_path") or "").strip()
    if not file_path_raw:
        raise HTTPException(status_code=400, detail="Selected Qwen voice has no stored file_path")

    voice_path = Path(file_path_raw).resolve()
    if not voice_path.exists():
        raise HTTPException(status_code=400, detail="Stored Qwen voice sample file is missing")

    wav, sample_rate = await prepare_reference_audio_from_path(voice_path)
    return voice_record, wav, sample_rate


def guess_audio_media_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def serialize_voice_record(voice: dict) -> dict:
    payload = dict(voice)
    payload["type"] = "global" if payload.get("voice_type") == "global" else "user"
    payload["is_global"] = payload.get("voice_type") == "global"
    return payload


def resolve_voice_file_path(app: FastAPI, voice: dict) -> Path:
    file_path_raw = str(voice.get("file_path") or "").strip()
    if not file_path_raw:
        raise HTTPException(status_code=400, detail="Voice has no file_path")

    voice_path = Path(file_path_raw).resolve()
    try:
        voice_path.relative_to(app.state.voice_files_dir.resolve())
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid voice file path") from error
    return voice_path


def cleanup_voice_file(app: FastAPI, voice: dict) -> None:
    file_path_raw = str(voice.get("file_path") or "").strip()
    if not file_path_raw:
        return
    try:
        voice_path = Path(file_path_raw).resolve()
        voice_path.relative_to(app.state.voice_files_dir.resolve())
        voice_path.unlink(missing_ok=True)
    except Exception:
        return


async def retranscribe_voice_record(request: Request, voice_id: int) -> dict[str, object]:
    voice = await request.app.state.voice_store.get_voice_by_id(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_path = resolve_voice_file_path(request.app, voice)
    try:
        reference_text = await transcribe_voice_file(voice_path)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    updated = await request.app.state.voice_store.update_voice_settings(
        voice_id,
        {"reference_text": reference_text},
    )
    return {
        "success": True,
        "status": "success",
        "voice": serialize_voice_record(updated or voice),
        "reference_text": reference_text,
    }


async def admin_voice_groups(app: FastAPI) -> dict[str, object]:
    voices = await app.state.voice_store.list_all_voices()
    serialized = [serialize_voice_record(voice) for voice in voices]
    global_voices = [voice for voice in serialized if voice.get("voice_type") == "global"]
    user_voices = [voice for voice in serialized if voice.get("voice_type") != "global"]
    return {
        "success": True,
        "voices": serialized,
        "global_voices": global_voices,
        "user_voices": user_voices,
    }
