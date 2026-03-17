import io
import json
import mimetypes
import struct
import numpy as np
import gc
import uuid
from collections import deque
import torch
import soundfile as sf
import traceback
import os
import asyncio
import time
import threading
import sys
import argparse
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, BackgroundTasks, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph
from nano_qwen3tts_vllm.utils.prompt import prepare_custom_voice_prompt, _tokenize_texts
from nano_qwen3tts_vllm.utils.generation import prepare_inputs, generate_speaker_prompt, generate_icl_prompt
from runtime_models import (
    QWEN_MODEL_CATALOG,
    build_runtime_model_catalog,
    pick_runtime_model,
)
from voice_store import FileVoiceStore
from voice_uploads import (
    prepare_reference_audio_from_path,
    prepare_uploaded_voice_file,
    sanitize_voice_name,
    transcribe_voice_file,
)

# Global state
current_model_name = None
interface = None
_zmq_bridge = None # Renamed from zmq_bridge
_tokenizer = None # New global for tokenizer
USE_ZMQ = False # New global for ZMQ status
ENFORCE_EAGER = False

# Global lock for safe decoding across async requests
decode_lock = threading.Lock()

# Serialize model switch/load so concurrent requests do not race.
model_switch_lock = asyncio.Lock()

# Stream registry:
# stream_id -> {
#   "tenant_id": str,
#   "request_data": dict,
#   "state": "queued|running|finished|cancelled|failed",
#   "stream_requested": bool,
#   "error": Optional[str],
#   "created_at": float,
#   "started_at": Optional[float],
#   "finished_at": Optional[float]
# }
active_streams = {}

# Fair queue over tenants (single generation slot).
tenant_queues = {}  # tenant_id -> deque[stream_id]
tenant_rr = deque()  # round-robin ring of tenant_ids
tenant_rr_set = set()
queue_condition = asyncio.Condition()
active_stream_id = None

MAX_QUEUE_PER_TENANT = int(os.environ.get("MAX_QUEUE_PER_TENANT", "20"))
MAX_TOTAL_QUEUED = int(os.environ.get("MAX_TOTAL_QUEUED", "200"))
STREAM_WAIT_TIMEOUT_SEC = float(os.environ.get("STREAM_WAIT_TIMEOUT_SEC", "0"))
SERVER_HOST = os.environ.get("QWEN_TTS_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("QWEN_TTS_PORT", os.environ.get("PORT", "8012")))
VOICE_STORAGE_DIR = Path(os.environ.get("QWEN_VOICE_STORAGE_DIR", "./runtime/qwen_voices")).resolve()
VOICE_FILES_DIR = VOICE_STORAGE_DIR / "files"
VOICE_PREVIEW_DIR = VOICE_STORAGE_DIR / "previews"
VOICE_STATE_PATH = VOICE_STORAGE_DIR / "state.json"


def _runtime_configured_model() -> str | None:
    candidate = str(os.environ.get("QWEN3_TTS_MODEL_PATH", "")).strip()
    return candidate or None


def _runtime_allowed_models_raw() -> str | None:
    for env_name in ("QWEN_TTS_ALLOWED_MODELS", "QWEN_ALLOWED_MODELS"):
        candidate = str(os.environ.get(env_name, "")).strip()
        if candidate:
            return candidate
    return None


def _runtime_allowed_models_source() -> str | None:
    for env_name in ("QWEN_TTS_ALLOWED_MODELS", "QWEN_ALLOWED_MODELS"):
        candidate = str(os.environ.get(env_name, "")).strip()
        if candidate:
            return env_name
    return None


def _runtime_model_catalog() -> list[dict[str, object]]:
    return build_runtime_model_catalog(
        configured_model=_runtime_configured_model(),
        allowed_models_raw=_runtime_allowed_models_raw(),
    )


def _resolve_runtime_model(model_name: str | None, *, required_family: str | None = None) -> str:
    try:
        return pick_runtime_model(
            requested_model=model_name,
            runtime_catalog=_runtime_model_catalog(),
            required_family=required_family,
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

def load_model(model_name):
    global interface, _tokenizer, USE_ZMQ, _zmq_bridge, current_model_name
    
    USE_ZMQ = os.environ.get("USE_ZMQ", "1") == "1"
    
    if USE_ZMQ:
        _zmq_bridge = ZMQOutputBridge(auto_find_port=True)
        interface = Qwen3TTSInterface.from_pretrained(
            model_name,
            zmq_bridge=_zmq_bridge,
            enforce_eager=ENFORCE_EAGER,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9  # Set high to avoid KV cache negative calculation on heavy models
        )
    else:
        interface = Qwen3TTSInterface.from_pretrained(
            model_name,
            enforce_eager=ENFORCE_EAGER,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        )
        
    _tokenizer = SpeechTokenizerCUDAGraph(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device="cuda:0",
    )
    current_model_name = model_name


def _dispose_loaded_model_state():
    global interface, _zmq_bridge, _tokenizer, current_model_name

    if interface is not None and hasattr(interface, "shutdown"):
        interface.shutdown()

    if _tokenizer is not None and hasattr(_tokenizer, "shutdown"):
        _tokenizer.shutdown()

    interface = None
    _tokenizer = None
    _zmq_bridge = None
    current_model_name = None

    torch._dynamo.reset()

    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

async def switch_model_if_needed(model_name: str):
    global current_model_name, interface, _zmq_bridge, USE_ZMQ, _tokenizer
    if interface is not None and current_model_name != model_name:
        print(f"\n🔄 Switching model from {current_model_name} to {model_name}...")
        if USE_ZMQ and hasattr(interface, 'zmq_bridge') and interface.zmq_bridge:
            await interface.stop_zmq_tasks()
            # Forcefully close ZMQ context to release TCP ports instantly
            if hasattr(interface.zmq_bridge, 'context'):
                interface.zmq_bridge.context.destroy(linger=0)
            interface.zmq_bridge.close()

        # Shutdown/model cleanup can take noticeable time and must not block the event loop.
        await asyncio.to_thread(_dispose_loaded_model_state)
        
    if interface is None:
        print(f"\n🚀 Loading model (ZMQ Mode): {model_name}...")
        try:
            await asyncio.to_thread(load_model, model_name)
            if USE_ZMQ and hasattr(interface, 'zmq_bridge') and interface.zmq_bridge:
                # Start background ZMQ loop (sync context)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(interface.start_zmq_tasks())
                else:
                    asyncio.run(interface.start_zmq_tasks())
            current_model_name = model_name
        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"Error loading model:\n{err_msg}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


def _parse_optional_user_id(raw_value: object) -> int | None:
    try:
        value = int(str(raw_value or "").strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


async def _resolve_base_model_reference_voice(
    app: FastAPI,
    *,
    user_id_raw: object,
    requested_voice: str | None,
) -> tuple[dict, np.ndarray, int]:
    user_id = _parse_optional_user_id(user_id_raw)
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


def _guess_audio_media_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def _serialize_voice_record(voice: dict) -> dict:
    payload = dict(voice)
    payload["type"] = "global" if payload.get("voice_type") == "global" else "user"
    payload["is_global"] = payload.get("voice_type") == "global"
    return payload


def _resolve_voice_file_path(app: FastAPI, voice: dict) -> Path:
    file_path_raw = str(voice.get("file_path") or "").strip()
    if not file_path_raw:
        raise HTTPException(status_code=400, detail="Voice has no file_path")

    voice_path = Path(file_path_raw).resolve()
    try:
        voice_path.relative_to(app.state.voice_files_dir.resolve())
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid voice file path") from error
    return voice_path


def _cleanup_voice_file(app: FastAPI, voice: dict) -> None:
    file_path_raw = str(voice.get("file_path") or "").strip()
    if not file_path_raw:
        return
    try:
        voice_path = Path(file_path_raw).resolve()
        voice_path.relative_to(app.state.voice_files_dir.resolve())
        voice_path.unlink(missing_ok=True)
    except Exception:
        return


async def _retranscribe_voice_record(request: Request, voice_id: int) -> dict[str, object]:
    voice = await request.app.state.voice_store.get_voice_by_id(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_path = _resolve_voice_file_path(request.app, voice)
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
        "voice": _serialize_voice_record(updated or voice),
        "reference_text": reference_text,
    }


async def _admin_voice_groups(app: FastAPI) -> dict[str, object]:
    voices = await app.state.voice_store.list_all_voices()
    serialized = [_serialize_voice_record(voice) for voice in voices]
    global_voices = [voice for voice in serialized if voice.get("voice_type") == "global"]
    user_voices = [voice for voice in serialized if voice.get("voice_type") != "global"]
    return {
        "success": True,
        "voices": serialized,
        "global_voices": global_voices,
        "user_voices": user_voices,
    }

def _resample_to_24k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    TARGET_SAMPLE_RATE = 24000
    if orig_sr == TARGET_SAMPLE_RATE:
        return wav
    n_orig = len(wav)
    n_new = int(round(n_orig * TARGET_SAMPLE_RATE / orig_sr))
    if n_new == 0:
        return wav
    indices = np.linspace(0, n_orig - 1, n_new, dtype=np.float64)
    return np.interp(indices, np.arange(n_orig), wav).astype(np.float32)

def _decode_batch(codes: list):
    with decode_lock:
        wav_list, sr = _tokenizer.decode([{"audio_codes": codes}])
    wav = wav_list[0]
    wav_24k = _resample_to_24k(wav, sr)
    wav_24k = np.clip(wav_24k, -1.0, 1.0)
    return (wav_24k * 32767.0).astype(np.int16)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    VOICE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    VOICE_FILES_DIR.mkdir(parents=True, exist_ok=True)
    VOICE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    app.state.voice_store = FileVoiceStore(VOICE_STATE_PATH, VOICE_FILES_DIR)
    await app.state.voice_store.startup()
    app.state.voice_files_dir = VOICE_FILES_DIR
    app.state.voice_preview_dir = VOICE_PREVIEW_DIR
    yield
    # Cleanup on shutdown
    global interface, _zmq_bridge, USE_ZMQ
    await app.state.voice_store.close()
    if interface is not None:
        if USE_ZMQ and hasattr(interface, 'zmq_bridge') and interface.zmq_bridge:
            await interface.stop_zmq_tasks()
            interface.zmq_bridge.close()

app = FastAPI(title="Qwen3-TTS Streaming API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health/live")
async def health_live():
    return {
        "status": "healthy",
        "service": "nano-qwen3tts-vllm",
        "current_model": current_model_name,
    }


@app.get("/health/ready")
async def health_ready():
    return {
        "status": "healthy",
        "service": "nano-qwen3tts-vllm",
        "model_loaded": interface is not None,
        "current_model": current_model_name,
    }


@app.get("/api/models")
async def get_supported_models():
    models = _runtime_model_catalog()
    return {
        "success": True,
        "provider": "qwen",
        "current_model": current_model_name,
        "configured_model": _runtime_configured_model(),
        "model_policy": "restricted" if _runtime_allowed_models_raw() or _runtime_configured_model() else "dynamic",
        "allowed_models_source": _runtime_allowed_models_source(),
        "models": models,
    }


@app.get("/api/audio/{filename}")
async def get_generated_audio(filename: str):
    preview_root = VOICE_PREVIEW_DIR.resolve()
    target_path = (preview_root / filename).resolve()
    try:
        target_path.relative_to(preview_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(
        target_path,
        media_type=_guess_audio_media_type(target_path),
        filename=target_path.name,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/tts/voices/global")
async def voices_global(request: Request):
    voices = await request.app.state.voice_store.list_global_voices()
    return [_serialize_voice_record(voice) for voice in voices]


@app.get("/api/tts/voices")
async def voices_all(request: Request, user_id: int | None = None):
    voices = await request.app.state.voice_store.list_available_voices(user_id=user_id)
    return [_serialize_voice_record(voice) for voice in voices]


@app.get("/api/tts/voices/{voice_id}")
async def get_voice_info(request: Request, voice_id: int):
    voice = await request.app.state.voice_store.get_voice_by_id(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
    return _serialize_voice_record(voice)


@app.get("/api/tts/user/voices/{user_id}")
async def user_voices(request: Request, user_id: int):
    voices = await request.app.state.voice_store.list_user_voices(user_id)
    return [_serialize_voice_record(voice) for voice in voices]


@app.post("/api/tts/user/voices/upload")
async def upload_user_voice(
    request: Request,
    file: UploadFile = File(...),
    voice_name: str = Form(""),
    name: str = Form(""),
    user_id: int = Form(...),
    reference_text: str = Form(""),
    sample_text: str = Form(""),
):
    try:
        clean_name = sanitize_voice_name(voice_name or name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    all_voices = await request.app.state.voice_store.list_all_voices()
    if any(
        int(item.get("owner_id") or 0) == int(user_id)
        and str(item.get("name") or "").strip().lower() == clean_name.lower()
        for item in all_voices
    ):
        raise HTTPException(status_code=400, detail=f"Voice with name '{clean_name}' already exists")

    try:
        target_path, resolved_reference_text = await prepare_uploaded_voice_file(
            voices_dir=request.app.state.voice_files_dir,
            upload=file,
            filename_prefix=f"user_{user_id}_{clean_name}",
            reference_text=reference_text or sample_text or None,
        )
        voice = await request.app.state.voice_store.create_voice(
            name=clean_name,
            owner_id=int(user_id),
            voice_type="user",
            file_path=str(target_path),
            is_public=False,
            reference_text=resolved_reference_text or None,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    return {"success": True, "status": "success", "voice": voice}


@app.delete("/api/tts/user/voices/{voice_id}")
async def delete_user_voice(request: Request, voice_id: int, user_id: int):
    voice = await request.app.state.voice_store.get_voice_by_id(voice_id)
    if not voice or int(voice.get("owner_id") or 0) != int(user_id):
        raise HTTPException(status_code=404, detail="Voice not found")

    file_path_raw = str(voice.get("file_path") or "").strip()
    await request.app.state.voice_store.delete_voice(voice_id)
    if file_path_raw:
        voice_path = Path(file_path_raw).resolve()
        try:
            voice_path.relative_to(request.app.state.voice_files_dir.resolve())
            voice_path.unlink(missing_ok=True)
        except ValueError:
            pass
    return {"success": True}


@app.put("/api/tts/user/voices/{voice_id}/settings")
async def update_user_voice_settings(request: Request, voice_id: int, settings: dict):
    updated = await request.app.state.voice_store.update_voice_settings(voice_id, settings)
    if not updated:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"success": True, "voice": updated}


@app.get("/api/tts/user/voices/enabled/{user_id}")
async def get_enabled_voices(request: Request, user_id: int):
    voice_ids = await request.app.state.voice_store.get_enabled_voice_ids(user_id)
    return {"success": True, "voice_ids": voice_ids, "enabled_voice_ids": voice_ids}


@app.post("/api/tts/user/voices/enabled/{user_id}")
async def set_enabled_voices(request: Request, user_id: int, voice_ids: list[int]):
    stored = await request.app.state.voice_store.set_enabled_voice_ids(user_id, voice_ids)
    return {"success": True, "voice_ids": stored, "enabled_voice_ids": stored}


@app.put("/api/tts/user/voices/{voice_id}/rename")
async def rename_user_voice(
    request: Request,
    voice_id: int,
    user_id: int,
    new_name: str = Form(...),
):
    voice = await request.app.state.voice_store.get_voice_by_id(voice_id)
    if not voice or int(voice.get("owner_id") or 0) != int(user_id):
        raise HTTPException(status_code=404, detail="Voice not found")
    try:
        clean_name = sanitize_voice_name(new_name)
        updated = await request.app.state.voice_store.rename_voice(voice_id, clean_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {"success": True, "voice": updated}


@app.post("/api/tts/user/voices/{voice_id}/retranscribe")
async def retranscribe_user_voice(request: Request, voice_id: int, user_id: int):
    voice = await request.app.state.voice_store.get_voice_by_id(voice_id)
    if not voice or int(voice.get("owner_id") or 0) != int(user_id):
        raise HTTPException(status_code=404, detail="Voice not found")
    return await _retranscribe_voice_record(request, voice_id)


@app.get("/api/admin/voices")
async def admin_list_global_voices(request: Request):
    return await _admin_voice_groups(request.app)


@app.post("/api/admin/voices/upload")
async def upload_admin_voice(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(""),
    voice_name: str = Form(""),
):
    raw_name = name or voice_name or file.filename or "global_voice"
    target_path: Path | None = None
    try:
        clean_name = sanitize_voice_name(raw_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        target_path, resolved_reference_text = await prepare_uploaded_voice_file(
            voices_dir=request.app.state.voice_files_dir,
            upload=file,
            filename_prefix=f"global_{clean_name}",
        )
        voice = await request.app.state.voice_store.create_voice(
            name=clean_name,
            owner_id=None,
            voice_type="global",
            file_path=str(target_path),
            is_public=True,
            reference_text=resolved_reference_text or None,
        )
    except ValueError as error:
        if target_path is not None:
            target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception:
        if target_path is not None:
            target_path.unlink(missing_ok=True)
        raise

    return {"success": True, "status": "success", "voice": _serialize_voice_record(voice)}


@app.delete("/api/admin/voices/{voice_id}")
async def delete_admin_voice(request: Request, voice_id: int):
    voice = await request.app.state.voice_store.get_voice_by_id(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    deleted = await request.app.state.voice_store.delete_voice(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Voice not found")

    _cleanup_voice_file(request.app, voice)
    return {"success": True}


@app.put("/api/admin/voices/{voice_id}/rename")
async def rename_admin_voice(request: Request, voice_id: int, new_name: str):
    try:
        clean_name = sanitize_voice_name(new_name)
        updated = await request.app.state.voice_store.rename_voice(voice_id, clean_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if not updated:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"success": True, "voice": _serialize_voice_record(updated)}


@app.put("/api/admin/voices/{voice_id}/settings")
async def update_admin_voice_settings(request: Request, voice_id: int, settings: dict):
    updated = await request.app.state.voice_store.update_voice_settings(voice_id, settings)
    if not updated:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"success": True, "voice": _serialize_voice_record(updated)}


@app.post("/api/admin/voices/{voice_id}/toggle")
async def toggle_admin_voice(request: Request, voice_id: int):
    updated = await request.app.state.voice_store.toggle_voice(voice_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"success": True, "voice": _serialize_voice_record(updated)}


@app.post("/api/admin/voices/{voice_id}/retranscribe")
async def retranscribe_admin_voice(request: Request, voice_id: int):
    return await _retranscribe_voice_record(request, voice_id)


@app.post("/api/admin/voices/{voice_id}/transcribe")
async def transcribe_admin_voice(request: Request, voice_id: int):
    return await _retranscribe_voice_record(request, voice_id)


@app.get("/api/admin/stats")
async def admin_stats(request: Request, days: int = 7):
    _ = days
    stats_payload = await request.app.state.voice_store.stats()
    return {"success": True, **stats_payload}


@app.post("/api/admin/voices/test")
async def admin_test_voice(
    request: Request,
    voice_name: str = Form(...),
    user_id: int = Form(...),
    test_text: str = Form(...),
    model: str = Form("Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
):
    if not test_text.strip():
        raise HTTPException(status_code=400, detail="test_text is required")

    resolved_model = _resolve_runtime_model(model, required_family="base")

    voice_record, ref_wav, ref_sr = await _resolve_base_model_reference_voice(
        request.app,
        user_id_raw=user_id,
        requested_voice=voice_name,
    )
    ref_text = str(voice_record.get("reference_text") or "").strip()
    if not ref_text:
        voice_path = Path(str(voice_record.get("file_path") or "")).resolve()
        ref_text = await transcribe_voice_file(voice_path)
        await request.app.state.voice_store.update_voice_settings(
            int(voice_record["id"]),
            {"reference_text": ref_text},
        )

    request_data = {
        "model": resolved_model,
        "text": test_text,
        "language": "Russian",
        "temperature": 0.9,
        "instruction": "",
        "speaker": str(voice_record.get("name") or ""),
        "ref_audio": ref_wav,
        "ref_sr": ref_sr,
        "ref_text": ref_text,
        "cancel_event": asyncio.Event(),
        "channel_name": "voice_test",
        "author": "voice_test",
        "user_id": str(user_id),
        "request_id": str(uuid.uuid4()),
        "event_id": "",
    }

    preview_stream_id = f"preview-{uuid.uuid4()}"
    preview_bytes = bytearray()
    async with model_switch_lock:
        await switch_model_if_needed(resolved_model)
    try:
        async for chunk in audio_stream_generator_async(preview_stream_id, request_data):
            if chunk:
                preview_bytes.extend(chunk)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Preview synthesis failed: {error}") from error

    filename = f"{preview_stream_id}.wav"
    target_path = request.app.state.voice_preview_dir / filename
    await asyncio.to_thread(target_path.write_bytes, bytes(preview_bytes))
    return {
        "success": True,
        "audio_url": f"/api/audio/{filename}",
        "voice": str(voice_record.get("name") or voice_name),
        "selected_voice": str(voice_record.get("name") or voice_name),
        "model": resolved_model,
    }

# Global state
# These are re-declared here, but the ones above are the actual global state.
# Keeping them for now as per instruction to not make unrelated edits, but ideally they should be removed.
# current_model_name = None
# interface = None
# zmq_bridge = None
# ENFORCE_EAGER = False

# Global lock for safe decoding across async requests
# decode_lock = threading.Lock() # Already declared above

# Store active generators and cancellation events for streaming
# active_streams = {} # Already declared above

# The original load_model is removed as per instruction 1.
# The new load_model and switch_model_if_needed handle the logic.

def generate_wav_header(sample_rate: int, num_channels: int = 1, bit_depth: int = 16) -> bytes:
    """Generate a standard WAV header with an unknown data size (0xFFFFFFFF) for streaming."""
    byte_rate = sample_rate * num_channels * (bit_depth // 8)
    block_align = num_channels * (bit_depth // 8)
    
    header = b'RIFF'
    header += struct.pack('<I', 0xFFFFFFFF) # ChunkSize (unknown)
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)         # Subchunk1Size
    header += struct.pack('<H', 1)          # AudioFormat (PCM)
    header += struct.pack('<H', num_channels) # NumChannels
    header += struct.pack('<I', sample_rate)  # SampleRate
    header += struct.pack('<I', byte_rate)    # ByteRate
    header += struct.pack('<H', block_align)  # BlockAlign
    header += struct.pack('<H', bit_depth)    # BitsPerSample
    header += b'data'
    header += struct.pack('<I', 0xFFFFFFFF) # Subchunk2Size (unknown)
    return header


def _normalize_tenant_id(tenant_id: str, channel_name: str) -> str:
    tenant = (tenant_id or "").strip()
    if tenant:
        return tenant
    channel = (channel_name or "").strip()
    if channel:
        return f"channel:{channel.lower()}"
    return "default"


def _queued_total_locked() -> int:
    return sum(len(q) for q in tenant_queues.values())


def _remove_tenant_from_rr_locked(tenant_id: str) -> None:
    if tenant_id in tenant_rr_set:
        tenant_rr_set.discard(tenant_id)
        try:
            tenant_rr.remove(tenant_id)
        except ValueError:
            pass


def _prune_tenant_head_locked(tenant_id: str):
    queue = tenant_queues.get(tenant_id)
    if queue is None:
        return None
    while queue:
        stream_id = queue[0]
        job = active_streams.get(stream_id)
        if job is None or job.get("state") != "queued":
            queue.popleft()
            continue
        return job
    tenant_queues.pop(tenant_id, None)
    _remove_tenant_from_rr_locked(tenant_id)
    return None


def _pop_next_ready_stream_locked():
    tenants_count = len(tenant_rr)
    for _ in range(tenants_count):
        tenant_id = tenant_rr.popleft()
        head_job = _prune_tenant_head_locked(tenant_id)
        if head_job is None:
            continue

        if not head_job.get("stream_requested", False):
            tenant_rr.append(tenant_id)
            continue

        stream_id = tenant_queues[tenant_id].popleft()
        head_after = _prune_tenant_head_locked(tenant_id)
        if head_after is not None:
            tenant_rr.append(tenant_id)
        return stream_id
    return None


def _try_activate_next_locked():
    global active_stream_id
    if active_stream_id is not None:
        return None

    next_stream_id = _pop_next_ready_stream_locked()
    if next_stream_id is None:
        return None

    job = active_streams.get(next_stream_id)
    if job is None:
        return None

    job["state"] = "running"
    job["started_at"] = time.time()
    active_stream_id = next_stream_id
    return next_stream_id


async def _wait_until_stream_can_run(stream_id: str):
    deadline = None
    if STREAM_WAIT_TIMEOUT_SEC > 0:
        deadline = time.monotonic() + STREAM_WAIT_TIMEOUT_SEC

    async with queue_condition:
        job = active_streams.get(stream_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Stream ID not found")
        if job.get("state") == "queued":
            job["stream_requested"] = True

        while True:
            job = active_streams.get(stream_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Stream ID not found")

            state = job.get("state")
            if state == "running" and active_stream_id == stream_id:
                return job
            if state == "cancelled":
                raise HTTPException(status_code=409, detail="Stream was cancelled")
            if state == "failed":
                raise HTTPException(status_code=500, detail=job.get("error", "Stream failed"))
            if state == "finished":
                raise HTTPException(status_code=410, detail="Stream already finished")

            _try_activate_next_locked()
            if job.get("state") == "running" and active_stream_id == stream_id:
                return job

            timeout = None
            if deadline is not None:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    raise HTTPException(status_code=408, detail="Timed out waiting in queue")

            try:
                await asyncio.wait_for(queue_condition.wait(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise HTTPException(status_code=408, detail="Timed out waiting in queue") from exc


async def _mark_stream_done(stream_id: str, final_state: str, error: str | None = None):
    global active_stream_id
    async with queue_condition:
        job = active_streams.get(stream_id)
        if job is not None:
            if job.get("state") not in {"cancelled", "failed"}:
                job["state"] = final_state
            job["finished_at"] = time.time()
            if error:
                job["error"] = error

        if active_stream_id == stream_id:
            active_stream_id = None

        _try_activate_next_locked()
        queue_condition.notify_all()

async def audio_stream_generator_async(stream_id: str, request_data: dict):
    """Generator that leverages ZMQ async generator, decodes natively, and yields PCM bytes."""
    global interface
    final_state = "finished"
    final_error = None
    
    # Send WAV header first (24000 Hz, Mono, 16-bit PCM)
    yield generate_wav_header(24000, 1, 16)
    
    try:
        # We must implement VoiceDesign and VoiceClone Async logic ourselves 
        # because nano-qwen3tts-vllm only shipped with generate_custom_voice_async!
        
        model = request_data["model"]
        text = request_data["text"]
        language = request_data["language"]
        instruction = request_data["instruction"]
        speaker = request_data["speaker"]
        temperature = request_data.get("temperature", 0.9)
        ref_audio = request_data.get("ref_audio")
        ref_sr = request_data.get("ref_sr")
        ref_text = request_data.get("ref_text")

        if "VoiceDesign" in model:
            # 1. Custom ZMQ wrapper for Voice Design
            def _prep_voice_design() -> tuple:
                with interface._prep_lock:
                    input_ids, instruct_ids, speakers, languages = prepare_custom_voice_prompt(
                        text=[text], speaker=[""], language=[language], instruct=[instruction],
                        processor=interface.processor, device=interface.device,
                    )
                    return prepare_inputs(
                        config=interface.model_config,
                        input_ids=input_ids, instruct_ids=instruct_ids, languages=languages,
                        speakers=None, non_streaming_mode=True,
                        text_embedding=interface.text_embedding, input_embedding=interface.input_embedding,
                        text_projection=interface.text_projection, device=interface.device,
                    )
            loop = asyncio.get_event_loop()
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = await loop.run_in_executor(None, _prep_voice_design)
            async_gen = interface.generate_async(
                talker_input_embeds,
                trailing_text_hiddens,
                tts_pad_embed,
                talker_attention_mask,
                temperature=temperature,
            )
            
        elif "CustomVoice" in model:
            async_gen = interface.generate_custom_voice_async(
                text=text,
                language=language,
                speaker=speaker,
                temperature=temperature,
            )
            
        elif "Base" in model:
            loop = asyncio.get_running_loop()
            prompt = await loop.run_in_executor(
                None,
                lambda: interface.create_voice_clone_prompt(
                    ref_audio=(ref_audio, ref_sr),
                    ref_text=ref_text if ref_text else None,
                    x_vector_only_mode=False
                ),
            )
            
            # 2. Custom ZMQ wrapper for Voice Clone
            def _prep_voice_clone() -> tuple:
                with interface._prep_lock:
                    input_txt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
                    input_ids = _tokenize_texts([input_txt], interface.processor, interface.device)
                    
                    ref_ids = None
                    if ref_text and ref_text.strip():
                        ref_ids = [_tokenize_texts([interface._build_ref_text(ref_text)], interface.processor, interface.device)[0]]
                        
                    voice_clone_prompt_lists = {
                        "ref_code": [prompt["ref_code"]],
                        "ref_spk_embedding": [prompt["ref_spk_embedding"]],
                        "x_vector_only_mode": [prompt["x_vector_only_mode"]],
                        "icl_mode": [prompt["icl_mode"]],
                    }
                    
                    def generate_speaker_prompt_fn(p, **kwargs):
                        return generate_speaker_prompt(p, interface.device)
                    def generate_icl_prompt_fn(text_id, ref_id, ref_code, tts_pad_embed, tts_eos_embed, non_streaming_mode, **kwargs):
                        return generate_icl_prompt(
                            text_id=text_id, ref_id=ref_id, ref_code=ref_code, tts_pad_embed=tts_pad_embed,
                            tts_eos_embed=tts_eos_embed, non_streaming_mode=non_streaming_mode,
                            config=interface.model_config, text_embedding=interface.text_embedding,
                            input_embedding=interface.input_embedding, text_projection=interface.text_projection,
                            code_predictor_embeddings=interface.predictor_input_embeddings, device=interface.device,
                        )
                        
                    return prepare_inputs(
                        config=interface.model_config,
                        input_ids=input_ids,
                        ref_ids=ref_ids,
                        voice_clone_prompt=voice_clone_prompt_lists,
                        languages=[language],
                        non_streaming_mode=True,
                        text_embedding=interface.text_embedding,
                        input_embedding=interface.input_embedding,
                        text_projection=interface.text_projection,
                        device=interface.device,
                        generate_speaker_prompt_fn=generate_speaker_prompt_fn,
                        generate_icl_prompt_fn=generate_icl_prompt_fn,
                    )
                    
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = await loop.run_in_executor(None, _prep_voice_clone)
            async_gen = interface.generate_async(
                talker_input_embeds,
                trailing_text_hiddens,
                tts_pad_embed,
                talker_attention_mask,
                temperature=temperature,
            )
        else:
            return

        # ZMQ async loop bridging
        # To avoid blocking event loop with decode, we use a Producer-Consumer architecture:
        
        codes_queue = asyncio.Queue(maxsize=4) # backpressure
        loop = asyncio.get_event_loop()
        cancel_event = request_data.get("cancel_event", asyncio.Event())
        
        print(f"\n[STREAM {stream_id[:8]}] Started generating: {model}")
        
        async def producer():
            audio_codes = []
            chunk_count = 0
            start_time = time.time()
            first_chunk_time = None
            last_chunk_time = None
            
            try:
                async for chunk in async_gen:
                    if cancel_event.is_set():
                        print(f"[STREAM {stream_id[:8]}] 🛑 Generation cancelled by user.")
                        break
                        
                    current_time = time.time()
                    chunk_count += 1
                    
                    if first_chunk_time is None:
                        first_chunk_time = current_time
                        ttft = first_chunk_time - start_time
                        print(f"[STREAM {stream_id[:8]}] ⚡ First chunk received! TTFT: {ttft*1000:.2f}ms")
                    else:
                        latency = current_time - last_chunk_time
                        if latency > 1.0:
                            print(f"[STREAM {stream_id[:8]}] >> Chunk #{chunk_count} arrived (+{latency*1000:.2f}ms) ⚠️")
                    
                    last_chunk_time = current_time
                    audio_codes.append(chunk)
                    
                    # Offload to queue every 4 chunks (Exactly like official repo)
                    if len(audio_codes) % 4 == 0:
                        await codes_queue.put(list(audio_codes))
                
                # Push any last remaining chunks
                if len(audio_codes) % 4 != 0:
                    await codes_queue.put(list(audio_codes))
                    
            except Exception as e:
                print(f"[STREAM {stream_id[:8]}] Producer error: {e}")
                traceback.print_exc()
            finally:
                await codes_queue.put(None)  # Sentinel to denote generation end

        producer_task = asyncio.create_task(producer())
        total_start = time.time()
        prev_len_chunks = 0
        
        try:
            while True:
                if cancel_event.is_set():
                     break
                     
                item = await codes_queue.get()
                if item is None:
                    break
                
                num_total_chunks = len(item)
                new_chunks_count = num_total_chunks - prev_len_chunks
                
                # O(1) SLIDING WINDOW: Limit context to the last 48 chunks.
                # Why 48? The `SpeechTokenizerCUDAGraph` natively caches execution graphs for T <= 50.
                # Passing the full history natively breaks the CUDA Graph limit, forcing slow eager evaluation
                # and causing O(N^2) latency (which backs up the queue and stalls the stream).
                MAX_WINDOW = 48
                codes_to_decode = item[-MAX_WINDOW:] if num_total_chunks > MAX_WINDOW else item

                pcm16 = await loop.run_in_executor(None, _decode_batch, codes_to_decode)
                
                # Neural vocoders map sequence lengths symmetrically. We dynamically measure the exact
                # samples-per-chunk mapping (e.g., 1920 at 24kHz) to mathematically perfectly truncate the
                # left-historical context, yielding ONLY the mathematically perfect newly synthesized audio tail.
                spc = len(pcm16) // len(codes_to_decode)
                new_samples_count = new_chunks_count * spc
                
                if new_samples_count > 0:
                    new_chunk_array = pcm16[-new_samples_count:]
                    new_chunk = new_chunk_array.tobytes()
                    prev_len_chunks = num_total_chunks
                    
                    if new_chunk:
                        yield new_chunk
                    
            print(f"[STREAM {stream_id[:8]}] 🏁 Stream complete. Total Time: {(time.time() - total_start):.2f}s")
        finally:
            # Ensure producer task is cancelled if consumer loop breaks early
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass # Expected if cancelled
            except Exception as e:
                print(f"[STREAM {stream_id[:8]}] Error awaiting producer task: {e}")

    except Exception as e:
        final_state = "failed"
        final_error = str(e)
        print(f"Streaming error: {e}")
        traceback.print_exc()

    finally:
        cancel_event = request_data.get("cancel_event")
        if final_state != "failed" and cancel_event is not None and cancel_event.is_set():
            final_state = "cancelled"
        await _mark_stream_done(stream_id, final_state=final_state, error=final_error)

@app.post("/api/prepare")
async def prepare_stream(
    model: str = Form(...),
    text: str = Form(...),
    language: str = Form(...),
    temperature: float = Form(0.9),
    instruction: str = Form(""),
    speaker: str = Form(""),
    ref_audio: UploadFile = File(None),
    ref_text: str = Form(""),
    tenant_id: str = Form(""),
    channel_name: str = Form(""),
    author: str = Form(""),
    user_id: str = Form(""),
    request_id: str = Form(""),
    event_id: str = Form(""),
):
    """
    Endpoint 1: Receives form data and enqueues a stream request.
    """
    if temperature <= 0:
        raise HTTPException(status_code=400, detail="temperature must be > 0")

    required_family = "base" if "Base" in model else None
    resolved_model = _resolve_runtime_model(model, required_family=required_family)
    stream_id = str(uuid.uuid4())

    # We delay the generator creation to the GET request because async generation
    # must be instantiated inside the same event loop task the StreamingResponse consumes it from.
    request_data = {
        "model": resolved_model,
        "text": text,
        "language": language,
        "temperature": float(temperature),
        "instruction": instruction,
        "speaker": speaker,
        "cancel_event": asyncio.Event(),
        "channel_name": channel_name,
        "author": author,
        "user_id": user_id,
        "request_id": request_id,
        "event_id": event_id,
    }
    
    if "Base" in resolved_model:
        if ref_audio:
            audio_bytes = await ref_audio.read()
            ref_wav, ref_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if getattr(ref_wav, "ndim", 1) > 1:
                ref_wav = np.mean(ref_wav, axis=1).astype(np.float32)
            request_data["ref_audio"] = ref_wav
            request_data["ref_sr"] = ref_sr

            resolved_ref_text = str(ref_text or "").strip()
            if not resolved_ref_text:
                print(f"[STREAM {stream_id[:8]}] Auto-transcribing uploaded reference audio...")
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = Path(tmp.name)
                    try:
                        resolved_ref_text = await transcribe_voice_file(tmp_path)
                    finally:
                        tmp_path.unlink(missing_ok=True)
                    print(f"[STREAM {stream_id[:8]}] Auto-transcription result: '{resolved_ref_text}'")
                except Exception as error:
                    print(f"[STREAM {stream_id[:8]}] Auto-transcription failed: {error}")
                    raise HTTPException(
                        status_code=400,
                        detail="Reference text was empty and auto-transcription failed. Please provide text manually.",
                    ) from error
            request_data["ref_text"] = resolved_ref_text
        else:
            voice_record, ref_wav, ref_sr = await _resolve_base_model_reference_voice(
                app,
                user_id_raw=user_id,
                requested_voice=speaker,
            )
            request_data["ref_audio"] = ref_wav
            request_data["ref_sr"] = ref_sr
            request_data["speaker"] = str(voice_record.get("name") or speaker or "")

            resolved_ref_text = str(ref_text or voice_record.get("reference_text") or "").strip()
            if not resolved_ref_text:
                voice_path = Path(str(voice_record.get("file_path") or "")).resolve()
                try:
                    resolved_ref_text = await transcribe_voice_file(voice_path)
                    await app.state.voice_store.update_voice_settings(
                        int(voice_record["id"]),
                        {"reference_text": resolved_ref_text},
                    )
                except Exception as error:
                    raise HTTPException(
                        status_code=400,
                        detail="Stored Qwen clone voice has no reference_text and retranscription failed.",
                    ) from error
            request_data["ref_text"] = resolved_ref_text

    tenant_key = _normalize_tenant_id(tenant_id, channel_name)

    async with queue_condition:
        if _queued_total_locked() >= MAX_TOTAL_QUEUED:
            raise HTTPException(status_code=429, detail=f"Queue is full (MAX_TOTAL_QUEUED={MAX_TOTAL_QUEUED})")

        tenant_queue = tenant_queues.get(tenant_key)
        if tenant_queue is None:
            tenant_queue = deque()
            tenant_queues[tenant_key] = tenant_queue

        if len(tenant_queue) >= MAX_QUEUE_PER_TENANT:
            raise HTTPException(status_code=429, detail=f"Tenant queue is full (MAX_QUEUE_PER_TENANT={MAX_QUEUE_PER_TENANT})")

        active_streams[stream_id] = {
            "tenant_id": tenant_key,
            "request_data": request_data,
            "state": "queued",
            "stream_requested": False,
            "stream_opened": False,
            "error": None,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
        }
        tenant_queue.append(stream_id)

        if tenant_key not in tenant_rr_set:
            tenant_rr.append(tenant_key)
            tenant_rr_set.add(tenant_key)

        queue_condition.notify_all()

        return {
            "stream_id": stream_id,
            "tenant_id": tenant_key,
            "state": "queued",
            "tenant_queue_depth": len(tenant_queue),
            "global_queue_depth": _queued_total_locked(),
            "message": "Queued. Connect GET /api/stream/{stream_id} and wait for your fair turn.",
        }

@app.get("/api/stream/{stream_id}")
async def stream_tts(stream_id: str):
    """
    Endpoint 2: Client connects via GET stream endpoint and waits fair queue turn.
    """
    job = await _wait_until_stream_can_run(stream_id)
    request_data = job["request_data"]

    try:
        async with model_switch_lock:
            await switch_model_if_needed(request_data["model"])
    except HTTPException as e:
        await _mark_stream_done(stream_id, final_state="failed", error=str(e.detail))
        raise e
    except Exception as e:
        await _mark_stream_done(stream_id, final_state="failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load model for stream: {e}")

    async with queue_condition:
        if job.get("stream_opened"):
            raise HTTPException(status_code=409, detail="Stream already consumed")
        job["stream_opened"] = True

    return StreamingResponse(audio_stream_generator_async(stream_id, request_data), media_type="audio/wav")


@app.get("/api/status/{stream_id}")
async def stream_status(stream_id: str):
    """
    Endpoint 2.5: Poll queue/execution status.
    """
    async with queue_condition:
        job = active_streams.get(stream_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Stream ID not found")
        tenant_id = job["tenant_id"]
        tenant_queue_depth = len(tenant_queues.get(tenant_id, ()))
        return {
            "stream_id": stream_id,
            "tenant_id": tenant_id,
            "state": job.get("state"),
            "active_stream_id": active_stream_id,
            "tenant_queue_depth": tenant_queue_depth,
            "global_queue_depth": _queued_total_locked(),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "error": job.get("error"),
        }

@app.post("/api/cancel/{stream_id}")
async def cancel_stream(stream_id: str):
    """
    Endpoint 3: Cancel queued or ongoing stream.
    """
    async with queue_condition:
        job = active_streams.get(stream_id)
        if job is None:
            return {"message": "Stream not found or already cancelled"}

        request_data = job.get("request_data", {})
        cancel_event = request_data.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()

        state = job.get("state")
        if state == "queued":
            tenant_id = job["tenant_id"]
            queue = tenant_queues.get(tenant_id)
            if queue is not None:
                try:
                    queue.remove(stream_id)
                except ValueError:
                    pass
                if not queue:
                    tenant_queues.pop(tenant_id, None)
                    _remove_tenant_from_rr_locked(tenant_id)
            job["state"] = "cancelled"
            job["finished_at"] = time.time()
            _try_activate_next_locked()
            queue_condition.notify_all()
            return {"message": "Queued stream cancelled", "state": "cancelled"}

        if state == "running":
            queue_condition.notify_all()
            return {"message": "Stream cancellation requested", "state": "running"}

        return {"message": "Stream already completed", "state": state}


HTML_UI = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Qwen3-TTS Streaming API Server ⚡</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #f9fafb; color: #111827; }
        .card { background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
        h1 { margin-top: 0; color: #2563eb; }
        label { display: block; font-weight: 600; margin-top: 16px; margin-bottom: 8px; }
        select, input, textarea { width: 100%; padding: 10px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 16px; box-sizing: border-box; }
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 24px;
        }
        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            flex: 1;
            transition: 0.2s;
        }
        button:hover { background: #1d4ed8; }
        button:disabled { background: #9ca3af; cursor: not-allowed; }
        .btn-stop { background-color: #f44336; }
        .btn-stop:hover { background-color: #da190b; }
        #audioPlayer { width: 100%; margin-top: 24px; display: none; }
        .dynamic-field { display: none; }
        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: bold;
            display: none; /* Hidden by default */
        }
        .status.loading { background: #eff6ff; border-left: 4px solid #3b82f6; color: #3b82f6; }
        .status.error { background: #fee2e2; border-left: 4px solid #ef4444; color: #ef4444; }
        .status.success { background: #dcfce7; border-left: 4px solid #22c55e; color: #22c55e; }
        .info { padding: 12px; background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 4px; margin-bottom: 20px; font-size: 14px; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🎙️ Настоящий Streaming API</h1>
        <div class="info">
            Звук начинает воспроизводиться прямо в браузере сразу после получения ПЕРВОГО миллисекундного чанка (TTFT), не дожидаясь генерации всего текста! <br><br>
            <b>Для программистов:</b> Это работает через обычный REST API: сначала POST-запрос на <code>/api/prepare</code>, затем GET-стриминг <code>/api/stream/{id}</code>.
        </div>
        
        <form id="tts-form">
            <label>Выбор модели</label>
            <select id="modelSelect" name="model" onchange="updateUI()">
                <option value="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign">⭐️ Qwen3-TTS-12Hz-1.7B-VoiceDesign (Дизайн по тексту)</option>
                <option value="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice">⚡ Qwen3-TTS-12Hz-0.6B-CustomVoice (Готовые дикторы, Быстро)</option>
                <option value="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice">🎙️ Qwen3-TTS-12Hz-1.7B-CustomVoice (Готовые дикторы, Качественно)</option>
                <option value="Qwen/Qwen3-TTS-12Hz-0.6B-Base">⚡ Qwen3-TTS-12Hz-0.6B-Base (Быстрое Клонирование)</option>
                <option value="Qwen/Qwen3-TTS-12Hz-1.7B-Base">🎭 Qwen3-TTS-12Hz-1.7B-Base (Качественное Клонирование)</option>
            </select>

            <label>Язык текста</label>
            <select id="languageSelect" name="language">
                <option value="Russian">Russian</option>
                <option value="English">English</option>
            </select>
            
            <label>Temperature</label>
            <input id="temperature" name="temperature" type="number" min="0.1" max="2.0" step="0.05" value="0.9">

            <label>Текст для озвучки</label>
            <textarea id="textInput" name="text" rows="4">Привет! Это потоковое воспроизведение через API. Вы начнете слышать звук еще до того, как весь этот длинный текст будет полностью сгенерирован нашей нейросетью. Это работает очень быстро и круто!</textarea>

            <div id="field-instruction" class="dynamic-field">
                <label>Описание голоса (Instruction)</label>
                <input type="text" id="instruction" name="instruction" value="Уверенный мужской голос, профессиональный диктор, очень радостный">
            </div>

            <div id="field-speaker" class="dynamic-field">
                <label>Выбор встроенного диктора (CustomVoice)</label>
                <select id="speaker" name="speaker">
                    <option value="serena">serena</option>
                    <option value="vivian">vivian</option>
                    <option value="uncle_fu">uncle_fu</option>
                    <option value="ryan">ryan</option>
                    <option value="aiden">aiden</option>
                    <option value="ono_anna">ono_anna</option>
                    <option value="sohee">sohee</option>
                    <option value="eric">eric</option>
                    <option value="dylan">dylan</option>
                </select>
            </div>

            <div id="field-clone" class="dynamic-field">
                <label>Загрузите аудио-оригинал (.wav, .mp3)</label>
                <input type="file" id="ref_audio" name="ref_audio" accept="audio/*">
                
                <label>Оригинальный текст с аудио (буква в букву)</label>
                <textarea id="ref_text" name="ref_text" rows="2" placeholder="Оставьте пустым для АВТОМАТИЧЕСКОГО РАСПОЗНАВАНИЯ (Whisper)"></textarea>
            </div>
            
            <div class="btn-group">
                <button id="generateBtn" type="button" onclick="generateTTS()">Генерировать поток 🔊</button>
                <button id="stopBtn" type="button" onclick="stopTTS()" class="btn-stop" disabled>Остановить</button>
            </div>
        </form>

        <div id="status" class="status"></div>
        <audio id="audioPlayer" controls autoplay></audio>
    </div>

    <script>
        function updateUI() {
            const model = document.getElementById('modelSelect').value;
            document.getElementById('field-instruction').style.display = model.includes('VoiceDesign') ? 'block' : 'none';
            document.getElementById('field-speaker').style.display = model.includes('CustomVoice') ? 'block' : 'none';
            document.getElementById('field-clone').style.display = model.includes('Base') ? 'block' : 'none';
        }
        
        // Init UI state
        updateUI();

        let currentStreamId = null;
        let audioContext = null; // Not used in this version, but kept from snippet

        async function stopTTS() {
            if (currentStreamId) {
                try {
                    await fetch(`/api/cancel/${currentStreamId}`, { method: 'POST' });
                    const statusDiv = document.getElementById('status');
                    statusDiv.textContent = "Воспроизведение остановлено вручную.";
                    statusDiv.className = "status error";
                    statusDiv.style.display = 'block';
                    
                    // Stop HTML5 audio player
                    const player = document.getElementById('audioPlayer');
                    player.pause();
                    player.currentTime = 0;
                    player.src = "";
                    
                } catch (e) {
                    console.error("Cancel failed:", e);
                    const statusDiv = document.getElementById('status');
                    statusDiv.textContent = "Ошибка при отмене: " + e.message;
                    statusDiv.className = "status error";
                    statusDiv.style.display = 'block';
                }
            }
            resetButtons();
        }
        
        function resetButtons() {
            document.getElementById('generateBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            currentStreamId = null;
        }

        async function generateTTS() {
            const generateBtn = document.getElementById('generateBtn');
            const stopBtn = document.getElementById('stopBtn');
            const statusDiv = document.getElementById('status');
            const player = document.getElementById('audioPlayer');
            
            generateBtn.disabled = true;
            stopBtn.disabled = false;
            statusDiv.style.display = 'block';
            statusDiv.className = 'status loading';
            statusDiv.textContent = 'Загрузка модели и подготовка потока...';
            player.src = '';
            player.style.display = 'none'; // Hide player until stream starts

            try {
                const formData = new FormData();
                formData.append('model', document.getElementById('modelSelect').value);
                formData.append('text', document.getElementById('textInput').value);
                formData.append('language', document.getElementById('languageSelect').value);
                formData.append('temperature', document.getElementById('temperature').value);
                formData.append('instruction', document.getElementById('instruction').value);
                formData.append('speaker', document.getElementById('speaker').value);

                const refAudioInput = document.getElementById('ref_audio');
                if (refAudioInput && refAudioInput.files.length > 0) {
                    formData.append('ref_audio', refAudioInput.files[0]);
                    formData.append('ref_text', document.getElementById('ref_text').value);
                }

                // 1. Send inputs to the server to initialize the generator and allocate memory
                const t0 = performance.now();
                const response = await fetch('/api/prepare', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || "Неизвестная ошибка API");
                }
                
                const t1 = performance.now();
                
                // 2. We received a stream_id! Now connect the HTML5 Audio player directly to the stream.
                // The browser will handle the HTTP chunked stream and start playing immediately!
                currentStreamId = data.stream_id;
                statusDiv.textContent = `Подготовка заняла ${Math.round(t1 - t0)}мс. Подключаемся к потоку звука... Слушайте!`;
                statusDiv.className = 'status success';
                
                player.src = `/api/stream/${currentStreamId}`;
                player.style.display = 'block';
                player.play().catch(e => console.error("Автовоспроизведение заблокировано:", e));
                
                // Setup event listeners for audio end/error
                player.onended = () => {
                    statusDiv.textContent = "Воспроизведение завершено.";
                    statusDiv.className = "status success";
                    resetButtons();
                };
                player.onerror = (e) => {
                    console.error("Ошибка воспроизведения аудио:", e);
                    statusDiv.textContent = "Ошибка воспроизведения аудио.";
                    statusDiv.className = "status error";
                    resetButtons();
                };
                
            } catch (err) {
                statusDiv.textContent = "Ошибка: " + err.message;
                statusDiv.className = 'status error';
                resetButtons();
            }
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def serve_ui():
    """Serves the main HTML Web UI interface."""
    return HTMLResponse(content=HTML_UI, status_code=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS streaming API server")
    parser.add_argument("--host", default=SERVER_HOST)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    args = parser.parse_args()

    print("\nStarting Qwen3-TTS Streaming API Server...")
    print(f"Go to: http://127.0.0.1:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)

