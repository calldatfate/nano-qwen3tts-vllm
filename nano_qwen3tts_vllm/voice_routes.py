from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .voice_service import (
    admin_voice_groups,
    cleanup_voice_file,
    guess_audio_media_type,
    resolve_base_model_reference_voice,
    retranscribe_voice_record,
    serialize_voice_record,
)
from .voice_uploads import prepare_uploaded_voice_file, sanitize_voice_name, transcribe_voice_file


class VoiceApiService:
    def __init__(
        self,
        *,
        runtime: Any,
        logger: logging.Logger,
        model_switch_lock: asyncio.Lock,
        switch_model_if_needed: Callable[[str], Awaitable[None]],
        resolve_runtime_model: Callable[..., str],
        runtime_model_catalog: Callable[[], list[dict[str, object]]],
        runtime_configured_model: Callable[[], str | None],
        runtime_allowed_models_raw: Callable[[], str | None],
        runtime_allowed_models_source: Callable[[], str | None],
        audio_stream_generator: Callable[[str, dict], AsyncIterator[bytes]],
    ) -> None:
        self.runtime = runtime
        self.logger = logger
        self.model_switch_lock = model_switch_lock
        self.switch_model_if_needed = switch_model_if_needed
        self.resolve_runtime_model = resolve_runtime_model
        self.runtime_model_catalog = runtime_model_catalog
        self.runtime_configured_model = runtime_configured_model
        self.runtime_allowed_models_raw = runtime_allowed_models_raw
        self.runtime_allowed_models_source = runtime_allowed_models_source
        self.audio_stream_generator = audio_stream_generator

    def build_router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/health/live")
        async def health_live():
            return {
                "status": "healthy",
                "service": "nano-qwen3tts-vllm",
                "current_model": self.runtime.current_model_name,
            }

        @router.get("/health/ready")
        async def health_ready():
            return {
                "status": "healthy",
                "service": "nano-qwen3tts-vllm",
                "model_loaded": self.runtime.is_loaded,
                "current_model": self.runtime.current_model_name,
            }

        @router.get("/api/models")
        async def get_supported_models():
            models = self.runtime_model_catalog()
            return {
                "success": True,
                "provider": "qwen",
                "current_model": self.runtime.current_model_name,
                "configured_model": self.runtime_configured_model(),
                "model_policy": (
                    "restricted"
                    if self.runtime_allowed_models_raw() or self.runtime_configured_model()
                    else "dynamic"
                ),
                "allowed_models_source": self.runtime_allowed_models_source(),
                "models": models,
            }

        @router.get("/api/audio/{filename}")
        async def get_generated_audio(request: Request, filename: str):
            preview_root = request.app.state.voice_preview_dir.resolve()
            target_path = (preview_root / filename).resolve()
            try:
                target_path.relative_to(preview_root)
            except ValueError as error:
                raise HTTPException(status_code=400, detail="Invalid filename") from error
            if not target_path.exists():
                raise HTTPException(status_code=404, detail="Audio not found")
            return FileResponse(
                target_path,
                media_type=guess_audio_media_type(target_path),
                filename=target_path.name,
                headers={"Cache-Control": "no-store"},
            )

        @router.get("/api/tts/voices/global")
        async def voices_global(request: Request):
            voices = await request.app.state.voice_store.list_global_voices()
            return [serialize_voice_record(voice) for voice in voices]

        @router.get("/api/tts/voices")
        async def voices_all(request: Request, user_id: int | None = None):
            voices = await request.app.state.voice_store.list_available_voices(user_id=user_id)
            return [serialize_voice_record(voice) for voice in voices]

        @router.get("/api/tts/voices/{voice_id}")
        async def get_voice_info(request: Request, voice_id: int):
            voice = await request.app.state.voice_store.get_voice_by_id(voice_id)
            if not voice:
                raise HTTPException(status_code=404, detail="Voice not found")
            return serialize_voice_record(voice)

        @router.get("/api/tts/user/voices/{user_id}")
        async def user_voices(request: Request, user_id: int):
            voices = await request.app.state.voice_store.list_user_voices(user_id)
            return [serialize_voice_record(voice) for voice in voices]

        @router.post("/api/tts/user/voices/upload")
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
                raise HTTPException(status_code=400, detail=str(error)) from error

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
                raise HTTPException(status_code=400, detail=str(error)) from error

            return {"success": True, "status": "success", "voice": voice}

        @router.delete("/api/tts/user/voices/{voice_id}")
        async def delete_user_voice(request: Request, voice_id: int, user_id: int):
            voice = await request.app.state.voice_store.get_voice_by_id(voice_id)
            if not voice or int(voice.get("owner_id") or 0) != int(user_id):
                raise HTTPException(status_code=404, detail="Voice not found")

            cleanup_voice_file(request.app, voice)
            await request.app.state.voice_store.delete_voice(voice_id)
            return {"success": True}

        @router.put("/api/tts/user/voices/{voice_id}/settings")
        async def update_user_voice_settings(request: Request, voice_id: int, settings: dict):
            updated = await request.app.state.voice_store.update_voice_settings(voice_id, settings)
            if not updated:
                raise HTTPException(status_code=404, detail="Voice not found")
            return {"success": True, "voice": updated}

        @router.get("/api/tts/user/voices/enabled/{user_id}")
        async def get_enabled_voices(request: Request, user_id: int):
            voice_ids = await request.app.state.voice_store.get_enabled_voice_ids(user_id)
            return {"success": True, "voice_ids": voice_ids, "enabled_voice_ids": voice_ids}

        @router.post("/api/tts/user/voices/enabled/{user_id}")
        async def set_enabled_voices(request: Request, user_id: int, voice_ids: list[int]):
            stored = await request.app.state.voice_store.set_enabled_voice_ids(user_id, voice_ids)
            return {"success": True, "voice_ids": stored, "enabled_voice_ids": stored}

        @router.put("/api/tts/user/voices/{voice_id}/rename")
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
                raise HTTPException(status_code=400, detail=str(error)) from error
            return {"success": True, "voice": updated}

        @router.post("/api/tts/user/voices/{voice_id}/retranscribe")
        async def retranscribe_user_voice(request: Request, voice_id: int, user_id: int):
            voice = await request.app.state.voice_store.get_voice_by_id(voice_id)
            if not voice or int(voice.get("owner_id") or 0) != int(user_id):
                raise HTTPException(status_code=404, detail="Voice not found")
            return await retranscribe_voice_record(request, voice_id)

        @router.get("/api/admin/voices")
        async def admin_list_global_voices(request: Request):
            return await admin_voice_groups(request.app)

        @router.post("/api/admin/voices/upload")
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

            return {"success": True, "status": "success", "voice": serialize_voice_record(voice)}

        @router.delete("/api/admin/voices/{voice_id}")
        async def delete_admin_voice(request: Request, voice_id: int):
            voice = await request.app.state.voice_store.get_voice_by_id(voice_id)
            if not voice:
                raise HTTPException(status_code=404, detail="Voice not found")

            deleted = await request.app.state.voice_store.delete_voice(voice_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Voice not found")

            cleanup_voice_file(request.app, voice)
            return {"success": True}

        @router.put("/api/admin/voices/{voice_id}/rename")
        async def rename_admin_voice(request: Request, voice_id: int, new_name: str):
            try:
                clean_name = sanitize_voice_name(new_name)
                updated = await request.app.state.voice_store.rename_voice(voice_id, clean_name)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error

            if not updated:
                raise HTTPException(status_code=404, detail="Voice not found")
            return {"success": True, "voice": serialize_voice_record(updated)}

        @router.put("/api/admin/voices/{voice_id}/settings")
        async def update_admin_voice_settings(request: Request, voice_id: int, settings: dict):
            updated = await request.app.state.voice_store.update_voice_settings(voice_id, settings)
            if not updated:
                raise HTTPException(status_code=404, detail="Voice not found")
            return {"success": True, "voice": serialize_voice_record(updated)}

        @router.post("/api/admin/voices/{voice_id}/toggle")
        async def toggle_admin_voice(request: Request, voice_id: int):
            updated = await request.app.state.voice_store.toggle_voice(voice_id)
            if not updated:
                raise HTTPException(status_code=404, detail="Voice not found")
            return {"success": True, "voice": serialize_voice_record(updated)}

        @router.post("/api/admin/voices/{voice_id}/retranscribe")
        async def retranscribe_admin_voice(request: Request, voice_id: int):
            return await retranscribe_voice_record(request, voice_id)

        @router.post("/api/admin/voices/{voice_id}/transcribe")
        async def transcribe_admin_voice(request: Request, voice_id: int):
            return await retranscribe_voice_record(request, voice_id)

        @router.get("/api/admin/stats")
        async def admin_stats(request: Request, days: int = 7):
            _ = days
            stats_payload = await request.app.state.voice_store.stats()
            return {"success": True, **stats_payload}

        @router.post("/api/admin/voices/test")
        async def admin_test_voice(
            request: Request,
            voice_name: str = Form(...),
            user_id: int = Form(...),
            test_text: str = Form(...),
            model: str = Form("Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        ):
            if not test_text.strip():
                raise HTTPException(status_code=400, detail="test_text is required")

            resolved_model = self.resolve_runtime_model(model, required_family="base")

            voice_record, ref_wav, ref_sr = await resolve_base_model_reference_voice(
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
            async with self.model_switch_lock:
                await self.switch_model_if_needed(resolved_model)
            try:
                async for chunk in self.audio_stream_generator(preview_stream_id, request_data):
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

        return router
