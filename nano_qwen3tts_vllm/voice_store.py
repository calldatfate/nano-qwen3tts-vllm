from __future__ import annotations

import json
import os
import random
import tempfile
from asyncio import Lock
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_VOICE_NAME = "default"
DEFAULT_VOICE_ALIASES = {DEFAULT_VOICE_NAME, "default_voice"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FileVoiceStore:
    def __init__(self, state_path: Path, voices_dir: Path) -> None:
        self.state_path = state_path
        self.voices_dir = voices_dir
        self._lock = Lock()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write_state(
                {
                    "next_id": 1,
                    "updated_at": _utc_now_iso(),
                    "voices": [],
                    "enabled": {},
                }
            )

    async def startup(self) -> None:
        return

    async def close(self) -> None:
        return

    async def list_global_voices(self) -> list[dict[str, Any]]:
        state = self._read_state()
        return [
            voice
            for voice in state["voices"]
            if voice.get("voice_type") == "global" and self._is_visible_voice(voice)
        ]

    async def list_user_voices(self, user_id: int) -> list[dict[str, Any]]:
        state = self._read_state()
        return [
            voice
            for voice in state["voices"]
            if int(voice.get("owner_id") or 0) == int(user_id) and self._is_visible_voice(voice)
        ]

    async def list_available_voices(self, user_id: int | None) -> list[dict[str, Any]]:
        return self._active_voices_for_user(user_id)

    async def list_all_voices(self) -> list[dict[str, Any]]:
        return self._read_state()["voices"]

    async def get_voice_by_id(self, voice_id: int) -> dict[str, Any] | None:
        for voice in self._read_state()["voices"]:
            if int(voice.get("id") or 0) == int(voice_id):
                return voice
        return None

    async def get_voice_by_name(self, name: str, user_id: int | None = None) -> dict[str, Any] | None:
        lowered = str(name or "").strip().lower()
        for voice in self._active_voices_for_user(user_id):
            if str(voice.get("name") or "").strip().lower() == lowered:
                return voice
        return None

    async def create_voice(
        self,
        *,
        name: str,
        owner_id: int | None,
        voice_type: str,
        file_path: str,
        is_public: bool,
        reference_text: str | None = None,
        cfg_strength: float | None = None,
        speed_preset: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            state = self._read_state()
            for existing in state["voices"]:
                if self._is_name_conflict(
                    existing=existing,
                    candidate_name=name,
                    candidate_type=voice_type,
                    candidate_owner_id=owner_id,
                ):
                    raise ValueError(f"Voice '{name}' already exists")

            voice_id = int(state["next_id"])
            voice = {
                "id": voice_id,
                "name": name,
                "file_path": file_path,
                "voice_type": voice_type,
                "owner_id": owner_id,
                "is_public": bool(is_public),
                "is_active": True,
                "reference_text": reference_text,
                "created_at": _utc_now_iso(),
                "cfg_strength": cfg_strength,
                "speed_preset": speed_preset,
                "enabled_user_ids": [],
            }
            state["voices"].append(voice)
            state["next_id"] = voice_id + 1
            state["updated_at"] = _utc_now_iso()
            self._write_state(state)
            return voice

    async def update_voice_settings(self, voice_id: int, patch: dict[str, Any]) -> dict[str, Any] | None:
        async with self._lock:
            state = self._read_state()
            for voice in state["voices"]:
                if int(voice.get("id") or 0) != int(voice_id):
                    continue
                for key in ("reference_text", "cfg_strength", "speed_preset"):
                    if key in patch:
                        voice[key] = patch[key]
                state["updated_at"] = _utc_now_iso()
                self._write_state(state)
                return voice
        return None

    async def rename_voice(self, voice_id: int, new_name: str) -> dict[str, Any] | None:
        async with self._lock:
            state = self._read_state()
            target: dict[str, Any] | None = None
            for voice in state["voices"]:
                if int(voice.get("id") or 0) == int(voice_id):
                    target = voice
                    break
            if target is None:
                return None
            for existing in state["voices"]:
                if self._is_name_conflict(
                    existing=existing,
                    candidate_name=new_name,
                    candidate_type=str(target.get("voice_type") or "user"),
                    candidate_owner_id=target.get("owner_id"),
                    exclude_id=voice_id,
                ):
                    raise ValueError(f"Voice '{new_name}' already exists")
            target["name"] = new_name
            state["updated_at"] = _utc_now_iso()
            self._write_state(state)
            return target

    async def toggle_voice(self, voice_id: int) -> dict[str, Any] | None:
        async with self._lock:
            state = self._read_state()
            for voice in state["voices"]:
                if int(voice.get("id") or 0) != int(voice_id):
                    continue
                voice["is_active"] = not bool(voice.get("is_active", True))
                state["updated_at"] = _utc_now_iso()
                self._write_state(state)
                return voice
        return None

    async def delete_voice(self, voice_id: int) -> bool:
        async with self._lock:
            state = self._read_state()
            before = len(state["voices"])
            state["voices"] = [voice for voice in state["voices"] if int(voice.get("id") or 0) != int(voice_id)]
            if len(state["voices"]) == before:
                return False
            state["updated_at"] = _utc_now_iso()
            self._write_state(state)
            return True

    async def get_enabled_voice_ids(self, user_id: int) -> list[int]:
        values = self._read_state().get("enabled", {}).get(str(int(user_id)), [])
        return [int(item) for item in values]

    async def set_enabled_voice_ids(self, user_id: int, voice_ids: list[int]) -> list[int]:
        async with self._lock:
            state = self._read_state()
            valid_ids = {int(voice.get("id") or 0) for voice in state["voices"]}
            filtered = sorted({int(item) for item in voice_ids if int(item) in valid_ids})
            state.setdefault("enabled", {})[str(int(user_id))] = filtered
            state["updated_at"] = _utc_now_iso()
            self._write_state(state)
            return filtered

    async def toggle_enabled_voice_id(self, user_id: int, voice_id: int, is_enabled: bool) -> list[int]:
        current = set(await self.get_enabled_voice_ids(user_id))
        valid_ids = {int(voice.get("id") or 0) for voice in self._read_state()["voices"]}
        if int(voice_id) not in valid_ids:
            return sorted(current)
        if is_enabled:
            current.add(int(voice_id))
        else:
            current.discard(int(voice_id))
        return await self.set_enabled_voice_ids(user_id, sorted(current))

    async def resolve_voice_record_for_user(
        self,
        user_id: int | None,
        requested_voice: str | None,
    ) -> dict[str, Any] | None:
        active = self._active_voices_for_user(user_id)
        if not active:
            return None

        enabled_pool = self._filter_by_enabled(user_id, active)
        effective_pool = enabled_pool if enabled_pool else active
        usable_pool = [voice for voice in effective_pool if self._is_usable_voice(voice)]
        if not usable_pool:
            return None

        requested = str(requested_voice or "").strip()
        normalized_requested = requested.lower()
        if normalized_requested == "random":
            return random.choice(usable_pool)
        if requested and normalized_requested not in DEFAULT_VOICE_ALIASES:
            matched = self._find_by_name(usable_pool, requested)
            if matched:
                return matched

        default_voice = self._find_by_name(usable_pool, DEFAULT_VOICE_NAME)
        if default_voice:
            return default_voice
        return usable_pool[0]

    async def stats(self) -> dict[str, Any]:
        state = self._read_state()
        voices = state["voices"]
        return {
            "total_voices": len(voices),
            "global_voices": len([voice for voice in voices if voice.get("voice_type") == "global"]),
            "user_voices": len([voice for voice in voices if voice.get("voice_type") != "global"]),
            "active_voices": len([voice for voice in voices if bool(voice.get("is_active", True))]),
            "updated_at": _utc_now_iso(),
        }

    def _read_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f"{self.state_path.name}.",
            suffix=".tmp",
            dir=str(self.state_path.parent),
            text=True,
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                handle.write(body)
            try:
                os.replace(tmp_name, self.state_path)
            except PermissionError:
                # Some Windows environments deny atomic replace inside sandboxed or watched
                # directories even though plain file writes are allowed. Fall back to a direct
                # write so the local file-backed store stays usable.
                self.state_path.write_text(body, encoding="utf-8")
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except PermissionError:
                    pass

    def _active_voices_for_user(self, user_id: int | None) -> list[dict[str, Any]]:
        voices = self._read_state()["voices"]
        active = [voice for voice in voices if self._is_visible_voice(voice)]
        if user_id is None:
            return [voice for voice in active if voice.get("voice_type") == "global"]
        return [
            voice
            for voice in active
            if voice.get("voice_type") == "global" or int(voice.get("owner_id") or 0) == int(user_id)
        ]

    def _filter_by_enabled(self, user_id: int | None, voices: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if user_id is None:
            return voices
        enabled_ids = set(int(item) for item in self._read_state().get("enabled", {}).get(str(int(user_id)), []))
        if not enabled_ids:
            return voices
        return [voice for voice in voices if int(voice.get("id") or 0) in enabled_ids]

    @staticmethod
    def _find_by_name(voices: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        lowered = str(name or "").strip().lower()
        for voice in voices:
            if str(voice.get("name") or "").strip().lower() == lowered:
                return voice
        return None

    @classmethod
    def _has_reference_file(cls, voice: dict[str, Any]) -> bool:
        file_path = str(voice.get("file_path") or "").strip()
        if not file_path:
            return False
        try:
            return Path(file_path).expanduser().resolve().exists()
        except Exception:
            return False

    @classmethod
    def _is_visible_voice(cls, voice: dict[str, Any]) -> bool:
        return bool(voice.get("is_active", True))

    @classmethod
    def _is_usable_voice(cls, voice: dict[str, Any]) -> bool:
        return cls._is_visible_voice(voice) and cls._has_reference_file(voice)

    @staticmethod
    def _is_name_conflict(
        *,
        existing: dict[str, Any],
        candidate_name: str,
        candidate_type: str,
        candidate_owner_id: int | None,
        exclude_id: int | None = None,
    ) -> bool:
        if exclude_id is not None and int(existing.get("id") or 0) == int(exclude_id):
            return False
        if str(existing.get("name") or "").strip().lower() != str(candidate_name or "").strip().lower():
            return False

        existing_type = str(existing.get("voice_type") or "user").strip().lower()
        existing_owner = existing.get("owner_id")
        normalized_type = str(candidate_type or "user").strip().lower()
        owner = int(candidate_owner_id) if candidate_owner_id is not None else None

        if normalized_type == "global":
            return True
        if existing_type == "global":
            return True
        return int(existing_owner or 0) == int(owner or 0)
