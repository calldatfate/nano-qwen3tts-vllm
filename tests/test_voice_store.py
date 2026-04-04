from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from nano_qwen3tts_vllm.voice_store import FileVoiceStore


def make_repo_temp_dir(case_name: str) -> Path:
    root = Path(__file__).resolve().parent / "_tmp"
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = root / f"{case_name}-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


@pytest.mark.asyncio
async def test_voice_store_toggle_and_stats():
    temp_path = make_repo_temp_dir("voice-store-toggle-and-stats")
    try:
        store = FileVoiceStore(temp_path / "state.json", temp_path / "voices")

        global_voice = await store.create_voice(
            name="global_voice",
            owner_id=None,
            voice_type="global",
            file_path=str(temp_path / "voices" / "global.wav"),
            is_public=True,
        )
        user_voice = await store.create_voice(
            name="user_voice",
            owner_id=42,
            voice_type="user",
            file_path=str(temp_path / "voices" / "user.wav"),
            is_public=False,
        )

        toggled = await store.toggle_voice(int(global_voice["id"]))
        stats = await store.stats()

        assert toggled is not None
        assert toggled["is_active"] is False
        assert stats["total_voices"] == 2
        assert stats["global_voices"] == 1
        assert stats["user_voices"] == 1
        assert stats["active_voices"] == 1

        all_voices = await store.list_all_voices()
        assert {int(voice["id"]) for voice in all_voices} == {int(global_voice["id"]), int(user_voice["id"])}
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_list_global_voices_returns_only_active_global_entries():
    temp_path = make_repo_temp_dir("voice-store-global-only")
    try:
        store = FileVoiceStore(temp_path / "state.json", temp_path / "voices")

        global_voice = await store.create_voice(
            name="global_voice",
            owner_id=None,
            voice_type="global",
            file_path=str(temp_path / "voices" / "global.wav"),
            is_public=True,
        )
        await store.create_voice(
            name="user_voice",
            owner_id=42,
            voice_type="user",
            file_path=str(temp_path / "voices" / "user.wav"),
            is_public=False,
        )

        await store.toggle_voice(int(global_voice["id"]))
        global_voices = await store.list_global_voices()

        assert global_voices == []
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)
