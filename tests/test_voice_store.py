from __future__ import annotations

import pytest

from nano_qwen3tts_vllm.voice_store import FileVoiceStore


@pytest.mark.asyncio
async def test_voice_store_toggle_and_stats(tmp_path):
    store = FileVoiceStore(tmp_path / "state.json", tmp_path / "voices")

    global_voice = await store.create_voice(
        name="global_voice",
        owner_id=None,
        voice_type="global",
        file_path=str(tmp_path / "voices" / "global.wav"),
        is_public=True,
    )
    user_voice = await store.create_voice(
        name="user_voice",
        owner_id=42,
        voice_type="user",
        file_path=str(tmp_path / "voices" / "user.wav"),
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


@pytest.mark.asyncio
async def test_list_global_voices_returns_only_active_global_entries(tmp_path):
    store = FileVoiceStore(tmp_path / "state.json", tmp_path / "voices")

    global_voice = await store.create_voice(
        name="global_voice",
        owner_id=None,
        voice_type="global",
        file_path=str(tmp_path / "voices" / "global.wav"),
        is_public=True,
    )
    await store.create_voice(
        name="user_voice",
        owner_id=42,
        voice_type="user",
        file_path=str(tmp_path / "voices" / "user.wav"),
        is_public=False,
    )

    await store.toggle_voice(int(global_voice["id"]))
    global_voices = await store.list_global_voices()

    assert global_voices == []
