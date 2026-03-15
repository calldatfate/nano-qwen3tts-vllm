from __future__ import annotations

from runtime_models import build_runtime_model_catalog, pick_runtime_model


def test_runtime_model_catalog_uses_single_configured_model():
    catalog = build_runtime_model_catalog(
        configured_model="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        allowed_models_raw=None,
    )

    assert [item["id"] for item in catalog] == ["Qwen/Qwen3-TTS-12Hz-0.6B-Base"]


def test_runtime_model_catalog_expands_family_aliases():
    catalog = build_runtime_model_catalog(
        configured_model=None,
        allowed_models_raw="base,customvoice",
    )

    assert [item["id"] for item in catalog] == [
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ]


def test_pick_runtime_model_resolves_same_family_when_exact_model_is_not_exposed():
    catalog = build_runtime_model_catalog(
        configured_model="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        allowed_models_raw=None,
    )

    resolved = pick_runtime_model(
        requested_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        runtime_catalog=catalog,
        required_family="base",
    )

    assert resolved == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
