from __future__ import annotations

import os

from fastapi import HTTPException

from .runtime_models import build_runtime_model_catalog, pick_runtime_model


def runtime_configured_model_from_env() -> str | None:
    candidate = str(os.environ.get("QWEN3_TTS_MODEL_PATH", "")).strip()
    return candidate or None


def runtime_allowed_models_raw_from_env() -> str | None:
    for env_name in ("QWEN_TTS_ALLOWED_MODELS", "QWEN_ALLOWED_MODELS"):
        candidate = str(os.environ.get(env_name, "")).strip()
        if candidate:
            return candidate
    return None


def runtime_allowed_models_source_from_env() -> str | None:
    for env_name in ("QWEN_TTS_ALLOWED_MODELS", "QWEN_ALLOWED_MODELS"):
        candidate = str(os.environ.get(env_name, "")).strip()
        if candidate:
            return env_name
    return None


def runtime_model_catalog_from_env() -> list[dict[str, object]]:
    return build_runtime_model_catalog(
        configured_model=runtime_configured_model_from_env(),
        allowed_models_raw=runtime_allowed_models_raw_from_env(),
    )


def resolve_runtime_model_or_409(
    model_name: str | None,
    *,
    required_family: str | None = None,
) -> str:
    try:
        return pick_runtime_model(
            requested_model=model_name,
            runtime_catalog=runtime_model_catalog_from_env(),
            required_family=required_family,
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
