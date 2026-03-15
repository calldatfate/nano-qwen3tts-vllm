from __future__ import annotations

from pathlib import Path
from typing import Iterable

QWEN_MODEL_CATALOG = [
    {
        "id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "label": "0.6B CustomVoice",
        "family": "custom_voice",
        "supports_voice_cloning": False,
        "requires_ref_audio": False,
        "requires_prompt": False,
    },
    {
        "id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "label": "1.7B CustomVoice",
        "family": "custom_voice",
        "supports_voice_cloning": False,
        "requires_ref_audio": False,
        "requires_prompt": False,
    },
    {
        "id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "label": "1.7B VoiceDesign",
        "family": "voice_design",
        "supports_voice_cloning": False,
        "requires_ref_audio": False,
        "requires_prompt": True,
    },
    {
        "id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "label": "0.6B Base",
        "family": "base",
        "supports_voice_cloning": True,
        "requires_ref_audio": True,
        "requires_prompt": False,
    },
    {
        "id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "label": "1.7B Base",
        "family": "base",
        "supports_voice_cloning": True,
        "requires_ref_audio": True,
        "requires_prompt": False,
    },
]

QWEN_MODEL_META_BY_ID = {
    str(item["id"]): dict(item)
    for item in QWEN_MODEL_CATALOG
}

_FAMILY_ALIASES = {
    "base": "base",
    "voice_design": "voice_design",
    "voicedesign": "voice_design",
    "prompt": "voice_design",
    "custom_voice": "custom_voice",
    "customvoice": "custom_voice",
}

_EXACT_MODEL_ALIASES = {
    "qwen/qwen3-tts-12hz-06b-base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "qwen/qwen3-tts-12hz-17b-base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "qwen/qwen3-tts-12hz-06b-customvoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "qwen/qwen3-tts-12hz-17b-customvoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "qwen/qwen3-tts-12hz-17b-voicedesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "06bbase": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "17bbase": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "06bcustomvoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "17bcustomvoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "17bvoicedesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


def _compact_token(value: str) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def infer_qwen_model_family(model_name: str | None) -> str | None:
    candidate = str(model_name or "").strip()
    if not candidate:
        return None

    known = QWEN_MODEL_META_BY_ID.get(candidate)
    if known:
        return str(known["family"])

    compact = _compact_token(candidate)
    if "voicedesign" in compact or "prompt" in compact:
        return "voice_design"
    if "customvoice" in compact:
        return "custom_voice"
    if "base" in compact:
        return "base"
    return None


def describe_qwen_model(model_name: str) -> dict[str, object]:
    candidate = str(model_name or "").strip()
    if not candidate:
        raise ValueError("model_name is required")

    known = QWEN_MODEL_META_BY_ID.get(candidate)
    if known is not None:
        return dict(known)

    inferred_family = infer_qwen_model_family(candidate) or "custom"
    label = Path(candidate.rstrip("/\\")).name or candidate
    return {
        "id": candidate,
        "label": label,
        "family": inferred_family,
        "supports_voice_cloning": inferred_family == "base",
        "requires_ref_audio": inferred_family == "base",
        "requires_prompt": inferred_family == "voice_design",
    }


def _expand_model_token(token: str) -> list[str]:
    candidate = str(token or "").strip()
    if not candidate:
        return []

    if candidate in QWEN_MODEL_META_BY_ID:
        return [candidate]

    compact = _compact_token(candidate)
    exact = _EXACT_MODEL_ALIASES.get(compact)
    if exact:
        return [exact]

    family = _FAMILY_ALIASES.get(compact)
    if family:
        return [
            str(item["id"])
            for item in QWEN_MODEL_CATALOG
            if str(item["family"]) == family
        ]

    return [candidate]


def build_runtime_model_catalog(
    *,
    configured_model: str | None,
    allowed_models_raw: str | None,
) -> list[dict[str, object]]:
    allowed_raw = str(allowed_models_raw or "").strip()
    configured = str(configured_model or "").strip()

    selected_model_ids: list[str] = []
    if allowed_raw:
        for token in allowed_raw.split(","):
            selected_model_ids.extend(_expand_model_token(token))
    elif configured:
        selected_model_ids = [configured]
    else:
        return [dict(item) for item in QWEN_MODEL_CATALOG]

    runtime_catalog: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for model_id in selected_model_ids:
        described = describe_qwen_model(model_id)
        normalized_id = str(described["id"])
        if normalized_id in seen_ids:
            continue
        runtime_catalog.append(described)
        seen_ids.add(normalized_id)
    return runtime_catalog


def pick_runtime_model(
    *,
    requested_model: str | None,
    runtime_catalog: Iterable[dict[str, object]],
    required_family: str | None = None,
) -> str:
    catalog = [dict(item) for item in runtime_catalog]
    if not catalog:
        raise ValueError("No Qwen models are exposed by this runtime")

    requested = str(requested_model or "").strip()
    allowed_ids = {str(item["id"]): dict(item) for item in catalog}

    if requested:
        exact = allowed_ids.get(requested)
        if exact is not None:
            family = str(exact.get("family") or "")
            if required_family is None or family == required_family:
                return requested

    target_family = required_family or infer_qwen_model_family(requested)
    if target_family:
        family_matches = [
            str(item["id"])
            for item in catalog
            if str(item.get("family") or "") == target_family
        ]
        if family_matches:
            return family_matches[0]

    if not requested and len(catalog) == 1:
        return str(catalog[0]["id"])

    if requested:
        raise ValueError(f"Model '{requested}' is not exposed by this runtime")
    raise ValueError("A Qwen model must be selected")
