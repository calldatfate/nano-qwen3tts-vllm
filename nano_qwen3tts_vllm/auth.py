from __future__ import annotations

import hmac
import os
from typing import Iterable

from fastapi import Request
from fastapi.responses import JSONResponse, Response

PUBLIC_PATHS = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}
PUBLIC_PREFIXES = ("/health",)


def _split_keys(raw: str | None) -> set[str]:
    return {part.strip() for part in str(raw or "").split(",") if part.strip()}


def get_allowed_api_keys() -> set[str]:
    keys = _split_keys(os.environ.get("API_KEYS"))
    if keys:
        return keys
    return _split_keys(os.environ.get("API_KEY"))


def extract_api_token(
    *,
    authorization: str | None,
    x_api_key: str | None,
    query_api_key: str | None = None,
) -> str | None:
    for raw in (authorization, x_api_key, query_api_key):
        if not raw:
            continue
        candidate = raw.strip()
        if not candidate:
            continue
        if candidate.lower().startswith("bearer "):
            candidate = candidate[7:].strip()
        if candidate:
            return candidate
    return None


def is_public_path(path: str) -> bool:
    normalized = str(path or "").strip() or "/"
    if normalized in PUBLIC_PATHS:
        return True
    return any(normalized.startswith(prefix) for prefix in PUBLIC_PREFIXES)


def is_authorized_token(token: str | None, allowed_keys: Iterable[str]) -> bool:
    if not token:
        return False
    return any(hmac.compare_digest(token, allowed) for allowed in allowed_keys)


def reject_unauthorized_request(request: Request) -> Response | None:
    if is_public_path(request.url.path):
        return None

    allowed_keys = get_allowed_api_keys()
    if not allowed_keys:
        return None

    token = extract_api_token(
        authorization=request.headers.get("authorization"),
        x_api_key=request.headers.get("x-api-key"),
        query_api_key=request.query_params.get("api_key"),
    )
    if is_authorized_token(token, allowed_keys):
        return None

    return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
