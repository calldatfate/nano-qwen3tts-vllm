from __future__ import annotations

from nano_qwen3tts_vllm.auth import (
    extract_api_token,
    get_allowed_api_keys,
    is_authorized_token,
    is_public_path,
)


def test_get_allowed_api_keys_from_api_key(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("API_KEY", "alpha,beta")

    assert get_allowed_api_keys() == {"alpha", "beta"}


def test_extract_api_token_prefers_authorization_bearer():
    token = extract_api_token(
        authorization="Bearer secret-token",
        x_api_key=None,
    )

    assert token == "secret-token"


def test_extract_api_token_accepts_x_api_key():
    token = extract_api_token(
        authorization=None,
        x_api_key="secret-token",
    )

    assert token == "secret-token"


def test_extract_api_token_accepts_query_api_key():
    token = extract_api_token(
        authorization=None,
        x_api_key=None,
        query_api_key="secret-token",
    )

    assert token == "secret-token"


def test_is_public_path_whitelists_health_only():
    assert is_public_path("/health/ready") is True
    assert is_public_path("/health/live") is True
    assert is_public_path("/") is True
    assert is_public_path("/api/prepare") is False


def test_is_authorized_token_matches_exact_key():
    assert is_authorized_token("beta", {"alpha", "beta"}) is True
    assert is_authorized_token("wrong", {"alpha", "beta"}) is False
