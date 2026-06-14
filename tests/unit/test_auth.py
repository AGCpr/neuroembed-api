"""Unit tests for the API-key auth module — TDD cycle 5 (RED)."""
from __future__ import annotations

import pytest

from neuroembed.auth.apikey import (
    generate_api_key,
    hash_api_key,
    verify_api_key,
    ApiKey,
)


def test_generate_api_key_has_prefix_and_minimum_length() -> None:
    k = generate_api_key()
    assert k.startswith("nmb_")
    assert len(k) >= 32


def test_hash_api_key_uses_bcrypt_format_and_salt() -> None:
    """bcrypt output starts with $2b$ and includes a per-hash salt."""
    k = generate_api_key()
    h = hash_api_key(k)
    assert h.startswith("$2b$")
    # Different salts → different hashes for the same plaintext
    h2 = hash_api_key(k)
    assert h != h2
    assert len(h) >= 60


def test_hash_api_key_different_for_different_inputs() -> None:
    h1 = hash_api_key(generate_api_key())
    h2 = hash_api_key(generate_api_key())
    assert h1 != h2


def test_verify_api_key_accepts_correct() -> None:
    k = generate_api_key()
    h = hash_api_key(k)
    assert verify_api_key(k, h) is True


def test_verify_api_key_rejects_wrong() -> None:
    k = generate_api_key()
    h = hash_api_key(k)
    assert verify_api_key("nmb_wrong_value_here_xxxxxxxx", h) is False


def test_apikey_dataclass_round_trip() -> None:
    """ApiKey has the fields we expose via the API and the dashboard."""
    k = ApiKey(key_id="k_abc", hashed_secret=hash_api_key(generate_api_key()), user_id="u_1", tier="hobby")
    assert k.tier == "hobby"
    assert k.user_id == "u_1"
