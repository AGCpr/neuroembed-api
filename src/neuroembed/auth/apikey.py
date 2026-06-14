"""API-key generation, hashing, and verification.

Keys are prefixed ``nmb_`` (NeuroEmbed) so they're grep-friendly in logs
and accidental commits. The plaintext key is only ever returned to the
user **once** at creation; only the bcrypt hash is stored.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass

import bcrypt

_KEY_PREFIX = "nmb_"
_KEY_BODY_BYTES = 32  # 256 bits of entropy


def generate_api_key() -> str:
    """Return a fresh API key in the form ``nmb_<43-char-base64>``."""
    return _KEY_PREFIX + secrets.token_urlsafe(_KEY_BODY_BYTES)


def hash_api_key(key: str) -> str:
    """Hash a plaintext key with bcrypt. Returns a string suitable for storage."""
    return bcrypt.hashpw(key.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_api_key(key: str, hashed: str) -> bool:
    """Constant-time verification of a plaintext key against its stored hash."""
    if not key or not hashed:
        return False
    try:
        return bcrypt.checkpw(key.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True)
class ApiKey:
    """An API key record as exposed to the API and dashboard.

    NEVER log or return ``hashed_secret`` to the user.
    """

    key_id: str
    hashed_secret: str
    user_id: str
    tier: str = "free"
