"""In-memory API key store.

In v1.1 this is swapped for a Postgres-backed store. The interface is
the same: ``lookup(plaintext_key)`` returns an :class:`ApiKey` or None.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from neuroembed.auth.apikey import ApiKey, hash_api_key, verify_api_key

# Each registered entry: (key_id, hashed_secret, tier)
@dataclass(frozen=True)
class _Entry:
    api_key: ApiKey


class ApiKeyStore:
    def __init__(self) -> None:
        self._by_id: dict[str, _Entry] = {}
        self._lock = Lock()

    def add_key(self, key_id: str, hashed: str, tier: str = "free", user_id: str = "u_local") -> None:
        with self._lock:
            self._by_id[key_id] = _Entry(ApiKey(key_id=key_id, hashed_secret=hashed, user_id=user_id, tier=tier))

    def add_plaintext(self, key_id: str, plaintext: str, tier: str = "free", user_id: str = "u_local") -> None:
        """Convenience: hash and register in one call. Used in tests."""
        self.add_key(key_id, hash_api_key(plaintext), tier=tier, user_id=user_id)

    def lookup(self, plaintext: str) -> ApiKey | None:
        with self._lock:
            for entry in self._by_id.values():
                if verify_api_key(plaintext, entry.api_key.hashed_secret):
                    return entry.api_key
        return None

    def reset(self) -> None:
        with self._lock:
            self._by_id.clear()


_STORE = ApiKeyStore()


def get_store() -> ApiKeyStore:
    return _STORE


def reset() -> None:
    """Test helper."""
    _STORE.reset()


def add_key(key_id: str, hashed: str, tier: str = "free", user_id: str = "u_local") -> None:
    """Module-level convenience for tests and bootstrap scripts."""
    _STORE.add_key(key_id, hashed, tier=tier, user_id=user_id)


def lookup(plaintext: str) -> ApiKey | None:
    """Module-level convenience for the auth dependency."""
    return _STORE.lookup(plaintext)
