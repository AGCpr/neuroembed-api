"""FastAPI dependencies for auth."""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from neuroembed.auth.apikey import ApiKey
from neuroembed.auth.store import lookup


from neuroembed.auth.store import lookup as _lookup


async def require_api_key(authorization: str | None = Header(default=None)) -> ApiKey:
    """Verify the ``Authorization: Bearer ***`` header against the store.

    Returns the matching :class:`ApiKey` on success. Raises a 401 otherwise.
    Public endpoints (health, metrics) MUST NOT include this dependency.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    plaintext = authorization[len("Bearer ") :].strip()
    api_key = _lookup(plaintext)
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return api_key
