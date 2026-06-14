"""Configuration via environment variables.

All settings are read from env vars prefixed with ``NEUROEMBED_``. The
``Settings`` model is constructed once at startup and passed via FastAPI
dependencies; it is never re-parsed per request.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from NEUROEMBED_* env vars."""

    model_config = SettingsConfigDict(
        env_prefix="NEUROEMBED_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG/INFO/WARNING/ERROR).",
    )
    model_id: str = Field(
        default="brain-bzh/reve-base",
        description="HuggingFace model id for the REVE foundation model.",
    )
    position_bank_id: str = Field(
        default="brain-bzh/reve-positions",
        description="HuggingFace model id for the REVE electrode position bank.",
    )
    hf_token: str | None = Field(
        default=None,
        description="HuggingFace token for accessing the gated REVE model.",
    )
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Comma-separated list of allowed CORS origins.",
    )
    cache_dir: str = Field(
        default="/tmp/neuroembed-cache",
        description="Where to cache downloaded REVE weights.",
    )

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        v_up = v.upper()
        if v_up not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"log_level must be one of DEBUG/INFO/WARNING/ERROR/CRITICAL, got {v!r}")
        return v_up

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, v: object) -> object:
        """Allow ``NEUROEMBED_CORS_ORIGINS='a,b,c'`` from the env."""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


_cached_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance (env is parsed once)."""
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = Settings()
    return _cached_settings


def reset_settings_cache() -> None:
    """Test helper: drop the cache so the next ``get_settings()`` re-reads env."""
    global _cached_settings
    _cached_settings = None
