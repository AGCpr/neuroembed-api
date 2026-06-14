"""Unit tests for neuroembed.config — Phase 3, TDD cycle 1 (RED)."""
import pytest
from pydantic import ValidationError


def test_config_loads_with_defaults_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh config load with no env vars should pick the documented defaults."""
    # Ensure no env overrides
    for k in ("NEUROEMBED_API_KEY", "NEUROEMBED_LOG_LEVEL", "NEUROEMBED_MODEL_ID"):
        monkeypatch.delenv(k, raising=False)
    from neuroembed.config import Settings

    s = Settings()

    assert s.log_level == "INFO"
    assert s.model_id == "brain-bzh/reve-base"
    assert s.cors_origins == []


def test_config_overrides_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env vars prefixed NEUROEMBED_ should override defaults."""
    monkeypatch.setenv("NEUROEMBED_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("NEUROEMBED_MODEL_ID", "brain-bzh/reve-large")
    from neuroembed.config import Settings

    s = Settings()

    assert s.log_level == "DEBUG"
    assert s.model_id == "brain-bzh/reve-large"


def test_config_rejects_invalid_log_level() -> None:
    """An unrecognised log level must be rejected at construction time."""
    from neuroembed.config import Settings

    with pytest.raises(ValidationError):
        Settings(log_level="BANANA")


def test_config_cors_origins_parsed_from_csv_string() -> None:
    """CORS origins should parse a comma-separated string into a list of URLs."""
    from neuroembed.config import Settings

    s = Settings(cors_origins="https://app.example.com,https://dev.example.com")

    assert s.cors_origins == [
        "https://app.example.com",
        "https://dev.example.com",
    ]
