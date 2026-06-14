"""Unit tests for auth on the v1 endpoints — TDD cycle 6 (RED)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def valid_key() -> str:
    return "nmb_testkey_valid_xxxxxxxxxxxxxx"


@pytest.fixture
def app_with_auth(monkeypatch: pytest.MonkeyPatch, valid_key: str):
    """Create an app where a known key is registered in the in-memory store."""
    from neuroembed.auth.apikey import hash_api_key
    from neuroembed.main import create_app
    from neuroembed.auth import store

    store.reset()
    store.add_key(key_id="k_test", hashed=hash_api_key(valid_key), tier="hobby")
    return TestClient(create_app()), valid_key


def test_embeddings_without_auth_header_returns_401() -> None:
    from neuroembed.main import create_app
    from neuroembed.auth import store

    store.reset()
    client = TestClient(create_app())
    r = client.post("/v1/embeddings", json={})
    assert r.status_code == 401


def test_embeddings_with_invalid_token_returns_401() -> None:
    from neuroembed.main import create_app
    from neuroembed.auth import store

    store.reset()
    client = TestClient(create_app())
    r = client.post(
        "/v1/embeddings",
        json={},
        headers={"Authorization": "Bearer nmb_wrong_key"},
    )
    assert r.status_code == 401


def test_embeddings_with_valid_token_succeeds(app_with_auth) -> None:
    client, key = app_with_auth
    import numpy as np

    rng = np.random.default_rng(7)
    body = {
        "electrode_names": ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "F3", "F4"],
        "samples": rng.standard_normal((8, 8 * 200)).tolist(),
        "sample_rate_hz": 200,
        "window_seconds": 4,
        "return_per_window": False,
    }
    r = client.post("/v1/embeddings", json=body, headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200, r.text
    body_out = r.json()
    assert body_out["embedding_dim"] == 256


def test_metrics_endpoint_does_not_require_auth() -> None:
    """/metrics is public for Prometheus scrape."""
    from neuroembed.main import create_app
    from neuroembed.auth import store

    store.reset()
    client = TestClient(create_app())
    r = client.get("/metrics")
    assert r.status_code == 200


def test_healthz_does_not_require_auth() -> None:
    """/healthz and /readyz are public for liveness/readiness probes."""
    from neuroembed.main import create_app
    from neuroembed.auth import store

    store.reset()
    client = TestClient(create_app())
    r = client.get("/healthz")
    assert r.status_code == 200
