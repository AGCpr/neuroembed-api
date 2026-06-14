"""Unit tests for neuroembed.api.v1.embeddings — TDD cycle 4 (RED)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _clear_embedding_cache() -> None:
    """Reset the in-process cache between tests so cached=True assertions are real."""
    from neuroembed.api.v1.embeddings import _CACHE
    from neuroembed.auth import store as auth_store

    _CACHE.clear()
    auth_store.reset()


@pytest.fixture
def client() -> TestClient:
    from neuroembed.main import create_app
    from neuroembed.config import reset_settings_cache

    reset_settings_cache()
    return TestClient(create_app())


def _ok_payload() -> dict[str, object]:
    """A minimal valid embeddings request body."""
    import numpy as np

    rng = np.random.default_rng(42)
    return {
        "electrode_names": ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "F3", "F4"],
        "samples": rng.standard_normal((8, 8 * 200)).tolist(),
        "sample_rate_hz": 200,
        "window_seconds": 4,
        "return_per_window": True,
    }


def _seed_auth() -> str:
    from neuroembed.auth.store import get_store
    from neuroembed.auth.apikey import hash_api_key

    key = "nmb_testkey_valid_xxxxxxxxxxxxxx"
    get_store().add_key("k_test", hash_api_key(key), tier="hobby")
    return key


@pytest.fixture(autouse=True)
def _seed_default_auth() -> str:
    return _seed_auth()


def test_post_embeddings_returns_mean_and_windows(client: TestClient) -> None:
    r = client.post(
        "/v1/embeddings",
        json=_ok_payload(),
        headers={"Authorization": f"Bearer {_seed_auth()}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "mean_embedding" in body
    assert "window_embeddings" in body
    assert "window_count" in body
    assert body["embedding_dim"] == 256
    assert body["model"] == "brain-bzh/reve-base"
    assert body["cached"] is False
    assert body["window_count"] == 2
    assert len(body["mean_embedding"]) == 256
    assert len(body["window_embeddings"]) == 2
    assert all(len(w) == 256 for w in body["window_embeddings"])


def test_post_embeddings_without_per_window_returns_only_mean(
    client: TestClient,
) -> None:
    payload = _ok_payload()
    payload["return_per_window"] = False
    r = client.post(
        "/v1/embeddings",
        json=payload,
        headers={"Authorization": f"Bearer {_seed_auth()}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "mean_embedding" in body
    assert body["window_embeddings"] is None


def test_post_embeddings_rejects_short_recording(client: TestClient) -> None:
    """A recording shorter than one window must return 422 with a helpful error."""
    import numpy as np

    payload = _ok_payload()
    payload["samples"] = np.random.default_rng(0).standard_normal((8, 200)).tolist()  # 1s
    r = client.post(
        "/v1/embeddings",
        json=payload,
        headers={"Authorization": f"Bearer {_seed_auth()}"},
    )
    assert r.status_code == 422
    body = r.json()
    assert "detail" in body


def test_post_embeddings_rejects_wrong_channel_count(client: TestClient) -> None:
    payload = _ok_payload()
    payload["electrode_names"] = ["Fp1"]
    r = client.post(
        "/v1/embeddings",
        json=payload,
        headers={"Authorization": f"Bearer {_seed_auth()}"},
    )
    assert r.status_code == 422


def test_post_embeddings_returns_cached_on_second_call(
    client: TestClient,
) -> None:
    """Identical inputs should be served from the in-process cache on the 2nd call."""
    headers = {"Authorization": f"Bearer {_seed_auth()}"}
    body = _ok_payload()
    r1 = client.post("/v1/embeddings", json=body, headers=headers)
    assert r1.status_code == 200
    assert r1.json()["cached"] is False
    r2 = client.post("/v1/embeddings", json=body, headers=headers)
    assert r2.status_code == 200
    assert r2.json()["cached"] is True
    assert r2.json()["mean_embedding"] == r1.json()["mean_embedding"]


def test_post_embeddings_unwraps_to_unit_norm(client: TestClient) -> None:
    """Each embedding returned should have unit norm (the FakeReve contract)."""
    import math

    r = client.post(
        "/v1/embeddings",
        json=_ok_payload(),
        headers={"Authorization": f"Bearer {_seed_auth()}"},
    )
    body = r.json()
    for vec in body["window_embeddings"]:
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4
