"""Unit tests for /v1/cognitive — TDD cycle 7."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from neuroembed.auth import store as auth_store
    from neuroembed.auth.apikey import hash_api_key
    from neuroembed.config import reset_settings_cache
    from neuroembed.main import create_app

    reset_settings_cache()
    auth_store.reset()
    auth_store.add_key("k_test", hash_api_key("nmb_testkey_xxxxxxxxxxxx"), tier="hobby")
    return TestClient(create_app())


def _ok_payload() -> dict[str, object]:
    import numpy as np

    rng = np.random.default_rng(11)
    return {
        "electrode_names": ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "F3", "F4"],
        "samples": rng.standard_normal((8, 8 * 200)).tolist(),
        "sample_rate_hz": 200,
        "window_seconds": 4,
    }


def test_cognitive_returns_five_scores(client: TestClient) -> None:
    r = client.post(
        "/v1/cognitive",
        json=_ok_payload(),
        headers={"Authorization": "Bearer nmb_testkey_xxxxxxxxxxxx"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "scores" in body
    s = body["scores"]
    for k in ("sleep_stage", "pvt_lapse_prob", "valence", "arousal", "seizure_risk"):
        assert k in s, k
    for stage in ("wake", "n1", "n2", "n3", "rem"):
        assert stage in s["sleep_stage"]
    # Sleep stage probs must sum to ~1
    total = sum(s["sleep_stage"].values())
    assert abs(total - 1.0) < 1e-3
    # Confidence in [0, 1]
    assert 0.0 <= body["confidence"] <= 1.0


def test_cognitive_requires_auth(client: TestClient) -> None:
    r = client.post("/v1/cognitive", json=_ok_payload())
    assert r.status_code == 401
