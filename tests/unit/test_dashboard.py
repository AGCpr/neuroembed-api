"""Unit tests for the static dashboard mount."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from neuroembed.main import create_app
    return TestClient(create_app())


def test_dashboard_root_returns_html(client: TestClient) -> None:
    r = client.get("/dashboard/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "NeuroEmbed" in r.text
    assert "Playground" in r.text


def test_dashboard_assets_load(client: TestClient) -> None:
    r = client.get("/dashboard/dashboard.html")
    assert r.status_code == 200
