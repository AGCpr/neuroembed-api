"""Unit tests for neuroembed.main — Phase 3, TDD cycle 2 (RED)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_model_loaded_flag() -> None:
    """Reset the module-global model_loaded flag so /readyz assertions are real."""
    import neuroembed.core.reve as reve

    reve._MODEL_LOADED = False


@pytest.fixture
def client() -> TestClient:
    """A test client against the in-process FastAPI app."""
    from neuroembed.main import create_app

    return TestClient(create_app())


def test_healthz_returns_ok(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_returns_status_and_model_loaded(client: TestClient) -> None:
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ready", "not_ready"}
    assert "model_loaded" in body
    assert isinstance(body["model_loaded"], bool)


def test_readyz_status_is_not_ready_when_model_not_loaded(client: TestClient) -> None:
    """In tests we never load the REVE weights; /readyz must report not_ready."""
    r = client.get("/readyz")
    body = r.json()
    # We assert it doesn't pretend to be ready
    assert body["model_loaded"] is False
    assert body["status"] == "not_ready"


def test_404_returns_problem_json(client: TestClient) -> None:
    """Unknown routes return a 404 with a JSON body (not HTML)."""
    r = client.get("/this/does/not/exist")
    assert r.status_code == 404
    assert "application/json" in r.headers["content-type"]


def test_metrics_endpoint_exposes_prometheus_text(client: TestClient) -> None:
    """A /metrics endpoint should serve prometheus exposition format."""
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus exposition format always starts with `# HELP` or `# TYPE`
    assert r.text.startswith("#") or "process_" in r.text or "python_" in r.text
