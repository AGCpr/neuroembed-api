"""FastAPI app factory + health endpoints.

The app is built via ``create_app()`` so tests can spin up a fresh instance
without polluting global state. REVE weights are NOT loaded here — they
are loaded on demand by the worker process. This keeps the API process
lean (no torch import at startup) and CPU-only-runnable for development.
"""
from __future__ import annotations

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

from neuroembed.api.v1 import cognitive as v1_cognitive
from neuroembed.api.v1 import embeddings as v1_embeddings
from neuroembed.config import Settings, get_settings
from neuroembed.observability.logging import configure_logging, get_logger
from neuroembed.observability.metrics import render_metrics




log = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI app."""
    if settings is None:
        settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="NeuroEmbed API",
        version="0.1.0",
        description=(
            "Hosted REVE EEG foundation-model inference. "
            "Research use only — not a medical device."
        ),
        docs_url="/docs",
        redoc_url=None,
    )

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        """Liveness — does the process respond at all."""
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> dict[str, object]:
        """Readiness — is the model loaded and the API serving traffic?

        In v1 the model is lazy-loaded by the worker, so this returns
        ``model_loaded=False`` until the first inference request completes
        successfully. Once the worker has been warmed at least once, a
        module-level flag flips to True.
        """
        from neuroembed.core.reve import is_model_loaded

        loaded = is_model_loaded()
        return {
            "status": "ready" if loaded else "not_ready",
            "model_loaded": loaded,
            "model_id": settings.model_id,
        }

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        """Prometheus exposition endpoint."""
        body, content_type = render_metrics()
        return Response(content=body, media_type=content_type)

    # Mount the v1 API. Models/health are added as M0 grows.
    app.include_router(v1_embeddings.router)
    app.include_router(v1_cognitive.router)

    # Mount the static dashboard at /dashboard
    import os as _os
    _static_dir = _os.path.join(_os.path.dirname(__file__), "static")
    if _os.path.isdir(_static_dir):
        app.mount("/dashboard", StaticFiles(directory=_static_dir, html=True), name="dashboard")

    log.info("neuroembed_app_created", model_id=settings.model_id)
    return app
