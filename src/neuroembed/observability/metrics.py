"""Prometheus metrics for NeuroEmbed API.

Exposed via the /metrics endpoint. Uses the default process and platform
collectors so a Prometheus server can compute rates out of the box.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

# Single, explicit registry so tests and prod share a view
REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS_TOTAL = Counter(
    "neuroembed_requests_total",
    "Total HTTP requests served.",
    labelnames=("method", "path", "status"),
    registry=REGISTRY,
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "neuroembed_inference_latency_seconds",
    "Time spent in the model worker, in seconds.",
    labelnames=("model", "task"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY,
)

EMBEDDINGS_PRODUCED = Counter(
    "neuroembed_embeddings_produced_total",
    "Number of embeddings produced (per model, per task).",
    labelnames=("model", "task"),
    registry=REGISTRY,
)


def render_metrics() -> tuple[bytes, str]:
    """Render the current registry in Prometheus exposition format."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
