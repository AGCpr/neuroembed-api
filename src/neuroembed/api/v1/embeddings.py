"""POST /v1/embeddings — accept EEG samples, return 256-dim REVE embeddings.

The endpoint is a thin wrapper over :func:`neuroembed.core.reve.embed`
plus an in-process LRU cache keyed by (sha256(samples), params).

For v1 the data arrives inline in the request body. v1.1 will switch to
presigned S3 URLs (``file_url``) for files >50MB.
"""
from __future__ import annotations

import hashlib
import struct
import time
from collections import OrderedDict
from threading import Lock

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from neuroembed.auth.apikey import ApiKey
from neuroembed.auth.deps import require_api_key
from neuroembed.config import get_settings
from neuroembed.core.eeg import parse_recording, resample_to
from neuroembed.core.reve import embed as reve_embed
from neuroembed.observability.metrics import (
    EMBEDDINGS_PRODUCED,
    INFERENCE_LATENCY_SECONDS,
    REQUESTS_TOTAL,
)
from neuroembed.schemas import EmbeddingsRequest, EmbeddingsResponse

router = APIRouter(prefix="/v1", tags=["embeddings"])


# A simple in-process LRU cache. Redis-backed in v1.1.
class _EmbedCache:
    def __init__(self, capacity: int = 128) -> None:
        self._data: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cap = capacity
        self._lock = Lock()

    def get_or_compute(
        self,
        key: str,
        compute,  # callable returning (mean, windows)
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        with self._lock:
            cached = self._data.get(key)
            if cached is not None:
                self._data.move_to_end(key)
                return cached[0], cached[1], True
        mean, windows = compute()
        with self._lock:
            self._data[key] = (mean, windows)
            self._data.move_to_end(key)
            while len(self._data) > self._cap:
                self._data.popitem(last=False)
        return mean, windows, False

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_CACHE = _EmbedCache()


def _cache_key(req: EmbeddingsRequest) -> str:
    """Stable hash of the request inputs that affect the embedding."""
    h = hashlib.sha256()
    h.update(",".join(req.electrode_names).encode())
    h.update(struct.pack("<III", req.sample_rate_hz, req.window_seconds, len(req.samples[0])))
    arr = np.asarray(req.samples, dtype=np.float32)
    h.update(arr.tobytes())
    return h.hexdigest()


@router.post("/embeddings", response_model=EmbeddingsResponse)
async def post_embeddings(
    req: EmbeddingsRequest,
    api_key: ApiKey = Depends(require_api_key),
) -> EmbeddingsResponse:
    """Embed a multi-channel EEG recording."""
    settings = get_settings()
    model_id = req.model or settings.model_id
    started = time.perf_counter()

    # 1. Parse + resample to 200 Hz (REVE pretraining rate)
    try:
        parsed = parse_recording(
            {
                "data": np.asarray(req.samples, dtype=np.float32),
                "channels": req.electrode_names,
                "sample_rate_hz": req.sample_rate_hz,
            }
        )
        if parsed.sample_rate_hz != 200:
            parsed = resample_to(parsed, target_hz=200)
    except ValueError as e:
        # Surface input-shape problems as a 422 with a helpful message
        raise HTTPException(
            status_code=422,
            detail=str(e),
        ) from e

    # 2. Cache lookup
    key = _cache_key(req)

    def _compute() -> tuple[np.ndarray, np.ndarray]:
        with INFERENCE_LATENCY_SECONDS.labels(model=model_id, task="embed").time():
            try:
                return reve_embed(
                    parsed.data,
                    parsed.channel_names,
                    settings=settings,
                    window_seconds=req.window_seconds,
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=422,
                    detail=str(e),
                ) from e

    mean, windows, cached = _CACHE.get_or_compute(key, _compute)

    EMBEDDINGS_PRODUCED.labels(model=model_id, task="embed").inc(windows.shape[0])
    REQUESTS_TOTAL.labels(method="POST", path="/v1/embeddings", status="200").inc()

    return EmbeddingsResponse(
        model=model_id,
        window_count=windows.shape[0],
        embedding_dim=windows.shape[1],
        mean_embedding=mean.tolist(),
        window_embeddings=windows.tolist() if req.return_per_window else None,
        processing_ms=int((time.perf_counter() - started) * 1000),
        cached=cached,
    )
