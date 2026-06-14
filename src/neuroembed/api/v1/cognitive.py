"""POST /v1/cognitive — embeddings + zero-shot cognitive-state scores."""
from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from neuroembed.auth.apikey import ApiKey
from neuroembed.auth.deps import require_api_key
from neuroembed.config import get_settings
from neuroembed.core.cognitive import score_from_embedding
from neuroembed.core.eeg import parse_recording, resample_to
from neuroembed.core.reve import embed as reve_embed
from neuroembed.observability.metrics import (
    EMBEDDINGS_PRODUCED,
    INFERENCE_LATENCY_SECONDS,
    REQUESTS_TOTAL,
)
from neuroembed.schemas import CognitiveRequest, CognitiveResponse, CognitiveScores

router = APIRouter(prefix="/v1", tags=["cognitive"])


@router.post("/cognitive", response_model=CognitiveResponse)
async def post_cognitive(
    req: CognitiveRequest,
    api_key: ApiKey = Depends(require_api_key),
) -> CognitiveResponse:
    settings = get_settings()
    model_id = req.model or settings.model_id
    started = time.perf_counter()

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
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        with INFERENCE_LATENCY_SECONDS.labels(model=model_id, task="cognitive").time():
            mean, _windows = reve_embed(
                parsed.data,
                parsed.channel_names,
                settings=settings,
                window_seconds=req.window_seconds,
            )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    scores_dict, versions, confidence = score_from_embedding(mean, parsed.channel_names)

    EMBEDDINGS_PRODUCED.labels(model=model_id, task="cognitive").inc(1)
    REQUESTS_TOTAL.labels(method="POST", path="/v1/cognitive", status="200").inc()

    return CognitiveResponse(
        model=model_id,
        scores=CognitiveScores(**scores_dict),
        confidence=confidence,
        model_versions=versions,
        processing_ms=int((time.perf_counter() - started) * 1000),
        cached=False,  # cognitive layer is not yet cached
    )
