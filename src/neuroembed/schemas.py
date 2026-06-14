"""Pydantic request/response schemas for the v1 API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EmbeddingsRequest(BaseModel):
    """Request body for POST /v1/embeddings."""

    electrode_names: list[str] = Field(
        min_length=1,
        max_length=512,
        description="Electrode names corresponding to the channel axis of `samples`.",
    )
    samples: list[list[float]] = Field(
        description="EEG samples as a 2D list shaped (channels, samples), float values.",
    )
    sample_rate_hz: int = Field(
        default=200,
        ge=1,
        le=4096,
        description="Sample rate of the recording in Hz. Will be resampled to 200 Hz.",
    )
    window_seconds: int = Field(
        default=4,
        ge=1,
        le=60,
        description="Window size in seconds. Default 4s matches REVE pretraining.",
    )
    return_per_window: bool = Field(
        default=True,
        description="If true, also return per-window embeddings.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model id override; defaults to the server-configured model.",
    )

    @field_validator("samples")
    @classmethod
    def _rectangular(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("samples must be a non-empty 2D list")
        widths = {len(row) for row in v}
        if len(widths) != 1:
            raise ValueError(
                f"all channels must have the same sample count; got widths {sorted(widths)}"
            )
        return v


class EmbeddingsResponse(BaseModel):
    """Response body for POST /v1/embeddings."""

    model: str
    window_count: int
    embedding_dim: int
    mean_embedding: list[float]
    window_embeddings: Optional[list[list[float]]] = None
    processing_ms: int
    cached: bool


class CognitiveScores(BaseModel):
    """Sub-scores for each cognitive-state dimension."""

    sleep_stage: dict[str, float]
    pvt_lapse_prob: float
    valence: float
    arousal: float
    seizure_risk: float


class CognitiveRequest(EmbeddingsRequest):
    """EmbeddingsRequest + a flag to also return cognitive-state scores."""


class CognitiveResponse(BaseModel):
    model: str
    scores: CognitiveScores
    confidence: float
    model_versions: dict[str, str]
    processing_ms: int
    cached: bool
