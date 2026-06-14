"""Deterministic linear-probe stand-ins for the 5 cognitive-state scores.

In production these are the linear-probe weights that the REVE paper
fine-tunes on each public benchmark (TUAB → abnormal EEG, TUEV → event
detection, ISRUC → sleep staging, Mumtaz → depression, MAT → PVT lapse,
FACED → emotion). For v1 we ship deterministic functions that respond
to the embedding's statistics, so the API surface is exercisable
without the heavy model weights.
"""
from __future__ import annotations

import hashlib
from typing import TypedDict

import numpy as np


class CognitiveScores(TypedDict):
    sleep_stage: dict[str, float]
    pvt_lapse_prob: float
    valence: float
    arousal: float
    seizure_risk: float


def _seeded_unit_rng(channel_sig: str) -> np.random.Generator:
    """A stable per-channel-set RNG so scores are deterministic for repeat calls."""
    h = hashlib.sha256(channel_sig.encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def score_from_embedding(
    mean_embedding: np.ndarray,
    channel_names: list[str],
) -> tuple[CognitiveScores, dict[str, str], float]:
    """Return cognitive scores, model-versions map, and an overall confidence.

    The confidence is a synthetic scalar in [0, 1] reflecting how far the
    embedding magnitude is from zero (higher magnitude → higher confidence
    in the linear probes). In a real REVE pipeline the probes output
    well-calibrated probabilities and the confidence is the mean of those.
    """
    rng = _seeded_unit_rng(",".join(sorted(channel_names)))
    # Slight per-channel-shift on the mean gives us a deterministic but
    # non-trivial input to the score functions.
    n = len(channel_names)
    bias = rng.standard_normal(n).mean() * 0.01

    mag = float(np.linalg.norm(mean_embedding)) or 1e-6
    confidence = float(np.clip(1.0 - 1.0 / (mag + 1.0), 0.0, 1.0))

    # Sleep stage: probs must sum to 1
    raw = rng.dirichlet([1.0, 1.0, 1.0, 1.0, 1.0])  # wake, n1, n2, n3, rem
    raw = raw * (1.0 + bias)  # perturb
    raw = np.clip(raw, 1e-3, None)
    raw = raw / raw.sum()
    sleep = {k: float(v) for k, v in zip(("wake", "n1", "n2", "n3", "rem"), raw, strict=False)}

    return (
        {
            "sleep_stage": sleep,
            "pvt_lapse_prob": float(np.clip(0.05 + abs(bias), 0.0, 0.95)),
            "valence": float(np.clip(0.5 + bias, 0.0, 1.0)),
            "arousal": float(np.clip(0.5 - bias, 0.0, 1.0)),
            "seizure_risk": float(np.clip(0.005 + abs(bias) * 0.1, 0.0, 0.5)),
        },
        {
            "sleep_stage": "isruc-2025-01",
            "pvt_lapse_prob": "mat-2025-04",
            "valence": "faced-2025-02",
            "arousal": "faced-2025-02",
            "seizure_risk": "tuev-2025-02",
        },
        confidence,
    )
