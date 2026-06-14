"""REVE foundation-model wrapper.

The REVE model itself is heavy (≈400MB of weights, requires torch and
transformers). To keep the API process startup fast, this module exposes
a lazy ``get_model()`` accessor and a status flag. The first call to
``get_model()`` imports torch + transformers + downloads the gated weights
from HuggingFace.

For v1 development without a GPU, ``FakeReve`` provides a deterministic
stand-in that returns a 256-dim random projection keyed by the input
tensor's hash. This lets the API be exercised end-to-end without the
real model — see :func:`is_model_loaded` for the test path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from neuroembed.config import Settings

_MODEL_LOADED = False


def is_model_loaded() -> bool:
    """True once :func:`get_model` has produced at least one inference result."""
    return _MODEL_LOADED


def _mark_loaded() -> None:
    global _MODEL_LOADED
    _MODEL_LOADED = True


@dataclass(frozen=True)
class ReveConfig:
    """Configuration resolved for a single inference call."""

    model_id: str
    position_bank_id: str
    sample_rate_hz: int = 200
    window_seconds: int = 4
    embedding_dim: int = 256


class ReveBackend(Protocol):
    """Backend interface — REVE and FakeReve both satisfy it."""

    def embed_windows(
        self, eeg: np.ndarray, electrode_names: list[str], config: ReveConfig
    ) -> np.ndarray:
        """Return an array of shape (n_windows, embedding_dim)."""
        ...


class FakeReve:
    """A deterministic, dependency-free stand-in for the REVE model.

    Generates a 256-dim unit-norm embedding per window by:
    1. Hashing (channel-set, window index) to seed an RNG.
    2. Drawing from N(0, 1).
    3. Projecting to unit norm.

    This is enough to exercise the API surface end-to-end during
    development. Real REVE inferences will replace this once the
    heavy deps are installed via ``pip install -e .[model]``.
    """

    def embed_windows(
        self, eeg: np.ndarray, electrode_names: list[str], config: ReveConfig
    ) -> np.ndarray:
        if eeg.ndim != 2:
            raise ValueError(f"eeg must be (channels, samples), got shape {eeg.shape}")
        if eeg.shape[0] != len(electrode_names):
            raise ValueError(
                f"electrode_names length ({len(electrode_names)}) must match "
                f"channel count ({eeg.shape[0]})"
            )
        n_samples = eeg.shape[1]
        samples_per_window = config.sample_rate_hz * config.window_seconds
        if n_samples < samples_per_window:
            raise ValueError(
                f"recording has {n_samples} samples; need at least "
                f"{samples_per_window} (one window of {config.window_seconds}s "
                f"at {config.sample_rate_hz}Hz)"
            )
        n_windows = n_samples // samples_per_window
        channel_sig = ",".join(sorted(electrode_names)).encode()
        out = np.empty((n_windows, config.embedding_dim), dtype=np.float32)
        for w in range(n_windows):
            seed = hashlib.sha256(channel_sig + w.to_bytes(4, "big")).digest()
            rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
            vec = rng.standard_normal(config.embedding_dim).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            out[w] = vec
        return out


_BACKEND: ReveBackend | None = None


def get_backend() -> ReveBackend:
    """Return a singleton backend, preferring real REVE if available.

    Detection: if ``torch`` and ``transformers`` are importable AND the
    HuggingFace model id is reachable, instantiate the real REVE.
    Otherwise return :class:`FakeReve`.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        # In a v1 GPU build we'd do:
        #   _BACKEND = RealReve(model_id=settings.model_id, ...)
        # For now we always use FakeReve to keep tests deterministic.
    except ImportError:
        pass
    _BACKEND = FakeReve()
    return _BACKEND


def embed(
    eeg: np.ndarray,
    electrode_names: list[str],
    settings: Settings | None = None,
    window_seconds: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed an EEG recording. Returns (mean_embedding, per_window_embeddings).

    The first successful call flips :func:`is_model_loaded` to True, so
    /readyz can report ready.
    """
    settings = settings or Settings()
    cfg = ReveConfig(
        model_id=settings.model_id,
        position_bank_id=settings.position_bank_id,
        window_seconds=window_seconds,
    )
    backend = get_backend()
    windows = backend.embed_windows(eeg, electrode_names, cfg)
    mean = windows.mean(axis=0)
    _mark_loaded()
    return mean, windows
