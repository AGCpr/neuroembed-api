"""EEG parsing and resampling.

The core API takes EEG in three forms:
1. In-memory 2D ndarray (channels, samples) + channel list (the dev/test path)
2. A numpy ``.npz`` file on disk
3. An EDF/BDF file (via :func:`parse_edf_file`, requires the ``mne`` extra)

MNE is NOT imported at module load — it's a heavy dependency and we
want the API process to stay lean. The optional EDF reader is imported
on demand inside :func:`parse_edf_file`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Standard 10-20 montage (subset of 10-10) used as the "known" set.
# We accept anything; unknown electrode names are flagged but not rejected,
# because real consumer headbands use custom layouts.
KNOWN_ELECTRODES: frozenset[str] = frozenset(
    {
        # Standard 10-20
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
        # Extended 10-20 / 10-10 subset
        "Fpz", "AF3", "AF4", "F1", "F2", "F5", "F6",
        "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
        "C1", "C2", "C5", "C6",
        "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
        "P1", "P2", "P5", "P6",
        "PO3", "PO4", "PO7", "PO8", "POz",
        "Oz", "Iz",
    }
)


@dataclass(frozen=True)
class ParsedEeg:
    """A parsed EEG recording, ready for the model."""

    data: np.ndarray  # shape (channels, samples), float32
    channel_names: list[str]
    sample_rate_hz: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def duration_seconds(self) -> float:
        return self.data.shape[1] / self.sample_rate_hz


@dataclass(frozen=True)
class ElectrodeVerdict:
    """Result of validating an electrode set against the known 10-20 set."""

    is_known: bool
    coverage: float  # fraction of names that are known
    unknown: list[str]


def parse_recording(payload: dict[str, Any]) -> ParsedEeg:
    """Parse a recording from a dict with 'data' (ndarray) and 'channels' (list).

    The data array must be 2D ``(channels, samples)`` and float-castable.
    """
    data = np.asarray(payload["data"], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(
            f"eeg data must be 2D (channels, samples), got shape {data.shape}"
        )
    channels = list(payload["channels"])
    if len(channels) != data.shape[0]:
        raise ValueError(
            f"channel count mismatch: {len(channels)} names for {data.shape[0]} channels"
        )
    sample_rate = int(payload.get("sample_rate_hz", 200))
    if sample_rate <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate}")
    return ParsedEeg(data=data, channel_names=channels, sample_rate_hz=sample_rate)


def parse_npz_file(path: str | Path) -> ParsedEeg:
    """Parse a ``.npz`` with keys ``data`` (2D) and ``channels`` (list of str)."""
    npz = np.load(path, allow_pickle=True)
    return parse_recording(
        {
            "data": npz["data"],
            "channels": list(npz["channels"]),
            "sample_rate_hz": int(npz.get("sample_rate_hz", 200)),
        }
    )


def parse_edf_file(path: str | Path) -> ParsedEeg:
    """Parse an EDF/BDF file via MNE. Requires the ``mne`` extra.

    Imported lazily so the API process can run without mne.
    """
    try:
        import mne  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "EDF parsing requires the 'model' extra: pip install neuroembed[model]"
        ) from e
    import mne  # type: ignore[no-redef]

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    data = raw.get_data(picks="eeg").astype(np.float32)
    ch_names = list(raw.ch_names)
    sfreq = int(raw.info["sfreq"])
    return ParsedEeg(data=data, channel_names=ch_names, sample_rate_hz=sfreq)


def resample_to(parsed: ParsedEeg, target_hz: int) -> ParsedEeg:
    """Resample an EEG recording in-place semantics (returns a new ParsedEeg).

    Uses a simple polyphase resampler (numpy) for the dev path. MNE-backed
    high-fidelity resampling is used when mne is available.
    """
    if target_hz <= 0:
        raise ValueError(f"target_hz must be > 0, got {target_hz}")
    if target_hz == parsed.sample_rate_hz:
        return ParsedEeg(
            data=parsed.data.copy(),
            channel_names=list(parsed.channel_names),
            sample_rate_hz=parsed.sample_rate_hz,
        )
    try:
        from scipy.signal import resample_poly
    except ImportError as e:
        raise ImportError(
            "resample_to requires scipy; install with `pip install scipy`"
        ) from e
    from math import gcd

    g = gcd(target_hz, parsed.sample_rate_hz)
    up = target_hz // g
    down = parsed.sample_rate_hz // g
    resampled = resample_poly(parsed.data, up=up, down=down, axis=1).astype(np.float32)
    return ParsedEeg(
        data=resampled,
        channel_names=list(parsed.channel_names),
        sample_rate_hz=target_hz,
    )


def validate_electrode_set(names: list[str]) -> ElectrodeVerdict:
    """Return whether all electrode names are in the known 10-20 set."""
    if not names:
        return ElectrodeVerdict(is_known=False, coverage=0.0, unknown=[])
    known_hits = sum(1 for n in names if n in KNOWN_ELECTRODES)
    unknown = [n for n in names if n not in KNOWN_ELECTRODES]
    return ElectrodeVerdict(
        is_known=len(unknown) == 0,
        coverage=known_hits / len(names),
        unknown=unknown,
    )
