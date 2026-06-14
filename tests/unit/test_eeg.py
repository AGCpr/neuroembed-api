"""Unit tests for neuroembed.core.eeg — TDD cycle 3 (RED)."""
from __future__ import annotations

import numpy as np
import pytest

from neuroembed.core.eeg import (
    ParsedEeg,
    parse_recording,
    resample_to,
    validate_electrode_set,
)


# A minimal 8-channel, 4-second, 200Hz array — exactly one window.
SAMPLE_NPZ = {
    "data": np.random.default_rng(0).standard_normal((8, 800)).astype(np.float32),
    "channels": ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "F3", "F4"],
}


def test_parse_recording_from_in_memory_ndarray() -> None:
    """A 2D ndarray + matching channel list parses cleanly."""
    parsed = parse_recording(
        {"data": SAMPLE_NPZ["data"], "channels": SAMPLE_NPZ["channels"]}
    )
    assert parsed.shape == (8, 800)
    assert parsed.channel_names == SAMPLE_NPZ["channels"]
    assert parsed.sample_rate_hz == 200
    assert parsed.duration_seconds == pytest.approx(4.0, abs=0.05)


def test_parse_recording_rejects_wrong_channel_count() -> None:
    """If channel list length does not match data, raise ValueError."""
    with pytest.raises(ValueError, match="channel"):
        parse_recording(
            {
                "data": SAMPLE_NPZ["data"],
                "channels": ["Fp1", "Fp2"],  # only 2 names for 8 channels
            }
        )


def test_parse_recording_rejects_1d_data() -> None:
    """A 1D input has no channel axis; reject it."""
    with pytest.raises(ValueError, match="2D"):
        parse_recording(
            {"data": np.zeros(800), "channels": ["Fp1"] * 1}  # type: ignore[list-item]
        )


def test_resample_to_changes_sample_rate_and_shape() -> None:
    """Downsampling 4s @ 200Hz to 100Hz halves the time axis."""
    parsed = parse_recording(
        {"data": SAMPLE_NPZ["data"], "channels": SAMPLE_NPZ["channels"]}
    )
    out = resample_to(parsed, target_hz=100)
    assert out.sample_rate_hz == 100
    assert out.shape == (8, 400)
    assert out.duration_seconds == pytest.approx(4.0, abs=0.05)


def test_resample_to_is_idempotent_when_target_equals_current() -> None:
    """If already at target rate, resample_to returns an equal-valued copy."""
    parsed = parse_recording(
        {"data": SAMPLE_NPZ["data"], "channels": SAMPLE_NPZ["channels"]}
    )
    out = resample_to(parsed, target_hz=200)
    np.testing.assert_array_equal(out.data, parsed.data)


def test_validate_electrode_set_returns_known_set_for_canonical_10_20() -> None:
    """A canonical 10-20 subset validates as 'standard'."""
    verdict = validate_electrode_set(
        ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "F3", "F4", "P3", "P4"]
    )
    assert verdict.is_known is True
    assert verdict.coverage >= 0.5
    assert verdict.unknown == []


def test_validate_electrode_set_flags_unknown_electrode_names() -> None:
    """Custom electrode names are flagged as unknown but still accepted."""
    verdict = validate_electrode_set(["CUSTOM1", "CUSTOM2", "C3"])
    assert verdict.is_known is False
    assert set(verdict.unknown) == {"CUSTOM1", "CUSTOM2"}
    assert verdict.coverage == pytest.approx(1 / 3, abs=0.01)
