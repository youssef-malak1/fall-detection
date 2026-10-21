"""Tests for the per-frame sequence builder (no torch required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sequences import (  # noqa: E402
    CHANNELS,
    apply_transform,
    build_sequences,
    impute_masked,
    load_sequence,
    normalise_transform,
)

CLIP = "fall-01-cam0"


@pytest.fixture(scope="module")
def clip_available() -> bool:
    return (PROJECT_ROOT / "data" / "interim" / "pose" / f"{CLIP}.csv").exists()


def test_channels_are_unique():
    names = [name for name, _ in CHANNELS]
    assert len(names) == len(set(names))


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "interim" / "pose" / f"{CLIP}.csv").exists(),
    reason="pose features not extracted",
)
def test_sequence_shape_and_no_nan():
    x, mask = load_sequence(CLIP)
    assert x.ndim == 2
    assert x.shape[1] == len(CHANNELS)
    assert mask.shape == (x.shape[0],)
    assert not np.isnan(x).any(), "NaN leaked out of the sequence builder"
    assert not np.isnan(mask).any()
    assert set(np.unique(mask)).issubset({0.0, 1.0})


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "interim" / "pose" / f"{CLIP}.csv").exists(),
    reason="pose features not extracted",
)
def test_mask_matches_detected_column():
    x, mask = load_sequence(CLIP)
    detected = x[:, [name for name, _ in CHANNELS].index("detected")] == 1.0
    # mask can never be 1 where the pose was not detected
    assert not (mask.astype(bool) & ~detected).any()


def test_impute_holds_last_value_and_zeroes_the_lead():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    mask = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    out = impute_masked(x, mask)
    assert np.allclose(out[0], 0.0), "frames before the first detection must be zero"
    assert np.allclose(out[2], x[1]), "a gap holds the last valid frame"
    assert np.allclose(out[1], x[1])


def test_impute_all_invalid_is_all_zero():
    x = np.ones((5, 3), dtype=np.float32)
    out = impute_masked(x, np.zeros(5, dtype=np.float32))
    assert np.allclose(out, 0.0)


def test_transform_uses_only_valid_frames():
    good = {"a": np.array([[1.0], [3.0]], dtype=np.float32)}
    mask = {"a": np.ones(2, dtype=np.float32)}
    mean, std = normalise_transform(good, mask)
    assert np.isclose(mean[0], 2.0)
    assert np.isclose(std[0], 1.0)

    # an extreme garbage frame excluded by the mask must not move the statistics
    noisy = {"a": np.array([[1.0], [3.0], [1000.0]], dtype=np.float32)}
    noisy_mask = {"a": np.array([1.0, 1.0, 0.0], dtype=np.float32)}
    mean2, _ = normalise_transform(noisy, noisy_mask)
    assert np.isclose(mean2[0], 2.0)


def test_transform_clips_outliers():
    x = np.array([[100.0, -100.0]], dtype=np.float32)
    mean = np.zeros(2, dtype=np.float32)
    std = np.ones(2, dtype=np.float32)
    out = apply_transform(x, mean, std)
    assert out.max() <= 5.0
    assert out.min() >= -5.0
