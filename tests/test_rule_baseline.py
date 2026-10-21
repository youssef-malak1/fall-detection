"""Unit tests for the rule baseline (synthetic time series, no dataset needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.rule_baseline import RuleParams, detect_fall


def make_series(angle, hip_y, detected=None) -> pd.DataFrame:
    angle = list(angle)
    hip_y = list(hip_y)
    n = len(angle)
    df = pd.DataFrame(
        {
            "frame": np.arange(n),
            "detected": np.ones(n, dtype=int) if detected is None else detected,
            "torso_angle": angle,
            "hip_y": hip_y,
            "conf": np.full(n, 0.9),
            "body_aspect": np.zeros(n),
        }
    )
    return df


def collapse(n_descent: int, n_stay: int = 90) -> pd.DataFrame:
    """Upright, then a descent of `n_descent` frames, then lying still."""
    angle = [5.0] * 25 + list(np.linspace(5, 90, n_descent)) + [90.0] * n_stay
    hip_y = [0.35] * 25 + list(np.linspace(0.35, 0.80, n_descent)) + [0.80] * n_stay
    return make_series(angle, hip_y)


def test_fast_collapse_is_a_fall():
    assert detect_fall(collapse(10), RuleParams()).is_fall


def test_slow_voluntary_lying_down_is_not_a_fall():
    # ~6 seconds of controlled descent - the URFD "adl-30" pattern
    df = collapse(180, n_stay=120)
    for window in (0.6, 1.2, 2.0, 3.0):
        assert not detect_fall(df, RuleParams(max_transition_s=window)).is_fall


def test_upright_walking_is_not_a_fall():
    df = make_series([6.0] * 120, [0.4] * 120)
    assert not detect_fall(df, RuleParams()).is_fall


def test_bending_forward_without_hip_drop_is_not_a_fall():
    """Torso rotates but the hips stay at the same height -> not a fall."""
    angle = [5.0] * 20 + [40, 60, 75, 80, 75, 60, 40] + [10.0] * 60
    hip_y = [0.40] * (20 + 7 + 60)
    assert not detect_fall(make_series(angle, hip_y), RuleParams()).is_fall


def test_no_pose_detected_is_not_a_fall():
    df = make_series([np.nan] * 60, [np.nan] * 60, detected=np.zeros(60, dtype=int))
    assert not detect_fall(df, RuleParams()).is_fall


def test_pose_lost_after_descent_is_a_fall():
    """Person collapses towards the camera and the landmark model loses them."""
    n = 60
    angle = [5.0] * 20 + [15, 30, 50, 70] + [np.nan] * (n - 24)
    hip_y = [0.30] * 20 + [0.32, 0.40, 0.55, 0.70] + [np.nan] * (n - 24)
    detected = np.ones(n, dtype=int)
    detected[24:] = 0
    assert detect_fall(make_series(angle, hip_y, detected), RuleParams()).is_fall


def test_short_clip_does_not_crash():
    df = make_series([10.0, 12.0, 15.0], [0.4, 0.41, 0.42])
    assert detect_fall(df, RuleParams()).is_fall is False


@pytest.mark.parametrize(
    "n_descent, window, expected",
    [
        (12, 0.6, True),    # 25->70 deg in ~0.2 s
        (30, 1.2, True),    # ~0.5 s
        (90, 1.2, False),   # ~1.6 s -> too slow
        (180, 2.0, False),  # ~3.2 s -> voluntary
    ],
)
def test_transition_window(n_descent: int, window: float, expected: bool):
    df = collapse(n_descent)
    assert detect_fall(df, RuleParams(max_transition_s=window)).is_fall is expected
