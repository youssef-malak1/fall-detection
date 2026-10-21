"""Tests for the alerting state machine — pure logic, no model, no video."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.alerting import FallEventTracker  # noqa: E402


def test_single_strong_window_alerts_once():
    tracker = FallEventTracker(threshold=0.5, confirm_windows=1, cooldown_s=30)
    first = tracker.update(0.0, 0.9)
    assert first.alert
    assert first.reason == "fall_detected"

    # the same incident continues: no second alert
    second = tracker.update(3.0, 0.95)
    assert not second.alert
    assert second.reason == "already_alerted"


def test_weak_windows_never_alert():
    tracker = FallEventTracker(threshold=0.5)
    for t in range(0, 60, 3):
        assert not tracker.update(t, 0.1).alert


def test_confirm_windows_requires_a_streak():
    tracker = FallEventTracker(threshold=0.5, confirm_windows=2)
    assert not tracker.update(0.0, 0.7).alert
    assert tracker.update(0.5, 0.7).alert


def test_a_dip_breaks_the_streak():
    tracker = FallEventTracker(threshold=0.5, confirm_windows=2)
    assert not tracker.update(0.0, 0.7).alert
    assert not tracker.update(0.5, 0.4).alert  # dip
    assert not tracker.update(1.0, 0.7).alert  # streak restarted


def test_event_clears_and_rearms():
    tracker = FallEventTracker(threshold=0.5, cooldown_s=0, clear_below=0.3, clear_windows=2)
    assert tracker.update(0.0, 0.9).alert
    tracker.update(3.0, 0.1)
    tracker.update(6.0, 0.1)  # event cleared
    again = tracker.update(9.0, 0.9)
    assert again.alert, "a second, genuinely new fall must alert"


def test_cooldown_blocks_a_second_alert_for_the_same_incident():
    tracker = FallEventTracker(threshold=0.5, cooldown_s=30, clear_below=0.3, clear_windows=1)
    assert tracker.update(0.0, 0.9).alert
    tracker.update(3.0, 0.1)  # clears
    blocked = tracker.update(6.0, 0.9)  # within cooldown
    assert not blocked.alert
    assert blocked.reason == "cooldown"

    allowed = tracker.update(40.0, 0.9)
    assert allowed.alert, "after the cooldown an event must be able to alert again"


def test_short_dip_does_not_rearm():
    tracker = FallEventTracker(threshold=0.5, cooldown_s=0, clear_below=0.3, clear_windows=3)
    assert tracker.update(0.0, 0.9).alert
    tracker.update(3.0, 0.1)  # only one low window so far
    assert tracker.update(6.0, 0.9).reason == "already_alerted"


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        FallEventTracker(clear_below=0.8, threshold=0.5)
    with pytest.raises(ValueError):
        FallEventTracker(confirm_windows=0)
    with pytest.raises(ValueError):
        FallEventTracker(clear_windows=0)


def test_reset_clears_state():
    tracker = FallEventTracker(threshold=0.5, cooldown_s=100)
    tracker.update(0.0, 0.9)
    tracker.reset()
    assert tracker.update(1.0, 0.9).alert
