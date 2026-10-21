"""Alerting state machine: from per-window scores to at most one alert per fall.

The model answers "does this 3-second window contain a fall?". A deployed system needs
a different answer: "should I wake somebody up right now?". Those are not the same, and
conflating them is how fall detectors become unusable in practice:

* a single high window is not enough — a person sitting down fast can trigger it;
* the same fall must not produce an alert every 3 seconds for the next minute;
* after a person is helped up, the system has to re-arm.

The tracker below is deliberately tiny and dependency-free so it can be unit-tested and
ported to the backend as-is.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlertDecision:
    alert: bool
    reason: str
    score: float
    timestamp_s: float
    streak: int
    event_active: bool


class FallEventTracker:
    """Debounce + cooldown over a stream of window scores.

    Parameters
    ----------
    threshold:
        A window at or above this counts as "fall-like".
    confirm_windows:
        How many *consecutive* fall-like windows are needed before alerting. ``1``
        alerts immediately, which is right for a 3 s window that already carries the
        whole transition; raise it if the upstream window stride is small.
    cooldown_s:
        Minimum time between two alerts for what is probably the same incident.
    clear_below:
        A score under this ends the event and re-arms the tracker.
    clear_windows:
        Consecutive low windows required to consider the event over. Prevents a single
        bad frame in the middle of a fall from re-arming the system.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        confirm_windows: int = 1,
        cooldown_s: float = 30.0,
        clear_below: float = 0.3,
        clear_windows: int = 3,
    ) -> None:
        if confirm_windows < 1:
            raise ValueError("confirm_windows must be >= 1")
        if clear_windows < 1:
            raise ValueError("clear_windows must be >= 1")
        if clear_below >= threshold:
            raise ValueError("clear_below must be below threshold, otherwise the state oscillates")
        self.threshold = threshold
        self.confirm_windows = confirm_windows
        self.cooldown_s = cooldown_s
        self.clear_below = clear_below
        self.clear_windows = clear_windows
        self.reset()

    def reset(self) -> None:
        self._streak = 0
        self._low_streak = 0
        self._event_active = False
        self._alerted = False
        self._last_alert_s: float | None = None

    def update(self, timestamp_s: float, score: float) -> AlertDecision:
        if score >= self.threshold:
            self._streak += 1
            self._low_streak = 0
            self._event_active = True
        else:
            self._streak = 0
            self._low_streak += 1
            if self._low_streak >= self.clear_windows:
                self._event_active = False
                self._alerted = False

        cooled = (
            self._last_alert_s is None or timestamp_s - self._last_alert_s >= self.cooldown_s
        )
        should_alert = (
            self._event_active
            and not self._alerted
            and self._streak >= self.confirm_windows
            and cooled
        )
        if should_alert:
            self._alerted = True
            self._last_alert_s = timestamp_s
            reason = "fall_detected"
        elif self._event_active and self._alerted:
            reason = "already_alerted"
        elif self._event_active and self._streak >= self.confirm_windows and not cooled:
            # evidence is sufficient but a recent alert is still in its cooldown
            reason = "cooldown"
        elif self._event_active:
            reason = "collecting_evidence"
        else:
            reason = "no_event"

        return AlertDecision(
            alert=should_alert,
            reason=reason,
            score=float(score),
            timestamp_s=float(timestamp_s),
            streak=self._streak,
            event_active=self._event_active,
        )
