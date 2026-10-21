"""Rule-based fall detector — the honest baseline.

The rule encodes the operational definition used across the project:

    upright  --(fast)-->  horizontal on the floor  --(stays)-->  fall

It deliberately does **not** fire on the final posture alone. An elderly person may
lie down on the floor on purpose (URFD ``adl-30`` does exactly that), so what
separates a fall from a voluntary movement is the **duration of the transition**:

    a fall crosses from upright to horizontal in well under a second,
    lying down on purpose takes several seconds.

Signals used
------------
``torso_angle``  from pose landmarks: ~0 deg upright, ~90 deg horizontal.
``hip_y``        normalised vertical hip position (0 top, 1 bottom).
``detected``     whether the landmark model found a person in this frame.

If the pose is lost right after a fast descent, that is treated as **evidence**, not
as missing data: the landmark model is known to fail once a person is on the floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RuleParams:
    fps: int = 30
    smooth: int = 5
    upright_angle: float = 25.0
    horizontal_angle: float = 70.0
    max_transition_s: float = 1.2
    min_hip_drop: float = 0.08
    min_lying_s: float = 1.0
    search_back_s: float = 5.0
    allow_pose_loss: bool = True


@dataclass
class Detection:
    is_fall: bool = False
    onset_frame: int | None = None
    reason: str = ""
    evidence: dict = field(default_factory=dict)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) of consecutive True runs."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def detect_fall(pose: pd.DataFrame, params: RuleParams | None = None) -> Detection:
    """Run the state machine on one clip's per-frame pose features."""
    p = params or RuleParams()
    if pose.empty or int(pose["detected"].sum()) == 0:
        return Detection(reason="no pose detected")

    angle = pose["torso_angle"].astype(float).rolling(p.smooth, min_periods=1).mean()
    hip_y = pose["hip_y"].astype(float).rolling(p.smooth, min_periods=1).mean()
    detected = pose["detected"].astype(bool).to_numpy()

    horizontal = (angle >= p.horizontal_angle).fillna(False).to_numpy()
    upright = (angle <= p.upright_angle).fillna(False).to_numpy()

    window = int(round(p.max_transition_s * p.fps))
    search_back = int(round(p.search_back_s * p.fps))
    min_lying = int(round(p.min_lying_s * p.fps))

    best: Detection | None = None

    for start, end in _runs(horizontal):
        lookback = max(0, start - window)
        if not upright[lookback:start].any():
            # nobody was upright within the allowed transition time -> too slow
            continue

        upright_idx = max(i for i in range(lookback, start) if upright[i])

        hip_before = hip_y.iloc[upright_idx]
        hip_after = hip_y.iloc[min(len(hip_y) - 1, start + min_lying)]
        if np.isnan(hip_before) or np.isnan(hip_after):
            continue

        drop = float(hip_after - hip_before)
        if drop < p.min_hip_drop:
            continue

        duration = end - start
        tail = detected[min(len(detected), start + duration) :]
        lost_after = p.allow_pose_loss and len(tail) > 0 and not tail.any()

        if duration < min_lying and not lost_after:
            continue

        detection = Detection(
            is_fall=True,
            onset_frame=int(start),
            reason="fast upright->horizontal transition with hip drop",
            evidence={
                "transition_frames": int(start - upright_idx),
                "hip_drop": round(drop, 3),
                "horizontal_frames": int(duration),
                "pose_lost_after": bool(lost_after),
            },
        )
        if best is None or (detection.onset_frame or 0) < (best.onset_frame or 0):
            best = detection

    if best is not None:
        return best

    if p.allow_pose_loss:
        fallback = _pose_loss_after_descent(pose, p)
        if fallback is not None:
            return fallback

    return Detection(reason="no rule fired")


def _pose_loss_after_descent(pose: pd.DataFrame, p: RuleParams) -> Detection | None:
    """Person collapses towards the camera and the landmark model loses them."""
    detected = pose["detected"].astype(bool).to_numpy()
    if detected.all() or not detected.any():
        return None

    loss_idx = int(np.argmax(detected)) if not detected[0] else int(len(detected) - 1)
    if detected[loss_idx:].any():
        return None  # person came back -> not a permanent loss

    hip_y = pose["hip_y"].astype(float).rolling(p.smooth, min_periods=1).mean()
    window = int(round(p.max_transition_s * p.fps))
    before = hip_y.iloc[max(0, loss_idx - window) : loss_idx].dropna()
    if len(before) < 3:
        return None

    drop = float(before.iloc[-1] - before.iloc[0])
    if drop < p.min_hip_drop:
        return None

    return Detection(
        is_fall=True,
        onset_frame=loss_idx,
        reason="pose lost after fast hip descent",
        evidence={"hip_drop": round(drop, 3), "pose_lost_at": loss_idx},
    )
