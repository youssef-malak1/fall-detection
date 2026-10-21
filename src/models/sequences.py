"""Per-frame, multi-modal sequences for the temporal fall model.

The baseline in :mod:`src.eval.dataset` collapses every clip into a handful of
summary numbers, which throws away the *order* of events. A fall is defined by its
order: upright → fast descent → stays on the floor. This module keeps that order.

Two views of the same person are fused per frame:

* **pose** (MediaPipe, 15 fps) — where the body parts are;
* **motion** (OpenCV median background, 30 fps) — where the moving blob is, which
  still works when the pose model loses the person.

Pose frames are the timeline (every other motion frame), so the two are aligned by
frame index rather than by assuming a frame rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSE_DIR = PROJECT_ROOT / "data" / "interim" / "pose"
MOTION_DIR = PROJECT_ROOT / "data" / "interim" / "motion"

#: Channels fed to the model, in this exact order. Kept as a list of
#: ``(name, source)`` so a saved model can always be matched to its inputs.
CHANNELS: list[tuple[str, str]] = [
    ("detected", "pose"),
    ("conf", "pose"),
    ("torso_angle", "pose"),
    ("hip_y", "pose"),
    ("shoulder_y", "pose"),
    ("head_y", "pose"),
    ("ankle_y", "pose"),
    ("torso_len", "pose"),
    ("body_aspect", "pose"),
    ("torso_vertical", "pose"),  # hip_y - shoulder_y : how tall the torso looks
    ("leg_extension", "pose"),  # ankle_y - hip_y   : are the legs under the body
    ("d_hip_y", "pose"),
    ("d_shoulder_y", "pose"),
    ("d_torso_angle", "pose"),
    ("d_torso_len", "pose"),
    ("height_ratio", "motion"),
    ("aspect", "motion"),
    ("centroid_y", "motion"),
    ("drop_speed", "motion"),
]

POSE_BASE = ["torso_angle", "hip_y", "shoulder_y", "head_y", "ankle_y", "torso_len", "body_aspect"]
MOTION_BASE = ["height_ratio", "aspect", "centroid_y", "drop_speed"]
POSE_DELTAS = ["hip_y", "shoulder_y", "torso_angle", "torso_len"]


def load_pose_meta() -> dict:
    path = POSE_DIR / "meta.json"
    if not path.exists():
        return {"fps": 15.0, "stride": 2}
    return json.loads(path.read_text(encoding="utf-8"))


def _fill_gaps(series: pd.Series) -> pd.Series:
    """Fill short detection holes without inventing long trajectories.

    A pose that is missing for a frame or two is usually a detection dropout and
    interpolating it is honest. A pose missing for two seconds is a real absence of
    information, so the mask (not the value) is what the model should see.
    """
    return series.interpolate(method="linear", limit=4, limit_direction="both").bfill().ffill()


def build_frame_matrix(
    pose: pd.DataFrame, motion: pd.DataFrame | None
) -> tuple[np.ndarray, np.ndarray]:
    """Turn raw per-frame pose/motion tables into ``(features, mask)``.

    Shared by offline extraction (from CSV) and the inference service (from frames
    held in memory), so serving and evaluation cannot drift apart.
    """
    frames = pose["frame"].astype(int)

    if motion is not None:
        motion = motion.set_index("frame").reindex(frames)
        motion_ok = motion["detected"].fillna(False).astype(bool).to_numpy()
    else:  # pragma: no cover - every URFD clip has motion features
        motion = pd.DataFrame(index=frames)
        motion_ok = np.zeros(len(frames), dtype=bool)

    cols: dict[str, np.ndarray] = {}

    pose_detected = pose["detected"].astype(bool).to_numpy()
    cols["detected"] = pose_detected.astype(float)
    cols["conf"] = pose["conf"].fillna(0.0).to_numpy(dtype=float)

    filled = {}
    for name in POSE_BASE:
        s = _fill_gaps(pd.to_numeric(pose[name], errors="coerce"))
        filled[name] = s
        cols[name] = s.to_numpy(dtype=float)

    cols["torso_vertical"] = (filled["hip_y"] - filled["shoulder_y"]).to_numpy(dtype=float)
    cols["leg_extension"] = (filled["ankle_y"] - filled["hip_y"]).to_numpy(dtype=float)

    for name in POSE_DELTAS:
        # per-frame change, in units per frame; sign matters (a drop is negative)
        cols[f"d_{name}"] = filled[name].diff().fillna(0.0).to_numpy(dtype=float)

    for name in MOTION_BASE:
        values = pd.to_numeric(motion.get(name, pd.Series(np.nan, index=frames)), errors="coerce")
        cols[name] = _fill_gaps(values).to_numpy(dtype=float)

    features = np.stack([cols[name] for name, _ in CHANNELS], axis=1).astype(np.float32)
    mask = (pose_detected & motion_ok).astype(np.float32)

    # A NaN anywhere is a bug upstream, not something to paper over at training time.
    if np.isnan(features).any():
        bad = [name for (name, _), col in zip(CHANNELS, features.T) if np.isnan(col).any()]
        raise ValueError(f"unfilled NaN in channels {bad}")

    return features, mask


def load_sequence(clip: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features, mask)`` for one clip from the cached CSVs."""
    pose = pd.read_csv(POSE_DIR / f"{clip}.csv")
    motion_path = MOTION_DIR / f"{clip}.csv"
    motion = pd.read_csv(motion_path) if motion_path.exists() else None
    return build_frame_matrix(pose, motion)


def normalise_transform(sequences: dict[str, np.ndarray], masks: dict[str, np.ndarray]):
    """Per-channel mean/std over the *valid frames* of the training fold.

    Fitting the scaler on training folds only is the difference between a real
    cross-validation and a leak.
    """
    stacked = np.concatenate(list(sequences.values()), axis=0)
    valid = np.concatenate(list(masks.values()), axis=0) > 0.5
    if not valid.any():  # pragma: no cover - degenerate fold
        # No valid frame anywhere: there is nothing to normalise against. Falling back
        # to raw statistics is wrong but at least deterministic; the mask bug that
        # used to live here silently *ignored* the mask for small folds instead.
        raise ValueError("cannot fit normalisation: no valid frames in this fold")
    reference = stacked[valid]
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # 5 sigma is far beyond anything the model should care about; clipping keeps a
    # single misdetected frame from dominating the input.
    return np.clip((x - mean) / std, -5.0, 5.0).astype(np.float32)


def impute_masked(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill non-valid frames by holding the last known value (zero if none yet)."""
    out = x.copy()
    valid = mask > 0.5
    if not valid.any():
        return np.zeros_like(out)
    # forward fill: every frame borrows from the most recent valid frame; frames
    # before the first detection have nothing to borrow from and become zero.
    idx = np.where(valid, np.arange(len(out)), -1)
    np.maximum.accumulate(idx, out=idx)
    out = out[np.clip(idx, 0, None)]
    out[idx < 0] = 0.0
    return out


def build_sequences(clips: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    seqs, masks = {}, {}
    for clip in clips:
        x, m = load_sequence(clip)
        seqs[clip] = x
        masks[clip] = m
    return seqs, masks


def make_windows(
    x: np.ndarray,
    mask: np.ndarray,
    window: int,
    stride: int,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut a clip into fixed-length windows (padding the last one).

    Why windows at all
    ------------------
    Clip length is a giveaway in URFD: ADL clips average 112 frames, falls average 50,
    because recordings were stopped shortly after the event. A classifier that sees the
    whole clip can exploit that (`length alone scores F1 0.82`), which would collapse in
    deployment where clip length is arbitrary. Windows of a fixed length remove the
    shortcut and give the model one job: decide whether *this 3-second view* contains a
    fall. The clip score is then the max over its windows.

    ``rng`` given -> sample a single random window (training augmentation).
    ``rng`` None -> return every window at the requested stride (evaluation).
    """
    if len(x) <= window:
        pad = window - len(x)
        return [
            (
                np.pad(x, ((0, pad), (0, 0)), mode="constant"),
                np.pad(mask, (0, pad), mode="constant"),
            )
        ]

    if rng is not None:
        start = int(rng.integers(0, len(x) - window + 1))
        starts = [start]
    else:
        starts = list(range(0, len(x) - window + 1, stride))
        if starts[-1] != len(x) - window:
            starts.append(len(x) - window)

    return [(x[s : s + window], mask[s : s + window]) for s in starts]
