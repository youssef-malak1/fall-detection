"""Event-level dataset assembly and summary features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOTION_DIR = PROJECT_ROOT / "data" / "interim" / "motion"
POSE_DIR = PROJECT_ROOT / "data" / "interim" / "pose"
PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_pose_meta() -> dict:
    """Metadata written by the pose extraction (effective fps, stride, model)."""
    path = POSE_DIR / "meta.json"
    if not path.exists():
        return {"fps": 30.0, "stride": 1}
    return json.loads(path.read_text(encoding="utf-8"))


def load_motion_summary() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "motion_summary.csv")


def load_pose(clip: str) -> pd.DataFrame:
    return pd.read_csv(POSE_DIR / f"{clip}.csv")


def load_motion(clip: str) -> pd.DataFrame:
    return pd.read_csv(MOTION_DIR / f"{clip}.csv")


def pose_summary(pose: pd.DataFrame, fps: float = 30) -> dict:
    detected = pose["detected"].astype(bool)
    if not detected.any():
        return {
            "pose_detected_ratio": 0.0,
            "torso_angle_mean": np.nan,
            "torso_angle_max": np.nan,
            "torso_angle_std": np.nan,
            "hip_y_max": np.nan,
            "hip_y_range": np.nan,
            "max_hip_speed": np.nan,
            "body_aspect_max": np.nan,
            "conf_mean": 0.0,
            "longest_horizontal_run": 0,
            "pose_lost_permanently": 0,
        }

    view = pose[detected]
    # ~0.17 s of smoothing, expressed in frames so it survives an fps change
    win = max(3, int(round(0.17 * fps)))
    angle = view["torso_angle"].astype(float).rolling(win, min_periods=1).mean()
    hip = view["hip_y"].astype(float).rolling(win, min_periods=1).mean()
    speed = hip.diff().fillna(0.0) * fps

    horizontal = (angle >= 55).to_numpy()
    return {
        "pose_detected_ratio": float(detected.mean()),
        "torso_angle_mean": float(angle.mean()),
        "torso_angle_max": float(angle.max()),
        "torso_angle_std": float(angle.std()),
        "hip_y_max": float(hip.max()),
        "hip_y_range": float(hip.max() - hip.min()),
        "max_hip_speed": float(speed.abs().max()),
        "body_aspect_max": float(view["body_aspect"].max(skipna=True)),
        "conf_mean": float(view["conf"].mean()),
        "longest_horizontal_run": int(_longest_run(horizontal)),
        "pose_lost_permanently": int(not detected.iloc[-1] and detected.iloc[: max(1, len(detected) // 2)].any()),
    }


def _longest_run(flags: np.ndarray) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def build_feature_table(camera: int = 0) -> pd.DataFrame:
    """One row per clip: motion summary + pose summary + label."""
    motion = load_motion_summary()
    motion = motion[motion["camera"] == camera].copy()
    fps = float(load_pose_meta()["fps"])

    rows = []
    for row in motion.to_dict("records"):
        clip = Path(row["file"]).stem
        pose_path = POSE_DIR / f"{clip}.csv"
        if not pose_path.exists():
            continue
        summary = pose_summary(load_pose(clip), fps=fps)
        summary.update(row)
        summary["clip_id"] = clip
        rows.append(summary)

    df = pd.DataFrame(rows)
    df["label"] = (df["class"] == "fall").astype(int)
    return df


FEATURES = [
    # motion
    "detected_ratio",
    "max_horizontal_run",
    "max_lying_run",
    "max_drop_speed",
    "final_centroid_y",
    "final_aspect",
    "min_height_ratio",
    # pose
    "pose_detected_ratio",
    "torso_angle_mean",
    "torso_angle_max",
    "torso_angle_std",
    "hip_y_max",
    "hip_y_range",
    "max_hip_speed",
    "body_aspect_max",
    "conf_mean",
    "longest_horizontal_run",
    "pose_lost_permanently",
]
