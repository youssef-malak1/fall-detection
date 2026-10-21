"""Motion / posture features from plain OpenCV (no deep model).

Why this exists
---------------
70 events and 8 minutes of footage is not enough to train a video network. The
first model therefore has to be a transparent, cheap, classical baseline. It also
gives us the feature vocabulary the pose model will later replace:

  * ``aspect``      - bounding-box width / height. > 1 means a *horizontal* body.
  * ``height_ratio``- bbox height / frame height. Falls shrink this fast.
  * ``centroid_y``  - normalised vertical position. The floor is at the bottom.
  * ``drop_speed``  - how fast the centroid moves down.
  * ``extent``      - fraction of the bbox actually covered by the person.

The camera is static (URFD was recorded from a tripod), so the background can be
estimated per clip with a median over time. That is more robust than MOG2 on
clips that are only 2-13 seconds long.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.video import iter_rgb_frames

MIN_AREA = 250          # px, ignore noise
BG_SAMPLES = 50         # frames used for the median background
DIFF_THRESHOLD = 28     # gray-level difference that counts as foreground


@dataclass
class FrameFeatures:
    frame: int
    detected: bool
    area: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float


def estimate_background(path: str | Path) -> np.ndarray:
    """Per-pixel median background (grayscale) sampled across the clip."""
    frames = list(iter_rgb_frames(path))
    if not frames:
        raise IOError(f"no frames in {path}")
    if len(frames) > BG_SAMPLES:
        idx = np.linspace(0, len(frames) - 1, BG_SAMPLES).astype(int)
        frames = [frames[i] for i in idx]
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    return np.median(stack, axis=0).astype(np.uint8)


def _largest_person(mask: np.ndarray) -> tuple[bool, float, tuple[float, float, float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, (0.0, 0.0, 0.0, 0.0)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < MIN_AREA:
        return False, 0.0, (0.0, 0.0, 0.0, 0.0)
    x, y, w, h = cv2.boundingRect(contour)
    return True, area, (float(x), float(y), float(w), float(h))


def extract_frame_features(path: str | Path) -> pd.DataFrame:
    """Return one row per frame with bbox / posture descriptors."""
    background = estimate_background(path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    rows: list[FrameFeatures] = []
    for idx, frame in enumerate(iter_rgb_frames(path)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        detected, area, (x, y, w, h) = _largest_person(mask)
        rows.append(FrameFeatures(idx, detected, area, x, y, w, h))

    df = pd.DataFrame([r.__dict__ for r in rows])
    height = background.shape[0]
    width = background.shape[1]

    df["aspect"] = np.where(df["bbox_h"] > 0, df["bbox_w"] / df["bbox_h"].replace(0, np.nan), 0.0)
    df["height_ratio"] = df["bbox_h"] / height
    df["centroid_x"] = (df["bbox_x"] + df["bbox_w"] / 2) / width
    df["centroid_y"] = (df["bbox_y"] + df["bbox_h"] / 2) / height
    df["extent"] = np.where(
        (df["bbox_w"] * df["bbox_h"]) > 0,
        df["area"] / (df["bbox_w"] * df["bbox_h"]),
        0.0,
    )
    df["drop_speed"] = df["centroid_y"].diff().fillna(0.0) * 30.0  # per second @30fps
    df["aspect"] = df["aspect"].fillna(0.0)
    return df


def summarise(df: pd.DataFrame) -> dict[str, float]:
    """Clip-level summary used by the rule baseline and for quick EDA."""
    detected = df[df["detected"]]
    if detected.empty:
        return {
            "detected_frames": 0,
            "detected_ratio": 0.0,
            "max_horizontal_run": 0,
            "max_lying_run": 0,
            "max_drop_speed": 0.0,
            "final_centroid_y": 0.0,
            "final_aspect": 0.0,
            "min_height_ratio": 1.0,
        }

    horizontal = (detected["aspect"] > 1.0).to_numpy()
    lying = ((detected["aspect"] > 1.0) & (detected["centroid_y"] > 0.62)).to_numpy()

    return {
        "detected_frames": int(len(detected)),
        "detected_ratio": float(len(detected) / len(df)),
        "max_horizontal_run": int(_longest_run(horizontal)),
        "max_lying_run": int(_longest_run(lying)),
        "max_drop_speed": float(detected["drop_speed"].max()),
        "final_centroid_y": float(detected["centroid_y"].iloc[-1]),
        "final_aspect": float(detected["aspect"].iloc[-1]),
        "min_height_ratio": float(detected["height_ratio"].min()),
    }


def _longest_run(flags: np.ndarray) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best
