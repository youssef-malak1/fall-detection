"""Video reading utilities for URFD.

The URFD MP4 files are side-by-side composites:

    +-------------------+-------------------+
    |  depth  (320x240) |   RGB   (320x240) |
    +-------------------+-------------------+

Only the RGB half is used by this project. Everything downstream must go through
:func:`rgb_half` so the crop is applied exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "urfd"


def rgb_half(frame: np.ndarray) -> np.ndarray:
    """Return the RGB half of a URFD composite frame."""
    width = frame.shape[1]
    return frame[:, width // 2 :]


def depth_half(frame: np.ndarray) -> np.ndarray:
    """Return the depth half of a URFD composite frame (unused for now)."""
    width = frame.shape[1]
    return frame[:, : width // 2]


def iter_rgb_frames(path: str | Path, step: int = 1) -> Iterator[np.ndarray]:
    """Yield RGB frames of a URFD clip, cropping the depth half."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield rgb_half(frame)
            idx += 1
    finally:
        cap.release()


def count_frames(path: str | Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n
