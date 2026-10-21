"""Pose extraction with MediaPipe Tasks (PoseLandmarker).

MediaPipe gives 33 landmarks per person in normalized image coordinates
(x, y in [0, 1], z relative depth) plus a visibility score per point.

The URFD RGB frames are only 320x240 and the person can be ~45 px tall, which is
at the edge of what the landmarker can do. Frames are therefore upscaled before
inference; the returned coordinates stay normalized, so the scale does not matter.

Used as a *frozen* feature extractor: we never train or fine-tune this network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"

# MediaPipe pose landmark indices we care about.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

KEY_LANDMARKS = {
    "nose": NOSE,
    "l_shoulder": LEFT_SHOULDER,
    "r_shoulder": RIGHT_SHOULDER,
    "l_hip": LEFT_HIP,
    "r_hip": RIGHT_HIP,
    "l_knee": LEFT_KNEE,
    "r_knee": RIGHT_KNEE,
    "l_ankle": LEFT_ANKLE,
    "r_ankle": RIGHT_ANKLE,
}


@dataclass
class PoseFrame:
    detected: bool
    score: float
    landmarks: np.ndarray | None  # (33, 3) normalized x, y, z
    visibility: np.ndarray | None = None


class PoseExtractor:
    """Thin wrapper around ``vision.PoseLandmarker`` in VIDEO mode."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        upscale: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mode: str = "video",
    ) -> None:
        if mode not in ("video", "image"):
            raise ValueError("mode must be 'video' or 'image'")
        self.mode = mode
        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        self._options = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=(
                vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
            ),
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(self._options)
        self.upscale = upscale
        self._last_ts = -1
        self._ts_offset = 0

    def reset(self) -> None:
        """Drop the tracker state between clips.

        The landmarker keeps an internal tracker between ``detect_for_video`` calls.
        Reusing one instance across different videos makes it carry the previous
        clip's state into the next one, which produced frozen, anatomically
        impossible skeletons at the bottom of the frame. Recreating it per clip is
        the difference between valid data and garbage that still *looks* like data.
        """
        self._landmarker.close()
        self._landmarker = vision.PoseLandmarker.create_from_options(self._options)
        self._last_ts = -1
        self._ts_offset = 0

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseFrame:
        """Run the landmarker on one frame.

        VIDEO mode requires strictly increasing timestamps. Clips are processed one
        after another with timestamps that restart at 0, so an offset is applied that
        **preserves the real spacing between frames** (33 ms at 30 fps). Getting this
        wrong makes the tracker believe no time passes, and it returns frozen garbage
        poses instead of failing loudly.
        """
        ts = timestamp_ms + self._ts_offset
        if ts <= self._last_ts:
            self._ts_offset += (self._last_ts - ts) + 1
            ts = timestamp_ms + self._ts_offset
        self._last_ts = ts

        if self.upscale > 1:
            frame_bgr = cv2.resize(
                frame_bgr,
                None,
                fx=self.upscale,
                fy=self.upscale,
                interpolation=cv2.INTER_CUBIC,
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "video":
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)

        if not result.pose_landmarks:
            return PoseFrame(False, 0.0, None, None)

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks[0]], dtype=np.float32
        )
        visibility = np.array([lm.visibility for lm in result.pose_landmarks[0]])
        return PoseFrame(True, float(visibility.mean()), landmarks, visibility)

    def close(self) -> None:
        self._landmarker.close()


def frame_features(
    landmarks: np.ndarray | None,
    visibility: np.ndarray | None,
    min_conf: float = 0.5,
    min_torso: float = 0.02,
) -> dict:
    """Derive interpretable posture features from one pose.

    Shared by offline extraction and the inference service, so a feature cannot mean
    one thing in training and another in production.

    Poses that are anatomically implausible are reported as ``detected = 0``: a
    hallucinated skeleton (a chair read as a person) is worse than no skeleton, because
    the rule would happily track it. ``raw_detected`` keeps the original answer for
    debugging.
    """
    out = {
        "raw_detected": 0,
        "detected": 0,
        "conf": 0.0,
        "torso_angle": np.nan,
        "torso_len": np.nan,
        "hip_y": np.nan,
        "shoulder_y": np.nan,
        "ankle_y": np.nan,
        "head_y": np.nan,
        "pose_h": np.nan,
        "pose_w": np.nan,
        "body_aspect": np.nan,
    }
    if landmarks is None:
        return out

    out["raw_detected"] = 1
    mean_vis = float(np.mean(visibility)) if visibility is not None else 0.0
    out["conf"] = mean_vis

    shoulder = (landmarks[LEFT_SHOULDER, :2] + landmarks[RIGHT_SHOULDER, :2]) / 2
    hip = (landmarks[LEFT_HIP, :2] + landmarks[RIGHT_HIP, :2]) / 2
    ankle = (landmarks[LEFT_ANKLE, :2] + landmarks[RIGHT_ANKLE, :2]) / 2

    vec = hip - shoulder  # image coords: y grows downwards
    torso_len = float(np.linalg.norm(vec))
    # 0 deg = perfectly upright, 90 deg = perfectly horizontal
    out["torso_angle"] = float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)))
    out["torso_len"] = torso_len
    out["hip_y"] = float(hip[1])
    out["shoulder_y"] = float(shoulder[1])
    out["ankle_y"] = float(ankle[1])
    out["head_y"] = float(landmarks[NOSE, 1])

    visible = landmarks[:, :2][visibility > 0.5] if visibility is not None else landmarks[:, :2]
    if len(visible) > 3:
        out["pose_h"] = float(visible[:, 1].max() - visible[:, 1].min())
        out["pose_w"] = float(visible[:, 0].max() - visible[:, 0].min())
        if out["pose_h"] > 1e-6:
            out["body_aspect"] = out["pose_w"] / out["pose_h"]

    plausible = mean_vis >= min_conf and torso_len >= min_torso
    out["detected"] = 1 if plausible else 0
    return out
