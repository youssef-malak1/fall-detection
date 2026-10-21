"""Inference: turn a video (or a stream of frames) into a fall decision.

This is the only module the rest of the ElderLife system should need. It owns the
whole chain — OpenCV motion features, MediaPipe pose, the trained TCN — and returns a
plain dataclass that serialises to JSON without surprises.

Design rules
------------
* **One code path.** Frames are turned into features by
  :func:`src.models.sequences.build_frame_matrix`, the exact function used to build the
  training data. Serving cannot drift away from evaluation.
* **No hidden calibration.** The normalisation statistics come from the checkpoint,
  not from whatever video happens to be analysed.
* **Honest metadata.** Latency, pose coverage and the per-window scores are returned
  so the caller can decide whether to trust the answer.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.features.pose import PoseExtractor, frame_features
from src.models.sequences import CHANNELS, apply_transform, build_frame_matrix, make_windows

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "fall_tcn.pt"

TARGET_FPS = 30.0
POSE_STRIDE = 2            # pose every 2nd frame -> 15 fps, as in training
EVAL_STRIDE = 12           # window stride at inference
MIN_AREA = 250
BG_SAMPLES = 50
DIFF_THRESHOLD = 28


@dataclass
class WindowScore:
    start_s: float
    end_s: float
    score: float


@dataclass
class FallAnalysis:
    is_fall: bool
    score: float
    threshold: float
    onset_s: float | None
    windows: list[WindowScore]
    frames: int
    fps: float
    duration_s: float
    pose_coverage: float
    latency_ms: float
    model: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["windows"] = [asdict(w) for w in self.windows]
        return data


def _resample_to_30fps(frames: list[np.ndarray], fps: float) -> list[np.ndarray]:
    """The model was trained at 30 fps; every other frame rate is resampled to match.

    Skipping this would silently change every speed-based channel (`drop_speed`,
    `d_hip_y`, ...) and the duration of a "3 second" window.
    """
    if not frames or abs(fps - TARGET_FPS) < 1.0:
        return frames
    count = max(1, int(round(len(frames) * TARGET_FPS / fps)))
    idx = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[i] for i in idx]


def motion_from_frames(frames: list[np.ndarray]) -> pd.DataFrame:
    """Same median-background pipeline as ``src.features.motion``, from memory.

    A camera is assumed to be static for the duration of the clip (true for URFD and
    for a fixed wall camera; a moving camera needs a real background model).
    """
    if not frames:
        return pd.DataFrame(columns=["frame", "detected", "height_ratio", "aspect",
                                     "centroid_y", "drop_speed"])

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    sample = grays
    if len(sample) > BG_SAMPLES:
        idx = np.linspace(0, len(sample) - 1, BG_SAMPLES).astype(int)
        sample = [sample[i] for i in idx]
    background = np.median(np.stack(sample), axis=0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    height, width = background.shape
    rows = []
    for idx, gray in enumerate(grays):
        diff = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected, x, y, w, h = False, 0.0, 0.0, 0.0, 0.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) >= MIN_AREA:
                detected = True
                x, y, w, h = cv2.boundingRect(contour)
        rows.append({"frame": idx, "detected": detected, "bbox_w": float(w),
                     "bbox_h": float(h), "bbox_x": float(x), "bbox_y": float(y)})

    df = pd.DataFrame(rows)
    df["aspect"] = np.where(df["bbox_h"] > 0, df["bbox_w"] / df["bbox_h"].replace(0, np.nan), 0.0)
    df["height_ratio"] = df["bbox_h"] / height
    df["centroid_y"] = (df["bbox_y"] + df["bbox_h"] / 2) / height
    df["drop_speed"] = df["centroid_y"].diff().fillna(0.0) * TARGET_FPS
    df["aspect"] = df["aspect"].fillna(0.0)
    return df


@dataclass
class _RawFrame:
    index: int
    image: np.ndarray


class FallDetector:
    """Loads the checkpoint once and reuses it across requests."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        device: str = "cpu",
        pose_upscale: int = 2,
    ) -> None:
        import torch

        from src.models.temporal import build_model

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"no checkpoint at {self.model_path} — run `python -m scripts.train_temporal`"
            )
        self.device = device
        checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        self.threshold = float(checkpoint.get("threshold", 0.5))
        self.fps = float(checkpoint.get("pose_fps", 15.0))
        self.window = int(checkpoint.get("window", 48))
        self.channels = checkpoint.get("channels", [n for n, _ in CHANNELS])
        if self.channels != [n for n, _ in CHANNELS]:
            raise ValueError(
                "checkpoint channel order does not match src.models.sequences.CHANNELS; "
                "features would be silently misinterpreted"
            )
        self.mean = np.array(checkpoint["mean"], dtype=np.float32)
        self.std = np.array(checkpoint["std"], dtype=np.float32)
        self.metadata = {
            "name": "elderlife-fall-tcn",
            "version": "1.0.0",
            "architecture": checkpoint.get("architecture", "FallTCN"),
            "trained_events": checkpoint.get("trained_on", {}),
            "decision_threshold": self.threshold,
            "pose_fps": self.fps,
            "window_frames": self.window,
            "window_seconds": self.window / self.fps,
        }

        self.model = build_model(len(self.channels)).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self._pose: PoseExtractor | None = None
        self._pose_upscale = pose_upscale

    # ------------------------------------------------------------------ pose helper
    def _pose_extractor(self) -> PoseExtractor:
        if self._pose is None:
            self._pose = PoseExtractor(upscale=self._pose_upscale, mode="image")
        return self._pose

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    # ------------------------------------------------------------------ public API
    def analyze_video(self, path: str | Path, layout: str = "camera") -> FallAnalysis:
        """Analyse a video file.

        ``layout="urfd"`` crops the left (depth) half of the side-by-side URFD
        composites; ``layout="camera"`` (default) expects ordinary RGB frames.
        """
        started = time.perf_counter()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or TARGET_FPS
        frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if layout == "urfd":
                    frame = frame[:, frame.shape[1] // 2 :]
                frames.append(frame)
        finally:
            cap.release()
        analysis = self._analyze(frames, fps=fps)
        analysis.latency_ms = (time.perf_counter() - started) * 1000.0
        return analysis

    def analyze_frames(
        self, frames: list[np.ndarray], fps: float = TARGET_FPS
    ) -> FallAnalysis:
        """Analyse frames already decoded in memory (for a camera/streaming adapter)."""
        started = time.perf_counter()
        analysis = self._analyze(frames, fps=fps)
        analysis.latency_ms = (time.perf_counter() - started) * 1000.0
        return analysis

    # ------------------------------------------------------------------ internals
    def _analyze(self, frames: list[np.ndarray], fps: float) -> FallAnalysis:
        import torch

        if not frames:
            raise ValueError("empty video: nothing to analyse")

        frames = _resample_to_30fps(frames, fps)
        fps = TARGET_FPS
        motion = motion_from_frames(frames)

        extractor = self._pose_extractor()
        pose_rows = []
        for i in range(0, len(frames), POSE_STRIDE):
            pose = extractor.process(frames[i], int(i / fps * 1000))
            row = {"frame": i}
            row.update(frame_features(pose.landmarks, pose.visibility))
            pose_rows.append(row)
        pose_df = pd.DataFrame(pose_rows)

        features, mask = build_frame_matrix(pose_df, motion)
        features = apply_transform(features, self.mean, self.std)

        windows = make_windows(features, mask, self.window, EVAL_STRIDE)
        X = np.stack([w[0] for w in windows]).astype(np.float32)
        M = np.stack([w[1] for w in windows]).astype(np.float32)
        with torch.no_grad():
            probs = torch.sigmoid(
                self.model(torch.from_numpy(X).to(self.device), torch.from_numpy(M).to(self.device))
            ).cpu().numpy()

        starts = [0] if len(features) <= self.window else list(
            range(0, len(features) - self.window + 1, EVAL_STRIDE)
        )
        if len(features) > self.window and starts[-1] != len(features) - self.window:
            starts.append(len(features) - self.window)

        scored = [
            WindowScore(
                start_s=start * POSE_STRIDE / fps,
                end_s=(start + self.window) * POSE_STRIDE / fps,
                score=float(p),
            )
            for start, p in zip(starts, probs)
        ]
        best = max(scored, key=lambda w: w.score)

        return FallAnalysis(
            is_fall=bool(best.score >= self.threshold),
            score=float(best.score),
            threshold=self.threshold,
            onset_s=best.start_s,
            windows=scored,
            frames=len(frames),
            fps=fps,
            duration_s=len(frames) / fps,
            pose_coverage=float(mask.mean()),
            latency_ms=0.0,
            model=self.metadata,
        )
