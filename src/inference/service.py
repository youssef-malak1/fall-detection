"""REST service exposing fall detection to the rest of the ElderLife system.

    uvicorn src.inference.service:app --host 0.0.0.0 --port 8100

Endpoints
---------
``GET  /health``  liveness + whether a checkpoint is loaded
``GET  /model``   model metadata (version, threshold, training set size)
``POST /analyze`` multipart video upload -> fall decision as JSON
``POST /analyze/urfd`` same, but crops the depth half of URFD side-by-side files

The JSON contract is stable and versioned (``schema_version``) so the backend can code
against it without importing anything from this repository.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.inference.pipeline import FallDetector

SCHEMA_VERSION = "1.0"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB: a few minutes of compressed video

app = FastAPI(
    title="ElderLife Fall Detection",
    version="1.0.0",
    description="Computer-vision fall detection (pose + motion temporal model).",
)

_detector: FallDetector | None = None
_lock = threading.Lock()


def get_detector() -> FallDetector:
    """Load the model on first use, once, even under concurrent requests."""
    global _detector
    if _detector is None:
        with _lock:
            if _detector is None:
                _detector = FallDetector()
    return _detector


# --------------------------------------------------------------------------- schemas
class WindowScoreOut(BaseModel):
    start_s: float = Field(..., description="window start, seconds from clip start")
    end_s: float
    score: float = Field(..., ge=0.0, le=1.0, description="probability this window contains a fall")


class ModelInfo(BaseModel):
    name: str
    version: str
    architecture: str
    decision_threshold: float
    window_seconds: float
    pose_fps: float
    trained_events: dict


class AnalysisOut(BaseModel):
    schema_version: str = SCHEMA_VERSION
    is_fall: bool
    score: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    confidence: str = Field(..., description="high | medium | low, derived from score")
    onset_s: float | None = Field(..., description="start of the most fall-like window")
    windows: list[WindowScoreOut]
    frames: int
    fps: float
    duration_s: float
    pose_coverage: float = Field(..., description="fraction of frames where the person was tracked")
    latency_ms: float
    model: ModelInfo


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    schema_version: str


def _confidence(score: float, threshold: float) -> str:
    """Coarse bucket for the alerting layer; deliberately conservative at the boundary."""
    if score >= max(0.8, threshold + 0.25):
        return "high"
    if score >= max(0.6, threshold + 0.05):
        return "medium"
    return "low"


def _payload(analysis) -> AnalysisOut:
    return AnalysisOut(
        is_fall=analysis.is_fall,
        score=analysis.score,
        threshold=analysis.threshold,
        confidence=_confidence(analysis.score, analysis.threshold),
        onset_s=analysis.onset_s,
        windows=[WindowScoreOut(**w.__dict__) for w in analysis.windows],
        frames=analysis.frames,
        fps=analysis.fps,
        duration_s=analysis.duration_s,
        pose_coverage=analysis.pose_coverage,
        latency_ms=analysis.latency_ms,
        model=ModelInfo(**analysis.model),
    )


# --------------------------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(
        status="ok",
        model_loaded=_detector is not None,
        schema_version=SCHEMA_VERSION,
    )


@app.get("/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    try:
        return ModelInfo(**get_detector().metadata)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/analyze", response_model=AnalysisOut)
async def analyze(
    file: UploadFile = File(..., description="video file (mp4/avi/mov)"),
    layout: str = Query("camera", pattern="^(camera|urfd)$"),
) -> AnalysisOut:
    """Analyse an uploaded video and return a fall decision."""
    return await _analyze_upload(file, layout)


@app.post("/analyze/urfd", response_model=AnalysisOut)
async def analyze_urfd(file: UploadFile = File(...)) -> AnalysisOut:
    """Convenience wrapper for the URFD side-by-side (depth | RGB) layout."""
    return await _analyze_upload(file, "urfd")


async def _analyze_upload(file: UploadFile, layout: str) -> AnalysisOut:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")

    try:
        detector = get_detector()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        analysis = detector.analyze_video(tmp_path, layout=layout)
    except IOError as exc:
        raise HTTPException(status_code=422, detail=f"could not decode video: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return _payload(analysis)
