"""Tests for the inference pipeline and the REST contract.

The heavy tests need the trained checkpoint and the URFD clip, so they skip cleanly on a
fresh clone; the contract tests do not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT = PROJECT_ROOT / "models" / "fall_tcn.pt"
FALL_CLIP = PROJECT_ROOT / "data" / "raw" / "urfd" / "cam0" / "fall-14-cam0.mp4"
ADL_CLIP = PROJECT_ROOT / "data" / "raw" / "urfd" / "cam0" / "adl-26-cam0.mp4"

needs_model = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="no trained checkpoint (run scripts.train_temporal)"
)
needs_clips = pytest.mark.skipif(
    not (FALL_CLIP.exists() and ADL_CLIP.exists()), reason="URFD clips not downloaded"
)


@pytest.fixture(scope="module")
def detector():
    from src.inference.pipeline import FallDetector

    d = FallDetector()
    yield d
    d.close()


def test_motion_features_from_frames_contract():
    from src.inference.pipeline import motion_from_frames

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(5)]
    df = motion_from_frames(frames)
    for col in ("frame", "detected", "height_ratio", "aspect", "centroid_y", "drop_speed"):
        assert col in df.columns
    assert len(df) == 5
    assert not df[["height_ratio", "aspect", "centroid_y", "drop_speed"]].isna().any().any()


def test_empty_input_is_rejected():
    from src.inference.pipeline import motion_from_frames

    assert len(motion_from_frames([])) == 0


def test_resampling_leaves_30fps_untouched():
    from src.inference.pipeline import _resample_to_30fps

    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(10)]
    assert len(_resample_to_30fps(frames, 30.0)) == 10
    assert len(_resample_to_30fps(frames, 15.0)) == 20


@needs_model
def test_checkpoint_channel_order_matches_code(detector):
    from src.models.sequences import CHANNELS

    assert detector.channels == [name for name, _ in CHANNELS]


@needs_model
@needs_clips
def test_real_fall_is_detected(detector):
    analysis = detector.analyze_video(FALL_CLIP, layout="urfd")
    assert analysis.is_fall, f"missed a real fall (score {analysis.score:.3f})"
    assert analysis.score > 0.5
    assert 0.0 <= analysis.pose_coverage <= 1.0
    assert analysis.onset_s is not None


@needs_model
@needs_clips
def test_real_adl_is_not_flagged(detector):
    analysis = detector.analyze_video(ADL_CLIP, layout="urfd")
    assert not analysis.is_fall, f"false alarm on a normal activity (score {analysis.score:.3f})"


@needs_model
@needs_clips
def test_analysis_serialises_to_plain_json_types(detector):
    import json

    analysis = detector.analyze_video(FALL_CLIP, layout="urfd")
    payload = analysis.to_dict()
    text = json.dumps(payload)  # raises if anything is a numpy scalar
    assert '"is_fall"' in text
    assert isinstance(payload["windows"], list)


@needs_model
def test_health_and_model_endpoints():
    from fastapi.testclient import TestClient

    from src.inference.service import app

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    info = client.get("/model")
    assert info.status_code == 200
    body = info.json()
    assert body["name"] == "elderlife-fall-tcn"
    assert 0.0 < body["decision_threshold"] < 1.0


@needs_model
@needs_clips
def test_analyze_endpoint_returns_the_contract():
    from fastapi.testclient import TestClient

    from src.inference.service import app

    client = TestClient(app)
    with open(FALL_CLIP, "rb") as fh:
        response = client.post(
            "/analyze/urfd", files={"file": ("clip.mp4", fh, "video/mp4")}
        )
    assert response.status_code == 200, response.text
    body = response.json()
    for key in (
        "schema_version",
        "is_fall",
        "score",
        "threshold",
        "confidence",
        "onset_s",
        "windows",
        "pose_coverage",
        "latency_ms",
        "model",
    ):
        assert key in body
    assert body["confidence"] in {"high", "medium", "low"}


@needs_model
def test_analyze_rejects_garbage_bytes():
    from fastapi.testclient import TestClient

    from src.inference.service import app

    client = TestClient(app)
    response = client.post(
        "/analyze", files={"file": ("bad.mp4", b"not a video at all", "video/mp4")}
    )
    assert response.status_code in (422, 503)
