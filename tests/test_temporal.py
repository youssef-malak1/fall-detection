"""Tests for the temporal model: shape/orientation contracts that are easy to break."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from src.models.temporal import FallTCN, build_model  # noqa: E402

CHANNELS = 19


def test_forward_shape_and_length_independence():
    model = build_model(CHANNELS)
    model.eval()
    for length in (20, 60, 128):
        x = torch.randn(2, length, CHANNELS)
        mask = torch.ones(2, length)
        with torch.no_grad():
            out = model(x, mask)
        assert out.shape == (2,)


def test_masked_padding_does_not_change_the_score():
    """Padding a clip must not change its prediction — that is the whole point of the mask."""
    torch.manual_seed(0)
    model = build_model(CHANNELS)
    model.eval()

    x = torch.randn(1, 40, CHANNELS)
    mask = torch.ones(1, 40)

    padded_x = torch.cat([x, torch.randn(1, 25, CHANNELS)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(1, 25)], dim=1)

    with torch.no_grad():
        a = model(x, mask)
        b = model(padded_x, padded_mask)
    assert torch.allclose(a, b, atol=1e-4), "padded frames leaked into the pooled decision"


def test_receptive_field_covers_a_real_fall():
    model = build_model(CHANNELS)
    # a fall is roughly 0.5-1.5 s; at 15 fps that is 8-23 frames
    assert model.receptive_field >= 23


def test_fully_masked_clip_does_not_crash():
    model = build_model(CHANNELS)
    model.eval()
    x = torch.randn(1, 30, CHANNELS)
    mask = torch.zeros(1, 30)
    with torch.no_grad():
        out = model(x, mask)
    assert torch.isfinite(out).all()


def test_frame_scores_shape():
    model = build_model(CHANNELS)
    model.eval()
    x = torch.randn(1, 30, CHANNELS)
    mask = torch.ones(1, 30)
    scores = model.frame_scores(x, mask)
    assert scores.shape == (1, 30)
    assert torch.isfinite(scores).all()


def test_gradients_flow():
    model = build_model(CHANNELS)
    x = torch.randn(3, 40, CHANNELS)
    mask = torch.ones(3, 40)
    loss = model(x, mask).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_parameter_count_stays_small_for_70_events():
    model = build_model(CHANNELS)
    n = sum(p.numel() for p in model.parameters())
    # a model with more parameters than training examples is a memorisation machine
    assert n < 60_000, f"model has {n} parameters, too large for 70 events"
