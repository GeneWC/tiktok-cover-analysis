"""Shared pytest fixtures and import bootstrap for the test suite.

Adds the project root to sys.path (so `backend` imports work no matter where
pytest is launched) and provides a factory for building synthetic `FrameSample`
objects. Synthetic frames let the visual/motion feature tests run fast and
deterministically without decoding real videos or downloading ML models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import settings  # noqa: E402
from backend.features.frame_sampling import FrameSample, SampledFrame  # noqa: E402


@pytest.fixture(autouse=True)
def _generous_upload_rate_limit(monkeypatch):
    """Keep suite-wide POSTs from sharing one in-memory rate-limit bucket."""
    monkeypatch.setattr(settings, "upload_rate_limit", 10_000)


@pytest.fixture
def make_sample():
    """Factory: build a FrameSample from a list of images + their timestamps."""

    def _make(images: list[np.ndarray], timestamps: list[float]) -> FrameSample:
        if len(images) != len(timestamps):
            raise ValueError("images and timestamps must be the same length")
        height, width = images[0].shape[:2]
        frames = [
            SampledFrame(timestamp=float(t), image=img)
            for t, img in zip(timestamps, images)
        ]
        return FrameSample(
            source_width=width,
            source_height=height,
            fps=3.0,
            duration_seconds=float(timestamps[-1]) if timestamps else 0.0,
            proc_width=width,
            proc_height=height,
            frames=frames,
        )

    return _make


@pytest.fixture
def empty_sample() -> FrameSample:
    """A FrameSample with no frames (simulates a failed/empty decode)."""
    return FrameSample(0, 0, 0.0, 0.0, 0, 0)
