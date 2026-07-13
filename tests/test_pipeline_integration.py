"""End-to-end integration test (PRD 23.2 / 23.3 "golden sample").

Runs the real feature pipeline on a sample video from ./videos. Marked `slow`
because it decodes video and runs MediaPipe + EAST (downloading models on first
use). Skips automatically if no sample video is present.

Run only the fast suite with:   pytest -m "not slow"
Run everything (default):        pytest
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.features.extract_features import extract_all_features

_VIDEOS = sorted((Path(__file__).resolve().parents[1] / "videos").glob("*.mp4"))


@pytest.mark.slow
@pytest.mark.skipif(not _VIDEOS, reason="no sample videos found in ./videos")
def test_full_pipeline_on_real_video():
    result = extract_all_features(str(_VIDEOS[0]))

    # The pipeline completes without crashing and produces the full vector.
    assert result.frames_sampled > 0
    assert len(result.features) == 71

    # The deterministic, always-available stages must succeed on a normal clip.
    for step in ("metadata", "frame_sampling", "visual_quality", "framing", "motion"):
        assert result.steps[step] == "complete", f"{step} did not complete"

    # Audio/OCR are either complete or a clean, recorded failure - never missing.
    assert result.steps["audio"] in {"complete", "failed"}
    assert result.steps["ocr"] in {"complete", "failed"}

    # A few sanity checks on real extracted values.
    assert result.features["width"] > 0
    assert result.features["height"] > 0
    assert 0.0 <= result.features["person_visible_ratio"] <= 1.0
