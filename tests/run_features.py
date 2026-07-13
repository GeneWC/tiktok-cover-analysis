"""Manual feature-extraction runner / smoke test.

Runs the full Phase 2 feature pipeline built so far (frame sampling, metadata,
visual quality, framing, motion) on one or more videos and prints the combined
results as JSON. Useful for eyeballing real outputs and confirming nothing
crashes end to end.

Usage (from the project root):
    python tests/run_features.py                      # all videos in ./videos
    python tests/run_features.py videos/0601.mp4 ...  # specific files

No PYTHONPATH needed: this script adds the project root to sys.path itself.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# --- make `backend` importable regardless of the current working directory ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.features.audio_features import extract_audio_features  # noqa: E402
from backend.features.frame_sampling import WINDOW_SECONDS, sample_frames  # noqa: E402
from backend.features.framing_features import extract_framing_features  # noqa: E402
from backend.features.metadata_features import extract_metadata_features  # noqa: E402
from backend.features.motion_features import extract_motion_features  # noqa: E402
from backend.features.ocr_features import extract_ocr_features  # noqa: E402
from backend.features.visual_quality_features import (  # noqa: E402
    extract_visual_quality_features,
)
from backend.services.video_validation import (  # noqa: E402
    VideoValidationError,
    validate_video,
)


def run_one(path: str) -> dict:
    """Run every feature extractor on a single video and return a result dict."""
    result: dict = {"video": path, "timings_seconds": {}}

    def timed(label, fn):
        start = time.time()
        value = fn()
        result["timings_seconds"][label] = round(time.time() - start, 2)
        return value

    try:
        result["validation"] = timed(
            "validation", lambda: validate_video(path).model_dump()
        )
    except VideoValidationError as exc:
        result["validation"] = {"error_code": exc.code, "message": exc.message}

    sample = timed("frame_sampling", lambda: sample_frames(path))
    result["sampling"] = {
        "source": f"{sample.source_width}x{sample.source_height}",
        "fps": sample.fps,
        "duration_seconds": sample.duration_seconds,
        "processing": f"{sample.proc_width}x{sample.proc_height}",
        "frames_sampled": len(sample.frames),
        "window_counts": {w: len(sample.window(w)) for w in WINDOW_SECONDS},
    }

    result["metadata_features"] = timed(
        "metadata", lambda: extract_metadata_features(path)
    )
    result["visual_quality_features"] = timed(
        "visual_quality", lambda: extract_visual_quality_features(sample)
    )
    result["framing_features"] = timed(
        "framing", lambda: extract_framing_features(sample)
    )
    result["motion_features"] = timed(
        "motion", lambda: extract_motion_features(sample)
    )
    result["audio_features"] = timed(
        "audio", lambda: extract_audio_features(path)
    )
    result["ocr_features"] = timed(
        "ocr", lambda: extract_ocr_features(sample)
    )
    return result


def main(argv: list[str]) -> int:
    if argv:
        paths = argv
    else:
        videos_dir = PROJECT_ROOT / "videos"
        paths = [str(p) for p in sorted(videos_dir.glob("*.mp4"))]
        if not paths:
            print(f"No videos passed and none found in {videos_dir}")
            return 1

    for path in paths:
        print("=" * 70)
        print(json.dumps(run_one(path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
