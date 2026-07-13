"""Manual runner / smoke test for the feature-extraction orchestrator.

Runs `extract_all_features` (the unified per-video feature builder) on one or
more videos and prints the merged feature vector, per-step status, and timing.
This is the function both training and inference will call, so eyeballing its
output here confirms the end-to-end contract.

Usage (from the project root):
    python tests/run_pipeline.py                      # all videos in ./videos
    python tests/run_pipeline.py videos/0601.mp4 ...  # specific files

No PYTHONPATH needed: this script adds the project root to sys.path itself.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.features.extract_features import extract_all_features  # noqa: E402


def run_one(path: str) -> dict:
    start = time.time()
    result = extract_all_features(path)
    elapsed = round(time.time() - start, 2)
    return {
        "video": path,
        "elapsed_seconds": elapsed,
        "frames_sampled": result.frames_sampled,
        "duration_seconds": result.duration_seconds,
        "all_groups_ok": result.ok,
        "steps": result.steps,
        "feature_count": len(result.features),
        "features": result.features,
    }


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
