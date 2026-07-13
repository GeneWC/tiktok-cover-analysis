"""Feature extraction over training videos (PRD 9.2 / 20.2).

Reads the raw training CSV, runs the shared `extract_all_features` pipeline on
each video file, and writes `video_features.csv` - one row per video, columns =
`video_id` + `video_feature_extraction_status` + the 71-key feature vector.

Design choices:
- **Same extractor as inference.** Reusing `extract_all_features` guarantees the
  training feature vector matches what the live app computes (no train/serve skew).
- **Resumable.** Videos already present in the output CSV are skipped, so this
  multi-hour job can run in chunks and survive interruptions. Rows are flushed
  after each video.
- **Resilient + PRD 8.5.** A missing/undecodable video keeps its row but is
  marked `video_feature_extraction_status=failed` (blank features) and excluded
  from training later. An unexpected error on one video is caught and recorded as
  failed rather than aborting the whole run.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from backend.collectors.base_collector import load_raw_videos
from backend.core.config import PROJECT_ROOT
from backend.features.extract_features import extract_all_features

_STATUS_COLUMN = "video_feature_extraction_status"
_KEY_COLUMNS = ("video_id", _STATUS_COLUMN)


def feature_names() -> list[str]:
    """Canonical, ordered feature-vector keys (the stable training/inference schema).

    Derived by running the orchestrator on a non-existent path, which returns the
    full null vector without decoding anything - so the column set can never drift
    from what the extractor actually produces.
    """
    return list(extract_all_features("__schema_probe__.mp4").features.keys())


def _resolve_path(video_file: str) -> Path | None:
    """Resolve a (possibly relative) video path against the project root."""
    if not video_file:
        return None
    path = Path(video_file)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _existing_ids(out_path: Path) -> set[str]:
    """video_ids already written to the output CSV (for resuming)."""
    if not out_path.exists():
        return set()
    with out_path.open(newline="", encoding="utf-8") as handle:
        return {(row.get("video_id") or "").strip() for row in csv.DictReader(handle)}


def _process_one(video) -> dict[str, object]:
    """Extract features for one raw video record into a CSV row dict."""
    path = _resolve_path(video.video_file)
    if path is None or not path.exists():
        return {"video_id": video.video_id, _STATUS_COLUMN: "failed"}

    try:
        result = extract_all_features(str(path))
    except Exception:  # noqa: BLE001 - never let one video abort the batch
        return {"video_id": video.video_id, _STATUS_COLUMN: "failed"}

    # Frames are the prerequisite for every visual feature; zero => unusable file.
    status = "complete" if result.frames_sampled > 0 else "failed"
    row: dict[str, object] = {"video_id": video.video_id, _STATUS_COLUMN: status}
    row.update(result.features)
    return row


def build_video_features_csv(
    raw_csv: str | Path,
    out_path: str | Path,
    limit: int | None = None,
    video_ids: set[str] | None = None,
    progress: bool = True,
) -> dict[str, int]:
    """Extract features for all (unprocessed) videos in the raw CSV.

    Appends to `out_path`, skipping any `video_id` already present. Returns a
    summary of processed/complete/failed/skipped counts.
    """
    raw_csv = Path(raw_csv)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(_KEY_COLUMNS) + feature_names()
    videos = load_raw_videos(raw_csv)
    already_done = _existing_ids(out_path)

    todo = [v for v in videos if v.video_id not in already_done]
    if video_ids is not None:
        todo = [v for v in todo if v.video_id in video_ids]
    if limit is not None:
        todo = todo[:limit]

    summary = {
        "total": len(videos),
        "skipped_existing": sum(1 for v in videos if v.video_id in already_done),
        "processed": 0,
        "complete": 0,
        "failed": 0,
    }

    file_is_new = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="", extrasaction="ignore")
        if file_is_new:
            writer.writeheader()

        for index, video in enumerate(todo, start=1):
            started = time.time()
            row = _process_one(video)
            writer.writerow(row)
            handle.flush()

            summary["processed"] += 1
            status = row[_STATUS_COLUMN]
            summary[status] = summary.get(status, 0) + 1
            if progress:
                elapsed = time.time() - started
                print(
                    f"[{index}/{len(todo)}] {video.video_id} -> {status} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )

    return summary
