"""Resumable backfill of D-007 audio structure columns into video_features.csv.

Only re-decodes audio (skips visual/OCR). Safe to interrupt; re-run continues.

    python scripts/backfill_audio_structure_features.py
    python scripts/backfill_audio_structure_features.py --limit 20
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.collectors.base_collector import load_raw_videos  # noqa: E402
from backend.core.config import settings  # noqa: E402
from backend.features.audio_features import (  # noqa: E402
    AUDIO_STRUCTURE_FEATURE_KEYS,
    extract_audio_structure_features,
)

LOG_DEFAULT = Path("data/audio_structure_backfill.log")


def _needs_backfill(row: dict[str, str]) -> bool:
    for key in AUDIO_STRUCTURE_FEATURE_KEYS:
        val = (row.get(key) or "").strip()
        if val == "":
            return True
    return False


def _resolve_path(video_file: str) -> Path | None:
    if not video_file:
        return None
    path = Path(video_file)
    return path if path.is_absolute() else PROJECT_ROOT / path


def backfill(
    features_csv: Path,
    raw_csv: Path,
    limit: int | None,
    log_path: Path,
) -> dict[str, int]:
    features_csv = Path(features_csv)
    raw_csv = Path(raw_csv)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    videos = {v.video_id: v for v in load_raw_videos(raw_csv)}

    with features_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for key in AUDIO_STRUCTURE_FEATURE_KEYS:
        if key not in fieldnames:
            # Insert before audio status if present, else append.
            if "audio_feature_extraction_status" in fieldnames:
                idx = fieldnames.index("audio_feature_extraction_status")
                fieldnames[idx:idx] = [key]
            else:
                fieldnames.append(key)

    summary = {
        "total_rows": len(rows),
        "already_done": 0,
        "updated": 0,
        "failed": 0,
        "missing_video": 0,
        "skipped_limit": 0,
    }

    todo_indices = [i for i, row in enumerate(rows) if _needs_backfill(row)]
    if limit is not None:
        summary["skipped_limit"] = max(0, len(todo_indices) - limit)
        todo_indices = todo_indices[:limit]
    summary["already_done"] = summary["total_rows"] - len(todo_indices) - summary["skipped_limit"]

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n--- backfill start limit={limit} todo={len(todo_indices)} ---\n")
        for n, idx in enumerate(todo_indices, start=1):
            row = rows[idx]
            video_id = (row.get("video_id") or "").strip()
            video = videos.get(video_id)
            started = time.time()
            status = "failed"
            if video is None:
                summary["missing_video"] += 1
                for key in AUDIO_STRUCTURE_FEATURE_KEYS:
                    row[key] = ""
            else:
                path = _resolve_path(video.video_file)
                if path is None or not path.exists():
                    summary["missing_video"] += 1
                    for key in AUDIO_STRUCTURE_FEATURE_KEYS:
                        row[key] = ""
                else:
                    try:
                        feats = extract_audio_structure_features(str(path))
                        for key in AUDIO_STRUCTURE_FEATURE_KEYS:
                            val = feats.get(key)
                            row[key] = "" if val is None else str(val)
                        status = "ok" if feats.get("audio_feature_extraction_status") == "ok" else "failed"
                        if status == "ok":
                            summary["updated"] += 1
                        else:
                            summary["failed"] += 1
                    except Exception as exc:  # noqa: BLE001
                        summary["failed"] += 1
                        for key in AUDIO_STRUCTURE_FEATURE_KEYS:
                            row[key] = ""
                        status = f"error:{type(exc).__name__}"

            elapsed = time.time() - started
            msg = f"[{n}/{len(todo_indices)}] {video_id} -> {status} ({elapsed:.1f}s)\n"
            log.write(msg)
            log.flush()
            print(msg.strip(), flush=True)

            # Checkpoint every 10 videos (and always on the last) so crashes
            # lose little progress without rewriting the CSV every row.
            if n % 10 == 0 or n == len(todo_indices):
                tmp_path = features_csv.with_suffix(".csv.tmp")
                with tmp_path.open("w", newline="", encoding="utf-8") as out:
                    writer = csv.DictWriter(
                        out, fieldnames=fieldnames, restval="", extrasaction="ignore"
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                tmp_path.replace(features_csv)
                log.write(f"checkpoint at {n}/{len(todo_indices)}\n")
                log.flush()

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill audio structure features.")
    parser.add_argument("--features", default=str(settings.video_features_csv))
    parser.add_argument("--raw", default=str(settings.raw_training_csv))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log", default=str(LOG_DEFAULT))
    args = parser.parse_args(argv)

    summary = backfill(
        features_csv=Path(args.features),
        raw_csv=Path(args.raw),
        limit=args.limit,
        log_path=Path(args.log),
    )
    print("\nBackfill summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
