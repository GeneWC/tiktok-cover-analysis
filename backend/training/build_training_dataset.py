"""Training dataset builder (PRD 9.3 / 20.3).

Joins the per-video features (`video_features.csv`) with the computed
creator-relative labels (from the raw metrics) on `video_id`, producing
`training_dataset.csv` - the single table the model trainer consumes.

Join/exclusion rules (PRD 8.5):
- A video with no feature row, or `video_feature_extraction_status != complete`,
  is excluded from the dataset (it can't be trained on).
- Otherwise the row carries identifiers + labels + the full feature vector. The
  feature column set is read from the features CSV header (minus the id/status
  columns) so it always matches what was actually extracted.

Identifiers/labels are kept in the table for traceability and as training
targets, but Phase 4 must exclude them as model *inputs* (leakage, PRD 12.4).
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from backend.collectors.base_collector import load_raw_videos
from backend.training.compute_labels import LABEL_FIELDS, compute_labels

_STATUS_COLUMN = "video_feature_extraction_status"


def _load_feature_rows(features_csv: Path) -> tuple[dict[str, dict], list[str]]:
    """Return {video_id: row} and the feature column names (no id/status)."""
    with features_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = {(r.get("video_id") or "").strip(): r for r in reader}
    feature_cols = [c for c in header if c not in ("video_id", _STATUS_COLUMN)]
    return rows, feature_cols


def build_training_dataset(
    raw_csv: str | Path,
    features_csv: str | Path,
    out_path: str | Path,
) -> dict[str, int]:
    """Join labels + features into the training dataset CSV. Returns a summary."""
    raw_csv = Path(raw_csv)
    features_csv = Path(features_csv)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = compute_labels(load_raw_videos(raw_csv))
    feature_rows, feature_cols = _load_feature_rows(features_csv)

    columns = list(LABEL_FIELDS) + feature_cols + [_STATUS_COLUMN]
    summary = {
        "labeled_videos": len(labels),
        "written": 0,
        "skipped_no_features": 0,
        "skipped_failed": 0,
    }

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for label in labels:
            frow = feature_rows.get(label.video_id)
            if frow is None:
                summary["skipped_no_features"] += 1
                continue
            if (frow.get(_STATUS_COLUMN) or "").strip() != "complete":
                summary["skipped_failed"] += 1
                continue

            row: dict[str, object] = asdict(label)
            for col in feature_cols:
                row[col] = frow.get(col, "")
            row[_STATUS_COLUMN] = "complete"
            writer.writerow(row)
            summary["written"] += 1

    return summary
