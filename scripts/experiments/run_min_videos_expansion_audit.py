"""Exp C7: quantify risk of lowering --min-videos to grow the training set.

Uses local downloads + engagement/raw CSVs only. Does not re-download.

    python scripts/experiments/run_min_videos_expansion_audit.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.model_dataset import load_model_dataset  # noqa: E402

OUT = PROJECT_ROOT / "data" / "reports" / "min_videos_expansion_audit.json"


def main() -> int:
    dataset = load_model_dataset()
    frame = dataset.frame
    counts = frame["creator_username"].value_counts()
    y = (
        frame["top_quartile_for_creator"].astype(str).str.lower().eq("true").astype(int)
    )

    # Current inclusion is min 5. Simulate what adding creators with n in {3,4}
    # would look like IF they were in the dataset — they are not currently.
    # Audit current small creators and label degeneracy.
    rows = []
    for creator, n in counts.items():
        mask = frame["creator_username"] == creator
        pos = float(y[mask].mean())
        # Quartile degeneracy: with n=5, possible pos rates are multiples of 0.2
        unique_log = pd.to_numeric(frame.loc[mask, "log_views"], errors="coerce")
        n_unique_views = int(unique_log.nunique(dropna=True))
        rows.append(
            {
                "creator": creator,
                "n": int(n),
                "pos_rate": pos,
                "n_unique_log_views": n_unique_views,
                "degenerate_views": n_unique_views <= max(2, n // 3),
            }
        )

    # Extra mp4s not in training set
    downloads = PROJECT_ROOT / "downloads"
    mp4s = list(downloads.glob("*.mp4")) if downloads.exists() else []
    trained_ids = set(frame["video_id"].astype(str)) if "video_id" in frame.columns else set()

    # Creator roster from raw CSV if present
    raw_path = PROJECT_ROOT / "data" / "raw_tiktok_training_data.csv"
    expansion = {}
    if raw_path.exists():
        raw = pd.read_csv(raw_path, dtype=str)
        if "creator_username" in raw.columns:
            raw_counts = raw["creator_username"].value_counts()
            for min_v in (3, 4, 5):
                eligible = raw_counts[raw_counts >= min_v]
                expansion[str(min_v)] = {
                    "n_creators": int(eligible.shape[0]),
                    "n_videos": int(eligible.sum()),
                    "new_creators_vs_5": int(
                        (raw_counts[(raw_counts >= min_v) & (raw_counts < 5)]).shape[0]
                    ),
                    "new_videos_vs_5": int(
                        raw_counts[(raw_counts >= min_v) & (raw_counts < 5)].sum()
                    ),
                }

    small = [r for r in rows if r["n"] < 20]
    payload = {
        "current_creators": len(rows),
        "current_videos": int(len(frame)),
        "creators_n_lt_20": small,
        "n_degenerate_view_creators": sum(1 for r in rows if r["degenerate_views"]),
        "pos_rate_std_all": float(np.std([r["pos_rate"] for r in rows])),
        "pos_rate_std_n_ge_20": float(
            np.std([r["pos_rate"] for r in rows if r["n"] >= 20]) or 0
        ),
        "downloads_mp4_count": len(mp4s),
        "expansion_if_lower_min_videos": expansion,
        "recommendation": (
            "Keep min_videos=5"
            if expansion.get("4", {}).get("new_videos_vs_5", 0) < 50
            else "Consider min_videos=4 only after re-label audit"
        ),
    }
    # Refine recommendation
    new4 = expansion.get("4", {}).get("new_videos_vs_5", 0)
    new3 = expansion.get("3", {}).get("new_videos_vs_5", 0)
    if new4 == 0 and new3 == 0:
        payload["recommendation"] = (
            "No additional creators between n=3..4 in raw CSV; "
            "expansion requires ingesting unused downloads / new spreadsheet rows. "
            "Keep min_videos=5."
        )
    elif new4 > 0 and new4 < 80:
        payload["recommendation"] = (
            f"Lowering to min_videos=4 adds ~{new4} videos — modest gain with "
            "higher quartile noise. Prefer labeled-channel UX over lowering threshold."
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)[:2000])
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
