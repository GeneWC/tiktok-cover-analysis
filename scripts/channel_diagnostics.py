"""CLI: channel diagnostics for one creator in the training set (demo / research).

    python scripts/channel_diagnostics.py --creator onesemble
    python scripts/channel_diagnostics.py --creator austinkslam --json-out data/reports/diag.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.channel_diagnostics import diagnose_channel  # noqa: E402
from backend.training.feature_groups import select_group_features  # noqa: E402
from backend.training.model_dataset import PRIMARY_TARGET, load_model_dataset  # noqa: E402
from backend.training.model_specs import PRIMARY_SPEC  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Within-creator channel diagnostics.")
    parser.add_argument("--creator", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    dataset = load_model_dataset(args.dataset)
    mask = dataset.frame["creator_username"].astype(str) == args.creator
    if not mask.any():
        print(f"Creator '{args.creator}' not found.")
        return 1

    feats = select_group_features(dataset.feature_names, PRIMARY_SPEC.feature_groups)
    sub = dataset.frame.loc[mask].reset_index(drop=True)
    X = dataset.X.loc[mask, feats].reset_index(drop=True)
    labels = dataset.target(PRIMARY_TARGET).loc[mask]
    video_ids = (
        sub["video_id"].astype(str).tolist() if "video_id" in sub.columns else None
    )

    report = diagnose_channel(
        X,
        labels=labels,
        video_ids=video_ids,
        creator=args.creator,
        feature_names=feats,
    )
    print(f"Creator: {args.creator}")
    print(f"Videos: {report.n_videos}  hits: {report.n_hits}  pos_rate: {report.positive_rate}")
    if report.message:
        print(report.message)
        return 1
    print("\nTop |hit - miss| feature deltas:")
    for d in report.top_feature_deltas:
        print(
            f"  {d.feature:40s}  delta={d.delta:+.4f}  "
            f"hit={d.hit_mean:.4f} miss={d.miss_mean:.4f}"
        )
    print("\nTop 5 by presentation proxy:")
    for row in report.video_ranks[:5]:
        print(
            f"  {row['video_id']}  present={row['presentation_score']:.3f}  "
            f"resid_l2={row['residual_l2']:.3f}  label={row['label']}"
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
