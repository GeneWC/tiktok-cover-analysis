"""CLI: evaluate channel-mode residual scoring on frozen splits.

    python scripts/eval_channel_mode.py
    python scripts/eval_channel_mode.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.channel_mode import evaluate_channel_mode_on_split  # noqa: E402

OUT_DEFAULT = Path("data/reports/channel_mode_eval.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Channel-mode residual eval.")
    parser.add_argument("--split", choices=("val", "test", "both"), default="val")
    parser.add_argument("--json-out", default=str(OUT_DEFAULT))
    args = parser.parse_args(argv)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    payload = {"results": []}
    for split in splits:
        result = evaluate_channel_mode_on_split(eval_split=split)
        overall = result["overall"]
        print(
            f"=== channel mode [{split}] ===  "
            f"auc={overall.get('roc_auc', float('nan')):.4f}  "
            f"p@k={overall.get('precision_at_k', float('nan')):.4f}  "
            f"creators={result['n_creators']}"
        )
        for row in result["per_creator"]:
            if "roc_auc" in row:
                print(
                    f"  {row['creator']:20s} n={row['n']:3d}  "
                    f"auc={row['roc_auc']:.4f}  pos={row['positive_rate']:.3f}"
                )
            else:
                print(f"  {row['creator']:20s} n={row['n']:3d}  {row.get('message')}")
        payload["results"].append(result)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
