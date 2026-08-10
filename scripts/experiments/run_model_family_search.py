"""Exp C1 — model family + feature-group search on val creators only.

Train on train creators; select on val. Does NOT score the frozen test set and
does NOT write production artifacts under backend/models/.

Run (Windows):
  .venv\\Scripts\\python.exe scripts\\experiments\\run_model_family_search.py

Writes:
  data/reports/model_family_val.json
  appends ## Exp C1 section to docs/EXPERIMENTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.creator_splits import (  # noqa: E402
    DEFAULT_SPLITS_PATH,
    generate_and_save_splits,
    load_creator_splits,
)
from backend.training.model_dataset import load_model_dataset  # noqa: E402
from backend.training.model_family_search import (  # noqa: E402
    PROMISING_AUC_MARGIN,
    pick_winner,
    run_model_family_search,
    search_rows_to_jsonable,
)

REPORTS = PROJECT_ROOT / "data" / "reports"
DEFAULT_JSON_OUT = REPORTS / "model_family_val.json"
EXPERIMENTS_MD = PROJECT_ROOT / "docs" / "EXPERIMENTS.md"
SECTION_HEADER = "## Exp C1 — Model family + feature groups (val)"


def _fmt_auc(val: float | None) -> str:
    if val is None:
        return "n/a"
    return f"{val:.4f}"


def _ensure_splits(splits_path: Path, groups) -> object:
    if splits_path.exists():
        return load_creator_splits(splits_path)
    print(f"Splits missing at {splits_path}; generating...")
    return generate_and_save_splits(groups, path=splits_path)


def _render_markdown_section(payload: dict) -> str:
    rows = payload["results"]
    winner = payload.get("winner")
    lines = [
        "",
        "---",
        "",
        SECTION_HEADER,
        "",
        f"Date: {payload['date']}. Train on train creators; score **val only** "
        f"(test untouched). Margin for PROMISING: "
        f"+{PROMISING_AUC_MARGIN:.2f} ROC AUC vs `rf_control` and beat "
        "`simple_logistic`. No export to `backend/models/`.",
        "",
        "### Model family (framing+visual unless noted)",
        "",
        "| Model | n_features | ROC AUC | Precision@k | ΔAUC vs RF | PROMISING |",
        "|-------|------------|---------|-------------|------------|-----------|",
    ]
    for r in rows:
        if r["kind"] not in {"model_family", "baseline"}:
            continue
        m = r["metrics"] or {}
        delta = r.get("delta_auc_vs_rf_control")
        delta_s = "n/a" if delta is None else f"{delta:+.4f}"
        prom = "yes" if r.get("promising") else ""
        lines.append(
            f"| {r['name']} | {r['n_features']} | "
            f"{_fmt_auc(m.get('roc_auc'))} | "
            f"{_fmt_auc(m.get('precision_at_k'))} | {delta_s} | {prom} |"
        )

    lines += [
        "",
        "### Feature-group ablations (RF control hyperparams)",
        "",
        "| Feature set | n_features | ROC AUC | Precision@k | ΔAUC vs framing+visual |",
        "|-------------|------------|---------|-------------|------------------------|",
    ]
    control_auc = None
    for r in rows:
        if r["name"] == "rf_control":
            control_auc = (r["metrics"] or {}).get("roc_auc")
            break
    for r in rows:
        if r["kind"] != "feature_ablation":
            continue
        m = r["metrics"] or {}
        auc = m.get("roc_auc")
        if control_auc is None or auc is None:
            delta_s = "n/a"
        else:
            delta_s = f"{float(auc) - float(control_auc):+.4f}"
        lines.append(
            f"| {r['feature_set']} | {r['n_features']} | "
            f"{_fmt_auc(auc)} | {_fmt_auc(m.get('precision_at_k'))} | {delta_s} |"
        )

    lines += ["", "### Verdict", ""]
    if winner:
        lines.append(
            f"**PROMISING:** `{winner['name']}` val ROC AUC="
            f"{_fmt_auc((winner.get('metrics') or {}).get('roc_auc'))} "
            f"(ΔAUC vs RF={winner.get('delta_auc_vs_rf_control'):+.4f}). "
            "Do **not** run test yet; do **not** export to `backend/models/`."
        )
    else:
        lines.append(
            "**No winner.** No model-family candidate beat `rf_control` by "
            f">= {PROMISING_AUC_MARGIN:.2f} ROC AUC while also beating "
            "`simple_logistic` on val. Test set remains frozen."
        )
    lines.append("")
    return "\n".join(lines)


def _append_experiments_md(section: str) -> None:
    """Insert or replace the Exp C1 block (newest cycle near top of file)."""
    text = EXPERIMENTS_MD.read_text(encoding="utf-8") if EXPERIMENTS_MD.exists() else ""
    section_body = section.strip() + "\n"

    if SECTION_HEADER in text:
        start = text.index(SECTION_HEADER)
        prefix = text[:start].rstrip()
        if prefix.endswith("---"):
            prefix = prefix[: -len("---")].rstrip()
        rest = text[start:]
        # Drop this section through the line before the next ## heading (or EOF).
        lines = rest.splitlines(keepends=True)
        end = len(lines)
        for i in range(1, len(lines)):
            if lines[i].startswith("## "):
                end = i
                break
        suffix = "".join(lines[end:])
        text = prefix + "\n\n" + section_body
        if suffix:
            if not suffix.startswith("\n"):
                text += "\n"
            text += suffix
    else:
        marker = "## Protocol freeze"
        if marker in text:
            idx = text.index(marker)
            text = text[:idx] + section_body + "\n" + text[idx:]
        else:
            text = text.rstrip() + "\n\n" + section_body

    EXPERIMENTS_MD.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exp C1 model-family val search.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument(
        "--skip-md",
        action="store_true",
        help="Do not append/update docs/EXPERIMENTS.md",
    )
    args = parser.parse_args(argv)

    dataset = load_model_dataset(args.dataset)
    splits_path = Path(args.splits)
    membership = _ensure_splits(splits_path, dataset.groups)

    print(
        f"Splits: train/val/test="
        f"{len(membership.train_creators)}/"
        f"{len(membership.val_creators)}/"
        f"{len(membership.test_creators)}  "
        f"videos={membership.n_videos}"
    )
    print("Running model-family + feature-group search on VAL only...")

    rows = run_model_family_search(dataset, membership, eval_split="val")
    winner = pick_winner(rows)
    payload = {
        "date": str(date.today()),
        "experiment": "C1_model_family_feature_groups",
        "eval_split": "val",
        "target": "top_quartile_for_creator",
        "splits_path": str(splits_path).replace("\\", "/"),
        "promising_auc_margin": PROMISING_AUC_MARGIN,
        "n_videos": membership.n_videos,
        "train_creators": list(membership.train_creators),
        "val_creators": list(membership.val_creators),
        "test_creators": list(membership.test_creators),
        "results": search_rows_to_jsonable(rows),
        "winner": None
        if winner is None
        else search_rows_to_jsonable([winner])[0],
        "notes": (
            "Selection on val only. Test frozen. "
            "No artifacts written to backend/models/."
        ),
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    print("\n=== VAL results ===")
    for r in rows:
        m = r.metrics
        auc = m.get("roc_auc")
        flag = " PROMISING" if r.promising else ""
        msg = f"  ({r.message})" if r.message else ""
        print(
            f"  {r.name:36s} [{r.kind:16s}] "
            f"feats={r.n_features:3d}  "
            f"auc={_fmt_auc(auc)}  "
            f"p@k={_fmt_auc(m.get('precision_at_k'))}"
            f"{flag}{msg}"
        )

    if winner:
        print(
            f"\nWinner (PROMISING): {winner.name} "
            f"auc={_fmt_auc(winner.metrics.get('roc_auc'))} "
            f"— do NOT run test / export yet."
        )
    else:
        print("\nWinner: none (no PROMISING candidate on val).")

    if not args.skip_md:
        section = _render_markdown_section(payload)
        _append_experiments_md(section)
        print(f"Updated {EXPERIMENTS_MD}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
