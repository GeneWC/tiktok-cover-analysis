"""Stdlib-only check that trained serving artifacts exist.

Used by the Docker image build and GitHub Actions so a deploy fails fast with a
clear message instead of booting an API that cannot score videos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_JSON = (
    "feature_schema.json",
    "feature_importances.json",
    "calibration.json",
)


def models_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def missing_artifacts(directory: Path | None = None) -> list[str]:
    """Return artifact filenames that must be present but are not."""
    root = directory or models_dir()
    missing: list[str] = []
    for name in REQUIRED_JSON:
        if not (root / name).is_file():
            missing.append(name)
    schema_path = root / "feature_schema.json"
    if schema_path.is_file():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            missing.append("feature_schema.json (invalid JSON)")
        else:
            for entry in schema.get("models", {}).values():
                artifact = entry.get("artifact")
                if artifact and not (root / artifact).is_file():
                    missing.append(str(artifact))
    # Preserve order while dropping duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for name in missing:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def main() -> int:
    missing = missing_artifacts()
    if missing:
        print("Missing serving artifacts under backend/models/:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print(
            "Train locally (`python scripts/train_models.py`) and commit "
            "backend/models/ before deploying.",
            file=sys.stderr,
        )
        return 1
    print(f"Serving artifacts OK ({models_dir()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
