"""Tests for the stdlib serving-artifact gate used by Docker and CI."""

from __future__ import annotations

import json

from backend.check_serving_artifacts import missing_artifacts, models_dir


def test_repo_models_dir_is_complete():
    assert missing_artifacts() == []
    assert models_dir().is_dir()


def test_missing_artifacts_reports_schema_and_pkls(tmp_path):
    assert "feature_schema.json" in missing_artifacts(tmp_path)
    (tmp_path / "feature_schema.json").write_text(
        json.dumps({"models": {"toy": {"artifact": "toy.pkl"}}}),
        encoding="utf-8",
    )
    missing = missing_artifacts(tmp_path)
    assert "feature_importances.json" in missing
    assert "calibration.json" in missing
    assert "toy.pkl" in missing
