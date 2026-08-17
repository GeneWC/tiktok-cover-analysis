"""Analysis pipeline orchestrator (PRD 14.3 / 17.3 / 19).

Runs one uploaded video end to end: feature extraction -> model scoring ->
presentation scores -> signal breakdown -> assembled report. Along the way it
updates each `PIPELINE_STEPS` entry in the job store (so the status endpoint
reflects live progress), persists the final report JSON to `reports_dir`, and
sets the job's terminal status.

Fault handling (PRD 22.2): a per-group extraction failure is already isolated by
the extractor (that group's step is marked failed and the rest continue). If the
video is entirely undecodable, or scoring raises, we still persist a coherent
"unusable" report and mark the job failed rather than leaving it stuck.

Audio note: when there's no audio track, the audio step is marked `skipped`
(not `failed`) - a legitimate, neutral outcome (PRD 11.6).
"""

from __future__ import annotations

from pathlib import Path

from backend.core.config import settings
from backend.inference.explanation import build_explanation
from backend.inference.feature_assembly import AssembledFeatures, assemble_features
from backend.inference.generate_user_report import (
    build_analysis_report,
    build_unusable_report,
)
from backend.inference.model_registry import ModelRegistry, get_registry
from backend.inference.prediction import predict
from backend.inference.presentation import compute_presentation_scores
from backend.schemas.analysis import ReportResponse
from backend.services import analysis_store
from backend.services.job_cleanup import delete_job_video


def _primitive_features(raw: dict) -> dict:
    """Keep only JSON/report-safe primitive feature values (drop status strings)."""
    return {
        key: value
        for key, value in raw.items()
        if value is None or isinstance(value, (int, float, bool))
    }


def _apply_extraction_steps(analysis_id: str, assembled: AssembledFeatures) -> None:
    """Mirror the extractor's per-group statuses into the job store."""
    for step, status in assembled.steps.items():
        if step == "audio" and not assembled.has_audio:
            analysis_store.set_step(analysis_id, step, "skipped")
        else:
            analysis_store.set_step(analysis_id, step, status)


def _persist_report(analysis_id: str, report: ReportResponse) -> Path:
    """Write the report JSON to reports_dir and record its path on the job."""
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    path = settings.reports_dir / f"{analysis_id}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    analysis_store.update_analysis(analysis_id, report_json_path=str(path))
    return path


def _fail(analysis_id: str, record: dict) -> ReportResponse:
    """Persist an unusable report and mark the job failed."""
    report = build_unusable_report(record)
    _persist_report(analysis_id, report)
    analysis_store.update_analysis(analysis_id, status="failed")
    return report


def run_analysis(
    analysis_id: str, registry: ModelRegistry | None = None
) -> ReportResponse:
    """Execute the full analysis pipeline for a job and persist its report."""
    record = analysis_store.get_analysis(analysis_id)
    if record is None:
        raise ValueError(f"Analysis '{analysis_id}' not found.")
    registry = registry or get_registry()

    try:
        # --- feature extraction ---
        try:
            assembled = assemble_features(record["video_file_path"], registry=registry)
        except Exception:  # noqa: BLE001 - never leave the job stuck on a hard failure
            analysis_store.set_step(analysis_id, "prediction", "skipped")
            return _fail(analysis_id, record)

        _apply_extraction_steps(analysis_id, assembled)
        if not assembled.usable:
            analysis_store.set_step(analysis_id, "prediction", "skipped")
            return _fail(analysis_id, record)

        # --- model scoring ---
        analysis_store.set_step(analysis_id, "prediction", "running")
        try:
            predictions = predict(assembled, registry=registry)
        except Exception:  # noqa: BLE001
            analysis_store.set_step(analysis_id, "prediction", "failed")
            return _fail(analysis_id, record)
        analysis_store.set_step(analysis_id, "prediction", "complete")

        # --- presentation scores + signals (deterministic, non-ML) ---
        presentation = compute_presentation_scores(assembled.raw, registry.calibration)
        explanation = build_explanation(
            assembled.raw, registry.calibration, registry.importances, assembled.has_audio
        )

        # --- assemble + persist report ---
        analysis_store.set_step(analysis_id, "report", "running")
        report = build_analysis_report(
            record, predictions, presentation, explanation,
            features=_primitive_features(assembled.raw),
        )
        _persist_report(analysis_id, report)
        analysis_store.set_step(analysis_id, "report", "complete")
        analysis_store.update_analysis(analysis_id, status="complete")
        return report
    finally:
        delete_job_video(analysis_id)
