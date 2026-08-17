"""Analysis API routes.

Groups all `/api/analyze` endpoints behind an APIRouter, mounted onto the main
app. Step 2 implemented the upload endpoint (PRD 19.1); Step 3 adds full upload
validation (PRD 14.2 / 22.1): size limit, decodability, resolution, duration
bounds, and audio-track detection. The status and report endpoints
(PRD 19.2 / 19.3) follow in later steps.
"""

from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from backend.core.config import settings
from backend.inference.generate_user_report import build_report
from backend.inference.pipeline import run_analysis
from backend.schemas.analysis import AnalyzeResponse, ReportResponse, StatusResponse
from backend.services import analysis_store
from backend.services.job_ids import is_analysis_id
from backend.services.video_validation import VideoValidationError, validate_video

router = APIRouter(prefix="/api/analyze", tags=["analyze"])


def get_analysis_or_404(analysis_id: str) -> dict:
    """Dependency: load a job record by id or raise 404.

    Declaring `analysis_id` here lets FastAPI bind it from the path parameter of
    any route that depends on this function, so multiple endpoints can reuse the
    same fetch-or-404 logic without repeating it.
    """
    if not is_analysis_id(analysis_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' not found.",
        )
    record = analysis_store.get_analysis(analysis_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' not found.",
        )
    return record

# Read/write the upload in fixed-size chunks so a large video never has to sit
# in memory all at once.
_CHUNK_SIZE = 1024 * 1024  # 1 MB


@router.get("/ping")
def ping() -> dict[str, str]:
    """Trivial endpoint proving the analyze router is wired up."""
    return {"router": "analyze", "status": "ok"}


@router.get("/{analysis_id}/status", response_model=StatusResponse)
def get_status(record: dict = Depends(get_analysis_or_404)) -> StatusResponse:
    """Return the per-step progress of an analysis job (PRD 19.2).

    A safe, idempotent GET that clients poll while processing runs. The
    `analysis_id` path param is consumed by the `get_analysis_or_404` dependency.
    """
    return StatusResponse(
        analysis_id=record["analysis_id"],
        status=record["status"],
        steps=record["steps"],
    )


@router.get("/{analysis_id}/report", response_model=ReportResponse)
def get_report(record: dict = Depends(get_analysis_or_404)) -> ReportResponse:
    """Return the analysis report (PRD 19.3).

    Once the pipeline has run, the persisted report JSON is served as the single
    source of truth. While the job is still processing, a "pending" report (real
    metadata + null predictions + disclaimers) is returned so the client has a
    coherent shape to render; it polls the status endpoint to know when to refetch.
    """
    persisted = _load_persisted_report(record)
    return persisted or build_report(record)


def _load_persisted_report(record: dict) -> ReportResponse | None:
    """Load the report JSON written by the pipeline, if it exists yet."""
    path = record.get("report_json_path")
    if path and Path(path).exists():
        return ReportResponse.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    return None


@router.post("", response_model=AnalyzeResponse, status_code=status.HTTP_202_ACCEPTED)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(..., description="The cover video to analyze."),
    hashtags: str | None = Form(default=None),
    instrument: str | None = Form(default=None),
) -> AnalyzeResponse:
    """Accept a single video upload, validate it, and register an analysis job.

    Validation order (cheapest first): filename/extension, then a streamed
    size-limit check while saving, then a decode/metadata probe of the saved
    file. On success the analysis pipeline is scheduled as a background task so
    this endpoint returns `202 Accepted` immediately while processing continues;
    the client polls the status endpoint. `hashtags`/`instrument` are accepted
    but optional and unused by the MVP per PRD 14.1.
    """
    # --- 1. lightweight pre-storage validation: extension ---
    if not video_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided with the uploaded file.",
        )

    extension = Path(video_file.filename).suffix.lower()
    if extension not in settings.supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{extension}'. "
                f"Supported types: {', '.join(settings.supported_extensions)}."
            ),
        )

    # --- 2. register the job so we have an id to name the file with ---
    safe_name = Path(video_file.filename).name
    record = analysis_store.create_analysis(
        video_file_path="", original_filename=safe_name
    )
    analysis_id = record["analysis_id"]
    destination = settings.videos_dir / f"{analysis_id}{extension}"

    # --- 3. stream the upload to disk, enforcing the size limit as we go ---
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    bytes_written = 0
    try:
        with destination.open("wb") as buffer:
            while chunk := await video_file.read(_CHUNK_SIZE):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise VideoValidationError(
                        code="file_too_large",
                        message=f"File is too large. Maximum is {settings.max_file_size_mb} MB.",
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    )
                buffer.write(chunk)
    except VideoValidationError as exc:
        _cleanup(analysis_id, destination)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:  # noqa: BLE001 - surface any I/O failure as a 500
        _cleanup(analysis_id, destination)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save the uploaded video.",
        ) from exc
    finally:
        await video_file.close()

    # --- 4. probe/validate the saved file (decodability, resolution, duration, audio) ---
    try:
        metadata = validate_video(str(destination))
    except VideoValidationError as exc:
        _cleanup(analysis_id, destination)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    # --- 5. persist metadata and mark the metadata step complete ---
    analysis_store.update_analysis(
        analysis_id,
        video_file_path=str(destination),
        metadata=metadata.model_dump(),
    )
    analysis_store.set_step(analysis_id, "metadata", "complete")

    # --- 6. run the analysis pipeline in the background (returns 202 now) ---
    background_tasks.add_task(run_analysis, analysis_id)

    return AnalyzeResponse(analysis_id=analysis_id, status="processing")


def _cleanup(analysis_id: str, destination: Path) -> None:
    """Remove a partially-saved file and its job record after a failed upload."""
    destination.unlink(missing_ok=True)
    analysis_store.delete_analysis(analysis_id)
