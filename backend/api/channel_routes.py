"""Channel diagnostics API routes (D-018).

POST /api/channel/diagnose — multi-video upload + optional views
GET  /api/channel/{id}/status
GET  /api/channel/{id}/report
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from backend.core.config import settings
from backend.inference.channel_pipeline import run_channel_diagnose
from backend.schemas.channel import (
    ChannelAnalyzeResponse,
    ChannelReportResponse,
    ChannelStatusResponse,
)
from backend.services import channel_store
from backend.services.job_ids import is_channel_id
from backend.services.video_validation import VideoValidationError, validate_video
from backend.training.creator_residuals import MIN_CREATOR_VIDEOS_FOR_RESIDUALS

router = APIRouter(prefix="/api/channel", tags=["channel"])

_CHUNK_SIZE = 1024 * 1024
_MAX_VIDEOS = 30


@router.get("/ping")
def ping() -> dict[str, str]:
    return {"router": "channel", "status": "ok"}


@router.get("/{channel_id}/status", response_model=ChannelStatusResponse)
def get_channel_status(channel_id: str) -> ChannelStatusResponse:
    if not is_channel_id(channel_id):
        raise HTTPException(status_code=404, detail=f"Channel '{channel_id}' not found.")
    record = channel_store.get_channel_job(channel_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Channel '{channel_id}' not found.")
    return ChannelStatusResponse(
        channel_id=record["channel_id"],
        status=record["status"],
        steps=record["steps"],
        n_videos=len(record["videos"]),
        n_features_done=record.get("n_features_done") or 0,
        error=record.get("error"),
    )


@router.get("/{channel_id}/report", response_model=ChannelReportResponse)
def get_channel_report(channel_id: str) -> ChannelReportResponse:
    if not is_channel_id(channel_id):
        raise HTTPException(status_code=404, detail=f"Channel '{channel_id}' not found.")
    record = channel_store.get_channel_job(channel_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Channel '{channel_id}' not found.")

    path = record.get("report_json_path")
    if path and Path(path).exists():
        return ChannelReportResponse.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )

    # Pending placeholder while processing.
    return ChannelReportResponse(
        channel_id=channel_id,
        status=record["status"],
        n_videos=len(record["videos"]),
        n_labeled=0,
        limitations=[
            "Channel diagnostics are still processing. Poll status and retry.",
        ],
        message=record.get("error"),
    )


@router.post(
    "/diagnose",
    response_model=ChannelAnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def diagnose_channel_upload(
    background_tasks: BackgroundTasks,
    video_files: list[UploadFile] = File(
        ..., description="At least 5 videos from the same creator."
    ),
    metrics_json: str | None = Form(
        default=None,
        description='Optional JSON list aligned to files: [{"views": 1234}, ...]',
    ),
) -> ChannelAnalyzeResponse:
    """Accept a creator batch, validate, and run within-batch diagnostics."""
    if len(video_files) < MIN_CREATOR_VIDEOS_FOR_RESIDUALS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Channel diagnostics need at least {MIN_CREATOR_VIDEOS_FOR_RESIDUALS} "
                f"videos (got {len(video_files)})."
            ),
        )
    if len(video_files) > _MAX_VIDEOS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many videos (max {_MAX_VIDEOS}).",
        )

    metrics: list[dict] = []
    if metrics_json:
        try:
            parsed = json.loads(metrics_json)
            if not isinstance(parsed, list):
                raise ValueError("metrics_json must be a JSON list")
            metrics = parsed
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid metrics_json: {exc}"
            ) from exc
        if len(metrics) not in {0, len(video_files)}:
            raise HTTPException(
                status_code=400,
                detail="metrics_json length must match the number of video files.",
            )

    # Pre-create job id directory for videos
    # We'll create the job after saving files so we have paths.
    saved: list[dict] = []
    destinations: list[Path] = []

    try:
        # Save + validate each file first; then create the job.
        temp_dir = settings.videos_dir / "_channel_incoming"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for index, upload in enumerate(video_files):
            if not upload.filename:
                raise HTTPException(status_code=400, detail="Missing filename.")
            extension = Path(upload.filename).suffix.lower()
            if extension not in settings.supported_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file type '{extension}'. "
                        f"Supported: {', '.join(settings.supported_extensions)}."
                    ),
                )

            dest = temp_dir / f"pending_{index}_{Path(upload.filename).name}"
            max_bytes = settings.max_file_size_mb * 1024 * 1024
            bytes_written = 0
            with dest.open("wb") as buffer:
                while chunk := await upload.read(_CHUNK_SIZE):
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        raise VideoValidationError(
                            code="file_too_large",
                            message=(
                                f"File is too large. Maximum is "
                                f"{settings.max_file_size_mb} MB."
                            ),
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        )
                    buffer.write(chunk)
            await upload.close()
            destinations.append(dest)

            try:
                validate_video(str(dest))
            except VideoValidationError as exc:
                raise HTTPException(
                    status_code=exc.status_code, detail=exc.message
                ) from exc

            views = None
            if metrics:
                raw = metrics[index].get("views") if isinstance(metrics[index], dict) else None
                if raw is not None and str(raw).strip() != "":
                    views = int(raw)

            saved.append(
                {
                    "path": str(dest),
                    "filename": upload.filename,
                    "views": views,
                    "video_id": Path(upload.filename).stem,
                }
            )
    except HTTPException:
        for path in destinations:
            path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        for path in destinations:
            path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail="Failed to save channel videos."
        ) from exc

    record = channel_store.create_channel_job(saved)
    channel_id = record["channel_id"]

    # Move files into a stable per-job directory.
    job_dir = settings.videos_dir / channel_id
    job_dir.mkdir(parents=True, exist_ok=True)
    moved_videos: list[dict] = []
    for item in saved:
        src = Path(item["path"])
        dest = job_dir / src.name
        src.replace(dest)
        moved = dict(item)
        moved["path"] = str(dest)
        moved_videos.append(moved)
    channel_store.update_channel_job(channel_id, videos=moved_videos)

    background_tasks.add_task(run_channel_diagnose, channel_id)
    return ChannelAnalyzeResponse(
        channel_id=channel_id,
        status="processing",
        n_videos=len(moved_videos),
    )
