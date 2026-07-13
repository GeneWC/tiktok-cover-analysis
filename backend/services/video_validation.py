"""Video upload validation (PRD 14.2 / 22.1).

Probes a saved video file with PyAV (bundled ffmpeg) to confirm it can be
decoded and to read the metadata required for validation:
  - the file decodes and has a video stream
  - resolution can be read
  - duration is within configured bounds
  - audio-track presence can be determined

Validation failures raise `VideoValidationError`, which carries an HTTP status
code and a user-friendly message so the API layer can turn it into a clean
error response without knowing anything about ffmpeg.
"""

from __future__ import annotations

import av

from backend.core.config import settings
from backend.core.media import extract_duration_seconds
from backend.schemas.analysis import VideoMetadata


class VideoValidationError(Exception):
    """Raised when an uploaded video fails validation.

    `code` is a stable machine-readable tag; `message` is user-facing;
    `status_code` is the HTTP status the API should return.
    """

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def validate_video(path: str) -> VideoMetadata:
    """Probe and validate a video file, returning its basic metadata.

    Raises VideoValidationError on any validation failure.
    """
    # --- decode / open ---
    try:
        container = av.open(path)
    except Exception as exc:  # PyAV raises various errors for corrupt/unreadable files
        raise VideoValidationError(
            code="cannot_decode",
            message="The video file could not be decoded. Please upload a valid video.",
            status_code=422,
        ) from exc

    try:
        video_streams = container.streams.video
        if not video_streams:
            raise VideoValidationError(
                code="no_video_stream",
                message="The file does not contain a video stream.",
                status_code=422,
            )

        video_stream = video_streams[0]
        width = int(video_stream.codec_context.width or 0)
        height = int(video_stream.codec_context.height or 0)
        duration_seconds = extract_duration_seconds(container, video_stream)

        rate = video_stream.average_rate or video_stream.base_rate
        fps = float(rate) if rate else None

        has_audio = len(container.streams.audio) > 0
    finally:
        container.close()

    # --- validate what we read ---
    if width <= 0 or height <= 0:
        raise VideoValidationError(
            code="no_resolution",
            message="Could not read the video resolution.",
            status_code=422,
        )

    if duration_seconds is None or duration_seconds <= 0:
        raise VideoValidationError(
            code="no_duration",
            message="Could not determine the video duration.",
            status_code=422,
        )

    if duration_seconds < settings.min_duration_seconds:
        raise VideoValidationError(
            code="too_short",
            message=(
                f"Video is too short ({duration_seconds:.1f}s). "
                f"Minimum is {settings.min_duration_seconds:.0f} second."
            ),
            status_code=400,
        )

    if duration_seconds > settings.max_duration_seconds:
        raise VideoValidationError(
            code="too_long",
            message=(
                f"Video is too long ({duration_seconds:.1f}s). "
                f"Maximum is {settings.max_duration_seconds:.0f} seconds."
            ),
            status_code=400,
        )

    aspect_ratio = width / height if height else None
    return VideoMetadata(
        duration_seconds=round(duration_seconds, 3),
        width=width,
        height=height,
        fps=round(fps, 3) if fps is not None else None,
        has_audio=has_audio,
        aspect_ratio=round(aspect_ratio, 4) if aspect_ratio is not None else None,
        is_vertical_video=height > width,
        is_square_video=(abs(width - height) / max(width, height) < 0.05) if max(width, height) else False,
    )
