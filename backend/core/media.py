"""Shared PyAV helpers.

Small media-probing utilities shared by upload validation and metadata feature
extraction, so the duration-reading logic lives in exactly one place.
"""

from __future__ import annotations

# container.duration is expressed in microseconds (AV_TIME_BASE).
AV_TIME_BASE = 1_000_000.0


def extract_duration_seconds(container, video_stream) -> float | None:
    """Best-effort duration in seconds from the container or the video stream."""
    if container.duration is not None:
        return container.duration / AV_TIME_BASE
    if video_stream.duration is not None and video_stream.time_base is not None:
        return float(video_stream.duration * video_stream.time_base)
    return None
