"""Video metadata features (PRD 11.2).

Container-level features that need no frame analysis: dimensions, duration, fps,
aspect/area, bitrate, and audio presence. This is the ML-facing feature dict and
is intentionally a *superset* of the report's display `VideoMetadata`
(e.g. it adds `resolution_area` and `bitrate`).

Standalone by design: it re-probes the file with PyAV so the same function works
for both training videos and inference uploads, independent of upload validation.
Post date/hour are deliberately excluded as features (PRD 11.2).
"""

from __future__ import annotations

import av

from backend.core.media import extract_duration_seconds


def extract_metadata_features(video_path: str) -> dict[str, float | int | bool | None]:
    """Probe a video and return its metadata feature dict (PRD 11.2)."""
    with av.open(video_path) as container:
        video_streams = container.streams.video
        if not video_streams:
            # No video stream: return a fully-null/zero record rather than raising,
            # so the aggregator can mark the step failed and continue (PRD 22.2).
            return _empty_metadata()

        video_stream = video_streams[0]
        width = int(video_stream.codec_context.width or 0)
        height = int(video_stream.codec_context.height or 0)
        duration_seconds = extract_duration_seconds(container, video_stream)

        rate = video_stream.average_rate or video_stream.base_rate
        fps = float(rate) if rate else None

        has_audio = len(container.streams.audio) > 0
        bitrate = container.bit_rate  # bits/sec, may be None

    longest_side = max(width, height)
    aspect_ratio = (width / height) if height else None

    return {
        "duration_seconds": round(duration_seconds, 3) if duration_seconds else None,
        "fps": round(fps, 3) if fps else None,
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 4) if aspect_ratio is not None else None,
        "resolution_area": width * height,
        "bitrate": float(bitrate) if bitrate else None,
        "has_audio": has_audio,
        "is_vertical_video": height > width,
        "is_square_video": (abs(width - height) / longest_side < 0.05) if longest_side else False,
    }


def _empty_metadata() -> dict[str, float | int | bool | None]:
    """Null/zero metadata record used when no video stream is present."""
    return {
        "duration_seconds": None,
        "fps": None,
        "width": 0,
        "height": 0,
        "aspect_ratio": None,
        "resolution_area": 0,
        "bitrate": None,
        "has_audio": False,
        "is_vertical_video": False,
        "is_square_video": False,
    }
