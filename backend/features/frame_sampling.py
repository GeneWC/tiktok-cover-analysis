"""Frame sampling (PRD 11.1) - the shared foundation for all visual features.

Every visual feature module (visual quality, framing, motion) consumes the
output of this sampler, so they all see the *same* frames. Sampling once and
reusing the result is both correct (consistent features) and fast (decode once).

Strategy:
- Sample ~`sample_fps` frames per second (PRD recommends 2-5) by walking frames
  sequentially and keeping one whenever enough time has elapsed. Sequential
  decode avoids the unreliable frame/time seeking some codecs exhibit.
- Resize every kept frame to a consistent processing resolution (longer side
  <= `frame_processing_max_side`, preserving aspect ratio, never upscaling).
  All frames from one video share identical dimensions, which is required for
  frame-difference based motion features.
- Expose time windows (first_1s / first_3s / first_6s / full). When the video is
  shorter than a window, that window naturally contains the whole video.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.core.config import settings

# Window name -> upper time bound in seconds (exclusive). "full" = everything.
WINDOW_SECONDS: dict[str, float] = {
    "first_1s": 1.0,
    "first_3s": 3.0,
    "first_6s": 6.0,
    "full": math.inf,
}


def window_indices(timestamps: list[float], name: str) -> list[int]:
    """Indices of frames whose timestamp falls within a window.

    Shared by feature modules that hold per-frame metric arrays parallel to the
    sampled frames. Mirrors `FrameSample.window` semantics, including the
    short-video fallback of returning at least the first frame.
    """
    bound = WINDOW_SECONDS[name]
    if math.isinf(bound):
        indices = list(range(len(timestamps)))
    else:
        indices = [i for i, t in enumerate(timestamps) if t < bound]
    if not indices and timestamps:
        indices = [0]
    return indices


@dataclass
class SampledFrame:
    """A single sampled frame and the timestamp (seconds) it was taken at."""

    timestamp: float
    image: np.ndarray  # BGR uint8 at processing resolution


@dataclass
class FrameSample:
    """The full result of sampling a video."""

    source_width: int
    source_height: int
    fps: float
    duration_seconds: float
    proc_width: int
    proc_height: int
    frames: list[SampledFrame] = field(default_factory=list)

    def window(self, name: str) -> list[SampledFrame]:
        """Return the sampled frames whose timestamp falls within a window."""
        bound = WINDOW_SECONDS[name]
        if math.isinf(bound):
            return self.frames
        subset = [f for f in self.frames if f.timestamp < bound]
        # Guarantee at least the first frame for very short videos (PRD 11.1).
        if not subset and self.frames:
            subset = [self.frames[0]]
        return subset

    @property
    def is_empty(self) -> bool:
        return not self.frames


def _target_dimensions(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Scale (w, h) so the longer side is <= max_side; never upscale."""
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def sample_frames(
    video_path: str,
    sample_fps: float | None = None,
    max_side: int | None = None,
) -> FrameSample:
    """Decode a video and return frames sampled at ~`sample_fps` per second.

    Raises RuntimeError if the video cannot be opened (the caller decides how to
    map that onto a pipeline-step failure).
    """
    sample_fps = sample_fps or settings.sample_fps
    max_side = max_side or settings.frame_processing_max_side

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for frame sampling: {video_path}")

    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0 or math.isnan(source_fps):
            source_fps = 30.0  # sensible fallback when the container omits fps

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        proc_width, proc_height = _target_dimensions(source_width, source_height, max_side)
        sample_interval = 1.0 / sample_fps

        frames: list[SampledFrame] = []
        frame_index = 0
        next_sample_time = 0.0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            if timestamp + 1e-9 >= next_sample_time:
                if (proc_width, proc_height) != (source_width, source_height):
                    frame = cv2.resize(
                        frame, (proc_width, proc_height), interpolation=cv2.INTER_AREA
                    )
                frames.append(SampledFrame(timestamp=round(timestamp, 4), image=frame))
                next_sample_time += sample_interval
            frame_index += 1
    finally:
        capture.release()

    duration_seconds = (frame_count / source_fps) if frame_count else (
        frames[-1].timestamp if frames else 0.0
    )

    return FrameSample(
        source_width=source_width,
        source_height=source_height,
        fps=round(source_fps, 4),
        duration_seconds=round(duration_seconds, 4),
        proc_width=proc_width,
        proc_height=proc_height,
        frames=frames,
    )
