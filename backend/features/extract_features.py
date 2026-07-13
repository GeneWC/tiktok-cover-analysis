"""Feature-extraction orchestrator (PRD 11 / 12).

Single entry point that turns a video file into one flat, fixed-schema feature
vector consumed by *both* the offline training pipeline and online inference.
Building the vector in one place guarantees training and inference see the exact
same feature names, ordering, and null handling.

Responsibilities:
- Decode/sample frames ONCE and share the result across all visual extractors.
- Run hand detection ONCE: the framing step returns the per-frame hand centroids
  and we hand them to the motion step instead of running MediaPipe twice.
- Isolate failures per group (PRD 22.2): if one extractor raises, that group's
  features fall back to nulls and its step is marked "failed"; the rest continue.
- Report per-step status using the canonical PIPELINE_STEPS names the API and
  frontend already rely on.

Schema note: feature groups are merged in a fixed order, so `dict` insertion
order defines the vector layout. The only key collision across groups is
`hand_detection_failed` (emitted by both framing and motion); motion's copy is
renamed to `hand_motion_detection_failed` on merge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.features.audio_features import _empty_features as _empty_audio
from backend.features.audio_features import extract_audio_features
from backend.features.frame_sampling import FrameSample, sample_frames
from backend.features.framing_features import _empty_features as _empty_framing
from backend.features.framing_features import extract_framing_features
from backend.features.metadata_features import _empty_metadata
from backend.features.metadata_features import extract_metadata_features
from backend.features.motion_features import _empty_features as _empty_motion
from backend.features.motion_features import extract_motion_features
from backend.features.ocr_features import _empty_features as _empty_ocr
from backend.features.ocr_features import extract_ocr_features
from backend.features.visual_quality_features import _FEATURE_KEYS as _VQ_KEYS
from backend.features.visual_quality_features import extract_visual_quality_features

# Order in which feature groups are concatenated into the flat vector.
_GROUP_ORDER = ("metadata", "visual_quality", "framing", "motion", "audio", "ocr")

_STATUS_OK = "complete"
_STATUS_FAILED = "failed"


def _empty_visual_quality() -> dict[str, float | None]:
    return {key: None for key in _VQ_KEYS}


def _merge_motion(features: dict) -> dict:
    """Rename motion's `hand_detection_failed` to avoid clashing with framing's."""
    out = dict(features)
    if "hand_detection_failed" in out:
        out["hand_motion_detection_failed"] = out.pop("hand_detection_failed")
    return out


@dataclass
class FeatureExtractionResult:
    """The full per-video feature record plus per-step extraction status."""

    features: dict[str, object]
    steps: dict[str, str] = field(default_factory=dict)
    frames_sampled: int = 0
    duration_seconds: float | None = None

    @property
    def ok(self) -> bool:
        """True if every feature group extracted successfully."""
        return all(status == _STATUS_OK for status in self.steps.values())


def extract_all_features(
    video_path: str, sample: FrameSample | None = None
) -> FeatureExtractionResult:
    """Run the full Phase 2 feature pipeline on one video.

    `sample` may be passed in if frames were already decoded elsewhere; otherwise
    the video is sampled here. Never raises for per-group failures — those are
    captured in the returned `steps` map.
    """
    steps: dict[str, str] = {}
    groups: dict[str, dict] = {}

    # --- metadata (container-level, independent of frame sampling) ---
    try:
        groups["metadata"] = extract_metadata_features(video_path)
        steps["metadata"] = _STATUS_OK
    except Exception:  # noqa: BLE001 - isolate group failure (PRD 22.2)
        groups["metadata"] = _empty_metadata()
        steps["metadata"] = _STATUS_FAILED

    # --- frame sampling (shared foundation for all visual features) ---
    if sample is None:
        try:
            sample = sample_frames(video_path)
            steps["frame_sampling"] = _STATUS_OK
        except Exception:  # noqa: BLE001
            sample = FrameSample(0, 0, 0.0, 0.0, 0, 0)
            steps["frame_sampling"] = _STATUS_FAILED
    else:
        steps["frame_sampling"] = _STATUS_OK if not sample.is_empty else _STATUS_FAILED

    # --- visual quality ---
    try:
        groups["visual_quality"] = extract_visual_quality_features(sample)
        steps["visual_quality"] = _STATUS_OK if not sample.is_empty else _STATUS_FAILED
    except Exception:  # noqa: BLE001
        groups["visual_quality"] = _empty_visual_quality()
        steps["visual_quality"] = _STATUS_FAILED

    # --- framing (also yields hand centroids to reuse in motion) ---
    hand_positions = None
    try:
        groups["framing"], hand_positions = extract_framing_features(
            sample, return_hand_positions=True
        )
        steps["framing"] = _STATUS_OK if not sample.is_empty else _STATUS_FAILED
    except Exception:  # noqa: BLE001
        groups["framing"] = _empty_framing()
        steps["framing"] = _STATUS_FAILED

    # --- motion (reuses framing's hand detection) ---
    try:
        motion = extract_motion_features(sample, hand_positions=hand_positions)
        groups["motion"] = _merge_motion(motion)
        steps["motion"] = _STATUS_OK if len(sample.frames) >= 2 else _STATUS_FAILED
    except Exception:  # noqa: BLE001
        groups["motion"] = _merge_motion(_empty_motion())
        steps["motion"] = _STATUS_FAILED

    # --- audio ---
    try:
        audio = extract_audio_features(video_path)
        groups["audio"] = audio
        steps["audio"] = (
            _STATUS_OK
            if audio.get("audio_feature_extraction_status") == "ok"
            else _STATUS_FAILED
        )
    except Exception:  # noqa: BLE001
        groups["audio"] = _empty_audio(_STATUS_FAILED)
        steps["audio"] = _STATUS_FAILED

    # --- OCR text presence ---
    try:
        ocr = extract_ocr_features(sample)
        groups["ocr"] = ocr
        steps["ocr"] = _STATUS_FAILED if ocr.get("ocr_failed") else _STATUS_OK
    except Exception:  # noqa: BLE001
        groups["ocr"] = _empty_ocr(failed=True)
        steps["ocr"] = _STATUS_FAILED

    # Flatten groups into one ordered feature vector.
    features: dict[str, object] = {}
    for group in _GROUP_ORDER:
        features.update(groups[group])

    return FeatureExtractionResult(
        features=features,
        steps=steps,
        frames_sampled=len(sample.frames),
        duration_seconds=sample.duration_seconds or None,
    )
