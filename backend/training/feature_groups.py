"""Canonical feature-group partition (used for per-model feature selection).

Phase-4 experiments showed that feeding all 70 features to every model buries
the (small) cross-creator signal - narrowing to the right groups matters a lot
(e.g. `top_quartile` framing+visual >> all-features). This module is the single
source of truth mapping each feature to exactly one semantic group, so model
specs can request feature subsets by group name and inference can reproduce them.

Assignment is first-match over ordered keyword rules, which guarantees a clean
partition even if a name could match multiple keywords (e.g. `hand_motion_*`
resolves to motion, not framing).
"""

from __future__ import annotations

# (group name, keyword fragments). Order matters: first match wins.
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("audio", ("audio_",)),
    ("text", ("text_", "first_text", "average_text", "ocr_failed")),
    ("motion", ("motion_energy", "motion_consistency", "hand_motion", "camera_stability")),
    (
        "framing",
        (
            "person_visible", "face_visible", "hand_visible", "upper_body",
            "subject_centering", "subject_size", "face_size", "hand_detection_failed",
        ),
    ),
    ("visual", ("brightness", "contrast", "sharpness", "blur", "colorfulness")),
    (
        "metadata",
        (
            "duration_seconds", "fps", "width", "height", "aspect_ratio",
            "resolution_area", "bitrate", "has_audio", "is_vertical", "is_square",
        ),
    ),
)

GROUP_NAMES: tuple[str, ...] = tuple(name for name, _ in _RULES)


def assign_group(feature: str) -> str | None:
    """Return the group a feature belongs to (first-match), or None if unmapped."""
    for name, keywords in _RULES:
        if any(k in feature for k in keywords):
            return name
    return None


def select_group_features(all_features: list[str], groups: tuple[str, ...]) -> list[str]:
    """Features (in original order) belonging to any of the given groups.

    An empty `groups` means "all features" (no subsetting).
    """
    if not groups:
        return list(all_features)
    wanted = set(groups)
    return [f for f in all_features if assign_group(f) in wanted]
