"""On-screen text *content* features (role + lexicon), not presence.

EAST localizes boxes; this module optionally reads the crop (EasyOCR if
installed) then scores *what* the text is doing. Raw strings never enter the
model: handles, URLs, and an optional creator username are redacted first.

EasyOCR is optional. If boxes exist but no reader is available, role ratios
from geometry are still filled and lexicon/count fields are null with
`text_read_failed = 1`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from backend.features.frame_sampling import FrameSample, window_indices

ReadCrops = Callable[[list[np.ndarray]], list[str] | None]

TEXT_SEMANTIC_KEYS: tuple[str, ...] = (
    "text_titlecard_ratio",
    "text_caption_ratio",
    "text_corner_watermark_ratio",
    "text_char_count",
    "text_unique_token_count",
    "text_first_3s_char_count",
    "text_has_song_or_piece_cue",
    "text_has_cta",
    "text_has_question_hook",
    "text_has_social_handle",
    "text_script_latin_ratio",
    "text_has_cjk",
    "text_all_caps_ratio",
    "text_read_failed",
)

_HANDLE_RE = re.compile(r"@[\w.]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]"
)
_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")

_SONG_CUES = frozenset(
    {
        "cover",
        "piano",
        "violin",
        "cello",
        "guitar",
        "playing",
        "played",
        "song",
        "piece",
        "etude",
        "concerto",
        "sonata",
        "bpm",
        "op",
        "mvmt",
        "movement",
        "arrangement",
        "instrumental",
    }
)
_CTA_CUES = frozenset(
    {
        "follow",
        "like",
        "duet",
        "stitch",
        "subscribe",
        "comment",
        "share",
        "part",
        "bio",
        "link",
    }
)
_QUESTION_WORDS = frozenset({"who", "what", "why", "how", "when", "where", "which"})

_CROP_PAD = 0.08
_MAX_CROPS = 24


@dataclass(frozen=True)
class TextDetection:
    """One EAST box in normalized processing-frame coordinates (0-1)."""

    cx: float
    cy: float
    width: float
    height: float
    conf: float


def _empty_features(failed: bool) -> dict[str, float | int | None]:
    if failed:
        out: dict[str, float | int | None] = {key: None for key in TEXT_SEMANTIC_KEYS}
        out["text_read_failed"] = 1
        return out
    return {
        "text_titlecard_ratio": 0.0,
        "text_caption_ratio": 0.0,
        "text_corner_watermark_ratio": 0.0,
        "text_char_count": 0,
        "text_unique_token_count": 0,
        "text_first_3s_char_count": 0,
        "text_has_song_or_piece_cue": 0,
        "text_has_cta": 0,
        "text_has_question_hook": 0,
        "text_has_social_handle": 0,
        "text_script_latin_ratio": 0.0,
        "text_has_cjk": 0,
        "text_all_caps_ratio": 0.0,
        "text_read_failed": 0,
    }


def redact_ocr_text(text: str, creator_username: str | None = None) -> str:
    """Strip @handles, URLs, and the optional creator name (NLP leakage rule)."""
    cleaned = _HANDLE_RE.sub(" ", text)
    cleaned = _URL_RE.sub(" ", cleaned)
    if creator_username:
        cleaned = re.sub(re.escape(creator_username), " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str) -> list[str]:
    """Letter-run tokens; no stemming (lexicon match is exact/lowercased)."""
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def role_for_box(box: TextDetection) -> str | None:
    """titlecard / caption / corner / None from geometry only."""
    area = box.width * box.height
    cx, cy = box.cx, box.cy
    if (cx < 0.22 or cx > 0.78) and (cy < 0.22 or cy > 0.78):
        return "corner"
    if cy > 0.72:
        return "caption"
    if 0.25 <= cx <= 0.75 and 0.25 <= cy <= 0.75 and area >= 0.04:
        return "titlecard"
    return None


def _crop_box(image_bgr: np.ndarray, box: TextDetection) -> np.ndarray | None:
    height, width = image_bgr.shape[:2]
    x0 = int(max(0.0, (box.cx - box.width / 2 - _CROP_PAD) * width))
    x1 = int(min(width, (box.cx + box.width / 2 + _CROP_PAD) * width))
    y0 = int(max(0.0, (box.cy - box.height / 2 - _CROP_PAD) * height))
    y1 = int(min(height, (box.cy + box.height / 2 + _CROP_PAD) * height))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return image_bgr[y0:y1, x0:x1]


def default_read_crops(crops: list[np.ndarray]) -> list[str] | None:
    """EasyOCR if installed; None means 'reader unavailable' (missingness)."""
    if not crops:
        return []
    try:
        import easyocr  # type: ignore
    except ImportError:
        return None

    reader = getattr(default_read_crops, "_reader", None)
    if reader is None:
        reader = easyocr.Reader(["en", "ch_sim"], gpu=False, verbose=False)
        default_read_crops._reader = reader  # type: ignore[attr-defined]

    texts: list[str] = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        lines = reader.readtext(rgb, detail=0, paragraph=True)
        texts.append(" ".join(str(line) for line in lines))
    return texts


def _lexicon_flags(raw_text: str, redacted: str) -> dict[str, float | int]:
    lowered = redacted.lower()
    tokens = [t.lower() for t in tokenize(redacted)]
    token_set = set(tokens)
    letters = [ch for ch in redacted if ch.isalpha()]
    latin = sum(1 for ch in letters if _LATIN_RE.fullmatch(ch))
    latin_ratio = (latin / len(letters)) if letters else 0.0
    letter_tokens = [t for t in tokenize(redacted) if t.isalpha() and len(t) > 1]
    caps = sum(1 for t in letter_tokens if t.isupper())
    caps_ratio = (caps / len(letter_tokens)) if letter_tokens else 0.0
    has_question = ("?" in raw_text) or bool(token_set & _QUESTION_WORDS)
    return {
        "text_char_count": len(redacted),
        "text_unique_token_count": len(token_set),
        "text_has_song_or_piece_cue": int(bool(token_set & _SONG_CUES) or "key of" in lowered),
        "text_has_cta": int(
            bool(token_set & _CTA_CUES) or "part 2" in lowered or "link in bio" in lowered
        ),
        "text_has_question_hook": int(has_question),
        "text_has_social_handle": int(bool(_HANDLE_RE.search(raw_text))),
        "text_script_latin_ratio": round(latin_ratio, 4),
        "text_has_cjk": int(bool(_CJK_RE.search(raw_text))),
        "text_all_caps_ratio": round(caps_ratio, 4),
    }


def extract_text_semantics_features(
    sample: FrameSample,
    detections_per_frame: Sequence[Sequence[TextDetection]],
    *,
    read_crops: ReadCrops | None = None,
    creator_username: str | None = None,
) -> dict[str, float | int | None]:
    """Score text roles and (if readable) redacted lexicon flags."""
    if sample.is_empty:
        return _empty_features(failed=True)
    if len(detections_per_frame) != len(sample.frames):
        raise ValueError("detections_per_frame must align with sample.frames")

    timestamps = [f.timestamp for f in sample.frames]
    first_3s = set(window_indices(timestamps, "first_3s"))

    title_hits = 0
    caption_hits = 0
    corner_hits = 0
    n_first_3s = max(len(first_3s), 1)
    n_all = max(len(sample.frames), 1)

    crops: list[np.ndarray] = []
    crop_in_first_3s: list[bool] = []
    any_box = False
    saw_handle = False
    first_3s_raw_parts: list[str] = []

    for i, (frame, boxes) in enumerate(zip(sample.frames, detections_per_frame)):
        if not boxes:
            continue
        any_box = True
        roles = {role_for_box(box) for box in boxes}
        if i in first_3s and "titlecard" in roles:
            title_hits += 1
        if "caption" in roles:
            caption_hits += 1
        if "corner" in roles:
            corner_hits += 1
        for box in boxes:
            if len(crops) >= _MAX_CROPS:
                break
            crop = _crop_box(frame.image, box)
            if crop is None:
                continue
            crops.append(crop)
            crop_in_first_3s.append(i in first_3s)

    out = _empty_features(failed=False)
    out["text_titlecard_ratio"] = round(title_hits / n_first_3s, 4)
    out["text_caption_ratio"] = round(caption_hits / n_all, 4)
    out["text_corner_watermark_ratio"] = round(corner_hits / n_all, 4)

    if not any_box:
        return out

    reader = read_crops or default_read_crops
    texts = reader(crops)
    if texts is None:
        for key in (
            "text_char_count",
            "text_unique_token_count",
            "text_first_3s_char_count",
            "text_has_song_or_piece_cue",
            "text_has_cta",
            "text_has_question_hook",
            "text_has_social_handle",
            "text_script_latin_ratio",
            "text_has_cjk",
            "text_all_caps_ratio",
        ):
            out[key] = None
        out["text_read_failed"] = 1
        return out

    raw_joined = " ".join(texts)
    saw_handle = bool(_HANDLE_RE.search(raw_joined))
    redacted = redact_ocr_text(raw_joined, creator_username=creator_username)
    flags = _lexicon_flags(raw_joined, redacted)
    flags["text_has_social_handle"] = int(saw_handle)
    out.update(flags)

    for text, in_hook in zip(texts, crop_in_first_3s):
        if in_hook:
            first_3s_raw_parts.append(redact_ocr_text(text, creator_username))
    out["text_first_3s_char_count"] = sum(len(p) for p in first_3s_raw_parts)
    if first_3s_raw_parts:
        hook_raw = " ".join(first_3s_raw_parts)
        out["text_has_question_hook"] = int(
            ("?" in hook_raw)
            or bool(set(t.lower() for t in tokenize(hook_raw)) & _QUESTION_WORDS)
            or flags["text_has_question_hook"]
        )
    out["text_read_failed"] = 0
    return out
