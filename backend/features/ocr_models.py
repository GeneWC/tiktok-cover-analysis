"""EAST text-detector model management.

The OCR presence feature uses OpenCV's DNN module with the EAST text detector
(`frozen_east_text_detection.pb`). EAST localizes text *regions* (boxes), which
is exactly what PRD 11.7 needs (presence + approximate area, not content), and
runs on the OpenCV we already depend on - no extra Python package.

The frozen graph (~96 MB) is downloaded on first use and cached locally, like
the MediaPipe bundles. The file is git-ignored.
"""

from __future__ import annotations

import urllib.request

from backend.core.config import settings

_EAST_FILENAME = "frozen_east_text_detection.pb"
_EAST_URL = (
    "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/"
    "frozen_east_text_detection.pb"
)


def ensure_east_model() -> str:
    """Return a local path to the EAST graph, downloading it if missing."""
    destination = settings.ocr_models_dir / _EAST_FILENAME
    if not destination.exists():
        settings.ocr_models_dir.mkdir(parents=True, exist_ok=True)
        # Download to a temp file then rename, so an interrupted download never
        # leaves a corrupt "complete" file behind.
        tmp = destination.with_suffix(destination.suffix + ".part")
        urllib.request.urlretrieve(_EAST_URL, tmp)
        tmp.replace(destination)
    return str(destination)
