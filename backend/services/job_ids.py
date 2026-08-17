"""Server-generated job IDs. Never trust a client-supplied path fragment."""

from __future__ import annotations

import re

ANALYSIS_ID_RE = re.compile(r"^analysis_[0-9a-f]{12}$")
CHANNEL_ID_RE = re.compile(r"^channel_[0-9a-f]{12}$")


def is_analysis_id(value: str) -> bool:
    return bool(ANALYSIS_ID_RE.fullmatch(value or ""))


def is_channel_id(value: str) -> bool:
    return bool(CHANNEL_ID_RE.fullmatch(value or ""))
