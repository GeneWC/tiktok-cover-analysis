"""In-memory per-IP rate limit for the public upload endpoints."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from backend.core.config import settings

_LIMITED_PATHS = ("/api/analyze", "/api/channel/diagnose")


class UploadRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int | None = None, window_seconds: int | None = None):
        super().__init__(app)
        self._limit_override = limit
        self._window_override = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _limited(self, request: Request) -> bool:
        if request.method != "POST":
            return False
        path = request.url.path.rstrip("/")
        return path in {p.rstrip("/") for p in _LIMITED_PATHS}

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._limited(request):
            return await call_next(request)
        limit = (
            self._limit_override
            if self._limit_override is not None
            else settings.upload_rate_limit
        )
        window = (
            self._window_override
            if self._window_override is not None
            else settings.upload_rate_window_seconds
        )
        if limit <= 0:
            return await call_next(request)
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = self._hits[ip]
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= limit:
            return JSONResponse(
                {"detail": "Too many uploads. Try again in a few minutes."},
                status_code=429,
            )
        bucket.append(now)
        return await call_next(request)
