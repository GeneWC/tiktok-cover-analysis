"""Zukover FastAPI application entrypoint.

Creates the FastAPI app instance, configures middleware, registers routers, and
exposes basic health endpoints. Run locally with:

    uvicorn backend.app:app --reload

Interactive API docs are then served at http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import analyze_routes, channel_routes
from backend.core.config import PROJECT_ROOT, settings
from backend.core.database import init_db
from backend.core.rate_limit import UploadRateLimitMiddleware
from backend.inference.model_registry import get_registry
from backend.services.job_cleanup import cleanup_expired_jobs

# Built Vite output. Present in the Railway image; usually absent during local
# `uvicorn` so the JSON landing payload at `/` stays available for API-only use.
_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
_SPA_INDEX = _FRONTEND_DIST / "index.html"
_SPA_RESERVED = frozenset({"docs", "redoc", "openapi.json", "health"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook.

    Code before `yield` runs once on startup; code after runs on shutdown. We
    warm the model registry here so artifacts are read from disk once (not on the
    first request); if they're missing the app still boots (uploads/validation
    keep working) and the analysis pipeline will surface the error per-job.
    """
    # --- startup ---
    init_db()  # create SQLite tables if they don't exist yet
    cleanup_expired_jobs()
    try:
        get_registry()
        app.state.models_ready = True
    except FileNotFoundError:
        app.state.models_ready = False
    app.state.ready = True
    yield
    # --- shutdown ---
    # (nothing to clean up yet)


# The single application object. Uvicorn imports this as `backend.app:app`.
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "TikTok instrumental cover video analyzer. Upload a single video for "
        "exploratory presentation scores, or a channel batch (≥5 videos with "
        "optional views) for within-creator diagnostics. Not a virality forecast."
    ),
    lifespan=lifespan,
)

# CORS lets the browser-based frontend (a different origin) call this API.
# Without this, browsers block cross-origin requests for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(UploadRateLimitMiddleware)

# Mount the analyze + channel routers onto the app.
app.include_router(analyze_routes.router)
app.include_router(channel_routes.router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe used by tooling/deployments to check the app is up."""
    return {"status": "ok"}


def _serve_frontend(application: FastAPI) -> None:
    """Serve the built React app from the same origin as the API.

    Registered last so `/api/*`, `/health`, and `/docs` keep their handlers.
    Local `uvicorn` without a `frontend/dist` build keeps the JSON `/` payload.
    """
    if not _SPA_INDEX.is_file():
        @application.get("/", tags=["health"])
        def root() -> dict[str, str]:
            """Friendly landing payload identifying the service."""
            return {
                "service": settings.app_name,
                "version": settings.app_version,
                "docs": "/docs",
            }
        return

    assets = _FRONTEND_DIST / "assets"
    if assets.is_dir():
        application.mount(
            "/assets", StaticFiles(directory=assets), name="frontend-assets"
        )

    @application.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/") or full_path in _SPA_RESERVED:
            raise HTTPException(status_code=404, detail="Not found")
        if full_path:
            candidate = (_FRONTEND_DIST / full_path).resolve()
            try:
                candidate.relative_to(_FRONTEND_DIST.resolve())
            except ValueError as exc:
                raise HTTPException(status_code=404, detail="Not found") from exc
            if candidate.is_file():
                return FileResponse(candidate)
        return FileResponse(_SPA_INDEX)


_serve_frontend(app)
