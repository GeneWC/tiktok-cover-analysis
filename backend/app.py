"""CoverSignal FastAPI application entrypoint.

Creates the FastAPI app instance, configures middleware, registers routers, and
exposes basic health endpoints. Run locally with:

    uvicorn backend.app:app --reload

Interactive API docs are then served at http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import analyze_routes
from backend.core.config import settings
from backend.core.database import init_db
from backend.inference.model_registry import get_registry


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
        "TikTok instrumental cover video analyzer. Upload a single video to "
        "receive presentation scores, predicted performance tiers, and "
        "recommendations. Predictions are exploratory."
    ),
    lifespan=lifespan,
)

# CORS lets the browser-based frontend (a different origin) call this API.
# Without this, browsers block cross-origin requests for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the analyze router (and any future routers) onto the app.
app.include_router(analyze_routes.router)


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    """Friendly landing payload identifying the service."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe used by tooling/deployments to check the app is up."""
    return {"status": "ok"}
