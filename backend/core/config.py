"""Application configuration.

Centralizes every tunable value so the rest of the codebase never hardcodes
paths, limits, or magic numbers. Values can be overridden via environment
variables (or a .env file) without touching code.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (backend/core/config.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Strongly-typed app settings.

    Each attribute is validated by Pydantic. Override any of them with an env
    var of the same name (case-insensitive), e.g. ZUKOVER_MAX_DURATION_SECONDS=60.
    """

    model_config = SettingsConfigDict(
        env_prefix="ZUKOVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Zukover API"
    app_version: str = "0.1.0"

    # Where uploaded videos and generated reports are stored (local FS for MVP).
    data_dir: Path = PROJECT_ROOT / "data"
    videos_dir: Path = PROJECT_ROOT / "data" / "videos"
    reports_dir: Path = PROJECT_ROOT / "data" / "reports"

    # SQLite database file holding analysis job records (PRD section 21).
    db_path: Path = PROJECT_ROOT / "data" / "zukover.db"

    # Directory for downloaded MediaPipe Tasks model bundles (.task/.tflite).
    mediapipe_models_dir: Path = PROJECT_ROOT / "models" / "mediapipe"

    # Directory for the downloaded EAST text-detection model (OCR presence).
    ocr_models_dir: Path = PROJECT_ROOT / "models" / "ocr"
    # EAST confidence threshold for a text region to count (PRD 11.7).
    ocr_confidence_threshold: float = 0.5

    # Upload validation limits (see PRD section 14.2).
    max_duration_seconds: float = 120.0
    min_duration_seconds: float = 1.0
    supported_extensions: tuple[str, ...] = (".mp4", ".mov", ".m4v")

    # Reject uploads larger than this (PRD 22.1 "file too large").
    max_file_size_mb: int = 200

    # Frame sampling for visual feature extraction (PRD 11.1).
    # Sample this many frames per second (PRD recommends 2-5).
    sample_fps: float = 3.0
    # Resize sampled frames so the longer side is at most this many pixels
    # (consistent processing resolution; never upscales).
    frame_processing_max_side: int = 256

    # --- Offline training pipeline (PRD 8-10, 20) ---
    # User-provided real dataset: TikTok video files + their engagement metrics.
    downloads_dir: Path = PROJECT_ROOT / "downloads"
    engagement_csv: Path = PROJECT_ROOT / "engagement.csv"
    # Creator seed list (PRD 8.1) and the generated training artifacts (PRD 9).
    creators_csv: Path = PROJECT_ROOT / "data" / "creators.csv"
    raw_training_csv: Path = PROJECT_ROOT / "data" / "raw_tiktok_training_data.csv"
    video_features_csv: Path = PROJECT_ROOT / "data" / "video_features.csv"
    training_dataset_csv: Path = PROJECT_ROOT / "data" / "training_dataset.csv"

    # Trained model artifacts (PRD 13 / 18.2): feature schema, fitted
    # imputer/scaler, classifiers/regressors, importances, training metadata.
    models_dir: Path = PROJECT_ROOT / "backend" / "models"

    # Cross-origin requests allowed from the frontend dev server.
    cors_allow_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )


# A single shared instance imported throughout the app.
settings = Settings()

# Ensure storage directories exist at startup so later phases can write to them.
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.videos_dir.mkdir(parents=True, exist_ok=True)
settings.reports_dir.mkdir(parents=True, exist_ok=True)
