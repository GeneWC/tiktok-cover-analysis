"""Model dataset loading + leakage-safe feature selection (PRD 12.1 / 12.4).

Turns `training_dataset.csv` into the matrices the trainers consume:
- **X**: the leakage-safe feature matrix - only video-derived features that can
  also be computed from a user upload (PRD 6.2). Every identifier, raw metric,
  and derived label is dropped (PRD 12.4), so a label can never sneak in as an
  input feature.
- **y**: a chosen target column (primary classifier or a secondary regressor).
- **groups**: `creator_username`, used for GroupKFold so no creator appears in
  both train and test (PRD 12.7).

Why a dedicated module: feature/target/group selection is the one place leakage
can silently break the whole project, so it lives behind a single audited
function (`select_feature_columns`) that the trainers and tests share.

Type handling: the CSV stores booleans as "True"/"False" and missing values as
blanks. Feature columns are coerced to numeric here (bool-like -> 1/0, blank ->
NaN); the actual imputation/scaling happens in the Step 2 preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.config import settings
from backend.training.compute_labels import LABEL_FIELDS

# The per-video extraction status carried alongside the features, and the audio
# group's string status flag - neither is a model input feature.
_STATUS_COLUMN = "video_feature_extraction_status"
_AUDIO_STATUS_COLUMN = "audio_feature_extraction_status"

# Columns that must NEVER be used as model inputs (PRD 12.4): every identifier,
# raw metric, and derived label (the whole label block) plus the status columns.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(LABEL_FIELDS) | {
    _STATUS_COLUMN,
    _AUDIO_STATUS_COLUMN,
}

# Training targets (PRD 12.1 / 12.2). The classifier target is primary.
PRIMARY_TARGET = "top_quartile_for_creator"
CLASSIFICATION_TARGETS: tuple[str, ...] = (PRIMARY_TARGET,)
REGRESSION_TARGETS: tuple[str, ...] = (
    "creator_relative_log_views",
    "engagement_rate",
    "share_rate",
)

# Column carrying the GroupKFold grouping key.
GROUP_COLUMN = "creator_username"

_BOOL_STRINGS = {"true": 1.0, "false": 0.0}


def select_feature_columns(columns: list[str]) -> list[str]:
    """Return the ordered leakage-safe feature columns from a header.

    Drops everything in `NON_FEATURE_COLUMNS`; preserves the dataset's column
    order so the feature vector layout is stable and reproducible.
    """
    return [c for c in columns if c not in NON_FEATURE_COLUMNS]


def assert_no_leakage(feature_names: list[str]) -> None:
    """Guard: raise if any banned identifier/label/metric is among the features."""
    leaked = sorted(set(feature_names) & NON_FEATURE_COLUMNS)
    if leaked:
        raise ValueError(f"Leakage: banned column(s) used as features: {leaked}")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a column to float: bool-like -> 1/0, blanks/non-numeric -> NaN."""
    if series.dtype == bool:
        return series.astype(float)
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(_BOOL_STRINGS)
    numeric = pd.to_numeric(text, errors="coerce")
    return mapped.fillna(numeric).astype(float)


@dataclass
class ModelDataset:
    """Loaded training table split into features, targets, and groups."""

    frame: pd.DataFrame          # full source table (labels + features)
    X: pd.DataFrame              # coerced numeric, leakage-safe feature matrix
    feature_names: list[str]     # ordered model-input feature columns

    @property
    def groups(self) -> np.ndarray:
        """Creator usernames aligned to X rows (GroupKFold grouping key)."""
        return self.frame[GROUP_COLUMN].to_numpy()

    def target(self, name: str) -> pd.Series:
        """Coerced target column (classification -> 1/0, regression -> float)."""
        return _coerce_numeric(self.frame[name])

    def xy(self, target_name: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Return (X, y, groups) for a target, dropping rows where y is missing.

        Per PRD 8.5, a video with no value for the chosen target (e.g. no
        comments -> no engagement_rate) is excluded from that model only.
        """
        y = self.target(target_name)
        mask = y.notna().to_numpy()
        return self.X.loc[mask], y[mask], self.groups[mask]


def load_model_dataset(path: str | Path | None = None) -> ModelDataset:
    """Load `training_dataset.csv` into a leakage-safe `ModelDataset`."""
    path = Path(path or settings.training_dataset_csv)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    feature_names = select_feature_columns(list(frame.columns))
    assert_no_leakage(feature_names)

    X = pd.DataFrame(
        {name: _coerce_numeric(frame[name]) for name in feature_names},
        index=frame.index,
    )
    return ModelDataset(frame=frame, X=X, feature_names=feature_names)
