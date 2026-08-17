"""Model-family + feature-group search on frozen val creators (Exp C1).

Train on train creators only; score val only. Never touches the frozen test set
or production `backend/models/` artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.training.baselines import (
    available_simple_features,
    build_hist_gradient_pipeline,
    build_rf_control_pipeline,
    build_simple_logistic_pipeline,
)
from backend.training.classifier import CLASSIFIER_PARAMS, build_classifier
from backend.training.creator_splits import CreatorSplitMembership, load_creator_splits
from backend.training.feature_groups import select_group_features
from backend.training.heldout_eval import HeldoutResult, evaluate_estimator_heldout
from backend.training.model_dataset import ModelDataset
from backend.training.model_specs import PRIMARY_SPEC
from backend.training.preprocessing import build_preprocessor_for

# Val AUC lift vs RF control required to flag PROMISING (also must beat simple_logistic).
PROMISING_AUC_MARGIN = 0.02

EARLY_WINDOW_TOKENS: tuple[str, ...] = ("first_1s", "first_3s")


@dataclass(frozen=True)
class CandidateSpec:
    """One named estimator + feature subset to score on val."""

    name: str
    kind: str  # "model_family" | "feature_ablation" | "baseline"
    feature_set: str
    builder: Callable[[pd.DataFrame], object] = field(repr=False)


def select_early_window_features(all_features: list[str]) -> list[str]:
    """Features whose names contain early-window tokens (first_1s / first_3s)."""
    return [f for f in all_features if any(tok in f for tok in EARLY_WINDOW_TOKENS)]


def resolve_feature_set(all_features: list[str], feature_set: str) -> list[str]:
    """Map a named ablation key to concrete feature columns (original order)."""
    key = feature_set.strip().lower()
    if key in {"framing+visual", "framing_visual", "control"}:
        return select_group_features(all_features, PRIMARY_SPEC.feature_groups)
    if key in {"all", "all_features"}:
        return select_group_features(all_features, ())
    if key in {"audio", "audio-only", "audio_only"}:
        return select_group_features(all_features, ("audio",))
    if key in {"motion+visual", "motion_visual"}:
        return select_group_features(all_features, ("motion", "visual"))
    if key in {"early-window", "early_window", "early-window-only", "early_window_only"}:
        return select_early_window_features(all_features)
    if key == "simple_logistic":
        return available_simple_features(all_features)
    raise ValueError(f"Unknown feature_set '{feature_set}'")


def build_logistic_balanced_pipeline(feature_frame: pd.DataFrame) -> Pipeline:
    """Balanced logistic on the given feature columns (impute+scale)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor_for(feature_frame)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_rf_regularized_pipeline(feature_frame: pd.DataFrame) -> Pipeline:
    """RF control hyperparams with stronger leaf/depth regularization."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor_for(feature_frame)),
            (
                "model",
                build_classifier(min_samples_leaf=10, max_depth=8),
            ),
        ]
    )


def build_select_from_model_logistic_pipeline(
    feature_frame: pd.DataFrame,
    max_features: int = 15,
) -> Pipeline:
    """RF importance SelectFromModel (top-k) then balanced logistic."""
    selector_rf = RandomForestClassifier(**CLASSIFIER_PARAMS)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor_for(feature_frame)),
            (
                "select",
                SelectFromModel(
                    estimator=selector_rf,
                    max_features=max_features,
                    threshold=-np.inf,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def iter_candidate_specs() -> list[CandidateSpec]:
    """Model-family candidates + feature ablations + simple_logistic reference."""
    candidates: list[CandidateSpec] = [
        CandidateSpec(
            name="simple_logistic",
            kind="baseline",
            feature_set="simple_logistic",
            builder=build_simple_logistic_pipeline,
        ),
        CandidateSpec(
            name="rf_control",
            kind="model_family",
            feature_set="framing+visual",
            builder=build_rf_control_pipeline,
        ),
        CandidateSpec(
            name="logistic_balanced",
            kind="model_family",
            feature_set="framing+visual",
            builder=build_logistic_balanced_pipeline,
        ),
        CandidateSpec(
            name="hist_gradient_boosting",
            kind="model_family",
            feature_set="framing+visual",
            builder=build_hist_gradient_pipeline,
        ),
        CandidateSpec(
            name="rf_regularized",
            kind="model_family",
            feature_set="framing+visual",
            builder=build_rf_regularized_pipeline,
        ),
        CandidateSpec(
            name="select_from_model_logistic",
            kind="model_family",
            feature_set="framing+visual",
            builder=build_select_from_model_logistic_pipeline,
        ),
    ]

    ablation_sets = (
        "framing+visual",
        "all",
        "audio-only",
        "motion+visual",
        "early-window-only",
    )
    for fs in ablation_sets:
        # Control feature set also appears as rf_control; keep an explicit
        # ablation row for the feature-group table.
        name = f"rf_ablation_{fs.replace('+', '_').replace('-', '_')}"
        candidates.append(
            CandidateSpec(
                name=name,
                kind="feature_ablation",
                feature_set=fs,
                builder=build_rf_control_pipeline,
            )
        )
    return candidates


@dataclass
class SearchRow:
    name: str
    kind: str
    feature_set: str
    n_features: int
    n_train: int
    n_eval: int
    metrics: dict[str, float]
    delta_auc_vs_rf_control: float | None = None
    beats_simple_logistic: bool | None = None
    promising: bool = False
    message: str | None = None


def _auc(metrics: dict[str, float] | None) -> float | None:
    if not metrics:
        return None
    val = metrics.get("roc_auc")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)


def flag_promising(
    candidate_auc: float | None,
    rf_control_auc: float | None,
    simple_logistic_auc: float | None,
    margin: float = PROMISING_AUC_MARGIN,
) -> bool:
    """True if candidate beats RF control by >= margin AND beats simple_logistic."""
    if candidate_auc is None or rf_control_auc is None or simple_logistic_auc is None:
        return False
    return (
        candidate_auc >= rf_control_auc + margin
        and candidate_auc > simple_logistic_auc
    )


def run_model_family_search(
    dataset: ModelDataset,
    membership: CreatorSplitMembership | None = None,
    eval_split: str = "val",
) -> list[SearchRow]:
    """Fit each candidate on train creators; score `eval_split` (default val)."""
    if eval_split != "val":
        raise ValueError(
            "Exp C1 selection must use eval_split='val' "
            "(frozen test is reserved for the final cycle)."
        )
    membership = membership or load_creator_splits()
    specs = iter_candidate_specs()
    raw: list[tuple[CandidateSpec, HeldoutResult, int]] = []

    for spec in specs:
        feats = resolve_feature_set(dataset.feature_names, spec.feature_set)
        if not feats:
            raw.append(
                (
                    spec,
                    HeldoutResult(
                        name=spec.name,
                        split=eval_split,
                        message=f"No features for set '{spec.feature_set}'",
                    ),
                    0,
                )
            )
            continue
        X_sub = dataset.X[feats]
        estimator = spec.builder(X_sub)
        result = evaluate_estimator_heldout(
            spec.name,
            estimator,
            dataset,
            membership,
            eval_split=eval_split,
            feature_names=feats,
        )
        raw.append((spec, result, len(feats)))

    by_name = {spec.name: (spec, result, n_feat) for spec, result, n_feat in raw}
    rf_auc = _auc(by_name["rf_control"][1].metrics) if "rf_control" in by_name else None
    simple_auc = (
        _auc(by_name["simple_logistic"][1].metrics)
        if "simple_logistic" in by_name
        else None
    )

    rows: list[SearchRow] = []
    for spec, result, n_feat in raw:
        cand_auc = _auc(result.metrics)
        delta = None if cand_auc is None or rf_auc is None else cand_auc - rf_auc
        beats_simple = (
            None
            if cand_auc is None or simple_auc is None
            else cand_auc > simple_auc
        )
        # Ablation duplicate of control and the baseline itself are never PROMISING.
        can_promise = spec.kind == "model_family" and spec.name != "rf_control"
        promising = bool(
            can_promise
            and flag_promising(cand_auc, rf_auc, simple_auc)
        )
        rows.append(
            SearchRow(
                name=spec.name,
                kind=spec.kind,
                feature_set=spec.feature_set,
                n_features=n_feat,
                n_train=result.n_train,
                n_eval=result.n_eval,
                metrics=dict(result.metrics),
                delta_auc_vs_rf_control=delta,
                beats_simple_logistic=beats_simple,
                promising=promising,
                message=result.message,
            )
        )
    return rows


def search_rows_to_jsonable(rows: list[SearchRow]) -> list[dict]:
    out = []
    for row in rows:
        d = asdict(row)
        # JSON-friendly floats
        metrics = {}
        for k, v in d["metrics"].items():
            if isinstance(v, float) and np.isnan(v):
                metrics[k] = None
            else:
                metrics[k] = v
        d["metrics"] = metrics
        if d["delta_auc_vs_rf_control"] is not None and np.isnan(
            d["delta_auc_vs_rf_control"]
        ):
            d["delta_auc_vs_rf_control"] = None
        out.append(d)
    return out


def pick_winner(rows: list[SearchRow]) -> SearchRow | None:
    """Best PROMISING model-family candidate by val ROC AUC, else None."""
    promising = [r for r in rows if r.promising and r.kind == "model_family"]
    if not promising:
        return None

    def key(r: SearchRow) -> float:
        auc = _auc(r.metrics)
        return float("-inf") if auc is None else auc

    return max(promising, key=key)
