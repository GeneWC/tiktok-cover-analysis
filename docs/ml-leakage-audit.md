# Leakage and evaluation-integrity audit

This document records what the training and serving pipelines do today. It is
not a claim that validation metrics improved.

## Target

The product predicts **performance relative to the same creator**, not
universal virality or raw view count.

| Field | Meaning |
| --- | --- |
| `top_quartile_for_creator` | Binary. `log(views+1)` is at or above that creator's 75th percentile. Primary serving classifier. |
| `creator_relative_log_views` | Regression. `log(views+1)` minus the creator's median log views. |
| `engagement_rate` | `(likes + comments + shares) / views`. |
| `share_rate` | `shares / views`. |

Creator percentiles use **all labeled videos for that creator**. That is safe
only because creators are assigned wholly to one split. If a creator ever
spanned train and test, label construction would leak.

Labels do not use upload-time "future" features. They do use the creator's
full historical metric distribution in the labeled corpus, including videos
that may have been posted later than others in the same creator's set. That
is a within-creator peer-group definition, not a chronological forecast.

## Splits

`backend/training/creator_splits.py` assigns each creator to exactly one of
train / val / test (~60 / 20 / 20). Videos cannot cross splits.
`GroupKFold` by `creator_username` is used for out-of-fold estimates.

The frozen test set must not be used for feature selection, model selection,
thresholds, or ablation. Ablation (`scripts/run_feature_ablation.py`) refuses
`--split test`.

## Preprocessing

Each model is a `Pipeline(ColumnTransformer, estimator)`:

- numeric: median impute + `StandardScaler`
- boolean: constant-0 impute

During GroupKFold and held-out eval the pipeline is cloned and refit on the
training fold only. Tests in `tests/test_preprocessing_no_leakage.py` check
that scaler statistics change if held-out rows are included.

## Known integrity gaps (not silently "fixed" in serving)

1. **Default export still fits on all rows.** `export_models()` can take
   `fit_dataset` / `calibration_dataset` so a later retrain can use train or
   train+val only. Committed `.pkl` files were produced with the previous
   all-rows fit. Changing that without the local training CSV would change
   serving without a measured comparison.
2. **Calibration thresholds** (`calibration.json`) were computed on the same
   full matrix used to fit. Prefer val-only thresholds on the next retrain.
3. **Production `evaluate_all()`** reports full-dataset GroupKFold OOF, which
   is an honest unseen-creator estimate but is not the frozen val/test
   protocol. Use `scripts/eval_report.py --split val` for model selection.

## Feature contract

`feature_schema.json` now carries `schema_version` and `feature_fingerprint`.
Inference raises `FeatureSchemaError` if column names/order or the
fingerprint disagree with the loaded model.
