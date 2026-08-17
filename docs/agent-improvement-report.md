# Agent Improvement Report

## Executive Summary

Five named skills were installed (`scikit-learn-best-practices`,
`computer-vision-opencv`, `frontend-design`, `playwright-best-practices`,
`security-best-practices`), then the existing Zukover product was audited
before edits. The five phases harden evaluation integrity, separate camera
from performer motion, make the UI say creator-relative estimate (not view
forecast), add mocked Playwright coverage, and tighten the upload surface.

Committed serving models were **not** retrained. Val and train+val
GroupKFold numbers were measured on the local CSV; see Phase 1 Metrics
After. No frozen-test numbers are reported.

The product claim is unchanged: predict performance **relative to a
creator’s own historical videos**, not universal virality or raw views.

## Phase 1 — ML / Scikit-Learn

### Findings

- Creator-disjoint splits, `Pipeline` + `ColumnTransformer`, and GroupKFold
  by creator were already in place.
- Evaluation leaned on ROC-AUC / Precision@K (classification) and R² / MAE /
  RMSE / Spearman (regression). Brier existed on the held-out path only.
- Within-creator ranking lived in an experiment module and was not part of
  the default evaluation record.
- Histogram gradient boosting existed in model-family search but was not a
  registered baseline.
- Serving could silently impute if the extractor and schema drifted.
- Default artifact export still fits and calibrates on all creators,
  including the frozen test set.

### Changes

- Shared metrics module: PR-AUC, F1, balanced accuracy, log loss, Brier,
  expected calibration error, plus within-creator Spearman / pairwise rank
  accuracy.
- Registered HGB baseline alongside majority, random, logistic, and RF.
- Feature-group ablation runner that refuses the test set.
- `schema_version` + `feature_fingerprint` on the inference contract;
  `FeatureSchemaError` on mismatch. Committed schema is version 1,
  fingerprint `3b9ae0b273f1725d` (still the original 70 serving columns).
- Canonical seeds in `backend/training/reproducibility.py` (seed 42);
  `seed_everything()` is called from `scripts/train_models.py`.
- Optional `fit_dataset` / `calibration_dataset` on export (default
  unchanged so serving `.pkl` files stay as committed).
- Leakage audit: `docs/ml-leakage-audit.md`.

### Tests

- `tests/test_metrics.py`
- `tests/test_ablation.py`
- `tests/test_feature_schema_compat.py`
- `tests/test_preprocessing_no_leakage.py`
- Updates to evaluate / baseline / export tests

### Metrics Before

Committed `training_metadata.json` (full-dataset GroupKFold OOF, not rerun):

- `top_quartile`: ROC-AUC 0.531, P@K 0.319
- `engagement`: R² 0.066, Spearman 0.287
- `creator_relative`: R² -0.054, Spearman 0.036
- `shareability`: R² 0.001, Spearman 0.197

### Metrics After

Rerun 2026-08-15 on `data/training_dataset.csv` with frozen creator
splits. Test creators were not used for these numbers.

**Held-out val** (train 914 → val 291, 75 positives, rate 0.258):

| Model | ROC-AUC | PR-AUC | P@K | F1 | Bal. acc. | Brier | ECE | Within-creator Spearman | Pairwise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority | 0.500 | 0.258 | 0.347 | 0.000 | 0.500 | 0.191 | 0.002 | — | — |
| Random | 0.514 | 0.270 | 0.307 | 0.352 | 0.507 | 0.330 | 0.306 | −0.035 | 0.477 |
| Logistic | 0.556 | 0.328 | 0.267 | 0.239 | 0.507 | 0.235 | 0.217 | 0.189 | 0.622 |
| RF control | **0.586** | 0.315 | 0.333 | 0.232 | 0.531 | 0.204 | 0.125 | 0.208 | 0.636 |
| HGB | 0.560 | 0.273 | 0.253 | 0.186 | 0.499 | 0.216 | 0.122 | 0.166 | 0.607 |

RF val ROC-AUC matches the earlier `phase_a_eval_report.json` (0.5855).
HGB did not beat RF on val.

**GroupKFold by creator on train+val only** (1205 videos, 19 creators;
frozen test excluded). Committed `training_metadata.json` used all 1370
rows including test, so these are not a like-for-like lift:

| Model | Key metrics |
| --- | --- |
| `top_quartile` | ROC-AUC 0.537, PR-AUC 0.281, P@K 0.282, F1 0.224, Brier 0.214, ECE 0.124, within-creator Spearman 0.055 |
| `engagement` | R² 0.097, Spearman 0.335, within-creator Spearman 0.154 |
| `creator_relative` | R² −0.087, Spearman 0.021, within-creator Spearman 0.114 |
| `shareability` | R² 0.040, Spearman 0.310, within-creator Spearman 0.126 |

JSON: `data/reports/eval_report_val.json`,
`data/reports/evaluate_all_train_val.json`.

### Remaining Risks

- Committed models still trained on all creators.
- Calibration still reflects the previous full-data export.
- Do not treat the classifier score as a well-calibrated frequency until a
  val-only reliability check is run on the real dataset.

## Phase 2 — Computer Vision

### Findings

- `camera_stability_score` previously used mean pixel-frame difference as a
  proxy, so performer movement could look like camera shake.
- Motion energy mixed camera, performer, and lighting change.
- Temporal brightness/contrast variation and shot-cut heuristics were
  missing.
- Missing detections already used `None` plus failure flags in several
  framing/hand features; that pattern was kept.

### Changes

- `backend/features/camera_motion.py`: ORB keypoints, optional person-box
  mask, RANSAC affine, translation / rotation / scale, residual after warp.
- `camera_stability_score` uses the ORB aggregate when tracking succeeds
  and falls back to the old pixel proxy otherwise.
- Extra extractor keys (not in the 70-feature serving schema): camera
  translation/rotation/scale, tracking-failed flag, performer residual
  motion, subject-motion fraction, shot-cut count/frequency, average shot
  duration.
- Visual extras: `brightness_std_full`, `contrast_std_full`.
- Extractor output is now 89 keys (was 77). Serving still drops extras and
  scores the original 70.
- Definitions: `docs/visual-feature-definitions.md`.
- Feature-group map updated so new names land in `motion`.

### Tests

- `tests/test_camera_motion.py` (direct ORB translation / static pair)
- Updates in `tests/test_motion_features.py`,
  `tests/test_visual_quality_features.py`, `tests/test_extract_features.py`,
  `tests/test_pipeline_integration.py`
- Synthetic “translated background” on tiny frames was flaky; the relaxed
  motion test plus the direct ORB test cover the intent.

### Runtime Before/After

Not measured on real videos. ORB is capped at 400 features on frames
already downscaled to ≤256 px on the long side, sampled at ~3 fps. Cost
is expected to be small relative to MediaPipe, but that is an estimate,
not a benchmark.

### ML Impact

Not measured. Extra keys are extracted for reports and future training
only. The committed RF models still see the original 70 columns. Do not
treat the new CV features as predictive until a val-only ablation is run
on the real dataset.

### Remaining Risks

- Textureless / dark / heavily compressed frames fall back to the pixel
  proxy, so performer motion can still inflate “camera shake” there.
- Shot-cut heuristics are coarse at 3 fps.
- Multiple people: first detected box/hand is used.
- Small-frame synthetic translation can fail ORB; rely on
  `tests/test_camera_motion.py` for the method, not photorealism.

## Phase 3 — Frontend Design

### Findings

- Upload copy was close but could still be read as a view forecast.
- The relative result lived in a collapsed `<details>` block, so the
  five-second read started with scores instead of the claim.
- Processing progress is step-based; users could read the bar as a
  stopwatch. Reload already resumes the job URL but drops the local
  preview.
- Fire Nation / Zuko direction (dark lacquer, restrained gold, near-square
  radii) was already in place and was preserved.

### Changes

- Upload headline and body state creator-relative estimate, not view
  count (`UploadPage.tsx`, `VideoUploader.tsx`).
- Analyze stays disabled when a client-side error is set.
- `PredictionCard` is a hero section: resemblance percentage, relative
  band (weaker / typical / stronger), then pattern tiers.
- Report order: prediction → recommendations (observed signals) → scores
  → metadata → feature breakdown.
- Processing copy explains step-based progress and that a refresh can
  resume the same job link.
- Step label `prediction` → “Scoring against the creator baseline”.
- New motion/visual keys labeled in `featureCatalog.ts`.

### Accessibility

- `RelativeBand` has an `sr-only` score description.
- Upload control keeps `aria-label` / `aria-describedby`.
- Prediction card uses `aria-labelledby`.
- Existing `prefers-reduced-motion` rules in `index.css` were left in
  place.
- No `dangerouslySetInnerHTML`.
- Automated axe/Playwright a11y audit was not added; do not treat this
  as a full accessibility sign-off.

### Responsive Behavior

- Report hero stacks on small screens (`md:grid-cols-[1.2fr_1fr]`).
- Playwright Pixel 5 project covers upload → report readability.
- Desktop / laptop / tablet were not separately screenshot-reviewed in
  this pass.

### Remaining Risks

- Band thresholds (0.4 / 0.6) are UI interpretation of the classifier
  score, not calibrated creator percentiles.
- No visual-regression screenshots were checked in.
- Channel (multi-video) flow was not redesigned.

## Phase 4 — Playwright

### Coverage Added

- `@playwright/test` in `frontend/package.json` (`test:e2e`)
- `frontend/playwright.config.ts`: Vite on port 3000 with
  `VITE_API_BASE_URL=""`, Chromium + Pixel 5, traces / screenshots /
  video on failure
- `frontend/e2e/fixtures.ts` mocks `/api/analyze*` so CI never hits
  production or a live decoder

### Scenarios Tested

- Happy path: upload → processing steps → relative result (72% fixture)
- Unsupported file type stays on upload; Analyze disabled
- Backend upload failure shown as an alert
- Reload during processing resumes from the same job link
- Mobile (Pixel 5): upload and report remain readable

### CI Integration

`.github/workflows/ci.yml` frontend job now installs Chromium and runs
`npm run test:e2e`. Failure artifacts upload `frontend/test-results`.

### Remaining Gaps

- No live-backend / corrupt-video / missing-audio / oversized-file E2E
  (those stay in Python API tests).
- No automated accessibility smoke (axe) in Playwright.
- Tests use a synthetic in-memory “video” buffer, not a real decoder.

## Phase 5 — Security

### Threat Model

Untrusted browser
→ HTTP upload (`POST /api/analyze`, `POST /api/channel/diagnose`)
→ size / extension / decode validation
→ disk under a server-generated id
→ OpenCV / MediaPipe / librosa / OCR
→ model inference
→ JSON report
→ video delete + timed job cleanup

Trust boundary is the upload endpoint. There is no user auth; job IDs
are unguessable hex, not a capability system. Anyone who learns an id
can poll status/report until expiry.

### Findings

- Filenames from the client were too easy to treat as paths.
- Job IDs needed a strict allow-list in the path.
- CORS allowed more methods than the API uses.
- Uploads could accumulate on disk after success or failure.
- No per-IP upload cap.
- Frontend `npm audit --omit=dev`: 2 high-severity
  `react-router` / `react-router-dom` issues (RSC-mode CSRF,
  GHSA-qwww-vcr4-c8h2). This app is a Vite SPA and does not use RSC
  actions, so practical exploitability is low. Not upgraded blindly.
- `pip-audit` is not installed in the project venv; Python advisory
  scan was not run this pass.

### Fixes

- Job IDs: `analysis_[0-9a-f]{12}`, `channel_[0-9a-f]{12}`.
- Upload rate limit: 10 POSTs / 10 minutes / IP (in-memory).
- CORS methods limited to GET / POST / OPTIONS.
- Stored filename is `Path(name).name`; file is `{analysis_id}{ext}`.
- Video deleted in the analysis `finally` block; reports kept.
- `cleanup_expired_jobs()` on startup (`job_retention_hours=24`).
- Existing size / duration / decode limits kept (200 MB, 1–120 s,
  sample ≤256 px @ 3 fps).

### Regression Tests

`tests/test_upload_security.py`:

- path-traversal and malformed job IDs → 404
- `../../evil.mp4` stored as `evil.mp4` under `{analysis_id}.mp4`
- oversized upload → 413, nothing left on disk
- validation failure deletes the temp file
- isolated rate-limit middleware → 429 after 2 POSTs
- expired-job cleanup removes the record

### Remaining Risks

- In-memory rate limit does not hold across workers / restarts.
- No auth: leaked job IDs remain readable until cleanup.
- Channel-job media cleanup is the same retention path; not separately
  load-tested.
- Media decoders (OpenCV, etc.) still process untrusted bytes after
  extension/size checks.
- React Router advisory remains until a deliberate upgrade.
- Python dependency advisories not scanned with `pip-audit`.

## Files Changed

Skills (install only):

- `.agents/skills/` (five skills)
- `skills-lock.json`

ML:

- `backend/training/metrics.py` (new)
- `backend/training/ablation.py` (new)
- `backend/training/reproducibility.py` (new)
- `backend/training/baselines.py`
- `backend/training/evaluate.py`
- `backend/training/export_artifacts.py`
- `backend/training/feature_groups.py`
- `backend/training/heldout_eval.py`
- `backend/training/model_dataset.py`
- `backend/training/model_family_search.py`
- `backend/inference/feature_assembly.py`
- `backend/inference/prediction.py`
- `backend/inference/model_registry.py`
- `backend/models/feature_schema.json`
- `scripts/train_models.py`
- `scripts/eval_report.py`
- `scripts/run_feature_ablation.py` (new)
- `docs/ml-leakage-audit.md` (new)

CV:

- `backend/features/camera_motion.py` (new)
- `backend/features/motion_features.py`
- `backend/features/visual_quality_features.py`
- `docs/visual-feature-definitions.md` (new)

Frontend:

- `frontend/src/pages/UploadPage.tsx`
- `frontend/src/pages/ProcessingPage.tsx`
- `frontend/src/pages/ReportPage.tsx`
- `frontend/src/components/PredictionCard.tsx`
- `frontend/src/components/RelativeBand.tsx` (new)
- `frontend/src/components/VideoUploader.tsx`
- `frontend/src/lib/format.ts`
- `frontend/src/lib/featureCatalog.ts`

Playwright:

- `frontend/playwright.config.ts` (new)
- `frontend/e2e/` (new)
- `frontend/package.json`, `frontend/package-lock.json`
- `.github/workflows/ci.yml`

Security:

- `backend/services/job_ids.py` (new)
- `backend/core/rate_limit.py` (new)
- `backend/services/job_cleanup.py` (new)
- `backend/api/analyze_routes.py`
- `backend/api/channel_routes.py`
- `backend/app.py`
- `backend/core/config.py`
- `backend/inference/pipeline.py`
- `backend/services/analysis_store.py`
- `tests/test_upload_security.py` (new)
- `tests/conftest.py`

Tests / docs / ignore:

- `tests/test_metrics.py`, `tests/test_ablation.py`,
  `tests/test_feature_schema_compat.py`,
  `tests/test_preprocessing_no_leakage.py`,
  `tests/test_camera_motion.py`, plus updates to existing feature /
  baseline / export / pipeline tests
- `docs/agent-improvement-report.md`
- `.gitignore` (track the three docs above; keep other `docs/*` ignored)

## Commands Run

- `npx skills add` for the five named skills (mindrally, anthropics,
  currents-dev, openai)
- `.venv\Scripts\python.exe -m pytest -m "not slow"` — 222 passed, 2
  deselected
- `.venv\Scripts\python.exe -m pytest tests/test_upload_security.py` —
  7 passed (after the 429 case)
- `.venv\Scripts\python.exe -m compileall -q backend`
- `.venv\Scripts\python.exe backend/check_serving_artifacts.py` —
  Serving artifacts OK
- `frontend`: `npm test` — 25 passed; `npx tsc --noEmit`; `npm run build`
- `npx playwright test --project=chromium` — 5 passed
- `npx playwright test --project=mobile` — 5 passed
- `npm audit --omit=dev` — 2 high (react-router RSC CSRF)
- `python -m pip_audit` — module not installed; not run
- `python scripts/eval_report.py --split val --train-cv` — wrote
  `data/reports/eval_report_val.json`
- GroupKFold `evaluate_all` on train+val only — wrote
  `data/reports/evaluate_all_train_val.json`

## Final Test Results

| Suite | Result |
| --- | --- |
| Backend `pytest -m "not slow"` | 222 passed, 2 deselected |
| Upload security | 7 passed |
| Serving artifact check | OK |
| Frontend Vitest | 25 passed |
| Frontend `tsc` + Vite build | passed |
| Playwright Chromium | 5 passed |
| Playwright Pixel 5 | 5 passed |

Creator isolation, the 70-column serving schema, and committed `.pkl`
files were not changed in a way that retrains or peeks at the frozen
test set. New extractor keys are dropped at inference.

## Recommended Next Improvements

- Retrain with `fit_dataset` = train+val and val-only calibration. RF
  remains the val winner over HGB (ROC-AUC 0.586 vs 0.560).
- Run `scripts/run_feature_ablation.py` on val and paste the group table.
- Measure ORB vs pixel-proxy extraction time on a sample of real covers.
- Add a val-only ablation that includes the new motion keys before
  expanding `feature_schema.json`.
- Install `pip-audit` in CI and decide whether to bump `react-router`
  after checking the changelog.
- Shared-store rate limit if more than one API worker is deployed.
- Playwright axe smoke and a real tiny `.mp4` fixture if decoder-level
  E2E is needed later.
