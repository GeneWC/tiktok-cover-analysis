# CoverSignal Frontend

A Vite + React + TypeScript single-page app (styled with Tailwind CSS) for the
CoverSignal instrumental-cover video analyzer. It drives the existing FastAPI
backend: upload a video → poll processing status → render the analysis report.

## Prerequisites

- Node.js 18+ (tested on Node 24)
- The CoverSignal backend running on `http://localhost:8000` (see the repo root)

## Setup

```bash
cd frontend
npm install
```

Optionally copy the env template if your backend runs somewhere other than
`http://localhost:8000`:

```bash
cp .env.example .env
# then edit VITE_API_BASE_URL
```

## Run (development)

```bash
npm run dev
```

The dev server runs on **http://localhost:3000** (fixed via `server.port` in
`vite.config.ts`) because the backend's CORS allowlist
(`backend/core/config.py`) whitelists that origin.

In a second terminal, start the backend from the repo root:

```bash
uvicorn backend.app:app --reload --port 8000
```

Then open http://localhost:3000 and upload a cover video.

## Other scripts

```bash
npm run build     # type-check (tsc --noEmit) + production build to dist/
npm run preview   # serve the production build locally
npm run test      # run the Vitest unit tests (formatting + API client)
```

## Project structure

```
src/
  api/client.ts            typed analyze() / getStatus() / getReport()
  context/AnalysisContext  holds the uploaded File + object URL across routes
  hooks/useStatusPolling   polls /status ~every 1.5s until complete/failed
  lib/format.ts            tier/probability/score formatting, step labels
  lib/validation.ts        client-side extension/size/duration pre-checks
  lib/featureCatalog.ts    groups + human labels for the feature breakdown
  pages/                   UploadPage, ProcessingPage, ReportPage
  components/              VideoUploader, ProcessingStatus, PredictionCard,
                           ScoreCard, FeatureBreakdown, RecommendationPanel,
                           VideoSummary, LimitationsNote, TierBadge, Layout
  types.ts                 TS types mirroring the backend Pydantic schema
```

## API contract

- `POST /api/analyze` (multipart `video_file`, optional `hashtags`/`instrument`)
  → `202 { analysis_id, status }`
- `GET /api/analyze/{id}/status` → `{ analysis_id, status, steps }`
- `GET /api/analyze/{id}/report` → full `ReportResponse`

## Honest-UX notes

Predictions are shown as tiers and a probability, never fabricated view counts.
Null score/tier fields render as **"Not available"** (e.g. audio quality for a
video with no audio track), and every report shows the exploratory disclaimer.
