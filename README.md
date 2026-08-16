# Zukover

Formerly **CoverSignal**. Web app that analyzes how an instrumental TikTok cover is **presented** (filming + sound) and returns a coaching-style report. Machine learning scores are exploratory similarity to creator-relative top performers — not a virality guarantee.

Made by [@suibianmusic](https://www.tiktok.com/@suibianmusic).

## Stack

- **Backend:** FastAPI + OpenCV / MediaPipe / librosa + scikit-learn
- **Frontend:** Vite + React + TypeScript + Tailwind
- **Storage:** local filesystem uploads + SQLite job records

## Prerequisites

- Python 3.11+ (project uses a local `.venv`)
- Node.js 18+ (frontend; tested on Node 24)
- Trained model artifacts under `backend/models/` (`.pkl` files are gitignored — train locally or copy them in before full analysis works)

## Quick start

### 1. Clone and set up Python

```bash
git clone https://github.com/GeneWC/tiktok-cover-analysis.git
cd tiktok-cover-analysis

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the API

From the repo root:

```bash
uvicorn backend.app:app --reload --port 8000
```

- Health: http://127.0.0.1:8000/health  
- Interactive docs: http://127.0.0.1:8000/docs  

If model `.pkl` files are missing, the API still boots; uploads/validation work, but analysis jobs will fail until models are present.

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 (port is fixed to match backend CORS).

Optional: copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_BASE_URL` if the API is not at `http://localhost:8000`.

### 4. Use the app

1. Upload an instrumental cover (`.mp4`, `.mov`, or `.m4v`, max ~200 MB, 1–120 s).
2. Wait while the job processes (status polling).
3. Read the report: presentation scores, feature breakdown, recommendations, and exploratory ML tier.

## Tests

```bash
# Backend (from repo root, venv active)
pytest

# Frontend
cd frontend
npm test
npm run build   # same TypeScript + Vite step Railway runs
```

GitHub Actions (`.github/workflows/ci.yml`) runs the frontend tests/build plus `python backend/check_serving_artifacts.py` on every push and pull request.

## Training (optional)

Offline pipeline scripts live under `scripts/`. Typical flow when you have local videos + `engagement.csv`:

```bash
python scripts/extract_features.py
python scripts/build_training_dataset.py
python scripts/train_models.py
```

Dataset files, downloads, and trained `.pkl` artifacts are gitignored. Experiment helpers are under `scripts/experiments/`.

## Project layout

```text
backend/          FastAPI app, feature extraction, inference, training libs
frontend/         React SPA (upload → processing → report)
scripts/          CLI entry points for data / training / eval
tests/            pytest suite
data/             local DB, uploads, generated CSVs (gitignored)
downloads/        local video corpus (gitignored)
videos/           local media (gitignored)
```

## Deploy (Railway)

One Docker service serves the API and the built frontend on the same URL. GitHub Actions runs the same frontend `npm run build` (TypeScript + Vite) and a serving-artifact check on every push to `main`, so a broken UI build fails in CI instead of only on Railway.

1. Commit the serving artifacts under `backend/models/` (the `.pkl` files are required; they are no longer gitignored).
2. Push to GitHub, then on [railway.app](https://railway.app) create a project and deploy that repo. Railway will use the `Dockerfile`.
3. In the service settings, set memory to **at least 2 GB** (4 GB if you will use channel batches). The default is too small for OpenCV / MediaPipe.
4. Same settings page: enable **Wait for CI** so Railway only deploys commits whose GitHub Actions workflow passed.
5. Settings → Networking → generate a public domain.
6. Open that URL. `/health` should return `{"status":"ok"}`.

Uploads and the SQLite job DB live on the container disk and disappear when Railway restarts the service. First analysis after a cold start may download MediaPipe model bundles.

Hobby is usage-based (typically a few dollars a month at 2 GB). New accounts usually include a small trial credit.

## Configuration

Settings use the `ZUKOVER_` env prefix (see `backend/core/config.py`). Common knobs: upload size/duration limits, data paths, CORS origins, model directories. Older `COVERSIGNAL_` env vars are no longer read. The local SQLite file is `data/zukover.db`.

## Honest-UX notes

- Presentation scores (0–100) are percentile-style coaching signals from the video itself.
- ML output is framed as exploratory / similarity — not predicted view counts.
- Missing modalities (e.g. no audio track) show as **Not available**, not invented scores.

## License / data

Local video corpora, engagement CSVs, and private notes are intentionally excluded from the repo. Do not commit personal media or internal interview/resume materials.
