# Music Analyzer — AI Coding Agent Skill

A skill for AI coding agents (Claude Code, Codex, Gemini CLI, Copilot CLI) that performs comprehensive music analysis on audio files. It combines AI-powered vocal separation (HDEMUCS), speech-to-text lyrics transcription (Whisper), and signal processing (librosa) to extract detailed musical features — then lets the AI interpret the raw data into genre classifications, mood analysis, instrument detection, and a rich production description.

## Compatible Agents

| Agent | Skill System | Status |
|-------|-------------|--------|
| **Claude Code** | Skills (SKILL.md) | Fully supported |
| **Codex** (OpenAI) | Skills (SKILL.md) | Compatible |
| **Gemini CLI** (Google) | Skills via `activate_skill` | Compatible |
| **Copilot CLI** (GitHub) | Plugins | Compatible |

The `SKILL.md` defines the workflow. Any AI agent that can read Markdown instructions, run Python scripts, and interpret JSON output can use this skill.

## What It Does

Given any music file (MP3, WAV, FLAC, M4A, OGG), the skill produces:

**Lyrics Analysis** — Summary, moods, themes, language, explicit content detection
**Music Analysis** — Genres, subgenres, moods, instruments, BPM & key, vocal description
**Production Description** — Detailed prose describing drums, bass, harmony, melody, vocals, and mix characteristics

## How It Works

```
Audio File
    │
    ▼
[1] Load audio (librosa)
    │
    ▼
[2] Vocal separation (HDEMUCS) ──► vocals + accompaniment
    │                                    │
    ▼                                    ▼
[3] Lyrics transcription          Accompaniment feature
    (Whisper)                     extraction
    │
    ▼
[4] Audio feature extraction (librosa)
    BPM, key, spectral features, MFCCs, chroma, energy
    │
    ▼
[5] JSON output ──► AI interprets ──► Formatted analysis
```

The Python pipeline extracts raw audio features and transcribes lyrics. The AI agent then uses its broad music knowledge combined with a genre reference table to classify genres, detect moods, identify instruments, and write a detailed production description.

## Example Output

```
LYRICS ANALYSIS

Summary
A deeply personal and vulnerable confession where the speaker reveals their humble
origins, personal flaws, and fears about love, repeatedly asking their partner
for unconditional acceptance despite all shortcomings.
Moods: Vulnerable (85), Hopeful (70), Anxious (60), Tender (55), Pleading (45)
Themes: Vulnerability (90), Unconditional Love (80), Self-doubt (70), Honesty (65), Acceptance (55)
Language: English
Explicit: No

MUSIC ANALYSIS
Genres: Country (90), Pop (40), Rock (25), Folk (20), Blues (10)
Subgenres: Country Pop (85), Country Ballad (80), Contemporary Country (75), Nashville Sound (60), Americana (35)
Moods: Emotional (85), Warm (75), Hopeful (65), Intimate (60), Bittersweet (50)
Instruments: Acoustic Guitar, Electric Guitar, Bass Guitar, Drums, Piano, Pedal Steel
BPM & Key: 83BPM, A Major
Vocals: Strong male baritone, warm rich timbre, emotionally expressive

PRODUCTION DESCRIPTION
A warm emotional country ballad at 83 BPM in A major with a gentle driving rhythm
featuring a soft kick on the downbeats and brushed snare, a warm round bass guitar
laying down steady root-fifth patterns. Lush acoustic guitar strumming forms the
harmonic bed with bright steel-string shimmer, supported by tasteful electric guitar
fills and sustained pedal steel. Strong male baritone vocals sit front and center,
intimate in the verses, building to powerful emotional delivery in the choruses.
```

## Requirements

- **Python 3.10+** with CUDA-capable GPU recommended (works on CPU too, just slower)
- **ffmpeg** in PATH
- **~3GB disk space** for model downloads on first run

## Installation

### As a Claude Code / Codex Skill

```bash
# Clone into your skills folder
git clone https://github.com/rtfirst/music-analyzer.git ~/.claude/skills/music-analyzer
```

### As a standalone tool

```bash
git clone https://github.com/rtfirst/music-analyzer.git
cd music-analyzer
```

### Install Python dependencies

With NVIDIA GPU (recommended):
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

CPU only (slower, no GPU needed):
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Install ffmpeg

- Windows: `winget install Gyan.FFmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

### Verify

The script checks all dependencies on startup and reports what's missing:
```bash
python scripts/analyze_music.py --help
```

## Usage

### With an AI agent

Simply ask:
- "Analyze this song: /path/to/song.mp3"
- "What genre is /path/to/track.wav?"
- `/music-analyzer @"/path/to/file.mp3"`

The skill automatically triggers on music analysis requests.

### Standalone (no AI agent needed)

```bash
python scripts/analyze_music.py "/path/to/song.mp3"
```

Options:
- `--whisper-model tiny|base|small|medium` — Whisper model size (default: medium)
- `--skip-lyrics` — Skip vocal separation and transcription (faster, for instrumental analysis)

Output is JSON to stdout, status messages to stderr.

## How It Uses Your GPU

Models are loaded **sequentially** to fit in limited VRAM:

| Step | Model | VRAM | Purpose |
|------|-------|------|---------|
| 1 | HDEMUCS | ~320MB | Vocal separation |
| 2 | Whisper medium | ~2.5GB | Lyrics transcription |
| 3 | librosa | CPU only | Audio feature extraction |

Peak VRAM usage: **~2.5GB** — works on any GPU with 4GB+ VRAM.

## File Structure

```
music-analyzer/
├── SKILL.md                    # Skill definition (workflow for AI agents)
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── scripts/
│   ├── analyze_music.py        # Main analysis pipeline
│   └── audio_features.py       # librosa feature extraction
└── references/
    └── genre_profiles.md       # Genre feature reference table
```

## Supported Formats

MP3, WAV, FLAC, M4A, OGG (max 500MB file size)

## License

MIT
