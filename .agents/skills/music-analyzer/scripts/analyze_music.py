"""
Music Analysis Pipeline
Analyzes audio files using HDEMUCS (vocal separation), Whisper (lyrics),
and librosa (audio features). Outputs JSON to stdout.

Usage:
    python analyze_music.py <audio_path> [--whisper-model medium] [--skip-lyrics]
"""

import sys
import os
import json
import gc
import argparse
import tempfile
import warnings
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# Add scripts dir to path for local imports (append to avoid shadowing stdlib)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQ_FILE = os.path.join(SKILL_DIR, "requirements.txt")


def check_dependencies():
    """Check all required packages and report missing ones."""
    missing = []
    for name, pkg in [("torch", "torch"), ("torchaudio", "torchaudio"),
                       ("whisper", "openai-whisper"), ("librosa", "librosa"),
                       ("soundfile", "soundfile"), ("numpy", "numpy")]:
        try:
            __import__(name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(json.dumps({
            "error": f"Missing Python packages: {', '.join(missing)}",
            "install": f"pip install -r \"{REQ_FILE}\"",
            "hint": "For CUDA GPU support install PyTorch with: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128"
        }))
        sys.exit(1)

    # Check ffmpeg
    import shutil
    if not shutil.which("ffmpeg"):
        print(json.dumps({
            "error": "ffmpeg not found in PATH",
            "install": "Windows: winget install Gyan.FFmpeg | Mac: brew install ffmpeg | Linux: sudo apt install ffmpeg"
        }))
        sys.exit(1)


check_dependencies()


def log(msg):
    """Print status to stderr (stdout is reserved for JSON output)."""
    print(f"[music-analyzer] {msg}", file=sys.stderr, flush=True)


SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
MAX_FILE_SIZE_MB = 500


def load_audio(path):
    """Load audio file using librosa, return torch tensors (waveform_stereo, waveform_mono, sr)."""
    import torch
    import librosa

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {ext}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({file_size_mb:.0f}MB). Max supported: {MAX_FILE_SIZE_MB}MB")

    # Load stereo with librosa (preserves channels)
    y_stereo, sr = librosa.load(path, sr=None, mono=False)

    # Handle mono files
    if y_stereo.ndim == 1:
        y_stereo = y_stereo[None, :]  # (1, samples)

    waveform_stereo = torch.from_numpy(y_stereo).float()

    # Create mono version
    if waveform_stereo.shape[0] > 1:
        waveform_mono = waveform_stereo.mean(dim=0, keepdim=True)
    else:
        waveform_mono = waveform_stereo

    return waveform_stereo, waveform_mono, sr


def cleanup_gpu():
    """Free GPU memory."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def separate_vocals(waveform, sr):
    """Separate vocals from accompaniment using HDEMUCS."""
    import torch
    import torchaudio

    log("Loading HDEMUCS model for vocal separation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to(device)

    with torch.no_grad():
        model_sr = bundle.sample_rate  # 44100

        # Resample if needed
        if sr != model_sr:
            log(f"Resampling from {sr}Hz to {model_sr}Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)
            waveform_resampled = resampler(waveform)
        else:
            waveform_resampled = waveform

        # Ensure stereo (HDEMUCS expects 2 channels)
        if waveform_resampled.shape[0] == 1:
            waveform_resampled = waveform_resampled.repeat(2, 1)
        elif waveform_resampled.shape[0] > 2:
            waveform_resampled = waveform_resampled[:2]

        # Add batch dimension
        mix = waveform_resampled.unsqueeze(0).to(device)

        log("Separating vocals from accompaniment...")
        # Process in chunks if audio is long (>60s) to manage memory
        chunk_size = model_sr * 60  # 60 seconds
        n_samples = mix.shape[-1]

        if n_samples > chunk_size:
            overlap = model_sr * 2  # 2 second overlap
            stride = chunk_size - overlap
            pieces = []
            start = 0
            while start < n_samples:
                end = min(start + chunk_size, n_samples)
                chunk = mix[..., start:end]
                chunk_sources = model(chunk).cpu()
                if start == 0:
                    # First chunk: keep everything except the trailing overlap
                    keep_end = stride if end < n_samples else chunk_sources.shape[-1]
                    pieces.append(chunk_sources[..., :keep_end])
                elif end >= n_samples:
                    # Last chunk: discard the leading overlap
                    pieces.append(chunk_sources[..., overlap:])
                else:
                    # Middle chunks: discard both overlapping edges
                    pieces.append(chunk_sources[..., overlap:overlap + stride])
                start += stride
                cleanup_gpu()
            sources = torch.cat(pieces, dim=-1)
        else:
            sources = model(mix).cpu()

    # HDEMUCS outputs: drums, bass, other, vocals (index 3)
    vocals = sources[0, 3]  # shape: (2, samples)
    drums = sources[0, 0]
    bass = sources[0, 1]
    other = sources[0, 2]
    accompaniment = drums + bass + other

    # Convert to mono
    vocals_mono = vocals.mean(dim=0, keepdim=True)
    accompaniment_mono = accompaniment.mean(dim=0, keepdim=True)

    # Cleanup
    del model, mix, sources
    cleanup_gpu()

    log("Vocal separation complete.")
    return vocals_mono, accompaniment_mono, model_sr


def detect_vocal_presence(vocals_waveform):
    """Check if vocals track has meaningful content."""
    import torch

    rms = torch.sqrt(torch.mean(vocals_waveform ** 2)).item()
    return rms > 0.005, round(rms, 4)


def transcribe_lyrics(vocals_waveform, sr, model_name="medium"):
    """Transcribe lyrics from separated vocals using Whisper."""
    import torch
    import whisper
    import soundfile as sf

    log(f"Loading Whisper {model_name} model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = whisper.load_model(model_name, device=device)
    except torch.cuda.OutOfMemoryError:
        log(f"OOM with {model_name}, falling back to 'small'...")
        cleanup_gpu()
        model_name = "small"
        model = whisper.load_model(model_name, device=device)

    # Save vocals to temp file for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        vocals_np = vocals_waveform.squeeze().numpy()
        sf.write(temp_path, vocals_np, sr)

        log("Transcribing lyrics...")
        result = whisper.transcribe(model, temp_path, fp16=(device == "cuda"))

        text = result.get("text", "").strip()
        language = result.get("language", "unknown")

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": round(seg["start"], 1),
                "end": round(seg["end"], 1),
                "text": seg["text"].strip(),
            })
    finally:
        os.unlink(temp_path)
        del model
        cleanup_gpu()

    log("Lyrics transcription complete.")
    return {
        "text": text,
        "language": language,
        "segments": segments,
        "whisper_model_used": model_name,
    }


def extract_features(waveform_mono, sr):
    """Extract audio features using librosa."""
    import audio_features

    log("Extracting audio features with librosa...")
    y = waveform_mono.squeeze().numpy()
    features = audio_features.extract_all_features(y, sr)
    log("Feature extraction complete.")
    return features


def extract_accompaniment_features(accompaniment_mono, sr):
    """Extract features from accompaniment (non-vocal) track."""
    import librosa
    import numpy as np

    log("Extracting accompaniment features...")
    y = accompaniment_mono.squeeze().numpy()

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=y)[0]

    return {
        "spectral_centroid_mean": round(float(np.mean(cent)), 1),
        "mfcc_means": [round(float(x), 2) for x in np.mean(mfcc, axis=1)],
        "rms_mean": round(float(np.mean(rms)), 4),
    }


def build_output(metadata, features, lyrics, accompaniment_features, warnings_list):
    """Build the final JSON output."""
    return {
        "metadata": metadata,
        "lyrics": lyrics,
        "audio_features": features,
        "accompaniment_features": accompaniment_features,
        "warnings": warnings_list,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze music files")
    parser.add_argument("audio_path", help="Path to audio file (mp3/wav/flac/m4a/ogg)")
    parser.add_argument("--whisper-model", default="medium",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--skip-lyrics", action="store_true",
                        help="Skip vocal separation and lyrics transcription")
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio_path)
    warnings_list = []

    if not os.path.isfile(audio_path):
        print(json.dumps({"error": f"File not found: {audio_path}"}))
        sys.exit(1)

    log(f"Analyzing: {os.path.basename(audio_path)}")

    # Step 1: Load audio
    try:
        waveform_stereo, waveform_mono, sr = load_audio(audio_path)
    except Exception as e:
        print(json.dumps({"error": f"Could not load audio: {str(e)}"}))
        sys.exit(1)

    import librosa
    duration = round(librosa.get_duration(y=waveform_mono.squeeze().numpy(), sr=sr), 1)

    metadata = {
        "file_name": os.path.basename(audio_path),
        "duration_seconds": duration,
        "sample_rate": sr,
        "channels": waveform_stereo.shape[0],
        "analysis_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    lyrics = None
    accompaniment_features = None

    if not args.skip_lyrics:
        # Step 2: Vocal separation
        try:
            vocals_mono, accompaniment_mono, sep_sr = separate_vocals(waveform_stereo, sr)

            has_vocals, vocal_rms = detect_vocal_presence(vocals_mono)
            log(f"Vocal presence: {has_vocals} (RMS: {vocal_rms})")

            # Extract accompaniment features
            accompaniment_features = extract_accompaniment_features(accompaniment_mono, sep_sr)

            if has_vocals:
                # Step 3: Lyrics transcription
                lyrics = transcribe_lyrics(vocals_mono, sep_sr, args.whisper_model)
                lyrics["has_vocals"] = True
                lyrics["vocal_rms"] = vocal_rms

                if vocal_rms < 0.02:
                    warnings_list.append("Low vocal signal - lyrics may be unreliable")
            else:
                lyrics = {
                    "text": "",
                    "language": "unknown",
                    "segments": [],
                    "has_vocals": False,
                    "vocal_rms": vocal_rms,
                    "whisper_model_used": "skipped",
                }
                warnings_list.append("No vocals detected - instrumental track")

        except Exception as e:
            log(f"Vocal separation/transcription failed: {e}")
            warnings_list.append(f"Vocal separation failed: {str(e)}")
            lyrics = {
                "text": "",
                "language": "unknown",
                "segments": [],
                "has_vocals": False,
                "vocal_rms": 0.0,
                "whisper_model_used": "failed",
            }
            cleanup_gpu()
    else:
        lyrics = {
            "text": "",
            "language": "unknown",
            "segments": [],
            "has_vocals": False,
            "vocal_rms": 0.0,
            "whisper_model_used": "skipped",
        }
        warnings_list.append("Lyrics extraction skipped by user")

    # Step 4: Feature extraction (CPU only)
    # Resample to 22050 for librosa (standard)
    import torchaudio

    if sr != 22050:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform_librosa = resampler(waveform_mono)
        librosa_sr = 22050
    else:
        waveform_librosa = waveform_mono
        librosa_sr = sr

    features = extract_features(waveform_librosa, librosa_sr)

    # Build and output JSON
    output = build_output(metadata, features, lyrics, accompaniment_features, warnings_list)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    log("Analysis complete!")


if __name__ == "__main__":
    main()
