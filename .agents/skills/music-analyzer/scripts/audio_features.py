"""Audio feature extraction module using librosa."""

import numpy as np
import librosa


# Krumhansl-Kessler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_bpm(y, sr):
    """Detect BPM using librosa beat tracking."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    # Estimate confidence from onset strength autocorrelation
    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env))
    if len(ac) > 1 and ac[0] > 0:
        confidence = float(np.max(ac[1:]) / ac[0])
    else:
        confidence = 0.0
    return {"bpm": round(bpm, 1), "confidence": round(confidence, 2)}


def detect_key(y, sr):
    """Detect musical key using chroma CQT and Krumhansl-Kessler profiles."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    best_corr = -1.0
    best_key = "C"
    best_mode = "major"

    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        corr_major = float(np.corrcoef(rolled, MAJOR_PROFILE)[0, 1])
        corr_minor = float(np.corrcoef(rolled, MINOR_PROFILE)[0, 1])

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = PITCH_CLASSES[i]
            best_mode = "major"
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = PITCH_CLASSES[i]
            best_mode = "minor"

    return {"key": best_key, "mode": best_mode, "confidence": round(best_corr, 2)}


def extract_spectral_features(y, sr):
    """Extract spectral feature statistics."""
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    return {
        "centroid_mean": round(float(np.mean(cent)), 1),
        "centroid_std": round(float(np.std(cent)), 1),
        "rolloff_mean": round(float(np.mean(rolloff)), 1),
        "bandwidth_mean": round(float(np.mean(bandwidth)), 1),
        "contrast_means": [round(float(x), 1) for x in np.mean(contrast, axis=1)],
        "zero_crossing_rate_mean": round(float(np.mean(zcr)), 4),
    }


def extract_mfcc(y, sr, n_mfcc=13):
    """Extract mean MFCC coefficients."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return [round(float(x), 2) for x in np.mean(mfcc, axis=1)]


def extract_chroma(y, sr):
    """Extract mean chroma energy per pitch class."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    return {PITCH_CLASSES[i]: round(float(chroma_mean[i]), 3) for i in range(12)}


def estimate_energy_profile(y, sr):
    """Estimate energy profile over time."""
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_peak = float(np.max(rms))
    dynamic_range_db = 0.0
    if rms_mean > 0:
        ratio = rms_peak / rms_mean
        if ratio > 0:
            dynamic_range_db = 20.0 * np.log10(ratio)

    # Determine energy curve shape
    n = len(rms)
    if n >= 4:
        q1 = float(np.mean(rms[: n // 4]))
        q4 = float(np.mean(rms[3 * n // 4 :]))
        if q4 > q1 * 1.3:
            curve = "building"
        elif q1 > q4 * 1.3:
            curve = "declining"
        elif dynamic_range_db < 6:
            curve = "steady"
        else:
            curve = "variable"
    else:
        curve = "steady"

    return {
        "rms_mean": round(rms_mean, 4),
        "rms_peak": round(rms_peak, 4),
        "dynamic_range_db": round(dynamic_range_db, 1),
        "energy_curve": curve,
    }


def detect_onset_density(y, sr):
    """Detect rhythmic density as onsets per second."""
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    if duration > 0:
        return round(float(len(onset_frames)) / duration, 2)
    return 0.0


def extract_all_features(y, sr):
    """Extract all audio features and return as dict."""
    bpm_data = detect_bpm(y, sr)
    key_data = detect_key(y, sr)
    spectral = extract_spectral_features(y, sr)
    mfcc = extract_mfcc(y, sr)
    chroma = extract_chroma(y, sr)
    energy = estimate_energy_profile(y, sr)
    onset_density = detect_onset_density(y, sr)
    duration = round(librosa.get_duration(y=y, sr=sr), 1)

    return {
        "bpm": bpm_data["bpm"],
        "bpm_confidence": bpm_data["confidence"],
        "key": key_data["key"],
        "mode": key_data["mode"],
        "key_confidence": key_data["confidence"],
        "duration_seconds": duration,
        "spectral": spectral,
        "mfcc_means": mfcc,
        "chroma": chroma,
        "energy": energy,
        "onset_density": onset_density,
    }
