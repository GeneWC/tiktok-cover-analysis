"""Audio features (PRD 11.6 + D-007 musical structure cues).

Song-agnostic, production-focused audio metrics: loudness/RMS, dynamic range,
silence & clipping ratios, spectral statistics, onset strength, and energy
windows. Extended (docs/DECISIONS.md D-007) with structure cues that are still
not genre/melody IDs: onset density, tempo BPM, beat-interval stability, and
spectral flux — reusable for train/serve via `extract_all_features`.

Pipeline: decode the audio track to a mono float32 waveform with PyAV (its
bundled ffmpeg handles mp4/m4a), then analyze the NumPy array with librosa.

No-audio handling (PRD 11.6): if there is no audio stream, every audio feature
is null and `audio_feature_extraction_status = "failed"` (the report already
surfaces a no-audio warning).
"""

from __future__ import annotations

import av
import librosa
import numpy as np

_TARGET_SR = 22050  # librosa's default analysis rate
_SILENCE_DB = -40.0  # frames quieter than this (rel. to peak) count as silence
_CLIPPING_THRESHOLD = 0.99  # |sample| at/above this counts as clipping

# New columns introduced for Exp C2 (musical structure). Backfill scripts key off these.
AUDIO_STRUCTURE_FEATURE_KEYS: tuple[str, ...] = (
    "audio_onset_density",
    "audio_onset_density_first_3s",
    "audio_tempo_bpm",
    "audio_beat_interval_cv",
    "audio_spectral_flux_mean",
    "audio_spectral_flux_std",
)

_FEATURE_KEYS = (
    "audio_rms_mean",
    "audio_rms_std",
    "audio_peak_level",
    "audio_dynamic_range",
    "audio_silence_ratio",
    "audio_clipping_ratio",
    "audio_spectral_centroid_mean",
    "audio_spectral_bandwidth_mean",
    "audio_spectral_rolloff_mean",
    "audio_zero_crossing_rate_mean",
    "audio_onset_strength_mean",
    "audio_onset_strength_std",
    "audio_energy_first_1s",
    "audio_energy_first_3s",
    "audio_energy_first_6s",
    "audio_energy_full",
    "audio_energy_ratio_first_3s_to_full",
) + AUDIO_STRUCTURE_FEATURE_KEYS


def _empty_features(status: str) -> dict[str, float | str | None]:
    out: dict[str, float | str | None] = {key: None for key in _FEATURE_KEYS}
    out["audio_feature_extraction_status"] = status
    return out


def _decode_waveform(path: str, target_sr: int) -> np.ndarray | None:
    """Decode the first audio stream to a mono float32 waveform, or None."""
    container = av.open(path)
    try:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=target_sr
        )
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):  # flush
            chunks.append(resampled.to_ndarray().reshape(-1))
    finally:
        container.close()

    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32)


def _segment_energy(y: np.ndarray, sr: int, seconds: float | None) -> float | None:
    """RMS energy of the first `seconds` of audio (or the whole clip)."""
    segment = y if seconds is None else y[: int(sr * seconds)]
    if segment.size == 0:
        return None
    return float(np.sqrt(np.mean(segment**2)))


def extract_audio_features(path: str, target_sr: int = _TARGET_SR) -> dict[str, float | str | None]:
    """Compute audio features (PRD 11.6). Returns nulls + failed status if no audio."""
    y = _decode_waveform(path, target_sr)
    if y is None or y.size == 0:
        return _empty_features("failed")

    sr = target_sr
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    energy_first_3s = _segment_energy(y, sr, 3.0)
    energy_full = _segment_energy(y, sr, None)

    features: dict[str, float | str | None] = {
        "audio_rms_mean": round(float(np.mean(rms)), 6),
        "audio_rms_std": round(float(np.std(rms)), 6),
        "audio_peak_level": round(float(np.max(np.abs(y))), 6),
        "audio_dynamic_range": round(
            float(np.percentile(rms, 95) - np.percentile(rms, 5)), 6
        ),
        "audio_silence_ratio": round(float(np.mean(rms_db < _SILENCE_DB)), 4),
        "audio_clipping_ratio": round(
            float(np.mean(np.abs(y) >= _CLIPPING_THRESHOLD)), 6
        ),
        "audio_spectral_centroid_mean": round(
            float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))), 2
        ),
        "audio_spectral_bandwidth_mean": round(
            float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))), 2
        ),
        "audio_spectral_rolloff_mean": round(
            float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))), 2
        ),
        "audio_zero_crossing_rate_mean": round(
            float(np.mean(librosa.feature.zero_crossing_rate(y=y))), 6
        ),
        "audio_energy_first_1s": _round_or_none(_segment_energy(y, sr, 1.0)),
        "audio_energy_first_3s": _round_or_none(energy_first_3s),
        "audio_energy_first_6s": _round_or_none(_segment_energy(y, sr, 6.0)),
        "audio_energy_full": _round_or_none(energy_full),
    }

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features["audio_onset_strength_mean"] = round(float(np.mean(onset_env)), 6)
    features["audio_onset_strength_std"] = round(float(np.std(onset_env)), 6)
    features.update(_structure_features(y, sr, onset_env))

    if energy_full and energy_full > 0 and energy_first_3s is not None:
        features["audio_energy_ratio_first_3s_to_full"] = round(
            energy_first_3s / energy_full, 4
        )
    else:
        features["audio_energy_ratio_first_3s_to_full"] = None

    features["audio_feature_extraction_status"] = "ok"
    return features


def extract_audio_structure_features(
    path: str, target_sr: int = _TARGET_SR
) -> dict[str, float | str | None]:
    """Compute only the D-007 structure features (for cheap CSV backfill)."""
    y = _decode_waveform(path, target_sr)
    if y is None or y.size == 0:
        out = {key: None for key in AUDIO_STRUCTURE_FEATURE_KEYS}
        out["audio_feature_extraction_status"] = "failed"
        return out
    onset_env = librosa.onset.onset_strength(y=y, sr=target_sr)
    out = _structure_features(y, target_sr, onset_env)
    out["audio_feature_extraction_status"] = "ok"
    return out


def _structure_features(
    y: np.ndarray, sr: int, onset_env: np.ndarray
) -> dict[str, float | None]:
    """Onset density, tempo, beat stability, spectral flux (song-agnostic)."""
    duration = max(len(y) / float(sr), 1e-6)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_density = float(len(onset_frames) / duration)

    # First-3s onset density (hook window).
    n_3s = int(librosa.time_to_frames(3.0, sr=sr))
    onset_env_3s = onset_env[: max(n_3s, 1)]
    onset_frames_3s = librosa.onset.onset_detect(onset_envelope=onset_env_3s, sr=sr)
    hook_dur = min(3.0, duration)
    onset_density_3s = float(len(onset_frames_3s) / max(hook_dur, 1e-6))

    tempo_arr, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.atleast_1d(tempo_arr)[0])

    beat_cv: float | None = None
    if len(beat_frames) >= 3:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        intervals = np.diff(beat_times)
        mean_iv = float(np.mean(intervals))
        if mean_iv > 0:
            beat_cv = float(np.std(intervals) / mean_iv)

    # Spectral flux ≈ mean/std of the onset-strength envelope (librosa's flux proxy).
    flux_mean = float(np.mean(onset_env)) if onset_env.size else None
    flux_std = float(np.std(onset_env)) if onset_env.size else None

    return {
        "audio_onset_density": _round_or_none(onset_density),
        "audio_onset_density_first_3s": _round_or_none(onset_density_3s),
        "audio_tempo_bpm": _round_or_none(tempo, digits=3),
        "audio_beat_interval_cv": _round_or_none(beat_cv),
        "audio_spectral_flux_mean": _round_or_none(flux_mean),
        "audio_spectral_flux_std": _round_or_none(flux_std),
    }


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None
