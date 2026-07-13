"""Audio features (PRD 11.6).

Song-agnostic, production-focused audio metrics: loudness/RMS, dynamic range,
silence & clipping ratios, spectral statistics, onset strength, and energy
windows. We deliberately do NOT compute BPM, tempo, genre, melody, or motif
(PRD 11.6).

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
)


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

    if energy_full and energy_full > 0 and energy_first_3s is not None:
        features["audio_energy_ratio_first_3s_to_full"] = round(
            energy_first_3s / energy_full, 4
        )
    else:
        features["audio_energy_ratio_first_3s_to_full"] = None

    features["audio_feature_extraction_status"] = "ok"
    return features


def _round_or_none(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
