"""Speech vs music activity on the already-decoded mono waveform.

No Whisper. Frames that look voice-like (speech-band energy + moderate ZCR)
are counted; a late energy jump after early speech flags a dramatic open.
"""

from __future__ import annotations

import numpy as np

SPEECH_FEATURE_KEYS: tuple[str, ...] = (
    "speech_ratio",
    "speech_ratio_first_3s",
    "music_after_speech_gap",
)

_FRAME_MS = 25.0
_HOP_MS = 10.0
_SILENCE = 1e-4
_ZCR_LO = 0.04
_ZCR_HI = 0.28
_SPEECH_BAND = (300.0, 3400.0)


def _empty_features() -> dict[str, float | int | None]:
    return {key: None for key in SPEECH_FEATURE_KEYS}


def _frame_params(sr: int) -> tuple[int, int]:
    n_fft = max(int(sr * _FRAME_MS / 1000.0), 32)
    hop = max(int(sr * _HOP_MS / 1000.0), 1)
    return n_fft, hop


def _speech_mask(y: np.ndarray, sr: int) -> np.ndarray:
    """Per-hop boolean: energy + ZCR + not a single-tone peak (music note)."""
    n_fft, hop = _frame_params(sr)
    if y.size < n_fft:
        return np.zeros(0, dtype=bool)
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    windowed = frames * window
    rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1) / 2.0
    spec = np.abs(np.fft.rfft(windowed, axis=1))
    power = spec**2
    totals = power.sum(axis=1)
    peak = power.max(axis=1)
    peak_ratio = np.divide(peak, totals, out=np.ones_like(peak), where=totals > 0)
    return (
        (rms > _SILENCE)
        & (zcr > _ZCR_LO)
        & (zcr < _ZCR_HI)
        & (peak_ratio < 0.35)
    )


def extract_speech_activity(
    y: np.ndarray, sr: int
) -> dict[str, float | int | None]:
    """Speech occupancy and 'talk then hit' opening on a mono float waveform."""
    if y.size == 0 or sr <= 0:
        return _empty_features()

    mask = _speech_mask(y, sr)
    if mask.size == 0:
        return {
            "speech_ratio": 0.0,
            "speech_ratio_first_3s": 0.0,
            "music_after_speech_gap": 0,
        }

    n_fft, hop = _frame_params(sr)
    hop_s = hop / float(sr)
    times = np.arange(mask.size) * hop_s
    first_3s = times < 3.0
    ratio = float(mask.mean())
    ratio_3s = float(mask[first_3s].mean()) if np.any(first_3s) else 0.0

    # Dramatic open: speech in the first 1.5s, then RMS jumps after 1.5s.
    split = int(1.5 * sr)
    music_after = 0
    if 0 < split < y.size:
        early = y[:split]
        late = y[split : min(len(y), int(3.0 * sr))]
        early_speech = ratio_3s > 0.05 and float(np.mean(_speech_mask(early, sr))) > 0.08
        if late.size and early_speech:
            early_rms = float(np.sqrt(np.mean(early**2)))
            late_rms = float(np.sqrt(np.mean(late**2)))
            if late_rms > max(early_rms * 1.8, 2 * _SILENCE):
                music_after = 1

    return {
        "speech_ratio": round(ratio, 4),
        "speech_ratio_first_3s": round(ratio_3s, 4),
        "music_after_speech_gap": music_after,
    }


def speech_band_energy_ratio(y: np.ndarray, sr: int) -> float | None:
    """Share of FFT energy in the telephone/speech band (vocal-like, not a stem)."""
    if y.size == 0 or sr <= 0:
        return None
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    total = float(np.sum(spectrum**2))
    if total <= 0:
        return 0.0
    lo, hi = _SPEECH_BAND
    band = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(spectrum[band] ** 2) / total)
