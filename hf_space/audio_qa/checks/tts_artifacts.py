from __future__ import annotations
import numpy as np
import librosa


def check_metallic(audio: np.ndarray, sr: int,
                   flatness_threshold: float = 0.4) -> dict:
    """Detect metallic/robotic artifacts via spectral flatness.

    Silent frames are excluded since they naturally have high spectral
    flatness and would cause false positives.
    """
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]

    min_len = min(len(flatness), len(rms))
    flatness = flatness[:min_len]
    rms = rms[:min_len]

    rms_max = np.max(rms)
    if rms_max < 1e-10:
        return {
            "check": "tts_metallic", "passed": True,
            "mean_spectral_flatness": 0.0, "metallic_frame_ratio": 0.0,
            "threshold": flatness_threshold, "severity": "low",
        }

    active_mask = rms > (rms_max * 0.01)
    active_flatness = flatness[active_mask]

    if len(active_flatness) == 0:
        return {
            "check": "tts_metallic", "passed": True,
            "mean_spectral_flatness": 0.0, "metallic_frame_ratio": 0.0,
            "threshold": flatness_threshold, "severity": "low",
        }

    mean_flatness = float(np.mean(active_flatness))
    metallic_frames = int(np.sum(active_flatness > flatness_threshold))
    metallic_ratio = metallic_frames / len(active_flatness)
    passed = metallic_ratio < 0.5

    return {
        "check": "tts_metallic",
        "passed": passed,
        "mean_spectral_flatness": round(mean_flatness, 4),
        "metallic_frame_ratio": round(metallic_ratio, 4),
        "threshold": flatness_threshold,
        "severity": "high" if metallic_ratio > 0.7 else "medium" if metallic_ratio > 0.5 else "low",
    }


def check_repetition(audio: np.ndarray, sr: int,
                     min_repeat_s: float = 0.2,
                     max_repeat_s: float = 2.0,
                     correlation_threshold: float = 0.85) -> dict:
    """Detect word/phrase repetition via autocorrelation."""
    target_sr = min(sr, 8000)
    if sr > target_sr:
        audio_ds = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    else:
        audio_ds = audio
        target_sr = sr

    chunk_len = min(len(audio_ds), target_sr * 10)
    chunk = audio_ds[:chunk_len]

    min_lag = int(min_repeat_s * target_sr)
    max_lag = min(int(max_repeat_s * target_sr), len(chunk) // 2)

    if max_lag <= min_lag:
        return {
            "check": "tts_repetition", "passed": True,
            "max_correlation": 0.0, "severity": "low",
            "detail": "Audio too short for repetition analysis",
        }

    norm = np.sqrt(np.sum(chunk ** 2))
    if norm < 1e-6:
        return {
            "check": "tts_repetition", "passed": True,
            "max_correlation": 0.0, "severity": "low",
        }

    max_corr = 0.0
    max_corr_lag = 0
    step = max(1, target_sr // 50)
    for lag in range(min_lag, max_lag, step):
        shifted = chunk[lag:]
        original = chunk[:len(shifted)]
        if len(original) == 0:
            break
        corr = np.sum(original * shifted) / (
            np.sqrt(np.sum(original ** 2) * np.sum(shifted ** 2)) + 1e-10
        )
        if corr > max_corr:
            max_corr = corr
            max_corr_lag = lag

    passed = max_corr < correlation_threshold
    return {
        "check": "tts_repetition",
        "passed": passed,
        "max_correlation": round(float(max_corr), 4),
        "repeat_period_s": round(max_corr_lag / target_sr, 3) if max_corr_lag > 0 else 0,
        "threshold": correlation_threshold,
        "severity": "high" if max_corr > 0.95 else "medium" if max_corr > 0.85 else "low",
    }
