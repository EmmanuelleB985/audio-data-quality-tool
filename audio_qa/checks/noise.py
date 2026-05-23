from __future__ import annotations
import numpy as np
import librosa


def estimate_snr(audio: np.ndarray, sr: int, frame_length: int = 2048) -> float:
    """Estimate signal-to-noise ratio in dB using frame-level energy.

    Excludes frames below an absolute silence gate (-60 dB) so that
    intentional pauses are not counted as noise. The bottom 10 percent
    of remaining frames are treated as the noise floor.
    """
    frames = librosa.util.frame(audio, frame_length=frame_length,
                                hop_length=frame_length // 2)
    energy = np.sum(frames ** 2, axis=0)

    max_energy = np.max(energy)
    if max_energy < 1e-10:
        return 0.0

    silence_gate = max_energy * 1e-6
    active_energy = energy[energy > silence_gate]

    if len(active_energy) < 2:
        return 100.0

    sorted_energy = np.sort(active_energy)
    n_noise = max(1, len(sorted_energy) // 10)
    noise_energy = np.mean(sorted_energy[:n_noise])
    signal_energy = np.mean(sorted_energy[n_noise:])

    if noise_energy < 1e-10:
        return 100.0
    return float(10.0 * np.log10(signal_energy / noise_energy))


def check_noise(audio: np.ndarray, sr: int,
                snr_threshold: float = 15.0) -> dict:
    """Flag audio with low signal-to-noise ratio.

    Default threshold is 15 dB, a practical floor for usable
    TTS/ASR training data. For premium quality, HiFi-TTS uses
    32 dB (studio recordings). For noisy crowd-sourced data like
    Common Voice, 10 dB may be more appropriate. Adjust via
    --snr-threshold in the CLI.
    """
    snr = estimate_snr(audio, sr)
    return {
        "check": "background_noise",
        "passed": snr >= snr_threshold,
        "snr_db": round(snr, 2),
        "threshold_db": snr_threshold,
        "severity": "high" if snr < 10 else "medium" if snr < 15 else "low",
    }
