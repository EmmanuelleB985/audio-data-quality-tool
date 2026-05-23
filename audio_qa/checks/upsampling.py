from __future__ import annotations
import numpy as np


def check_upsampling(audio: np.ndarray, sr: int,
                     energy_ratio_threshold: float = 0.0005) -> dict:
    """Detect files that were upsampled from a lower sample rate.

    Computes the FFT and checks whether there is meaningful energy
    above the Nyquist frequency of common lower sample rates.
    Genuinely upsampled files have near-zero energy (ratio < 0.0001)
    in the upper band. Natural speech rolls off above ~8kHz but
    still has detectable energy from fricatives, breath noise, and
    room tone.

    The check requires BOTH a very low energy ratio AND a sharp
    spectral cutoff (measured by the ratio of energy just below vs
    just above the candidate Nyquist) to avoid false positives on
    soft audiobook recordings.
    """
    n = len(audio)
    if n < 256:
        return {
            "check": "upsampling",
            "passed": True,
            "severity": "low",
            "detail": "Audio too short for spectral analysis",
        }

    spectrum = np.abs(np.fft.rfft(audio))
    power = spectrum ** 2
    n_bins = len(spectrum)

    candidate_rates = [8000, 11025, 16000]
    detected_original_sr = None
    detected_ratio = None

    for orig_sr in candidate_rates:
        if orig_sr >= sr:
            continue
        nyquist_bin = int((orig_sr / 2) / (sr / 2) * n_bins)
        if nyquist_bin >= n_bins - 10 or nyquist_bin < 10:
            continue

        lower_energy = np.mean(power[1:nyquist_bin])
        upper_energy = np.mean(power[nyquist_bin:])

        if lower_energy < 1e-10:
            continue

        ratio = upper_energy / lower_energy

        # Check for sharp spectral cliff at the candidate Nyquist.
        # Upsampled files have an abrupt cutoff; natural speech
        # rolls off gradually.
        band_width = max(5, n_bins // 50)
        below_band = np.mean(power[max(1, nyquist_bin - band_width):nyquist_bin])
        above_band = np.mean(power[nyquist_bin:nyquist_bin + band_width])

        if below_band < 1e-10:
            continue

        cliff_ratio = above_band / below_band

        # Both conditions must hold:
        # 1. Overall upper energy is very low relative to lower
        # 2. Sharp cliff at the boundary (not gradual rolloff)
        if ratio < energy_ratio_threshold and cliff_ratio < 0.01:
            detected_original_sr = orig_sr
            detected_ratio = ratio
            break

    if detected_original_sr is not None:
        return {
            "check": "upsampling",
            "passed": False,
            "claimed_sr": sr,
            "estimated_original_sr": detected_original_sr,
            "upper_band_energy_ratio": round(detected_ratio, 6),
            "threshold": energy_ratio_threshold,
            "severity": "high",
        }

    return {
        "check": "upsampling",
        "passed": True,
        "claimed_sr": sr,
        "severity": "low",
    }