from __future__ import annotations
import numpy as np


def check_upsampling(audio: np.ndarray, sr: int,
                     energy_ratio_threshold: float = 0.001) -> dict:
    """Detect files that were upsampled from a lower sample rate.

    Computes the FFT and checks whether the upper half of the spectrum
    (above sr/4, i.e. Nyquist of half the current rate) contains
    meaningful energy. Genuinely recorded audio at a given sample rate
    will have energy across the full spectrum. Upsampled audio will
    have near-zero energy above the original Nyquist frequency.

    Common case: an 8kHz recording upsampled to 22050Hz will have
    no energy above 4kHz.
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
    n_bins = len(spectrum)

    # Check multiple candidate original rates
    candidate_rates = [8000, 11025, 16000, 22050]
    detected_original_sr = None

    for orig_sr in candidate_rates:
        if orig_sr >= sr:
            continue
        # Frequency bin corresponding to original Nyquist
        nyquist_bin = int((orig_sr / 2) / (sr / 2) * n_bins)
        if nyquist_bin >= n_bins - 1:
            continue

        lower_energy = np.mean(spectrum[1:nyquist_bin] ** 2)
        upper_energy = np.mean(spectrum[nyquist_bin:] ** 2)

        if lower_energy < 1e-10:
            continue

        ratio = upper_energy / lower_energy
        if ratio < energy_ratio_threshold:
            detected_original_sr = orig_sr
            break

    if detected_original_sr is not None:
        return {
            "check": "upsampling",
            "passed": False,
            "claimed_sr": sr,
            "estimated_original_sr": detected_original_sr,
            "upper_band_energy_ratio": round(ratio, 6),
            "threshold": energy_ratio_threshold,
            "severity": "high",
        }

    return {
        "check": "upsampling",
        "passed": True,
        "claimed_sr": sr,
        "severity": "low",
    }
