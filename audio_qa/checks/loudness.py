from __future__ import annotations
import numpy as np


def measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Simplified integrated LUFS using gated RMS (400ms blocks, 75% overlap).

    Approximates ITU-R BS.1770 without full K-weighting.
    Typical speech is -16 to -24 LUFS.
    """
    block_size = int(0.4 * sr)
    hop = block_size // 4

    if len(audio) < block_size:
        rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        return float(20.0 * np.log10(rms + 1e-10))

    n_blocks = (len(audio) - block_size) // hop + 1
    block_loudness = np.zeros(n_blocks)
    for i in range(n_blocks):
        start = i * hop
        block = audio[start:start + block_size]
        block_loudness[i] = np.mean(block ** 2)

    abs_gate = 10.0 ** (-70.0 / 10.0)
    above_gate = block_loudness[block_loudness > abs_gate]
    if len(above_gate) == 0:
        return -70.0

    ungated_mean = np.mean(above_gate)
    rel_gate = ungated_mean * 10.0 ** (-10.0 / 10.0)
    gated = above_gate[above_gate > rel_gate]
    if len(gated) == 0:
        return -70.0

    integrated = np.mean(gated)
    return float(-0.691 + 10.0 * np.log10(integrated + 1e-10))


def check_loudness(audio: np.ndarray, sr: int,
                   target_lufs: float = -23.0,
                   tolerance_db: float = 10.0) -> dict:
    """Check if audio loudness is within an acceptable range.

    Default target is -23 LUFS with 10 dB tolerance, covering both
    close-mic speech (~-16 to -20 LUFS) and audiobook recordings
    (~-24 to -28 LUFS). This aligns with EBU R128 broadcast standard
    (-23 LUFS) and avoids false positives on clean audiobook datasets
    like LibriTTS, MLS, and LJSpeech.

    Reference: YourTTS normalizes to -27dB RMS. HiFiTTS/HiFiTTS-2
    uses energy-based silence trimming at 50dB threshold. Streaming
    platforms target -14 LUFS (louder, for music/podcasts).
    """
    lufs = measure_lufs(audio, sr)
    deviation = abs(lufs - target_lufs)
    passed = deviation <= tolerance_db

    return {
        "check": "loudness",
        "passed": passed,
        "lufs": round(lufs, 2),
        "target_lufs": target_lufs,
        "deviation_db": round(deviation, 2),
        "tolerance_db": tolerance_db,
        "severity": "high" if deviation > 15 else "medium" if deviation > 10 else "low",
    }
