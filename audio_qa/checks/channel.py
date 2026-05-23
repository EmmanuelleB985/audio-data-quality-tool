from __future__ import annotations
import numpy as np
import soundfile as sf


def check_channel(filepath: str) -> dict:
    """Detect channel issues: stereo where mono expected, left-only,
    phase-inverted channels, or silent channels.
    """
    try:
        info = sf.info(filepath)
    except Exception as e:
        return {
            "check": "channel",
            "passed": False,
            "error": str(e),
            "severity": "high",
        }

    n_channels = info.channels
    issues = []

    if n_channels == 1:
        return {
            "check": "channel",
            "passed": True,
            "channels": 1,
            "issues": [],
            "severity": "low",
        }

    # Load stereo for analysis
    try:
        data, sr = sf.read(filepath, dtype="float32", always_2d=True)
    except Exception as e:
        return {
            "check": "channel",
            "passed": False,
            "error": str(e),
            "severity": "high",
        }

    issues.append("File is %d-channel (most TTS pipelines expect mono)" % n_channels)

    if n_channels >= 2:
        left = data[:, 0]
        right = data[:, 1]

        left_rms = np.sqrt(np.mean(left ** 2))
        right_rms = np.sqrt(np.mean(right ** 2))

        # Silent channel
        if left_rms < 1e-6:
            issues.append("Left channel is silent")
        if right_rms < 1e-6:
            issues.append("Right channel is silent")

        # Identical channels (dual mono)
        if left_rms > 1e-6 and right_rms > 1e-6:
            if np.allclose(left, right, atol=1e-6):
                issues.append("Channels are identical (dual mono)")

            # Phase inversion
            correlation = np.sum(left * right) / (
                np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)) + 1e-10
            )
            if correlation < -0.9:
                issues.append("Channels are phase-inverted (correlation=%.2f)" % correlation)

    return {
        "check": "channel",
        "passed": n_channels == 1,
        "channels": n_channels,
        "issues": issues,
        "severity": "medium" if n_channels > 1 else "low",
    }
