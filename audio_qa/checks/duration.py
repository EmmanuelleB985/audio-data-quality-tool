from __future__ import annotations


def check_duration(audio_length: int, sr: int,
                   min_seconds: float = 0.5,
                   max_seconds: float = 30.0) -> dict:
    """Check if audio duration falls within acceptable bounds for training."""
    duration = audio_length / sr
    too_short = duration < min_seconds
    too_long = duration > max_seconds
    passed = not too_short and not too_long

    issues = []
    if too_short:
        issues.append("Too short: %.2fs (min %.1fs)" % (duration, min_seconds))
    if too_long:
        issues.append("Too long: %.2fs (max %.1fs)" % (duration, max_seconds))

    return {
        "check": "duration",
        "passed": passed,
        "duration_s": round(duration, 3),
        "min_s": min_seconds,
        "max_s": max_seconds,
        "issues": issues,
        "severity": "high" if too_short else "medium" if too_long else "low",
    }
