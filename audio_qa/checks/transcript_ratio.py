from __future__ import annotations


def check_transcript_ratio(duration_s: float, transcript: str,
                           min_cps: float = 5.0,
                           max_cps: float = 25.0) -> dict:
    """Check if the transcript length is plausible for the audio duration.

    Uses characters-per-second as a proxy. Normal English speech is
    roughly 12-16 CPS. Values far outside 5-25 CPS usually indicate
    a misaligned transcript, truncated audio, or wrong pairing.

    Args:
        duration_s: Audio duration in seconds.
        transcript: The text transcript for this audio.
        min_cps: Minimum plausible characters per second.
        max_cps: Maximum plausible characters per second.
    """
    text = transcript.strip()
    n_chars = len(text)

    if duration_s <= 0 or n_chars == 0:
        return {
            "check": "transcript_ratio",
            "passed": False,
            "chars": n_chars,
            "duration_s": round(duration_s, 3),
            "cps": 0.0,
            "severity": "high",
            "detail": "Empty transcript or zero-length audio",
        }

    cps = n_chars / duration_s
    passed = min_cps <= cps <= max_cps

    issues = []
    if cps < min_cps:
        issues.append("Too few chars for duration (%.1f CPS, transcript may be truncated)" % cps)
    if cps > max_cps:
        issues.append("Too many chars for duration (%.1f CPS, audio may be truncated)" % cps)

    return {
        "check": "transcript_ratio",
        "passed": passed,
        "chars": n_chars,
        "duration_s": round(duration_s, 3),
        "cps": round(cps, 1),
        "min_cps": min_cps,
        "max_cps": max_cps,
        "issues": issues,
        "severity": "high" if not passed else "low",
    }
