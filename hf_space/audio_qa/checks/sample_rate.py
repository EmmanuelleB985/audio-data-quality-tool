from __future__ import annotations

STANDARD_RATES = {8000, 16000, 22050, 24000, 44100, 48000}
TTS_PREFERRED = {16000, 22050, 24000, 44100, 48000}


def check_sample_rate(sr: int, expected_sr: int | None = None) -> dict:
    """Validate sample rate against standard and TTS-preferred rates."""
    is_standard = sr in STANDARD_RATES
    is_tts = sr in TTS_PREFERRED
    matches_expected = expected_sr is None or sr == expected_sr
    passed = is_standard and matches_expected

    issues = []
    if not is_standard:
        issues.append("Non-standard rate: %dHz" % sr)
    if not is_tts:
        issues.append("%dHz is not a typical TTS rate" % sr)
    if not matches_expected:
        issues.append("Expected %dHz, got %dHz" % (expected_sr, sr))

    return {
        "check": "sample_rate",
        "passed": passed,
        "sample_rate": sr,
        "expected": expected_sr,
        "is_standard": is_standard,
        "is_tts_suitable": is_tts,
        "issues": issues,
        "severity": "high" if not is_standard else "medium" if not matches_expected else "low",
    }
