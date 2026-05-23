from __future__ import annotations
import numpy as np


def compute_quality_score(checks: list, weights: dict | None = None) -> dict:
    """Compute a composite quality score (0-10) from individual check results.

    The score is a weighted combination of continuous metrics extracted from
    check results, normalized to a 0-10 scale. This provides a single number
    comparable to NISQA MOS or PESQ, but computed entirely from signal-level
    features without any ML model or GPU.

    Score interpretation:
        9-10: Excellent -- studio quality, ready for any pipeline
        7-9:  Good -- suitable for most TTS/ASR training
        5-7:  Acceptable -- usable with caveats
        3-5:  Poor -- likely to degrade model quality
        0-3:  Bad -- should be excluded from training

    Args:
        checks: List of check result dicts from check_file().
        weights: Optional dict of {component_name: weight}. Defaults to
                 balanced weights for TTS training readiness.

    Returns:
        Dict with overall score, component scores, and grade.
    """
    default_weights = {
        "snr": 2.5,
        "clipping": 1.5,
        "silence": 1.0,
        "loudness": 1.5,
        "metallic": 1.5,
        "upsampling": 1.0,
        "channel": 0.5,
        "duration": 0.5,
    }
    w = weights if weights is not None else default_weights

    check_map = {}
    for c in checks:
        check_map[c.get("check", "")] = c

    components = {}

    # SNR: 0dB->0, 10dB->5, 20dB->8, 30+dB->10
    snr_check = check_map.get("background_noise", {})
    snr_db = snr_check.get("snr_db", 0)
    snr_score = _sigmoid_map(snr_db, midpoint=15, steepness=0.15)
    components["snr"] = round(snr_score, 2)

    # Clipping: 0 regions -> 10, any clipping degrades rapidly
    clip_check = check_map.get("clipping", {})
    clip_ratio = clip_check.get("clip_ratio", 0)
    clip_score = max(0, 10.0 * (1.0 - clip_ratio * 500))
    components["clipping"] = round(min(10, clip_score), 2)

    # Silence: penalize excess leading + trailing + internal
    # Thresholds aligned with defaults: 1.0s leading/trailing, 3.0s internal
    sil_check = check_map.get("silence", {})
    leading = sil_check.get("leading_silence_s", 0)
    trailing = sil_check.get("trailing_silence_s", 0)
    internal = sil_check.get("max_internal_silence_s", 0)
    sil_penalty = min(1.0, (max(0, leading - 1.0) + max(0, trailing - 1.0) + max(0, internal - 3.0)) / 5.0)
    sil_score = 10.0 * (1.0 - sil_penalty)
    components["silence"] = round(sil_score, 2)

    # Loudness: deviation from target. 0dB dev->10, 5dB->7, 10dB->4, 15+->1
    loud_check = check_map.get("loudness", {})
    deviation = loud_check.get("deviation_db", 0)
    loud_score = max(0, 10.0 - deviation * 0.6)
    components["loudness"] = round(loud_score, 2)

    # Metallic: low flatness ratio -> 10, high -> 0
    met_check = check_map.get("tts_metallic", {})
    met_ratio = met_check.get("metallic_frame_ratio", 0)
    met_score = max(0, 10.0 * (1.0 - met_ratio * 2.0))
    components["metallic"] = round(met_score, 2)

    # Upsampling: binary -- either genuine or fake
    up_check = check_map.get("upsampling", {})
    up_score = 10.0 if up_check.get("passed", True) else 1.0
    components["upsampling"] = round(up_score, 2)

    # Channel: mono=10, stereo with issues=3
    ch_check = check_map.get("channel", {})
    ch_issues = len(ch_check.get("issues", []))
    ch_score = 10.0 if ch_check.get("passed", True) else max(2.0, 8.0 - ch_issues * 2.0)
    components["channel"] = round(ch_score, 2)

    # Duration: in range=10, out of range=penalty
    dur_check = check_map.get("duration", {})
    dur_score = 10.0 if dur_check.get("passed", True) else 4.0
    components["duration"] = round(dur_score, 2)

    # Weighted average
    total_weight = sum(w.get(k, 0) for k in components)
    if total_weight == 0:
        overall = 0.0
    else:
        overall = sum(components[k] * w.get(k, 0) for k in components) / total_weight

    overall = round(min(10.0, max(0.0, overall)), 2)

    if overall >= 9.0:
        grade = "A"
    elif overall >= 7.0:
        grade = "B"
    elif overall >= 5.0:
        grade = "C"
    elif overall >= 3.0:
        grade = "D"
    else:
        grade = "F"

    return {
        "quality_score": overall,
        "grade": grade,
        "components": components,
        "weights": {k: w.get(k, 0) for k in components},
        "scale": "0-10 (10=best)",
    }


def _sigmoid_map(value: float, midpoint: float, steepness: float) -> float:
    """Map a value to 0-10 using a sigmoid curve."""
    x = steepness * (value - midpoint)
    return 10.0 / (1.0 + np.exp(-x))
