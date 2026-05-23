from __future__ import annotations
import numpy as np


def _rms_frames(audio: np.ndarray, frame_len: int = 1024,
                hop: int = 512) -> np.ndarray:
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_len]
        rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)
    return rms


def check_silence(audio: np.ndarray, sr: int,
                  silence_db: float = -40.0,
                  max_leading_s: float = 1.0,
                  max_trailing_s: float = 1.0,
                  max_internal_s: float = 3.0) -> dict:
    """Detect excessive leading, trailing, or internal silence.

    Defaults are tuned for audiobook and read-speech datasets.
    HiFiTTS-2 trims to 0.5s at start/end; we use 1.0s to avoid
    false positives on natural sentence-boundary pauses in datasets
    like LibriTTS, MLS, and Common Voice. Internal gap tolerance
    is 3.0s to accommodate paragraph breaks in long-form recordings.
    """
    hop = 512
    rms = _rms_frames(audio, frame_len=1024, hop=hop)
    silence_thresh = 10.0 ** (silence_db / 20.0)
    is_silent = rms < silence_thresh

    if len(is_silent) == 0:
        return {
            "check": "silence",
            "passed": False,
            "leading_silence_s": 0,
            "trailing_silence_s": 0,
            "max_internal_silence_s": 0,
            "severity": "high",
        }

    frame_dur = hop / sr

    leading_frames = 0
    for s in is_silent:
        if s:
            leading_frames += 1
        else:
            break
    leading_s = leading_frames * frame_dur

    trailing_frames = 0
    for s in reversed(is_silent):
        if s:
            trailing_frames += 1
        else:
            break
    trailing_s = trailing_frames * frame_dur

    max_gap = 0
    gap = 0
    for i, s in enumerate(is_silent):
        if i < leading_frames or i >= len(is_silent) - trailing_frames:
            continue
        if s:
            gap += 1
            max_gap = max(max_gap, gap)
        else:
            gap = 0
    max_internal_silence = max_gap * frame_dur

    issues = []
    if leading_s > max_leading_s:
        issues.append("leading=%.2fs (max %.1fs)" % (leading_s, max_leading_s))
    if trailing_s > max_trailing_s:
        issues.append("trailing=%.2fs (max %.1fs)" % (trailing_s, max_trailing_s))
    if max_internal_silence > max_internal_s:
        issues.append("internal_gap=%.2fs (max %.1fs)" % (max_internal_silence, max_internal_s))

    return {
        "check": "silence",
        "passed": len(issues) == 0,
        "leading_silence_s": round(leading_s, 3),
        "trailing_silence_s": round(trailing_s, 3),
        "max_internal_silence_s": round(max_internal_silence, 3),
        "issues": issues,
        "severity": "high" if len(issues) > 1 else "medium" if issues else "low",
    }
