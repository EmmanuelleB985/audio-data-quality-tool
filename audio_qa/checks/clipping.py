from __future__ import annotations
import numpy as np


def check_clipping(audio: np.ndarray, sr: int,
                   threshold: float = 0.99,
                   min_consecutive: int = 3) -> dict:
    """Detect digital clipping where consecutive samples hit max amplitude."""
    clipped = np.abs(audio) >= threshold
    clip_regions = []
    run_start = None
    run_len = 0

    for i, is_clip in enumerate(clipped):
        if is_clip:
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            if run_start is not None and run_len >= min_consecutive:
                clip_regions.append({
                    "start_sample": int(run_start),
                    "end_sample": int(run_start + run_len),
                    "start_time": round(run_start / sr, 4),
                    "duration_samples": run_len,
                })
            run_start = None
            run_len = 0

    if run_start is not None and run_len >= min_consecutive:
        clip_regions.append({
            "start_sample": int(run_start),
            "end_sample": int(run_start + run_len),
            "start_time": round(run_start / sr, 4),
            "duration_samples": run_len,
        })

    total_clipped = sum(r["duration_samples"] for r in clip_regions)
    clip_ratio = total_clipped / max(len(audio), 1)

    return {
        "check": "clipping",
        "passed": len(clip_regions) == 0,
        "clip_regions": len(clip_regions),
        "total_clipped_samples": total_clipped,
        "clip_ratio": round(clip_ratio, 6),
        "threshold": threshold,
        "severity": "high" if clip_ratio > 0.01 else "medium" if clip_ratio > 0.001 else "low",
    }
