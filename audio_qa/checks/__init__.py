from __future__ import annotations

from audio_qa.checks.noise import estimate_snr, check_noise
from audio_qa.checks.clipping import check_clipping
from audio_qa.checks.silence import check_silence
from audio_qa.checks.sample_rate import check_sample_rate
from audio_qa.checks.duration import check_duration
from audio_qa.checks.loudness import check_loudness
from audio_qa.checks.duplicates import compute_fingerprint, find_duplicates
from audio_qa.checks.tts_artifacts import check_metallic, check_repetition
from audio_qa.checks.channel import check_channel
from audio_qa.checks.upsampling import check_upsampling
from audio_qa.checks.transcript_ratio import check_transcript_ratio
from audio_qa.checks.quality_score import compute_quality_score

__all__ = [
    "estimate_snr", "check_noise",
    "check_clipping",
    "check_silence",
    "check_sample_rate",
    "check_duration",
    "check_loudness",
    "compute_fingerprint", "find_duplicates",
    "check_metallic", "check_repetition",
    "check_channel",
    "check_upsampling",
    "check_transcript_ratio",
    "compute_quality_score",
]
