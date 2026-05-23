from __future__ import annotations
import numpy as np
import tempfile
from scipy.io import wavfile
from audio_qa.checks.noise import estimate_snr, check_noise
from audio_qa.checks.clipping import check_clipping
from audio_qa.checks.silence import check_silence
from audio_qa.checks.duration import check_duration
from audio_qa.checks.sample_rate import check_sample_rate
from audio_qa.checks.loudness import check_loudness
from audio_qa.checks.upsampling import check_upsampling
from audio_qa.checks.transcript_ratio import check_transcript_ratio
from audio_qa.checks.channel import check_channel
from audio_qa.pipeline import check_file


def test_clean_has_high_snr(clean_speech):
    audio, sr = clean_speech
    snr = estimate_snr(audio, sr)
    assert snr > 8

def test_noisy_has_low_snr(noisy_signal):
    audio, sr = noisy_signal
    snr = estimate_snr(audio, sr)
    assert snr < 20

def test_check_noise_clean(clean_speech):
    audio, sr = clean_speech
    result = check_noise(audio, sr, snr_threshold=5)
    assert result["passed"]

def test_no_clipping_clean(clean_speech):
    audio, sr = clean_speech
    result = check_clipping(audio, sr)
    assert result["passed"]

def test_detects_clipping(clipped_signal):
    audio, sr = clipped_signal
    result = check_clipping(audio, sr)
    assert not result["passed"] or result["clip_regions"] > 0

def test_clean_no_silence_issues(clean_speech):
    audio, sr = clean_speech
    result = check_silence(audio, sr)
    assert result["passed"]

def test_silent_signal_flagged(silent_signal):
    audio, sr = silent_signal
    result = check_silence(audio, sr)
    assert result["leading_silence_s"] > 1.0 or result["trailing_silence_s"] > 1.0

def test_duration_in_range():
    result = check_duration(22050 * 5, 22050, min_seconds=1.0, max_seconds=30.0)
    assert result["passed"]

def test_duration_too_short():
    result = check_duration(22050, 22050 * 10, min_seconds=1.0)
    assert not result["passed"]

def test_sample_rate_standard():
    result = check_sample_rate(22050)
    assert result["passed"]

def test_sample_rate_nonstandard():
    result = check_sample_rate(11025)
    assert not result["passed"]

def test_upsampling_clean(clean_speech):
    audio, sr = clean_speech
    result = check_upsampling(audio, sr)
    assert result["passed"]

def test_transcript_ratio_normal():
    result = check_transcript_ratio(5.0, "This is a normal sentence for testing purposes.")
    assert result["passed"]

def test_transcript_ratio_too_fast():
    result = check_transcript_ratio(0.5, "This is way too much text for half a second of audio really.")
    assert not result["passed"]

def test_channel_mono():
    sr = 22050
    audio = np.random.randn(sr * 2).astype(np.float32) * 0.3
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, sr, (audio * 32767).astype(np.int16))
        result = check_channel(f.name)
    assert result["passed"]

def test_check_file_runs():
    sr = 22050
    n = int(sr * 3.0)
    f0_drift = np.cumsum(np.random.randn(n) * 0.05)
    f0_curve = 150 + np.clip(f0_drift - np.mean(f0_drift), -40, 40)
    phase = np.cumsum(2 * np.pi * f0_curve / sr)
    audio = 0.3 * np.sin(phase) + 0.02 * np.random.randn(n)
    audio = audio / np.max(np.abs(audio)) * 0.5
    audio_16 = (audio * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, sr, audio_16)
        result = check_file(f.name)
    assert result["num_checks"] >= 10
    assert "checks" in result
    assert "quality_score" in result
    assert "grade" in result
    assert 0 <= result["quality_score"] <= 10
    assert result["grade"] in ("A", "B", "C", "D", "F")

def test_nonexistent_file():
    result = check_file("/nonexistent.wav")
    assert "error" in result

def test_quality_score_perfect():
    from audio_qa.checks.quality_score import compute_quality_score
    checks = [
        {"check": "background_noise", "passed": True, "snr_db": 40},
        {"check": "clipping", "passed": True, "clip_ratio": 0},
        {"check": "silence", "passed": True, "leading_silence_s": 0.1, "trailing_silence_s": 0.1, "max_internal_silence_s": 0},
        {"check": "loudness", "passed": True, "deviation_db": 1},
        {"check": "tts_metallic", "passed": True, "metallic_frame_ratio": 0.02},
        {"check": "upsampling", "passed": True},
        {"check": "channel", "passed": True, "issues": []},
        {"check": "duration", "passed": True},
    ]
    result = compute_quality_score(checks)
    assert result["quality_score"] >= 8.0
    assert result["grade"] in ("A", "B")

def test_quality_score_bad():
    from audio_qa.checks.quality_score import compute_quality_score
    checks = [
        {"check": "background_noise", "passed": False, "snr_db": 3},
        {"check": "clipping", "passed": False, "clip_ratio": 0.05},
        {"check": "silence", "passed": False, "leading_silence_s": 3, "trailing_silence_s": 3, "max_internal_silence_s": 5},
        {"check": "loudness", "passed": False, "deviation_db": 15},
        {"check": "tts_metallic", "passed": False, "metallic_frame_ratio": 0.8},
        {"check": "upsampling", "passed": False},
        {"check": "channel", "passed": False, "issues": ["stereo", "silent left", "phase inverted"]},
        {"check": "duration", "passed": False},
    ]
    result = compute_quality_score(checks)
    assert result["quality_score"] <= 3.0
    assert result["grade"] in ("D", "F")
