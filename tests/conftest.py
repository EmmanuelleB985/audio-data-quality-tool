from __future__ import annotations
import numpy as np
import pytest


@pytest.fixture
def clean_speech():
    sr = 22050
    n = int(sr * 3.0)
    f0 = 150.0
    f0_drift = np.cumsum(np.random.randn(n) * 0.05)
    f0_curve = f0 + np.clip(f0_drift - np.mean(f0_drift), -40, 40)
    phase = np.cumsum(2 * np.pi * f0_curve / sr)
    signal = 0.3 * np.sin(phase) + 0.15 * np.sin(2 * phase)
    envelope = np.zeros(n)
    syl = n // 12
    for i in range(12):
        s = i * syl
        e = min(s + int(syl * 0.7), n)
        envelope[s:e] = np.hanning(e - s) * np.random.uniform(0.5, 1.0)
    signal *= np.clip(envelope, 0, 1)
    signal += 0.03 * np.random.randn(n)
    signal = signal / np.max(np.abs(signal)) * 0.5
    return signal, sr


@pytest.fixture
def noisy_signal():
    sr = 22050
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * 150 * t)
    noise = 0.3 * np.random.randn(len(signal))
    return signal + noise, sr


@pytest.fixture
def clipped_signal():
    sr = 22050
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    signal = np.sin(2 * np.pi * 150 * t) * 2.0
    return np.clip(signal, -0.99, 0.99), sr


@pytest.fixture
def silent_signal():
    sr = 22050
    return np.zeros(sr * 3), sr
