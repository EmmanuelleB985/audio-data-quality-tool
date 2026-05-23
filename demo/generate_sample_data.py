from __future__ import annotations
import numpy as np
from scipy.io import wavfile
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "sample_data"
SR = 22050


def make_speech_like(duration_s: float, sr: int = SR) -> np.ndarray:
    """Generate a speech-like signal with drifting pitch and irregular envelope."""
    n_samples = int(sr * duration_s)

    f0_base = 150.0
    f0_drift = np.cumsum(np.random.randn(n_samples) * 0.05)
    f0_drift = f0_drift - np.mean(f0_drift)
    f0_curve = f0_base + np.clip(f0_drift, -40, 40)
    phase = np.cumsum(2 * np.pi * f0_curve / sr)

    signal = 0.30 * np.sin(phase)
    signal += 0.15 * np.sin(2 * phase + np.random.uniform(0, np.pi))
    signal += 0.08 * np.sin(3 * phase + np.random.uniform(0, np.pi))
    signal += 0.04 * np.sin(5 * phase + np.random.uniform(0, np.pi))

    syllable_rate = np.random.uniform(3.0, 5.0)
    n_syllables = int(duration_s * syllable_rate) + 1
    envelope = np.zeros(n_samples)
    samples_per_syl = n_samples // n_syllables
    for i in range(n_syllables):
        start = i * samples_per_syl
        length = int(samples_per_syl * np.random.uniform(0.4, 0.9))
        end = min(start + length, n_samples)
        win = np.hanning(end - start)
        amplitude = np.random.uniform(0.5, 1.0)
        envelope[start:end] += win * amplitude

    envelope = np.clip(envelope, 0, 1)
    signal *= envelope
    noise_level = 0.03 * np.random.uniform(0.8, 1.2)
    signal += noise_level * np.random.randn(n_samples)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.5
    return signal


def save_wav(filename: str, audio: np.ndarray, sr: int = SR):
    audio_16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    filepath = OUTPUT_DIR / filename
    wavfile.write(str(filepath), sr, audio_16)
    print("  Created: %s (%.1fs, %dHz)" % (filename, len(audio) / sr, sr))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating sample audio data...\n")

    for i in range(1, 4):
        audio = make_speech_like(3.0 + i * 0.5)
        save_wav("good_%02d_clean_speech.wav" % i, audio)

    audio = make_speech_like(3.0)
    noisy_low = audio + 0.1 * np.random.randn(len(audio))
    save_wav("noisy_01_low_snr.wav", noisy_low / np.max(np.abs(noisy_low)) * 0.8)

    noisy_high = audio + 0.3 * np.random.randn(len(audio))
    save_wav("noisy_02_very_low_snr.wav", noisy_high / np.max(np.abs(noisy_high)) * 0.8)

    audio = make_speech_like(3.0)
    clipped = np.clip(audio * 2.0, -0.99, 0.99)
    save_wav("clipped_01_hard_clip.wav", clipped)

    audio2 = make_speech_like(2.5)
    clipped2 = np.clip(audio2 * 1.5, -0.99, 0.99)
    save_wav("clipped_02_soft_clip.wav", clipped2)

    audio = make_speech_like(2.0)
    padded_lead = np.concatenate([np.zeros(SR * 2), audio])
    save_wav("silence_01_long_leading.wav", padded_lead)

    padded_trail = np.concatenate([audio, np.zeros(SR * 3)])
    save_wav("silence_02_long_trailing.wav", padded_trail)

    audio1 = make_speech_like(1.5)
    audio2 = make_speech_like(1.5)
    gapped = np.concatenate([audio1, np.zeros(SR * 3), audio2])
    save_wav("silence_03_internal_gap.wav", gapped)

    tiny = make_speech_like(0.2)
    save_wav("duration_01_too_short.wav", tiny)

    long_audio = np.tile(make_speech_like(5.0), 8)
    save_wav("duration_02_too_long.wav", long_audio)

    audio = make_speech_like(3.0, sr=11025)
    save_wav("samplerate_01_wrong.wav", audio, sr=11025)

    audio = make_speech_like(3.0)
    quiet = audio * 0.02
    save_wav("loudness_01_too_quiet.wav", quiet)

    loud = audio * 0.99
    save_wav("loudness_02_too_loud.wav", loud)

    audio = make_speech_like(3.0)
    save_wav("duplicate_01_original.wav", audio)
    dup = audio + 0.001 * np.random.randn(len(audio))
    save_wav("duplicate_02_near_copy.wav", dup)

    t = np.linspace(0, 3.0, int(SR * 3.0), endpoint=False)
    metallic = 0.5 * np.random.randn(len(t))
    metallic *= 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    save_wav("artifact_01_metallic.wav", metallic / np.max(np.abs(metallic)) * 0.7)

    # Upsampled file: generate at 8kHz then save at 22050Hz
    audio_8k = make_speech_like(3.0, sr=8000)
    from scipy.signal import resample
    upsampled = resample(audio_8k, int(len(audio_8k) * SR / 8000))
    save_wav("upsampled_01_fake_22k.wav", upsampled / np.max(np.abs(upsampled)) * 0.5)

    # Stereo file (most TTS expects mono)
    mono = make_speech_like(3.0)
    stereo = np.stack([mono, mono], axis=-1)
    stereo_16 = np.clip(stereo * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(str(OUTPUT_DIR / "channel_01_stereo.wav"), SR, stereo_16)
    print("  Created: channel_01_stereo.wav (3.0s, %dHz, stereo)" % SR)

    print("\nDone. Generated %d sample files in %s" % (
        len(list(OUTPUT_DIR.glob("*.wav"))), OUTPUT_DIR))


if __name__ == "__main__":
    main()
