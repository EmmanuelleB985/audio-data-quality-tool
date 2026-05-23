from __future__ import annotations
import numpy as np
import librosa


def compute_fingerprint(audio: np.ndarray, sr: int,
                        n_chroma: int = 12) -> np.ndarray:
    """Compute a compact chromagram-based fingerprint (fixed-size vector)."""
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=n_chroma)
    fingerprint = np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)])
    norm = np.linalg.norm(fingerprint)
    if norm > 0:
        fingerprint = fingerprint / norm
    return fingerprint


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def find_duplicates(fingerprints: dict, threshold: float = 0.995) -> list:
    """Find near-duplicate pairs from {filename: fingerprint_vector}."""
    files = list(fingerprints.keys())
    duplicates = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            sim = cosine_similarity(fingerprints[files[i]], fingerprints[files[j]])
            if sim >= threshold:
                duplicates.append({
                    "file_a": files[i],
                    "file_b": files[j],
                    "similarity": round(sim, 4),
                })
    return duplicates
