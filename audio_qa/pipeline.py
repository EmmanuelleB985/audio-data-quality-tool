from __future__ import annotations
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np

from audio_qa.checks.noise import check_noise
from audio_qa.checks.clipping import check_clipping
from audio_qa.checks.silence import check_silence
from audio_qa.checks.sample_rate import check_sample_rate
from audio_qa.checks.duration import check_duration
from audio_qa.checks.loudness import check_loudness
from audio_qa.checks.tts_artifacts import check_metallic, check_repetition
from audio_qa.checks.channel import check_channel
from audio_qa.checks.upsampling import check_upsampling
from audio_qa.checks.quality_score import compute_quality_score
from audio_qa.report import Report


def check_file(filepath: str,
               expected_sr: int | None = None,
               min_duration: float = 0.5,
               max_duration: float = 30.0,
               snr_threshold: float = 15.0,
               target_lufs: float = -23.0) -> dict:
    """Run all quality checks on a single audio file."""
    try:
        audio, sr = librosa.load(filepath, sr=None, mono=True)
    except Exception as e:
        return {
            "file": str(filepath),
            "error": "Failed to load: %s" % str(e),
            "num_checks": 0,
            "num_passed": 0,
            "checks": [],
        }

    checks = []
    checks.append(check_noise(audio, sr, snr_threshold))
    checks.append(check_clipping(audio, sr))
    checks.append(check_silence(audio, sr))
    checks.append(check_sample_rate(sr, expected_sr))
    checks.append(check_duration(len(audio), sr, min_duration, max_duration))
    checks.append(check_loudness(audio, sr, target_lufs))
    checks.append(check_metallic(audio, sr))
    checks.append(check_repetition(audio, sr))
    checks.append(check_channel(filepath))
    checks.append(check_upsampling(audio, sr))

    num_passed = sum(1 for c in checks if c.get("passed"))
    score_result = compute_quality_score(checks)

    return {
        "file": str(filepath),
        "sample_rate": int(sr),
        "duration_s": round(len(audio) / sr, 3),
        "num_checks": len(checks),
        "num_passed": num_passed,
        "all_passed": num_passed == len(checks),
        "quality_score": score_result["quality_score"],
        "grade": score_result["grade"],
        "score_components": score_result["components"],
        "checks": checks,
    }


# Keep backward compat alias
check_single_file = check_file


def check_directory(directory: str,
                    expected_sr: int | None = None,
                    workers: int = 4,
                    extensions: tuple = (".wav", ".mp3", ".flac", ".ogg"),
                    **kwargs) -> Report:
    """Run quality checks on all audio files in a directory.

    Returns a Report object with export methods.
    """
    dir_path = Path(directory)
    audio_files = sorted([
        f for f in dir_path.rglob("*")
        if f.suffix.lower() in extensions
    ])

    if not audio_files:
        return Report({
            "directory": str(directory),
            "total_files": 0,
            "clean_files": 0,
            "clean_ratio": 0.0,
            "error": "No audio files found",
            "file_results": [],
            "check_summary": {},
            "duplicates": [],
        })

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(check_file, str(f), expected_sr, **kwargs): f
            for f in audio_files
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({
                    "file": str(futures[future]),
                    "error": str(e),
                    "num_checks": 0,
                    "num_passed": 0,
                })

    results.sort(key=lambda r: r.get("file", ""))

    # Cross-file duplicate detection
    from audio_qa.checks.duplicates import compute_fingerprint, find_duplicates
    fingerprints = {}
    for r in results:
        if r.get("error"):
            continue
        try:
            audio, sr = librosa.load(r["file"], sr=None, mono=True)
            fingerprints[r["file"]] = compute_fingerprint(audio, sr)
        except Exception:
            pass

    duplicates = find_duplicates(fingerprints) if fingerprints else []

    n_total = len(results)
    n_clean = sum(1 for r in results if r.get("all_passed"))
    n_errors = sum(1 for r in results if r.get("error"))

    check_summary = {}
    for r in results:
        for c in r.get("checks", []):
            name = c.get("check", "unknown")
            if name not in check_summary:
                check_summary[name] = {"passed": 0, "failed": 0}
            if c.get("passed"):
                check_summary[name]["passed"] += 1
            else:
                check_summary[name]["failed"] += 1

    return Report({
        "directory": str(directory),
        "total_files": n_total,
        "clean_files": n_clean,
        "clean_ratio": round(n_clean / max(n_total, 1), 3),
        "error_files": n_errors,
        "duplicates": duplicates,
        "check_summary": check_summary,
        "file_results": results,
    })


# Keep backward compat alias
check_dataset = check_directory


def audit_hf_dataset(dataset,
                     audio_column: str = "audio",
                     max_samples: int | None = None,
                     expected_sr: int | None = None,
                     **kwargs) -> Report:
    """Run quality checks on a HuggingFace Dataset with an audio column.

    Usage:
        from datasets import load_dataset
        from audio_qa import audit_hf_dataset

        ds = load_dataset("blabble-io/libritts_r", "clean",
                          split="train.clean.100", streaming=True)
        report = audit_hf_dataset(ds, max_samples=500)
        print(report.summary())
        report.to_csv("libritts_qa.csv")

    Note: For older datasets that use loading scripts, you may need to pass
    trust_remote_code=True to load_dataset().
    """
    import soundfile as sf

    results = []
    count = 0

    for sample in dataset:
        if max_samples is not None and count >= max_samples:
            break

        audio_data = sample.get(audio_column)
        if audio_data is None:
            count += 1
            continue

        # HF audio column is {"array": np.array, "sampling_rate": int, "path": str}
        if isinstance(audio_data, dict):
            array = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data["sampling_rate"]
            filepath = audio_data.get("path", "sample_%d" % count)

            # Write to temp file for channel check
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, array, sr)
            tmp.close()

            result = _check_array(array, sr, filepath, tmp.name, expected_sr, **kwargs)

            # Add transcript ratio check if transcript column exists
            transcript = sample.get("sentence") or sample.get("text") or sample.get("transcript")
            if transcript:
                from audio_qa.checks.transcript_ratio import check_transcript_ratio
                duration_s = len(array) / sr
                result["checks"].append(check_transcript_ratio(duration_s, transcript))
                result["num_checks"] = len(result["checks"])
                result["num_passed"] = sum(1 for c in result["checks"] if c.get("passed"))
                result["all_passed"] = result["num_passed"] == result["num_checks"]

            results.append(result)
        count += 1

    n_total = len(results)
    n_clean = sum(1 for r in results if r.get("all_passed"))

    check_summary = {}
    for r in results:
        for c in r.get("checks", []):
            name = c.get("check", "unknown")
            if name not in check_summary:
                check_summary[name] = {"passed": 0, "failed": 0}
            if c.get("passed"):
                check_summary[name]["passed"] += 1
            else:
                check_summary[name]["failed"] += 1

    return Report({
        "source": "huggingface_dataset",
        "total_files": n_total,
        "clean_files": n_clean,
        "clean_ratio": round(n_clean / max(n_total, 1), 3),
        "error_files": sum(1 for r in results if r.get("error")),
        "duplicates": [],
        "check_summary": check_summary,
        "file_results": results,
    })


def _check_array(audio: np.ndarray, sr: int, filepath: str,
                 tmp_path: str,
                 expected_sr: int | None = None,
                 min_duration: float = 0.5,
                 max_duration: float = 30.0,
                 snr_threshold: float = 15.0,
                 target_lufs: float = -23.0) -> dict:
    """Run checks on a numpy array (used internally by audit_hf_dataset)."""
    checks = []
    checks.append(check_noise(audio, sr, snr_threshold))
    checks.append(check_clipping(audio, sr))
    checks.append(check_silence(audio, sr))
    checks.append(check_sample_rate(sr, expected_sr))
    checks.append(check_duration(len(audio), sr, min_duration, max_duration))
    checks.append(check_loudness(audio, sr, target_lufs))
    checks.append(check_metallic(audio, sr))
    checks.append(check_repetition(audio, sr))
    checks.append(check_channel(tmp_path))
    checks.append(check_upsampling(audio, sr))

    num_passed = sum(1 for c in checks if c.get("passed"))
    score_result = compute_quality_score(checks)

    return {
        "file": str(filepath),
        "sample_rate": int(sr),
        "duration_s": round(len(audio) / sr, 3),
        "num_checks": len(checks),
        "num_passed": num_passed,
        "all_passed": num_passed == len(checks),
        "quality_score": score_result["quality_score"],
        "grade": score_result["grade"],
        "score_components": score_result["components"],
        "checks": checks,
    }


def save_report(report_or_data, output_path: str):
    """Save a report or dict as JSON."""
    data = report_or_data.raw if isinstance(report_or_data, Report) else report_or_data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
