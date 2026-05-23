from __future__ import annotations


def check_nisqa(filepath: str, min_mos: float = 3.0) -> dict:
    """Predict perceptual Mean Opinion Score using NISQA.

    Requires: pip install audio-data-quality-toolkit[perceptual]

    NISQA predicts MOS on a 1-5 scale for:
      - Noisiness, Coloration, Discontinuity, Loudness
    and an overall quality score.

    Args:
        filepath: Path to audio file.
        min_mos: Minimum acceptable MOS (default 3.0 = "fair").
    """
    try:
        import torch
        from nisqa.NISQA_lib import predict as nisqa_predict
    except ImportError:
        return {
            "check": "nisqa_mos",
            "passed": False,
            "error": "NISQA not installed. Run: pip install audio-data-quality-toolkit[perceptual]",
            "severity": "skipped",
        }

    try:
        result = nisqa_predict(filepath)
        mos = float(result.get("mos_pred", 0))
        return {
            "check": "nisqa_mos",
            "passed": mos >= min_mos,
            "mos": round(mos, 3),
            "noisiness": round(float(result.get("noi_pred", 0)), 3),
            "coloration": round(float(result.get("col_pred", 0)), 3),
            "discontinuity": round(float(result.get("dis_pred", 0)), 3),
            "loudness": round(float(result.get("loud_pred", 0)), 3),
            "min_mos": min_mos,
            "severity": "high" if mos < 2.0 else "medium" if mos < 3.0 else "low",
        }
    except Exception as e:
        return {
            "check": "nisqa_mos",
            "passed": False,
            "error": "NISQA prediction failed: %s" % str(e),
            "severity": "high",
        }


def check_utmos(filepath: str, min_mos: float = 3.0) -> dict:
    """Predict perceptual MOS using UTMOS (UTokyo-SaruLab).

    Requires PyTorch and the utmos model.
    Uses torchaudio for loading and the sarulab-speech/UTMOS22 model.

    Args:
        filepath: Path to audio file.
        min_mos: Minimum acceptable MOS (default 3.0).
    """
    try:
        import torch
        import torchaudio
    except ImportError:
        return {
            "check": "utmos",
            "passed": False,
            "error": "PyTorch/torchaudio not installed. Run: pip install torch torchaudio",
            "severity": "skipped",
        }

    try:
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
            trust_repo=True
        )
        wave, sr = torchaudio.load(filepath)
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)

        with torch.no_grad():
            mos = predictor(wave, sr=16000).item()

        return {
            "check": "utmos",
            "passed": mos >= min_mos,
            "mos": round(mos, 3),
            "min_mos": min_mos,
            "severity": "high" if mos < 2.0 else "medium" if mos < 3.0 else "low",
        }
    except Exception as e:
        return {
            "check": "utmos",
            "passed": False,
            "error": "UTMOS prediction failed: %s" % str(e),
            "severity": "high",
        }


def check_pesq(filepath: str, reference_filepath: str,
               min_score: float = 2.5) -> dict:
    """Compute PESQ (Perceptual Evaluation of Speech Quality).

    Requires a clean reference signal. Scores range from -0.5 to 4.5.

    Args:
        filepath: Path to degraded audio file.
        reference_filepath: Path to clean reference audio file.
        min_score: Minimum acceptable PESQ score (default 2.5).
    """
    try:
        from pesq import pesq as compute_pesq
        import soundfile as sf
    except ImportError:
        return {
            "check": "pesq",
            "passed": False,
            "error": "pesq not installed. Run: pip install pesq",
            "severity": "skipped",
        }

    try:
        ref, sr_ref = sf.read(reference_filepath)
        deg, sr_deg = sf.read(filepath)

        target_sr = 16000
        if sr_ref != target_sr or sr_deg != target_sr:
            import librosa
            ref = librosa.resample(ref, orig_sr=sr_ref, target_sr=target_sr)
            deg = librosa.resample(deg, orig_sr=sr_deg, target_sr=target_sr)

        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        score = compute_pesq(target_sr, ref, deg, "wb")

        return {
            "check": "pesq",
            "passed": score >= min_score,
            "pesq_score": round(float(score), 3),
            "min_score": min_score,
            "scale": "-0.5 to 4.5",
            "severity": "high" if score < 1.5 else "medium" if score < 2.5 else "low",
        }
    except Exception as e:
        return {
            "check": "pesq",
            "passed": False,
            "error": "PESQ computation failed: %s" % str(e),
            "severity": "high",
        }
