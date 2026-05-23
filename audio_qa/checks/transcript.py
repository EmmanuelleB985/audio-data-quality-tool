from __future__ import annotations
import difflib
import re


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def check_transcript(audio_path: str, expected_transcript: str,
                     model_name: str = "base") -> dict:
    """Verify audio matches expected transcript using Whisper."""
    try:
        import whisper
    except ImportError:
        return {
            "check": "transcript",
            "passed": False,
            "error": "whisper not installed (pip install openai-whisper)",
            "severity": "skipped",
        }

    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)
        actual = result["text"]
    except Exception as e:
        return {
            "check": "transcript",
            "passed": False,
            "error": "Whisper transcription failed: %s" % str(e),
            "severity": "high",
        }

    norm_expected = _normalize(expected_transcript)
    norm_actual = _normalize(actual)
    similarity = difflib.SequenceMatcher(
        None, norm_expected.split(), norm_actual.split()
    ).ratio()

    return {
        "check": "transcript",
        "passed": similarity >= 0.85,
        "expected_text": expected_transcript[:200],
        "actual_text": actual[:200],
        "word_similarity": round(similarity, 3),
        "threshold": 0.85,
        "severity": "high" if similarity < 0.5 else "medium" if similarity < 0.85 else "low",
    }
