"""
audio-data-quality-toolkit -- HuggingFace Space

Upload audio files and get instant quality reports.
No GPU required. Runs entirely on CPU.
"""
from __future__ import annotations
import json
from pathlib import Path

import gradio as gr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_qa.pipeline import check_file
from audio_qa.checks.transcript_ratio import check_transcript_ratio


def analyze_single(audio_path, transcript, expected_sr, min_dur, max_dur, snr_thresh):
    if audio_path is None:
        return "Upload an audio file to get started.", ""

    expected = int(expected_sr) if expected_sr and expected_sr != "Auto" else None

    result = check_file(
        audio_path,
        expected_sr=expected,
        min_duration=float(min_dur),
        max_duration=float(max_dur),
        snr_threshold=float(snr_thresh),
    )

    if result.get("error"):
        return "Error: %s" % result["error"], ""

    if transcript and transcript.strip():
        tr_check = check_transcript_ratio(result["duration_s"], transcript)
        result["checks"].append(tr_check)
        result["num_checks"] = len(result["checks"])
        result["num_passed"] = sum(1 for c in result["checks"] if c.get("passed"))
        result["all_passed"] = result["num_passed"] == result["num_checks"]

    lines = []
    status = "PASS" if result["all_passed"] else "ISSUES FOUND"
    score = result.get("quality_score", 0)
    grade = result.get("grade", "?")
    lines.append("## %s -- Quality Score: %.1f / 10 (Grade %s)" % (status, score, grade))
    lines.append("")
    lines.append("**Duration:** %.2fs | **Sample Rate:** %d Hz | **Checks:** %d/%d passed" % (
        result["duration_s"], result["sample_rate"],
        result["num_passed"], result["num_checks"],
    ))
    lines.append("")

    components = result.get("score_components", {})
    if components:
        lines.append("### Score Breakdown")
        lines.append("")
        for comp_name, comp_score in components.items():
            bar_len = int(comp_score)
            bar = "=" * bar_len + "." * (10 - bar_len)
            lines.append("- **%s:** %s %.1f" % (comp_name.title(), bar, comp_score))
        lines.append("")

    for c in result["checks"]:
        name = c["check"].replace("_", " ").title()
        if c.get("passed"):
            lines.append("- [PASS] **%s**" % name)
        else:
            severity = c.get("severity", "medium")
            icon = "FAIL" if severity == "high" else "WARN"
            detail_parts = []
            for k, v in c.items():
                if k in ("check", "passed", "severity"):
                    continue
                if isinstance(v, list) and v:
                    detail_parts.append("%s: %s" % (k, "; ".join(str(x) for x in v)))
                elif isinstance(v, float):
                    detail_parts.append("%s=%.2f" % (k, v))
                elif v is not None and not isinstance(v, list):
                    detail_parts.append("%s=%s" % (k, v))
            detail = " | ".join(detail_parts[:4])
            lines.append("- [%s] **%s**: %s" % (icon, name, detail))

    summary = "\n".join(lines)

    json_output = json.dumps(result, indent=2, default=str)
    return summary, json_output


def analyze_batch(files, expected_sr, min_dur, max_dur, snr_thresh):
    if not files:
        return "Upload one or more audio files."

    expected = int(expected_sr) if expected_sr and expected_sr != "Auto" else None
    all_results = []

    for f in files:
        filepath = f.name if hasattr(f, 'name') else str(f)
        result = check_file(
            filepath,
            expected_sr=expected,
            min_duration=float(min_dur),
            max_duration=float(max_dur),
            snr_threshold=float(snr_thresh),
        )
        all_results.append(result)

    n_total = len(all_results)
    n_clean = sum(1 for r in all_results if r.get("all_passed"))
    n_error = sum(1 for r in all_results if r.get("error"))

    check_stats = {}
    for r in all_results:
        for c in r.get("checks", []):
            name = c["check"]
            if name not in check_stats:
                check_stats[name] = {"passed": 0, "failed": 0}
            if c.get("passed"):
                check_stats[name]["passed"] += 1
            else:
                check_stats[name]["failed"] += 1

    lines = []
    lines.append("## Dataset Report")
    lines.append("")

    scores = [r.get("quality_score", 0) for r in all_results
              if r.get("quality_score") is not None and not r.get("error")]
    avg_score = sum(scores) / max(len(scores), 1) if scores else 0

    lines.append("**Total:** %d files | **Clean:** %d (%.0f%%) | **Avg Score:** %.1f/10 | **Errors:** %d" % (
        n_total, n_clean, n_clean / max(n_total, 1) * 100,
        avg_score, n_error,
    ))
    lines.append("")
    lines.append("### Per-Check Pass Rates")
    lines.append("")
    lines.append("| Check | Passed | Failed | Rate |")
    lines.append("|-------|--------|--------|------|")
    for name, counts in check_stats.items():
        total = counts["passed"] + counts["failed"]
        rate = counts["passed"] / max(total, 1) * 100
        lines.append("| %s | %d | %d | %.0f%% |" % (
            name.replace("_", " ").title(), counts["passed"], counts["failed"], rate
        ))

    lines.append("")
    lines.append("### Flagged Files")
    lines.append("")
    for r in all_results:
        if r.get("error"):
            lines.append("- **ERROR** %s: %s" % (Path(r["file"]).name, r["error"]))
        elif not r.get("all_passed"):
            fname = Path(r["file"]).name
            score = r.get("quality_score", 0)
            grade = r.get("grade", "?")
            failed = [c["check"] for c in r.get("checks", []) if not c.get("passed")]
            lines.append("- **%s** (%.1f/10 %s): %s" % (fname, score, grade, ", ".join(failed)))

    return "\n".join(lines)


TITLE = "🎙️ Audio Data Quality Toolkit for TTS/ASR Training Pipelines"

DESCRIPTION = """
Detect clipping, silence, noisy samples, duplicate clips, transcript mismatch,
speaker imbalance, and synthetic-data artifacts in speech datasets.

Designed for TTS, ASR, voice-cloning, and synthetic speech evaluation workflows.
"""

with gr.Blocks(title="Audio Data Quality Toolkit", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    gr.Markdown("""
**Lint your audio datasets before training.** Training-readiness checks for TTS, ASR, and voice-cloning pipelines, with roadmap support for duplicate detection, speaker balance, and ASR-based transcript alignment.
No GPU required. All checks run on CPU with numpy/scipy/librosa.

Unlike perceptual scoring tools such as NISQA, PESQ, or UTMOS, which answer *"how good does this sound?"*,
this toolkit answers *"is this file ready for training?"* by catching the data-engineering issues that silently degrade model quality.
""")

    with gr.Tabs():
        with gr.Tab("Single clip analysis"):
            gr.Markdown("Upload one audio clip and inspect training-readiness quality signals.")
            with gr.Row():
                with gr.Column(scale=2):
                    audio_input = gr.Audio(type="filepath", label="Upload audio clip")
                    transcript_input = gr.Textbox(
                        label="Optional transcript",
                        placeholder="Paste the expected transcript here to check chars-per-second alignment...",
                        lines=2,
                    )
                with gr.Column(scale=1):
                    sr_choice = gr.Dropdown(
                        choices=["Auto", "16000", "22050", "24000", "44100", "48000"],
                        value="Auto", label="Expected sample rate",
                    )
                    min_dur = gr.Number(value=0.5, label="Min duration (s)")
                    max_dur = gr.Number(value=30.0, label="Max duration (s)")
                    snr_thresh = gr.Number(value=20.0, label="SNR threshold (dB)")

            analyze_btn = gr.Button("Analyze audio quality", variant="primary")
            result_md = gr.Markdown(label="Quality report")
            result_json = gr.Code(label="Full JSON", language="json")

            analyze_btn.click(
                analyze_single,
                inputs=[audio_input, transcript_input, sr_choice, min_dur, max_dur, snr_thresh],
                outputs=[result_md, result_json],
            )

        with gr.Tab("Batch dataset audit"):
            gr.Markdown(
                "Upload multiple clips to generate a dataset-level QA report for TTS, ASR, voice-cloning, or synthetic speech pipelines."
            )
            batch_input = gr.File(
                file_count="multiple",
                file_types=["audio"],
                label="Upload audio files",
            )
            with gr.Row():
                b_sr = gr.Dropdown(
                    choices=["Auto", "16000", "22050", "24000", "44100", "48000"],
                    value="Auto", label="Expected sample rate",
                )
                b_min = gr.Number(value=0.5, label="Min duration (s)")
                b_max = gr.Number(value=30.0, label="Max duration (s)")
                b_snr = gr.Number(value=20.0, label="SNR threshold (dB)")

            batch_btn = gr.Button("Run batch audit", variant="primary")
            batch_result = gr.Markdown(label="Dataset quality report")

            batch_btn.click(
                analyze_batch,
                inputs=[batch_input, b_sr, b_min, b_max, b_snr],
                outputs=[batch_result],
            )

        with gr.Tab("Synthetic speech evaluation"):
            gr.Markdown(
                "Evaluate generated speech samples for clipping, silence, noise, duration anomalies, and transcript consistency."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    synthetic_audio = gr.Audio(type="filepath", label="Generated speech sample")
                    expected_text = gr.Textbox(
                        label="Expected text",
                        placeholder="Paste the prompt/text that the TTS system was supposed to speak...",
                        lines=3,
                    )
                with gr.Column(scale=1):
                    synth_sr = gr.Dropdown(
                        choices=["Auto", "16000", "22050", "24000", "44100", "48000"],
                        value="Auto", label="Expected sample rate",
                    )
                    synth_min = gr.Number(value=0.5, label="Min duration (s)")
                    synth_max = gr.Number(value=60.0, label="Max duration (s)")
                    synth_snr = gr.Number(value=20.0, label="SNR threshold (dB)")

            synth_button = gr.Button("Evaluate synthetic sample", variant="primary")
            synth_output = gr.Markdown(label="Synthetic speech QA")
            synth_json = gr.Code(label="Full JSON", language="json")

            synth_button.click(
                analyze_single,
                inputs=[synthetic_audio, expected_text, synth_sr, synth_min, synth_max, synth_snr],
                outputs=[synth_output, synth_json],
            )

        with gr.Tab("About"):
            gr.Markdown("""
## What this tool checks

- **Clipping:** waveform peaks too close to maximum amplitude
- **Silence:** long leading, trailing, or internal silent regions
- **Noise:** low signal quality, background hum, hiss, or abnormal energy profile
- **Transcript mismatch:** audio duration may not match the expected text length
- **Speaker imbalance:** some speakers may dominate the dataset *(roadmap / metadata-dependent)*
- **Duplicates:** repeated or near-identical clips *(roadmap / fingerprinting-dependent)*
- **Synthetic artifacts:** robotic, metallic, repeated, or degraded generated speech patterns

## Why this matters

Data quality directly affects TTS/ASR model stability, pronunciation, speaker consistency, alignment, and long-form generation quality.
This Space is designed as a practical QA dashboard for speech datasets used in training and evaluating voice AI systems.

## Current checks

| # | Check | What It Catches | GPU? |
|---|-------|----------------|------|
| 1 | SNR Estimation | Background noise, hum, hiss | No |
| 2 | Clipping Detection | Consecutive samples at max amplitude | No |
| 3 | Silence Analysis | Excessive leading, trailing, or internal silence | No |
| 4 | Sample Rate Validation | Non-standard or unexpected rates | No |
| 5 | Duration Bounds | Too short or too long clips | No |
| 6 | Loudness (LUFS) | Audio far from target loudness | No |
| 7 | Metallic Artifacts | Robotic/metallic TTS artifacts | No |
| 8 | Repetition Detection | Word/phrase loops via autocorrelation | No |
| 9 | Channel Issues | Stereo, silent channels, phase inversion | No |
| 10 | Upsampling Detection | Fake sample rates, e.g. 8kHz upsampled to 22kHz | No |
| 11 | Transcript Ratio | Misaligned transcripts using chars-per-second | No |
| 12 | Duplicate Detection | Near-duplicate files via fingerprinting | No |
| 13 | Transcript Alignment | Audio vs text mismatch with optional ASR | Optional |

## How is this different from NISQA / PESQ / DataSpeech?

| Tool | What it does | GPU | Output |
|------|-------------|-----|--------|
| **NISQA** | Perceptual MOS score | Yes | Quality score |
| **PESQ** | Reference-based quality score | No | Quality score |
| **DataSpeech** | Annotate datasets for Parler-TTS training | Yes | Natural-language descriptions |
| **This toolkit** | Pass/fail lint for training readiness | No | Report + clean manifest |

DataSpeech answers: *"describe this audio's characteristics for TTS conditioning."*  
This toolkit answers: *"should I include this file in my training set at all?"*

---
**Install:** `pip install audio-data-quality-toolkit`  
[GitHub](https://github.com/EmmanuelleB985/audio-data-quality-toolkit)  
**Python API:** `from audio_qa import check_file, check_directory, audit_hf_dataset`
""")


if __name__ == "__main__":
    demo.launch()
