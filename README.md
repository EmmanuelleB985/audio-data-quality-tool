# audio-data-quality-toolkit

**Lint your audio datasets before training.** 13 automated checks for TTS, ASR, and voice-cloning pipelines. Zero GPU required.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1xP6a21lmb1cF2suNOcW_ege9Ne9bn_CB/view?usp=sharing)

```python
from audio_qa import check_directory

report = check_directory("./my-dataset/")
print(report.summary())
report.export_clean_manifest("clean_files.txt")  # ready for training
```

## Why this exists

Existing tools answer the wrong question for dataset builders:

| Tool | Question it answers | GPU | Output |
|------|-------------------|-----|--------|
| NISQA (300+ stars) | "How does this audio sound?" (MOS 1-5) | Yes | Quality score |
| PESQ (600+ stars) | "How degraded is this vs reference?" | No | Quality score |
| DataSpeech (390+ stars) | "Describe this audio for Parler-TTS" | Yes | NL descriptions |
| **audio-qa** | **"Should this file be in my training set?"** | **No** | **Score (0-10) + pass/fail + clean manifest** |

DataSpeech annotates datasets for conditioned TTS training. NISQA predicts perceptual quality. Neither tells you that 12% of your training set has clipping, 8% is upsampled from 8kHz, and 3% has misaligned transcripts -- the data engineering problems that silently degrade your model.

## Quick start

```bash
pip install -e .
audio-qa ./my-dataset/ --csv report.csv --manifest clean_files.txt
```

## Quality score (0-10)

Every file gets a composite quality score on a 0-10 scale, computed from the signal-level checks. No GPU, no ML model -- just weighted signal metrics.

```python
from audio_qa import check_file

result = check_file("sample.wav")
print(result["quality_score"])   # 8.3
print(result["grade"])           # "B"
print(result["score_components"])
# {'snr': 9.1, 'clipping': 10.0, 'silence': 9.5, 'loudness': 7.2,
#  'metallic': 9.8, 'upsampling': 10.0, 'channel': 10.0, 'duration': 10.0}
```

| Score | Grade | Meaning |
|-------|-------|---------|
| 9-10 | A | Studio quality, ready for any pipeline |
| 7-9 | B | Good, suitable for most TTS/ASR training |
| 5-7 | C | Acceptable with caveats |
| 3-5 | D | Poor, likely to degrade model quality |
| 0-3 | F | Bad, exclude from training |

Directory reports show average score and grade distribution:

```python
report = check_directory("./data")
print(report.summary())
# Total files:  1000
# Clean files:  847 (85%)
# Quality Score: 7.8 / 10  (avg across 1000 files)
# Grade distribution: A=312, B=401, C=134, D=98, F=55
```

### Optional: perceptual MOS (NISQA / UTMOS / PESQ)

For ML-based perceptual scores alongside the signal checks:

```python
from audio_qa.checks.perceptual import check_nisqa, check_utmos, check_pesq

# NISQA: no-reference MOS (1-5 scale), needs PyTorch
result = check_nisqa("sample.wav")        # {"mos": 3.8, "noisiness": 4.1, ...}

# UTMOS: no-reference MOS, needs PyTorch
result = check_utmos("sample.wav")        # {"mos": 4.2, ...}

# PESQ: reference-based (-0.5 to 4.5), needs clean reference
result = check_pesq("degraded.wav", "clean_reference.wav")
```

Install with: `pip install audio-data-quality-toolkit[perceptual]`

## HuggingFace integration

```python
from datasets import load_dataset
from audio_qa import audit_hf_dataset

# LibriTTS-R -- cleaned audiobooks
ds = load_dataset("blabble-io/libritts_r", "clean", split="train.clean.100", streaming=True)
report = audit_hf_dataset(ds, max_samples=500)
print(report.summary())

# MLS English -- large-scale multilingual speech
ds = load_dataset("parler-tts/mls_eng", split="train", streaming=True)
report = audit_hf_dataset(ds, max_samples=500)
print(report.summary())

# LibriSpeech ASR -- classic ASR benchmark
ds = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True)
report = audit_hf_dataset(ds, max_samples=500)
print(report.summary())

# LJSpeech -- needs trust_remote_code with datasets>=4.0
# ds = load_dataset("keithito/lj_speech", split="train",
#                   streaming=True, trust_remote_code=True)

# Common Voice -- requires accepting terms + HF auth token
# ds = load_dataset("mozilla-foundation/common_voice_13_0", "en",
#                   split="train", streaming=True, token="hf_...")

# Filter to clean samples only
clean_ds = report.filter_hf_dataset(ds)

# Export for review
report.to_csv("qa_report.csv")
```

> **Setup for HuggingFace audio datasets:**
> ```bash
> # datasets v4.0+ requires torchcodec for audio decoding
> pip install torchcodec datasets huggingface-hub
>
> # OR pin to datasets v3.x to avoid the torchcodec dependency
> pip install "datasets>=2.14,<4.0"
> ```
> If you get `RuntimeError: Dataset scripts are no longer supported`,
> add `trust_remote_code=True` to `load_dataset()`.

## 13 checks

| # | Check | What it catches | GPU |
|---|-------|----------------|-----|
| 1 | SNR estimation | Background noise, hum, hiss | No |
| 2 | Clipping detection | Consecutive samples at max amplitude | No |
| 3 | Silence analysis | Excessive leading/trailing/internal silence | No |
| 4 | Sample rate | Non-standard or mismatched rates | No |
| 5 | Duration bounds | Too short or too long for training | No |
| 6 | Loudness (LUFS) | Audio far from target loudness | No |
| 7 | Metallic artifacts | Robotic/metallic TTS artifacts via spectral flatness | No |
| 8 | Repetition | Word/phrase loops via autocorrelation | No |
| 9 | Channel issues | Stereo, silent channel, phase inversion, dual mono | No |
| 10 | Upsampling detection | Files claiming 22kHz but upsampled from 8kHz | No |
| 11 | Transcript ratio | Misaligned transcripts (chars-per-second) | No |
| 12 | Duplicates | Near-duplicate files via chromagram fingerprinting | No |
| 13 | Transcript alignment | Audio vs expected text (optional, Whisper) | Optional |

Checks 1-12 run on CPU with numpy/scipy/librosa. Check 13 requires `pip install audio-data-quality-toolkit[transcript]`.

## Python API

### Single file

```python
from audio_qa import check_file

result = check_file("sample.wav", expected_sr=22050)
for check in result["checks"]:
    if not check["passed"]:
        print(check["check"], check.get("severity"))
```

### Directory with exports

```python
from audio_qa import check_directory

report = check_directory("./data", workers=8)
print(report.summary())

report.to_csv("qa_report.csv")         # spreadsheet
report.to_json("qa_report.json")       # structured
report.export_clean_manifest("clean.txt")  # filepaths that passed all checks
```

### HuggingFace dataset

```python
from datasets import load_dataset
from audio_qa import audit_hf_dataset

ds = load_dataset("keithito/lj_speech", split="train", streaming=True)
report = audit_hf_dataset(ds, max_samples=500)
print(report.summary())
```

### Transcript alignment (no ML)

```python
from audio_qa.checks.transcript_ratio import check_transcript_ratio

result = check_transcript_ratio(duration_s=4.2, transcript="Hello world test.")
# result["cps"] -> characters per second
# result["passed"] -> True if within normal speech range (5-25 CPS)
```

## CLI

```
audio-qa <file_or_directory> [options]

Options:
  --report PATH        Save full JSON report
  --csv PATH           Save per-file CSV summary
  --manifest PATH      Save clean file list (one per line)
  --expected-sr INT    Expected sample rate
  --min-duration FLOAT Min duration in seconds (default: 0.5)
  --max-duration FLOAT Max duration in seconds (default: 30.0)
  --snr-threshold FLOAT Min SNR in dB (default: 20.0)
  --target-lufs FLOAT  Target LUFS (default: -18.0)
  --workers INT        Parallel workers (default: 4)
```

## Optional extras

```bash
pip install -e ".[perceptual]"    # NISQA MOS scoring (PyTorch)
pip install -e ".[transcript]"    # Whisper transcript alignment
pip install -e ".[hf]"            # HuggingFace datasets
pip install -e ".[demo]"          # Gradio demo
pip install -e ".[all]"           # Everything
```

## HuggingFace Space

Try it without installing: upload audio files and get instant quality reports.

```bash
pip install -e ".[demo]"
python demo/hf_space_app.py
```

## Project structure

```
audio_qa/
  pipeline.py           # check_file, check_directory, audit_hf_dataset
  report.py             # Report class: to_csv, to_json, export_clean_manifest
  cli.py                # CLI entry point
  checks/
    quality_score.py    # Composite 0-10 score from signal checks
    noise.py            # SNR estimation (silence-gated)
    clipping.py         # Consecutive peak detection
    silence.py          # Leading/trailing/internal silence
    sample_rate.py      # Standard rate validation
    duration.py         # Min/max bounds
    loudness.py         # Simplified LUFS
    tts_artifacts.py    # Metallic + repetition detection
    channel.py          # Mono/stereo, phase, silent channel
    upsampling.py       # Fake SR detection via FFT
    transcript_ratio.py # CPS sanity check
    duplicates.py       # Chromagram fingerprinting
    transcript.py       # Whisper alignment (optional)
    perceptual.py       # NISQA, UTMOS, PESQ wrappers (optional)
demo/
  hf_space_app.py       # Gradio app for HuggingFace Spaces
  app.py                # Streamlit demo
  generate_sample_data.py
```

## License

MIT
