from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from audio_qa.pipeline import check_file, check_directory

st.set_page_config(page_title="Audio Data Quality Toolkit", layout="wide")
st.title("Audio Data Quality Toolkit")
st.markdown("Automated quality checks for TTS, ASR, and voice-cloning datasets.")
st.markdown("---")

st.sidebar.header("Settings")
expected_sr = st.sidebar.selectbox("Expected sample rate", [None, 16000, 22050, 24000, 44100, 48000])
min_dur = st.sidebar.slider("Min duration (s)", 0.1, 5.0, 0.5)
max_dur = st.sidebar.slider("Max duration (s)", 5.0, 120.0, 30.0)
snr_thresh = st.sidebar.slider("SNR threshold (dB)", 5.0, 40.0, 20.0)
target_lufs = st.sidebar.slider("Target LUFS", -30.0, -10.0, -18.0)

mode = st.radio("Mode", ["Single File", "Sample Data", "Directory"], horizontal=True)

if mode == "Single File":
    uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "ogg"])
    if uploaded:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        uploaded.seek(0)
        st.audio(uploaded, format="audio/wav")

        with st.spinner("Running quality checks..."):
            result = check_file(
                tmp_path,
                expected_sr=expected_sr,
                min_duration=min_dur,
                max_duration=max_dur,
                snr_threshold=snr_thresh,
                target_lufs=target_lufs,
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Checks Passed", "%d/%d" % (result["num_passed"], result["num_checks"]))
        c2.metric("Duration", "%.1fs" % result.get("duration_s", 0))
        c3.metric("Sample Rate", "%d Hz" % result.get("sample_rate", 0))
        c4.metric("Status", "Clean" if result["all_passed"] else "Issues Found")

        st.subheader("Check Results")
        for check in result.get("checks", []):
            name = check.get("check", "unknown")
            passed = check.get("passed", False)
            severity = check.get("severity", "low")
            icon = "[PASS]" if passed else ("[WARN]" if severity == "medium" else "[FAIL]")
            with st.expander("%s %s" % (icon, name.replace("_", " ").title()), expanded=not passed):
                display_keys = {k: v for k, v in check.items()
                               if k not in ("check", "passed", "severity")}
                for k, v in display_keys.items():
                    if isinstance(v, list):
                        for item in v:
                            st.markdown("- %s" % item)
                    else:
                        st.markdown("**%s:** `%s`" % (k, v))

elif mode == "Sample Data":
    sample_dir = Path(__file__).parent / "sample_data"
    if not sample_dir.exists() or not list(sample_dir.glob("*.wav")):
        st.warning("No sample data found. Generate it first:")
        st.code("python demo/generate_sample_data.py")
        st.stop()

    st.markdown("**Found %d sample files**" % len(list(sample_dir.glob("*.wav"))))

    if st.button("Run All Checks", type="primary"):
        with st.spinner("Checking all sample files..."):
            report = check_directory(
                str(sample_dir),
                expected_sr=expected_sr,
                workers=2,
                min_duration=min_dur,
                max_duration=max_dur,
                snr_threshold=snr_thresh,
                target_lufs=target_lufs,
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Files", report.total_files)
        c2.metric("Clean Files", "%d (%.0f%%)" % (report.clean_files, report.clean_ratio * 100))
        c3.metric("Duplicates", len(report.duplicates))
        c4.metric("Errors", report.raw.get("error_files", 0))

        st.subheader("Check Pass Rates")
        for name, counts in report.check_summary.items():
            total = counts["passed"] + counts["failed"]
            rate = counts["passed"] / max(total, 1)
            st.progress(rate, text="%s: %d/%d (%.0f%%)" % (name, counts["passed"], total, rate * 100))

        st.subheader("File Results")
        for r in report.file_results:
            fname = Path(r["file"]).name
            if r.get("error"):
                st.error("ERROR -- %s: %s" % (fname, r["error"]))
                continue
            status = "PASS" if r["all_passed"] else "FAIL"
            with st.expander("%s %s -- %d/%d passed" % (status, fname, r["num_passed"], r["num_checks"])):
                for c in r.get("checks", []):
                    if not c.get("passed"):
                        st.markdown("- FAIL **%s**: severity=%s" % (c["check"], c.get("severity")))

        dupes = report.duplicates
        if dupes:
            st.subheader("Duplicate Pairs")
            for d in dupes:
                st.markdown("- `%s` <-> `%s` (similarity: %s)" % (
                    Path(d["file_a"]).name, Path(d["file_b"]).name, d["similarity"]))

elif mode == "Directory":
    dir_path = st.text_input("Directory path", placeholder="/path/to/your/audio/dataset")
    if dir_path and st.button("Check Directory"):
        p = Path(dir_path)
        if not p.exists():
            st.error("Directory not found: %s" % dir_path)
        else:
            with st.spinner("Checking %s..." % dir_path):
                report = check_directory(
                    dir_path,
                    expected_sr=expected_sr,
                    workers=4,
                    min_duration=min_dur,
                    max_duration=max_dur,
                )
            st.text(report.summary())

st.markdown("---")
st.markdown("[GitHub](https://github.com/EmmanuelleB985/audio-data-quality-tool)")
