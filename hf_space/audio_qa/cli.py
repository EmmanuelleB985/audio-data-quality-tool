from __future__ import annotations
import argparse
import sys
from pathlib import Path

from audio_qa.pipeline import check_file, check_directory, save_report


def print_file_result(result: dict):
    fname = Path(result.get("file", "unknown")).name

    if result.get("error"):
        print("  ERROR %s: %s" % (fname, result["error"]))
        return

    status = "PASS" if result["all_passed"] else "FAIL"
    score = result.get("quality_score", 0)
    grade = result.get("grade", "?")
    print("  %s %s -- %d/%d passed | Score: %.1f/10 (%s) | %.1fs, %dHz" % (
        status, fname, result["num_passed"], result["num_checks"],
        score, grade,
        result.get("duration_s", 0), result.get("sample_rate", 0),
    ))

    for check in result.get("checks", []):
        if not check.get("passed"):
            icon = "WARN" if check.get("severity") == "medium" else "FAIL"
            detail = ""
            name = check["check"]
            if name == "background_noise":
                detail = "SNR=%.1fdB" % check.get("snr_db", 0)
            elif name == "clipping":
                detail = "ratio=%s" % check.get("clip_ratio")
            elif name == "silence":
                detail = ", ".join(check.get("issues", []))
            elif name == "loudness":
                detail = "LUFS=%.1fdB (dev=%.1f)" % (check.get("lufs", 0), check.get("deviation_db", 0))
            elif name == "duration":
                detail = ", ".join(check.get("issues", []))
            elif name == "sample_rate":
                detail = ", ".join(check.get("issues", []))
            elif name == "tts_metallic":
                detail = "flatness_ratio=%s" % check.get("metallic_frame_ratio")
            elif name == "tts_repetition":
                detail = "corr=%s" % check.get("max_correlation")
            elif name == "channel":
                detail = ", ".join(check.get("issues", []))
            elif name == "upsampling":
                detail = "claimed=%sHz, estimated_original=%sHz" % (
                    check.get("claimed_sr"), check.get("estimated_original_sr"))
            elif name == "transcript_ratio":
                detail = "%.1f CPS" % check.get("cps", 0)
            print("      %s %s: %s" % (icon, name, detail))


def main():
    parser = argparse.ArgumentParser(
        description="audio-qa -- lint audio datasets for TTS, ASR, and voice-cloning"
    )
    parser.add_argument("path", help="Audio file or directory")
    parser.add_argument("--report", help="Save JSON report to this path")
    parser.add_argument("--csv", help="Save CSV summary to this path")
    parser.add_argument("--manifest", help="Save clean file list to this path")
    parser.add_argument("--expected-sr", type=int, default=None)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--snr-threshold", type=float, default=15.0)
    parser.add_argument("--target-lufs", type=float, default=-23.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    path = Path(args.path)
    kwargs = dict(
        expected_sr=args.expected_sr,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        snr_threshold=args.snr_threshold,
        target_lufs=args.target_lufs,
    )

    if path.is_file():
        print("\nChecking: %s\n" % path.name)
        result = check_file(str(path), **kwargs)
        print_file_result(result)

        if args.report:
            save_report(result, args.report)
            print("\nReport saved to %s" % args.report)

    elif path.is_dir():
        print("\nChecking directory: %s\n" % path)
        report = check_directory(str(path), workers=args.workers, **kwargs)
        print(report.summary())

        # Show flagged files
        flagged = [r for r in report.file_results
                   if not r.get("all_passed") and not r.get("error")]
        if flagged:
            print("\nFlagged files (%d):" % len(flagged))
            for r in flagged[:20]:
                print_file_result(r)

        dupes = report.duplicates
        if dupes:
            print("\nDuplicate pairs (%d):" % len(dupes))
            for d in dupes[:10]:
                print("    DUP %s <-> %s (sim=%s)" % (
                    Path(d["file_a"]).name, Path(d["file_b"]).name, d["similarity"]))

        if args.report:
            report.to_json(args.report)
            print("\nJSON report saved to %s" % args.report)
        if args.csv:
            report.to_csv(args.csv)
            print("CSV report saved to %s" % args.csv)
        if args.manifest:
            report.export_clean_manifest(args.manifest)
            print("Clean manifest saved to %s (%d files)" % (args.manifest, len(report.clean_file_list())))
    else:
        print("Error: %s not found" % path)
        sys.exit(1)


if __name__ == "__main__":
    main()
