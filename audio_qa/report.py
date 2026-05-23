from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any


class Report:
    """Wraps pipeline results with export methods.

    Provides actionable outputs: clean file lists, CSV reports,
    JSON dumps, and HuggingFace dataset filtering.
    """

    def __init__(self, data: dict):
        self._data = data

    @property
    def total_files(self) -> int:
        return self._data.get("total_files", 0)

    @property
    def clean_files(self) -> int:
        return self._data.get("clean_files", 0)

    @property
    def clean_ratio(self) -> float:
        return self._data.get("clean_ratio", 0.0)

    @property
    def file_results(self) -> list:
        return self._data.get("file_results", [])

    @property
    def duplicates(self) -> list:
        return self._data.get("duplicates", [])

    @property
    def check_summary(self) -> dict:
        return self._data.get("check_summary", {})

    @property
    def raw(self) -> dict:
        return self._data

    def clean_file_list(self) -> list:
        """Return list of filepaths that passed all checks."""
        return [
            r["file"] for r in self.file_results
            if r.get("all_passed") and not r.get("error")
        ]

    def failed_file_list(self) -> list:
        """Return list of filepaths that failed at least one check."""
        return [
            r["file"] for r in self.file_results
            if not r.get("all_passed") or r.get("error")
        ]

    def export_clean_manifest(self, path: str):
        """Write a text file with one clean filepath per line."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for fp in self.clean_file_list():
                f.write(fp + "\n")

    def to_json(self, path: str):
        """Save the full report as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def to_csv(self, path: str):
        """Save a per-file summary as CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in self.file_results:
            row = {
                "file": r.get("file", ""),
                "passed": r.get("all_passed", False),
                "quality_score": r.get("quality_score", ""),
                "grade": r.get("grade", ""),
                "num_passed": r.get("num_passed", 0),
                "num_checks": r.get("num_checks", 0),
                "duration_s": r.get("duration_s", ""),
                "sample_rate": r.get("sample_rate", ""),
                "error": r.get("error", ""),
            }
            for c in r.get("checks", []):
                row["check_%s" % c["check"]] = "pass" if c.get("passed") else "fail"
            rows.append(row)

        if not rows:
            return

        fieldnames = list(rows[0].keys())
        for row in rows[1:]:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def filter_hf_dataset(self, dataset: Any) -> Any:
        """Return a filtered HuggingFace Dataset keeping only clean samples.

        Works with both Dataset (uses .select) and IterableDataset (uses .filter).
        """
        clean_indices = set()
        for i, r in enumerate(self.file_results):
            if r.get("all_passed") and not r.get("error"):
                clean_indices.add(i)

        if hasattr(dataset, "select"):
            return dataset.select(sorted(clean_indices))

        # IterableDataset: use .filter with index tracking
        counter = {"i": 0}

        def _keep(example):
            idx = counter["i"]
            counter["i"] += 1
            return idx in clean_indices

        return dataset.filter(_keep)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        scores = [r.get("quality_score", 0) for r in self.file_results
                  if r.get("quality_score") is not None and not r.get("error")]
        avg_score = sum(scores) / max(len(scores), 1) if scores else 0

        grade_counts = {}
        for r in self.file_results:
            g = r.get("grade", "?")
            grade_counts[g] = grade_counts.get(g, 0) + 1

        lines = [
            "Total files:  %d" % self.total_files,
            "Clean files:  %d (%.0f%%)" % (self.clean_files, self.clean_ratio * 100),
            "Failed files: %d" % (self.total_files - self.clean_files),
            "Duplicates:   %d" % len(self.duplicates),
            "",
            "Quality Score: %.1f / 10  (avg across %d files)" % (avg_score, len(scores)),
            "Grade distribution: %s" % ", ".join(
                "%s=%d" % (g, n) for g, n in sorted(grade_counts.items())
            ),
            "",
            "Per-check pass rates:",
        ]
        for name, counts in self.check_summary.items():
            total = counts["passed"] + counts["failed"]
            rate = counts["passed"] / max(total, 1)
            bar = "#" * int(rate * 20) + "." * (20 - int(rate * 20))
            lines.append("  %-20s %s %.0f%%" % (name, bar, rate * 100))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "Report(total=%d, clean=%d, ratio=%.1f%%)" % (
            self.total_files, self.clean_files, self.clean_ratio * 100
        )
