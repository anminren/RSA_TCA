#!/usr/bin/env python3
"""Collect schema-v2 FRANK subset diagnostics from fold directories."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


SUMMARY_FILENAME = "subset_comparison_summary.json"
TRANSITIONS = (
    "flagged_to_clear",
    "clear_to_flagged",
    "flagged_to_flagged",
    "clear_to_clear",
)
ALIGNMENTS = (
    "matched",
    "different_summary",
    "missing_original_summary",
    "missing_annotation",
)
SYSTEMS = ("baseline", "finetuned")

FIELDS = [
    "name",
    "schema_version",
    "n_samples",
    *(f"lexical_screen_{key}" for key in TRANSITIONS),
    *(f"{system}_annotation_{key}" for system in SYSTEMS for key in ALIGNMENTS),
    "mean_diff_rougeL",
    "mean_lexical_absence_delta",
]


class SummarySchemaError(ValueError):
    """Raised when a fold summary is not a complete schema-v2 document."""


def _require_mapping(value: Any, path: str, source: Path) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SummarySchemaError(f"{source}: {path} must be an object")
    return value


def _require_number(value: Any, path: str, source: Path) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummarySchemaError(f"{source}: {path} must be numeric")
    return value


def _require_count(value: Any, path: str, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SummarySchemaError(f"{source}: {path} must be a non-negative integer")
    return value


def _legacy_schema_message(source: Path, version: Any) -> str:
    return (
        f"{source}: expected schema_version 2, found {version!r}. "
        "Legacy summaries cannot be mixed with schema-v2 diagnostics or treated as zero. "
        "Recompute the diagnostics from the existing baseline_dir/output_dir generation JSON "
        "files with evaluate_frank_subset_compare.py --compare_only."
    )


def load_row(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SummarySchemaError(f"could not read {path}: {exc}") from exc

    summary = _require_mapping(summary, "summary", path)
    version = summary.get("schema_version")
    if version != 2:
        raise SummarySchemaError(_legacy_schema_message(path, version))

    transitions = _require_mapping(
        summary.get("lexical_screen_transition_counts"),
        "lexical_screen_transition_counts",
        path,
    )
    alignments = _require_mapping(
        summary.get("annotation_alignment_counts"),
        "annotation_alignment_counts",
        path,
    )

    row: Dict[str, Any] = {
        "name": path.parent.name,
        "schema_version": version,
        "n_samples": _require_count(summary.get("n_samples"), "n_samples", path),
        "mean_diff_rougeL": _require_number(
            summary.get("mean_diff_rougeL"), "mean_diff_rougeL", path
        ),
        "mean_lexical_absence_delta": _require_number(
            summary.get("mean_lexical_absence_delta"),
            "mean_lexical_absence_delta",
            path,
        ),
    }
    for key in TRANSITIONS:
        row[f"lexical_screen_{key}"] = _require_count(
            transitions.get(key), f"lexical_screen_transition_counts.{key}", path
        )
    for system in SYSTEMS:
        system_counts = _require_mapping(
            alignments.get(system), f"annotation_alignment_counts.{system}", path
        )
        for key in ALIGNMENTS:
            row[f"{system}_annotation_{key}"] = _require_count(
                system_counts.get(key),
                f"annotation_alignment_counts.{system}.{key}",
                path,
            )
    return row


def collect_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    return [load_row(path) for path in paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine schema-v2 FRANK subset diagnostics across fold directories."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing *_fold*/subset_comparison_summary.json files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.root.glob(f"*_fold*/{SUMMARY_FILENAME}"))
    if not paths:
        print(
            f"error: no *_fold*/{SUMMARY_FILENAME} files found under {args.root}",
            file=sys.stderr,
        )
        return 2
    try:
        rows = collect_rows(paths)
    except SummarySchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_path = args.root / "all_folds_subset_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
