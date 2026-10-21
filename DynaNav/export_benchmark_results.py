#!/usr/bin/env python3
"""Export aggregate benchmark results to one CSV row per experiment."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = [
    "experiment",
    "inference_method",
    "experiment_mode",
    "scenario",
    "config_file",
    "timestamp",
]
SUMMARY_FIELDS = [
    "total_episodes",
    "successful_episodes",
    "failed_episodes",
    "success_rate_percent",
    "total_human_collisions",
]
METRIC_FIELDS = [
    "navigation_error_m",
    "closest_goal_distance_m",
    "total_collision_rate_percent",
    "human_collision_rate_percent",
    "min_human_distance_m",
    "physical_collision_rate_percent",
    "spl",
    "path_length_m",
    "duration_s",
    "duration_frames",
]
CSV_FIELDS = IDENTITY_FIELDS + SUMMARY_FIELDS + METRIC_FIELDS + ["result_file"]


def select_result_file(experiment_dir: Path) -> Path | None:
    """Prefer the incremental latest result, otherwise use the newest snapshot."""
    latest = sorted(experiment_dir.glob("*_results_latest.json"))
    if latest:
        return latest[-1]
    snapshots = list(experiment_dir.glob("*_results_*.json"))
    return max(snapshots, key=lambda path: path.stat().st_mtime) if snapshots else None


def result_row(experiment_dir: Path, result_file: Path) -> dict[str, Any]:
    with result_file.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("top-level JSON value must be an object")

    config_file = result.get("config_file") or ""
    metrics = result.get("median_metrics") or {}
    row: dict[str, Any] = {
        "experiment": experiment_dir.name,
        "inference_method": result.get("inference_method", "unspecified"),
        "experiment_mode": result.get("experiment_mode", "unspecified"),
        "scenario": Path(config_file).stem if config_file else "unspecified",
        "config_file": config_file,
        "timestamp": result.get("timestamp", ""),
        "result_file": str(result_file),
    }
    row.update({field: result.get(field, "") for field in SUMMARY_FIELDS})
    row.update({field: metrics.get(field, "") for field in METRIC_FIELDS})
    return row


def find_experiment_dirs(root: Path, patterns: list[str]) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"results directory does not exist: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search benchmark experiment directories and export aggregate metrics."
    )
    parser.add_argument(
        "patterns",
        nargs="+",
        help="Shell-style experiment name patterns, for example 'hospital_*'.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_results",
        help="Benchmark results directory (default: DynaNav/benchmark_results).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="CSV output path (default: stdout).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_dirs = find_experiment_dirs(args.results_dir, args.patterns)
    if not experiment_dirs:
        print(f"No experiment directories matched: {', '.join(args.patterns)}", file=sys.stderr)
        return 1

    rows = []
    for experiment_dir in experiment_dirs:
        result_file = select_result_file(experiment_dir)
        if result_file is None:
            print(f"Skipping {experiment_dir.name}: no result JSON found", file=sys.stderr)
            continue
        try:
            rows.append(result_row(experiment_dir, result_file))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"Skipping {result_file}: {error}", file=sys.stderr)

    if not rows:
        print("No readable result JSON files found.", file=sys.stderr)
        return 1

    output_stream = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output_stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            output_stream.close()

    if args.output:
        print(f"Wrote {len(rows)} experiments to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
