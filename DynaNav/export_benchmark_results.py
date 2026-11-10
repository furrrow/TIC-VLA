#!/usr/bin/env python3
"""Export benchmark results to one CSV row per episode."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "experiment",
    "inference_method",
    "experiment_mode",
    "scenario",
    "config_file",
    "timestamp",
    "episode",
    "instruction",
    "start",
    "goal",
    "success",
    "manual_success",
    "collision",
    "human_collision",
    "human_collision_count",
    "collision_characters",
    "physical_collision",
    "physical_collision_count",
    "navigation_error",
    "closest_goal_distance_m",
    "min_human_distance_m",
    "spl",
    "path_length",
    "duration",
    "duration_frames",
    "termination_condition",
    "termination_reason",
    "failure_reason",
    "error",
    "isaac_sim_errors",
    "child_log",
    "result_file",
]

COLLISION_CHARACTER_PATTERN = re.compile(r"COLLISION WARNING: HUMAN[\s\S]*?character=([^\s|]+)")


def select_result_file(experiment_dir: Path) -> Path | None:
    """Prefer the incremental latest result, otherwise use the newest snapshot."""
    latest = sorted(experiment_dir.glob("*_results_latest.json"))
    if latest:
        return latest[-1]
    snapshots = list(experiment_dir.glob("*_results_*.json"))
    return max(snapshots, key=lambda path: path.stat().st_mtime) if snapshots else None


def csv_value(value: Any) -> Any:
    """Keep scalar values readable and serialize structured values consistently."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return "" if value is None else value


def collision_characters(experiment_dir: Path, episode: dict[str, Any]) -> str:
    """Return unique collided character paths found in the per-episode Isaac log."""
    candidates = []
    child_log = episode.get("child_log")
    if child_log:
        candidates.append(Path(child_log))
    episode_name = episode.get("episode")
    if episode_name:
        candidates.append(experiment_dir / "logs" / f"{episode_name}_isaac.log")

    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        characters = list(dict.fromkeys(COLLISION_CHARACTER_PATTERN.findall(text)))
        return ";".join(characters)
    return ""


def result_rows(experiment_dir: Path, result_file: Path) -> list[dict[str, Any]]:
    with result_file.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("top-level JSON value must be an object")
    episodes = result.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("top-level 'episodes' value must be a list")

    config_file = result.get("config_file") or ""
    rows = []
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        row = {field: csv_value(episode.get(field)) for field in CSV_FIELDS}
        row.update({
            "experiment": experiment_dir.name,
            "inference_method": episode.get(
                "inference_method", result.get("inference_method", "unspecified")
            ),
            "experiment_mode": episode.get(
                "experiment_mode", result.get("experiment_mode", "unspecified")
            ),
            "scenario": Path(config_file).stem if config_file else "unspecified",
            "config_file": config_file,
            "timestamp": result.get("timestamp", ""),
            "manual_success": "",
            "collision_characters": collision_characters(experiment_dir, episode),
            "result_file": str(result_file),
        })
        rows.append(row)
    return rows


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
        description="Search benchmark experiment directories and export one row per episode."
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
            rows.extend(result_rows(experiment_dir, result_file))
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
        print(f"Wrote {len(rows)} episode results to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
