"""Aggregate and present benchmark results without requiring Isaac Sim."""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from benchmark_runtime import SIMULATION_FPS


def binary_metric(result: dict, key: str, legacy_key: str | None = None) -> float:
    """Read a per-episode binary metric, accepting old *_rate keys."""
    value = result.get(key)
    if value is None and legacy_key is not None:
        value = result.get(legacy_key, 0.0)
    try:
        return 1.0 if float(value or 0.0) > 0.0 else 0.0
    except Exception:
        return 0.0


@dataclass
class EpisodeMetrics:
    """Measurements collected during one episode; defaults describe a failed run."""

    success: bool = False
    duration_frames: int = 0
    navigation_error: float = float("inf")
    spl: float = 0.0
    path_length: float = 0.0
    human_collision: bool = False
    human_collision_count: int = 0
    physical_collision: bool = False
    physical_collision_count: int = 0
    failure_reason: str | None = None


def build_episode_result(
    episode_name: str, episode_config: dict, metrics: EpisodeMetrics | None = None,
    *, error: str | None = None,
) -> dict:
    """Build the common child/legacy/error result schema."""
    metrics = metrics or EpisodeMetrics()
    result = {
        "episode": episode_name,
        "scene": episode_config.get("scene"),
        "start": episode_config.get("start"),
        "goal": episode_config.get("goal"),
        "instruction": episode_config.get("instruction", ""),
        "timeout": episode_config.get("timeout", 0),
        "robot_type": episode_config.get("robot_type", ""),
        "success": metrics.success,
        "duration": metrics.duration_frames / SIMULATION_FPS,
        "duration_frames": metrics.duration_frames,
        "navigation_error": metrics.navigation_error,
        "success_rate": 1.0 if metrics.success else 0.0,
        "collision": float(metrics.human_collision or metrics.physical_collision),
        "human_collision": float(metrics.human_collision),
        "human_collision_count": metrics.human_collision_count,
        "physical_collision": float(metrics.physical_collision),
        "spl": metrics.spl,
        "path_length": metrics.path_length,
        "physical_collision_detected": metrics.physical_collision,
        "physical_collision_count": metrics.physical_collision_count,
        "failure_reason": metrics.failure_reason,
    }
    if error is not None:
        result["error"] = error
    return result


def summarize_results(results: list[dict]) -> dict:
    """Return the aggregate fields shared by console, text, and JSON output."""
    total = len(results)
    successful = sum(1 for result in results if result.get("success"))
    navigation_errors = [result.get("navigation_error", float("inf")) for result in results]
    finite_errors = [value for value in navigation_errors if np.isfinite(value)]

    def average(values) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "total_episodes": total,
        "successful_episodes": successful,
        "failed_episodes": total - successful,
        "success_rate_percent": successful / total * 100 if total else 0.0,
        "total_human_collisions": sum(int(r.get("human_collision_count", 0) or 0) for r in results),
        "average_metrics": {
            "navigation_error_m": average(finite_errors),
            "total_collision_rate_percent": average([binary_metric(r, "collision", "collision_rate") for r in results]) * 100,
            "human_collision_rate_percent": average([binary_metric(r, "human_collision", "human_collision_rate") for r in results]) * 100,
            "physical_contact_rate_percent": average([binary_metric(r, "physical_collision", "physical_collision_rate") for r in results]) * 100,
            "spl": average([r.get("spl", 0.0) for r in results]),
            "path_length_m": average([r.get("path_length", 0.0) for r in results]),
            "duration_s": average([r.get("duration", 0.0) for r in results]),
            "duration_frames": average([r.get("duration_frames", 0.0) for r in results]),
        },
    }


def _aggregate_lines(summary: dict) -> list[str]:
    metrics = summary["average_metrics"]
    return [
        f"Total Episodes: {summary['total_episodes']}",
        f"Successful: {summary['successful_episodes']}",
        f"Failed: {summary['failed_episodes']}",
        f"Success Rate (SR): {summary['success_rate_percent']:.1f}%",
        f"Average Navigation Error (NE): {metrics['navigation_error_m']:.2f}m",
        f"Total Collision Rate (human OR physical): {metrics['total_collision_rate_percent']:.1f}%",
        f"Human Collision Rate: {metrics['human_collision_rate_percent']:.1f}%",
        f"Total Human Collisions: {summary['total_human_collisions']}",
        f"Physical Contact Rate: {metrics['physical_contact_rate_percent']:.1f}%",
        f"Average Success weighted by Path Length: {metrics['spl']:.3f}",
        f"Average Path Length: {metrics['path_length_m']:.2f}m",
        f"Average Simulation Duration: {metrics['duration_s']:.2f}s ({metrics['duration_frames']:.0f} frames)",
    ]


def _episode_lines(result: dict) -> list[str]:
    status = "✓ SUCCESS" if result.get("success") else "✗ FAILED"
    collision = "⚠ COLLISION" if binary_metric(result, "collision", "collision_rate") > 0 else "✓ NO COLLISION"
    physical = "⚠ PHYS" if result.get("physical_collision_detected", False) else ""
    duration = result.get("duration", 0.0)
    frames = result.get("duration_frames", 0)
    scene = str(result.get("scene", "")).split("/")[-1] if result.get("scene") else "N/A"
    lines = [
        f"{result.get('episode', 'UNKNOWN')}: {status} | {collision} {physical} ({duration:.1f}s, {int(frames)} frames)",
        f"  Scene: {scene}",
        f"  Robot: {result.get('robot_type', 'N/A')}",
        f"  Instruction: {result.get('instruction', '')}",
    ]
    if not result.get("success"):
        failure_reason = result.get("failure_reason") or result.get("error") or "Goal not reached"
        lines.append(f"  Failure Reason: {failure_reason}")
    lines.extend([
        f"  Human Collisions: {int(result.get('human_collision_count', 0) or 0)}",
        f"  Physical Collision: {'Yes' if binary_metric(result, 'physical_collision', 'physical_collision_rate') > 0 else 'No'}",
        f"  Navigation Error: {result.get('navigation_error', 0.0):.2f}m",
        f"  Success Rate: {result.get('success_rate', 0.0):.3f}",
        f"  Success weighted by Path Length: {result.get('spl', 0.0):.3f}",
        f"  Path Length: {result.get('path_length', 0.0):.2f}m",
    ])
    return lines


def print_final_results(results: list[dict]) -> None:
    """Print the aggregate and individual episode results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for line in _aggregate_lines(summarize_results(results)):
        print(line)
    print()
    for result in results:
        for line in _episode_lines(result):
            print(line)
        print()


def save_results_to_file(
    results: list[dict], config_file_path: str | None = None,
    incremental: bool = False, run_id: str = "default",
) -> str | None:
    """Save JSON and text results; return the text file path on success."""
    if not results:
        return None

    results_dir = Path(f"./benchmark_results/{run_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = "latest" if incremental else time.strftime("%Y%m%d_%H%M%S")
    config_name = Path(config_file_path).stem if config_file_path else "benchmark"
    summary = {
        "timestamp": timestamp,
        "config_file": config_file_path if config_file_path else None,
        **summarize_results(results),
        "episodes": results,
    }

    json_filename = results_dir / f"{config_name}_results_{timestamp}.json"
    try:
        with open(json_filename, "w") as output:
            json.dump(summary, output, indent=2)
        print(f"Results saved to JSON: {json_filename}")
        sys.stdout.flush()
    except Exception as error:
        print(f"Failed to save JSON file: {error}", file=sys.stderr)
        sys.stderr.flush()

    text_filename = results_dir / f"{config_name}_results_{timestamp}.txt"
    try:
        with open(text_filename, "w") as output:
            output.write("=" * 60 + "\nBENCHMARK RESULTS SUMMARY\n" + "=" * 60 + "\n\n")
            output.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if config_file_path:
                output.write(f"Config File: {config_file_path}\n")
            output.write("\nAGGREGATE METRICS\n" + "-" * 60 + "\n")
            output.write("\n".join(_aggregate_lines(summary)) + "\n\n")
            output.write("DETAILED EPISODE RESULTS\n" + "-" * 60 + "\n\n")
            for result in results:
                output.write("\n".join(_episode_lines(result)) + "\n\n")
        print(f"Results saved to text: {text_filename}")
        sys.stdout.flush()
        return str(text_filename)
    except Exception as error:
        print(f"Failed to save text file: {error}", file=sys.stderr)
        sys.stderr.flush()
        return None
