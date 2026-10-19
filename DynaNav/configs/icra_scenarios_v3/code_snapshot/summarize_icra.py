"""Export strict, manifest-scoped custom navigation results, including partial coverage.

Only explicitly named manifest result files can contribute. Old runs and pilots
are never discovered or included automatically. Missing trials are not failures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile

import yaml

EVAL = Path(__file__).resolve().parent
MODELS = ("vega76000", "omnivla120000", "navila", "ticvla")
SCENARIOS = ("scenario1", "scenario2", "scenario3")
VERSION = "physx_capsules_fov_v1"
EPISODE_FIELDS = [
    "model", "scenario", "run_id", "episode", "success", "collision",
    "human_contact", "obstacle_contact", "initial_goal_distance_m",
    "navigation_error_m", "goal_progress", "fov_clearance_m",
    "clearance_eligible", "benchmark_duration_s", "physical_duration_s", "physical_duration_measured", "duration_frames",
    "measurement_frames", "fov_clearance_frames", "path_length_m",
    "source", "source_sha256",
]
ISSUE_FIELDS = ["model", "scenario", "run_id", "episode", "reason", "source"]


def finite_number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid_{name}")
    if minimum is not None and value < minimum:
        raise ValueError(f"invalid_{name}")
    return float(value)


def integer(value, name, minimum=0):
    x = finite_number(value, name, minimum)
    if x != int(x):
        raise ValueError(f"invalid_{name}")
    return int(x)


def boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"invalid_{name}")
    return value


def mean(values):
    return sum(values) / len(values) if values else None


def digest(data):
    return hashlib.sha256(data).hexdigest()


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else base / path


def episode_row(data, expected, group, source, source_bytes):
    if not isinstance(data, dict) or data.get("error"):
        raise ValueError("episode_error")
    if data.get("episode") != expected["name"]:
        raise ValueError("episode_id_mismatch")
    if data.get("measurement_version") != VERSION:
        raise ValueError("measurement_version_mismatch")
    frames = integer(data.get("duration_frames"), "duration_frames", 1)
    measured = integer(data.get("measurement_frames"), "measurement_frames", 1)
    if measured != frames:
        raise ValueError("measurement_frame_coverage_mismatch")
    fov_frames = integer(data.get("fov_clearance_frames"), "fov_clearance_frames")
    if fov_frames > measured:
        raise ValueError("invalid_fov_frame_coverage")
    duration = finite_number(data.get("duration"), "duration", 0)
    if not math.isclose(duration, frames / 30.0, abs_tol=1e-6):
        raise ValueError("benchmark_clock_mismatch")
    physical_duration = data.get("physical_duration_s")
    if physical_duration is not None:
        physical_duration = finite_number(physical_duration, "physical_duration_s", 0)
        timing = data.get("timing_telemetry", {})
        if timing.get("version") != "physics_clock_v1":
            raise ValueError("unverified_physical_clock")
        measured_seconds = finite_number(timing.get("physics_duration_s"), "physics_duration_s", 0)
        if not math.isclose(physical_duration, measured_seconds, abs_tol=1e-6):
            raise ValueError("physical_clock_duration_mismatch")
        physics_steps = integer(timing.get("physics_steps"), "physics_steps")
        if physics_steps:
            dt_min = finite_number(timing.get("physics_dt_min_s"), "physics_dt_min_s", 0)
            dt_max = finite_number(timing.get("physics_dt_max_s"), "physics_dt_max_s", 0)
            if dt_min <= 0 or dt_max < dt_min or not dt_min * physics_steps - 1e-5 <= physical_duration <= dt_max * physics_steps + 1e-5:
                raise ValueError("physical_clock_step_mismatch")
        elif physical_duration != 0:
            raise ValueError("physical_clock_step_mismatch")
    for key in ("scene", "instruction", "robot_type", "timeout"):
        if data.get(key) != expected["config"][key]:
            raise ValueError(f"config_{key}_mismatch")
    for key in ("start", "goal"):
        actual = data.get(key)
        wanted = expected["config"][key]
        if not isinstance(actual, list) or len(actual) != len(wanted):
            raise ValueError(f"config_{key}_mismatch")
        if any(not math.isclose(finite_number(a, key), b, abs_tol=1e-6) for a, b in zip(actual, wanted)):
            raise ValueError(f"config_{key}_mismatch")
    initial = math.dist(data["start"][:2], data["goal"][:2])
    if initial <= 0:
        raise ValueError("nonpositive_initial_goal_distance")
    error = finite_number(data.get("navigation_error"), "navigation_error", 0)
    success = boolean(data.get("success"), "success")
    human = boolean(data.get("contact_human_collision"), "contact_human_collision")
    obstacle = boolean(data.get("contact_obstacle_collision"), "contact_obstacle_collision")
    collision = boolean(data.get("contact_collision"), "contact_collision")
    if collision != (human or obstacle):
        raise ValueError("inconsistent_contact_flags")
    body = data.get("human_collision_body", {})
    if body.get("shape") != "kinematic_capsule" or body.get("radius_m") != 0.25 or body.get("height_m") != 1.8:
        raise ValueError("human_collision_geometry_mismatch")
    if "minimum_fov_person_surface_clearance_m" not in data:
        raise ValueError("missing_fov_clearance")
    clearance = data["minimum_fov_person_surface_clearance_m"]
    if clearance is not None:
        clearance = finite_number(clearance, "fov_clearance")
    if (clearance is None) != (fov_frames == 0):
        raise ValueError("inconsistent_fov_clearance_coverage")
    path_length = data.get("path_length")
    if path_length is not None:
        path_length = finite_number(path_length, "path_length", 0)
    return {
        "model": group["model"], "scenario": group["scenario"], "run_id": group["run_id"],
        "episode": expected["name"], "success": success, "collision": collision,
        "human_contact": human, "obstacle_contact": obstacle,
        "initial_goal_distance_m": initial, "navigation_error_m": error,
        "goal_progress": (initial - error) / initial,
        "fov_clearance_m": clearance,
        "clearance_eligible": not collision and clearance is not None and fov_frames > 0,
        "benchmark_duration_s": duration, "physical_duration_s": physical_duration,
        "physical_duration_measured": physical_duration is not None, "duration_frames": frames,
        "measurement_frames": measured, "fov_clearance_frames": fov_frames,
        "path_length_m": path_length, "source": str(source), "source_sha256": digest(source_bytes),
    }


def aggregate(rows):
    eligible = [r["fov_clearance_m"] for r in rows if r["clearance_eligible"]]
    return {
        "success_count": sum(r["success"] for r in rows),
        "collision_count": sum(r["collision"] for r in rows),
        "success_rate": mean([r["success"] for r in rows]),
        "collision_rate": mean([r["collision"] for r in rows]),
        "human_contact_rate": mean([r["human_contact"] for r in rows]),
        "obstacle_contact_rate": mean([r["obstacle_contact"] for r in rows]),
        "normalized_goal_progress": mean([r["goal_progress"] for r in rows]),
        "mean_minimum_fov_person_surface_clearance_m": mean(eligible),
        "clearance_eligible_episodes": len(eligible),
        "mean_benchmark_duration_s": mean([r["benchmark_duration_s"] for r in rows]),
        "mean_physical_duration_s": mean([r["physical_duration_s"] for r in rows if r["physical_duration_measured"]]),
        "physical_duration_measured_episodes": sum(r["physical_duration_measured"] for r in rows),
        "mean_navigation_error_m": mean([r["navigation_error_m"] for r in rows]),
    }


def summarize(manifest_path, results_root):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    groups = manifest.get("groups", [])
    expected_pairs = {(m, s) for m in MODELS for s in SCENARIOS}
    pairs = [(g.get("model"), g.get("scenario")) for g in groups]
    if len(groups) != 12 or len(set(pairs)) != 12 or set(pairs) != expected_pairs:
        raise ValueError("Manifest must explicitly cover four models by three scenarios exactly once")
    repeats = manifest.get("trials_per_group", 20)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Invalid trials_per_group")
    if manifest.get("requested") != 12 * repeats or manifest.get("measurement_version") != VERSION:
        raise ValueError("Manifest requested total or measurement version mismatch")
    run_ids = [g.get("run_id") for g in groups]
    if len(set(run_ids)) != 12:
        raise ValueError("Manifest run IDs must be unique")
    rows, summaries, invalid, missing, coverage = [], [], [], [], []
    for group in groups:
        names = group.get("episodes", [])
        if len(names) != repeats or len(set(names)) != repeats or group.get("requested") != repeats:
            raise ValueError(f"Expected exactly {repeats} unique episode IDs in {group['run_id']}")
        if any(Path(n).name != n for n in names):
            raise ValueError("Episode IDs must be filenames without directory components")
        run_id = group["run_id"]
        if not isinstance(run_id, str) or Path(run_id).name != run_id:
            raise ValueError("Run IDs must be single directory names")
        config_path = resolve(group["config"], manifest_path.parent)
        config_error = None
        try:
            config_bytes = config_path.read_bytes()
            if digest(config_bytes) != group["config_sha256"]:
                raise ValueError("config_sha256_mismatch")
            config = yaml.safe_load(config_bytes)
            if set(names) != {k for k in config if k.startswith("episode_")}:
                raise ValueError("config_episode_coverage_mismatch")
            if any(config[n].get("physical_human_metrics") is not True for n in names):
                raise ValueError("physical_human_metrics_disabled")
        except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
            config_error = str(exc)
            config = {}
        validation_file = resolve(group["validation_file"], manifest_path.parent)
        validated = False
        try:
            validation = json.loads(validation_file.read_text())
            validated = validation.get("validated") is True and validation.get("config_sha256") == group["config_sha256"]
        except (OSError, ValueError):
            pass
        group_rows, group_invalid, group_missing = [], [], []
        # Explicit manifest names only: no discovery of old results or pilot folders.
        for name in names:
            source = results_root / run_id / f"{name}.json"
            info = {"model": group["model"], "scenario": group["scenario"], "run_id": run_id, "episode": name, "source": str(source)}
            if not source.exists():
                group_missing.append(dict(info, reason="result_missing"))
                continue
            try:
                if config_error:
                    raise ValueError(config_error)
                source_bytes = source.read_bytes()
                data = json.loads(source_bytes)
                row = episode_row(data, {"name": name, "config": config[name]}, group, source, source_bytes)
                group_rows.append(row)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                reason = "invalid_json_or_write_in_progress" if isinstance(exc, json.JSONDecodeError) else str(exc)
                group_invalid.append(dict(info, reason=reason))
        rows.extend(group_rows)
        invalid.extend(group_invalid)
        missing.extend(group_missing)
        complete = len(group_rows) == repeats and not group_missing and not group_invalid and not config_error and validated
        summary = {
            "model": group["model"], "scenario": group["scenario"], "run_id": run_id,
            "completed": len(group_rows), "requested": repeats, "missing": len(group_missing),
            "invalid": len(group_invalid), "scenario_validated": validated,
            "config_matches_manifest": config_error is None, "complete": complete,
            **aggregate(group_rows), "source_directory": str(results_root / run_id),
        }
        summaries.append(summary)
        coverage.append({
            **summary, "config": str(config_path), "config_error": config_error,
            "validation_file": str(validation_file),
            "missing_episodes": [r["episode"] for r in group_missing],
            "invalid_reason_counts": dict(Counter(r["reason"] for r in group_invalid)),
        })
    by_model = []
    for model in MODELS:
        selected = [r for r in rows if r["model"] == model]
        selected_groups = [g for g in summaries if g["model"] == model]
        by_model.append({
            "model": model, "completed": len(selected), "requested": 3 * repeats,
            "missing": sum(g["missing"] for g in selected_groups),
            "invalid": sum(g["invalid"] for g in selected_groups),
            "complete": all(g["complete"] for g in selected_groups),
            **aggregate(selected),
        })
    status = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": manifest.get("protocol"), "measurement_version": VERSION,
        "manifest": str(manifest_path), "manifest_sha256": digest(manifest_bytes),
        "complete": all(g["complete"] for g in summaries),
        "completed": len(rows), "requested": 12 * repeats, "missing": len(missing), "invalid": len(invalid),
        "completed_groups": sum(g["complete"] for g in summaries), "requested_groups": 12,
        "invalid_reason_counts": dict(Counter(r["reason"] for r in invalid)),
        "denominator": "Each displayed rate uses valid completed trials only; requested and missing counts are separate. Partial rates are provisional.",
        "benchmark_time": "Benchmark duration is duration_frames/30. Separately measured PhysX durations are exposed only when physics_clock_v1 telemetry exists; older missing measurements are not imputed. Neither field is wall-clock runtime.",
    }
    return {"status": status, "metrics": summaries, "models": by_model, "episodes": rows, "invalid": invalid, "missing": missing, "coverage": coverage, "manifest": manifest}


def csv_text(rows, columns=None):
    output = io.StringIO()
    fields = columns or list(rows[0])
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def atomic_write(path, text):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".summary-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def export_report(report, destinations):
    readme = (
        "Custom Isaac Sim navigation trials; not official DynaNav or VEGA-Bench benchmark scores.\n"
        "Four models x three distinct Hospital locations; see manifest.json for the fixed per-group counts and total. Read status.json for current coverage; missing/invalid trials do not count as model failures.\n"
        "Only explicit result paths in manifest.json contribute. Old proximity runs and pilots are excluded.\n"
        "SR = successful valid episodes / valid episodes. CR = episodes with any actual PhysX robot contact / valid episodes. SR and CR are independent; successful episodes may collide. Human contact uses kinematic capsules radius0.25m,totalheight1.8m; walls/furniture contacts also count. Expected wheel-ground support and robot self-contact are excluded by instrumentation.\n"
        "Goal progress = (initial planar goal distance - final planar goal distance) / initial planar goal distance, without clipping; negative values indicate moving farther from the goal.\n"
        "Clearance = mean of each eligible episode's minimum planar robot-convex-footprint to human-capsule surface gap. Eligible episodes are contact-free, including walls/furniture, with at least one human in the cropped camera FOV. All configured people, including the stationary goal person, are included. The active task and target are specified in manifest.json. FOV tests capsule bounds against the centered camera crop, with no occlusion check. This is human clearance, not all-obstacle ESDF clearance. Signed planar hull gaps can be negative without actual3Dcontact; the convex footprint is conservative and must not be used to infer collision. No eligible observations => blank CSV /null JSON.\n"
        "Benchmark durations are frame counts divided by30. Physical durations are separate measurements from summed PhysX callback timesteps when physics_clock_v1 telemetry exists. Each aggregate includes its physical_duration_measured_episodes count; older episodes without telemetry have null values and are never imputed. Neither duration is wall-clock runtime. Models retain their native action interface/controller; shared limits are1m/s and0.3rad/s.\n"
        "Repeats use paired pedestrian timing IDs at fixed geometry, rather than independently sampled maps. Native crowd routes may react differently to each robot. Model summaries pool valid episodes; compare complete per-scenario groups to avoid partial-coverage bias.\n"
        "A group is complete only with every manifest-requested valid record, matching config SHA256, and matching scenario validation. Partial/inconsistent JSON writes are excluded and retried on the next refresh. No imputed scores.\n"
    )
    contents = {name + ".json": json.dumps(report[name], indent=2, allow_nan=False) + "\n" for name in report}
    for name in ("metrics", "models"):
        contents[name + ".csv"] = csv_text(report[name])
    contents["episodes.csv"] = csv_text(report["episodes"], EPISODE_FIELDS)
    contents["invalid.csv"] = csv_text(report["invalid"], ISSUE_FIELDS)
    contents["missing.csv"] = csv_text(report["missing"], ISSUE_FIELDS)
    contents["README.txt"] = readme
    vega_groups = [g for g in report["metrics"] if g["model"] == "vega76000"]
    contents["vega_first.csv"] = csv_text(vega_groups)
    contents["vega_first.json"] = json.dumps(vega_groups, indent=2, allow_nan=False) + "\n"
    def percent(value):
        return "N/A" if value is None else f"{100 * value:.1f}%"
    vega_done = sum(g["completed"] for g in vega_groups)
    vega_total = sum(g["requested"] for g in vega_groups)
    lines = [f"VEGA checkpoint 76000: {vega_done}/{vega_total} trials complete.",
             "", f"Updated: {report['status']['generated_utc']}", "",
             "| Scene | Valid trials | Success | Actual contact | Goal progress | FOV clearance |",
             "|---|---:|---:|---:|---:|---:|"]
    for g in vega_groups:
        clearance = g['mean_minimum_fov_person_surface_clearance_m']
        clearance_text = 'N/A' if clearance is None else f'{clearance:.3f} m'
        lines.append(f"| {g['scenario']} | {g['completed']}/{g['requested']} | {percent(g['success_rate'])} | {percent(g['collision_rate'])} | {percent(g['normalized_goal_progress'])} | {clearance_text} |")
    lines += ["", "Custom Isaac Sim scenarios, not official benchmark scores. Rates use valid completed trials only; partial counts are provisional. Success and collision are independent. Clearance is averaged only over contact-free episodes with a person in the camera FOV. N/A indicates no eligible observations.", ""]
    contents['VEGA_FIRST_REPORT.md'] = "\n".join(lines)
    # status.json is written last, serving as the refresh marker for readers.
    status = contents.pop("status.json")
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for name, text in contents.items():
            atomic_write(destination / name, text)
        atomic_write(destination / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=EVAL / "icra_suite_manifest.json")
    parser.add_argument("--results-root", type=Path, default=EVAL / "results")
    parser.add_argument("--output", type=Path, default=EVAL / "icra_report")
    parser.add_argument("--export", type=Path, default=Path("/nfshomes/gershom/vega_exports/icra_three_scenarios"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = summarize(args.manifest.resolve(), args.results_root.resolve())
        export_report(report, (args.output.resolve(), args.export.resolve()))
    print(json.dumps(report["status"], allow_nan=False))
    return 2 if args.require_complete and not report["status"]["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
