"""Run complete episodes in one Isaac Sim app, with a fresh USD stage per episode.

Launch this through isaac_python.sh, not the policy Python environment. This
retains the Kit process and its caches, but deliberately closes the old stage:
robot/crowd physics, behavior state and measurement subscriptions cannot leak
from one trial into the next. The existing worker performs the usual setup,
200-frame warmup, policy calls, controller updates and measurements unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT.parent / "TIC-VLA" / "DynaNav"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--episodes", nargs="*", help="Optional subset of YAML episode names")
    parser.add_argument("--review-first-only", action="store_true")
    parser.add_argument("--episode-wall-timeout", type=float, default=1800)
    parser.add_argument("--max-episodes", type=int, help="Stop cleanly after this many new episodes")
    parser.add_argument("--cpu-threads", type=int, help="Kit CPU thread limit; otherwise use benchmark default")
    parser.add_argument("--disable-viewport-updates", action="store_true",
                        help="Skip the unused headless UI viewport; policy render products remain active")
    parser.add_argument("--debug-print", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List pending trials without importing Isaac")
    return parser.parse_args()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def valid_result(data):
    if data.get("error") or data.get("duration_frames", 0) <= 0:
        return False
    if data.get("measurement_version") == "physx_capsules_fov_v1":
        return data.get("measurement_frames") == data["duration_frames"]
    return True


@contextlib.contextmanager
def episode_log(path):
    """Capture Python and native Kit output; restore the batch log afterwards."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        with path.open("w") as stream:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def archive_evidence(output, run_id, name):
    source = BENCHMARK / "logs" / run_id / "carter_ticvla_data"
    if not source.exists():
        return
    evidence = output / f"{name}_evidence"
    evidence.mkdir(exist_ok=True)
    for state in source.glob("*.json"):
        shutil.copy2(state, evidence / state.name)
    pictures = sorted(source.glob("tp_frame_*.jpg"))
    for picture in pictures[::max(1, len(pictures) // 4)][:4]:
        shutil.copy2(picture, evidence / picture.name)


def main():
    args = parse_args()
    if args.disable_viewport_updates:
        raise ValueError(
            "Disabling the default viewport prevents hospital ASSETS_LOADED from completing. "
            "Omit --disable-viewport-updates; third-person capture may still be disabled."
        )
    config = Path(args.config)
    if not config.is_absolute():
        config = BENCHMARK / "configs" / config
    config = config.resolve()
    entries = yaml.safe_load(config.read_text())
    episode_names = [name for name, entry in entries.items() if isinstance(entry, dict)]
    if args.episodes is not None:
        unknown = set(args.episodes).difference(episode_names)
        if unknown:
            raise ValueError(f"Unknown episodes: {sorted(unknown)}")
        episode_names = [name for name in episode_names if name in args.episodes]
    output = ROOT / "results" / args.run_id
    pending = []
    for name in episode_names:
        result_path = output / f"{name}.json"
        try:
            complete = result_path.exists() and valid_result(json.loads(result_path.read_text()))
        except (ValueError, OSError):
            complete = False
        if complete:
            print("Already complete", name, flush=True)
        else:
            pending.append(name)
    if args.max_episodes is not None:
        pending = pending[:args.max_episodes]
    print(json.dumps({"run_id": args.run_id, "pending": pending,
                      "reset_mode": "fresh_stage_persistent_app"}), flush=True)
    if args.dry_run or not pending:
        return 0

    output.mkdir(parents=True, exist_ok=True)
    os.environ["BENCHMARK_RUN_ID"] = args.run_id
    os.environ["NAVIGATION_METHOD"] = "ticvla"
    if not os.environ.get("VEGA_TRACE_ROOT"):
        os.environ["VEGA_TRACE_ROOT"] = str(output / "predictions")
    os.chdir(BENCHMARK)
    sys.path.insert(0, str(BENCHMARK))
    from benchmark import APP_CONFIG, BenchmarkRunner
    from isaacsim import SimulationApp

    provenance = {"runner": "persistent_suite.py", "config": str(config),
                  "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                  "benchmark_sha256": hashlib.sha256((BENCHMARK / "benchmark.py").read_bytes()).hexdigest(),
                  "reset_mode": "new_runner_new_stage_same_simulation_app",
                  "front_camera_and_policy_frequency": "unchanged",
                  "cpu_threads": args.cpu_threads,
                  "disable_viewport_updates": args.disable_viewport_updates,
                  "controller_target": os.environ.get("VEGA_CONTROLLER_TARGET"),
                  "v_max": os.environ.get("VEGA_V_MAX"),
                  "w_max": os.environ.get("VEGA_W_MAX"),
                  "disable_third_person": os.environ.get("VEGA_DISABLE_THIRD_PERSON"),
                  "pid": os.getpid(), "start_unix": time.time()}
    atomic_json(output / "persistent_worker.json", provenance)
    app_start = time.monotonic()
    launch_config = dict(APP_CONFIG)
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive")
        launch_config["limit_cpu_threads"] = args.cpu_threads
    if args.disable_viewport_updates:
        launch_config["disable_viewport_updates"] = True
    sim_app = SimulationApp(launch_config=launch_config)
    app_startup_seconds = time.monotonic() - app_start
    from omni.kit.async_engine import run_coroutine
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep

    class TimedRunner(BenchmarkRunner):
        async def _set_carter_robot(self, prim_path):
            # The core physics manager is process-global. On a subsequent stage,
            # its old tensor view can survive USD close/open. Articulation's
            # constructor eagerly initializes whenever that view is non-null,
            # before the newly spawned robot has entered the new PhysX scene.
            # Restore the same pre-play state as a fresh process using Isaac's
            # own STOP handler; the next PLAY creates views for this stage.
            from isaacsim.core.simulation_manager import SimulationManager as CoreSimulationManager
            if omni.timeline.get_timeline_interface().is_playing():
                raise RuntimeError("Expected a stopped timeline before attaching robot behavior")
            had_view = CoreSimulationManager.get_physics_sim_view() is not None
            CoreSimulationManager._on_stop(None)
            if CoreSimulationManager.get_physics_sim_view() is not None:
                raise RuntimeError("Previous-stage physics tensor view was not cleared")
            print("PERSISTENT_PHYSICS_RESET", json.dumps({"stale_view_cleared": had_view}), flush=True)
            return await super()._set_carter_robot(prim_path)

        async def _setup_episode_simulation(self, episode_config):
            start = time.monotonic()
            try:
                return await super()._setup_episode_simulation(episode_config)
            finally:
                self._persistent_setup_seconds = time.monotonic() - start

        async def _run_navigation(self, timeout_frames):
            start = time.monotonic()
            try:
                return await super()._run_navigation(timeout_frames)
            finally:
                self._persistent_navigation_seconds = time.monotonic() - start

    def pump(coroutine):
        task = run_coroutine(coroutine)
        deadline = time.monotonic() + args.episode_wall_timeout
        while not task.done() and not sim_app.is_exiting():
            if time.monotonic() >= deadline:
                task.cancel()
                raise TimeoutError("Persistent episode exceeded wall timeout")
            sim_app.update()
        if not task.done():
            raise RuntimeError("Isaac Sim exited before task completion")
        return task.result()

    async def close_stage(runner=None):
        if runner is not None:
            metrics = getattr(runner, "_contact_fov_metrics", None)
            if metrics is not None:
                metrics.close()
                runner._contact_fov_metrics = None
            runner._cleanup_contact_sensor()
            runner._setup_sim_sub = None
        rep.orchestrator.set_capture_on_play(False)
        omni.timeline.get_timeline_interface().stop()
        await sim_app.app.next_update_async()
        context = omni.usd.get_context()
        if context.get_stage() is not None:
            success, error = await context.close_stage_async()
            if not success:
                raise RuntimeError(f"Cannot close previous episode stage: {error}")
        if context.get_stage() is not None:
            raise RuntimeError("Previous episode USD stage remains open")
        if runner is not None:
            runner._sim_manager = None
        await sim_app.app.next_update_async()

    completed_now = []
    try:
        pump(close_stage())
        for index, name in enumerate(pending):
            if args.review_first_only and completed_now:
                os.environ.pop("VEGA_REVIEW_ROOT", None)
            # Fresh stage guarantees fresh scene, bodies, behavior and human commands.
            # The remote policy's per-episode reset still uses the frame reset.
            os.environ["BENCHMARK_SIMULATION_FRAME_COUNT"] = "0"
            runner = TimedRunner(sim_app, str(config), None, args.debug_print)
            result_path, log_path = output / f"{name}.json", output / f"{name}.log"
            print("START", name, time.ctime(), flush=True)
            start = time.monotonic()
            with episode_log(log_path):
                try:
                    result = pump(runner.run_single_episode_worker(name))
                    result["execution_timing"] = {
                        "app_startup_s": app_startup_seconds if index == 0 else 0.0,
                        "setup_s": getattr(runner, "_persistent_setup_seconds", None),
                        "navigation_s": getattr(runner, "_persistent_navigation_seconds", None),
                        "episode_wall_s": time.monotonic() - start,
                        "runner": "persistent_app_fresh_stage"}
                    if not valid_result(result):
                        raise RuntimeError("Invalid episode or incomplete contact/FOV coverage")
                    # Archive before on_play() clears the next episode's temporary logs.
                    archive_evidence(output, args.run_id, name)
                    pump(close_stage(runner))
                except BaseException as error:
                    traceback.print_exc()
                    atomic_json(result_path, {"episode": name,
                                "error": f"persistent_worker:{type(error).__name__}:{error}",
                                "duration_frames": 0})
                    raise
            log_text = log_path.read_text()
            if "VEGA_INFERENCE_OK" not in log_text or "VEGA_INFERENCE_ERROR" in log_text:
                atomic_json(result_path, {"episode": name, "duration_frames": 0,
                            "error": f"persistent_worker:invalid_inference_evidence:{log_path}"})
                print("INVALID_EXECUTION", name, str(log_path), flush=True)
                raise RuntimeError(f"Invalid inference evidence: {log_path}")
            prediction_dir = Path(os.environ["VEGA_TRACE_ROOT"]) / name
            prediction_files = sorted(prediction_dir.glob("prediction_*.json"))
            reset_audit = {"maximum_first_frame": 10, "maximum_start_xy_error_m": 0.2}
            try:
                if not prediction_files:
                    raise ValueError("No prediction trace for episode reset validation")
                first_prediction = json.loads(prediction_files[0].read_text())
                first_frame = int(first_prediction["frame"])
                first_position = first_prediction["robot_pose"]["position"]
                configured_start = entries[name]["start"]
                start_error = math.hypot(float(first_position[0]) - float(configured_start[0]),
                                         float(first_position[1]) - float(configured_start[1]))
                reset_audit.update({"first_prediction_frame": first_frame,
                                    "start_xy_error_m": start_error,
                                    "first_prediction_file": str(prediction_files[0])})
                if not 0 <= first_frame <= 10 or not math.isfinite(start_error) or start_error > 0.2:
                    raise ValueError("First prediction is late or robot did not reset to configured start")
            except (KeyError, ValueError, TypeError, IndexError) as error:
                atomic_json(result_path, {"episode": name, "duration_frames": 0,
                            "error": f"persistent_worker:invalid_episode_reset:{error}",
                            "reset_validation": reset_audit})
                print("INVALID_EPISODE_RESET", name, json.dumps(reset_audit), flush=True)
                raise RuntimeError(f"Invalid reset evidence for {name}") from error
            result["reset_validation"] = reset_audit
            if os.environ.get("VEGA_RENDER_ON_SENSOR_ONLY", "0") == "1":
                from check_render_telemetry import summarize as summarize_render_telemetry
                telemetry_path = Path(os.environ["VEGA_TRACE_ROOT"]) / name / "front_render_telemetry.jsonl"
                audit = summarize_render_telemetry(telemetry_path) if telemetry_path.is_file() else {}
                if not audit.get("one_update_lookback_verified"):
                    atomic_json(result_path, {"episode": name, "duration_frames": 0,
                                "error": "persistent_worker:invalid_render_cadence_audit",
                                "render_cadence_audit": audit})
                    print("INVALID_RENDER_CADENCE", name, json.dumps(audit), flush=True)
                    raise RuntimeError(f"Unverified gated camera frames: {telemetry_path}")
                result["render_cadence_audit"] = audit
            atomic_json(result_path, result)
            completed_now.append(name)
            atomic_json(output / "persistent_progress.json", {
                "run_id": args.run_id, "completed_this_worker": completed_now,
                "remaining_this_worker": pending[index + 1:], "updated_unix": time.time()})
            print("RESULT", json.dumps(result), flush=True)
            del runner
            gc.collect()
        print("SUITE_COMPLETE", str(output), flush=True)
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
