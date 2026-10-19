"""
Benchmark script for testing robot navigation performance in different scenarios.

Usage (parent orchestrator):
    python benchmark.py -c benchmark_config.yaml

Parent will spawn a fresh child process per episode. Each child:
- Creates and closes its own SimulationApp safely
- Runs exactly one episode
- Writes back a JSON result

You can also run a single episode directly (child mode):
    python benchmark.py -c benchmark_config.yaml --child --episode_name EP1 --result_json /tmp/EP1.json
"""

import os
import sys
import json
import yaml
import argparse
import asyncio
import numpy as np
import time
import tempfile
import subprocess
from pathlib import Path
from generate_commands import *

# APP_CONFIG = {"renderer": "RayTracedLighting", "headless": True, "width": 640, "height": 480}
APP_CONFIG = {"renderer": "RayTracedLighting", "headless": True, "width": 640, "height": 480}

# Prefer writing outputs relative to this script (not cwd), since Isaac Sim subprocesses
# may be launched from different working directories.
SCRIPT_DIR = Path(__file__).resolve().parent

# Timeout adjustment: Add 50 seconds to all episode timeouts
TIMEOUT_EXTENSION_SECONDS = 0


def _binary_metric(result: dict, key: str, legacy_key: str | None = None) -> float:
    """Read a per-episode binary metric, accepting old *_rate keys for compatibility."""
    value = result.get(key)
    if value is None and legacy_key is not None:
        value = result.get(legacy_key, 0.0)
    try:
        return 1.0 if float(value or 0.0) > 0.0 else 0.0
    except Exception:
        return 0.0


def _get_results_dir(run_id: str) -> Path:
    return SCRIPT_DIR / "benchmark_results" / str(run_id)


def _benchmark_tmp_dir(config_file_path: str, run_id: str) -> Path:
    """Same directory for generated commands and YAML paths (independent of process cwd)."""
    return Path(config_file_path).resolve().parent / "tmp" / str(run_id)


def _load_latest_results_if_present(results_dir: Path, config_name: str) -> list:
    """Load existing incremental results file if present, otherwise return empty list."""
    latest_path = results_dir / f"{config_name}_results_latest.json"
    if not latest_path.is_file():
        return []
    try:
        with open(latest_path, "r") as f:
            data = json.load(f)
        episodes = data.get("episodes", [])
        return episodes if isinstance(episodes, list) else []
    except Exception as e:
        print(f"[Parent] Warning: Failed to load resume file '{latest_path}': {e}")
        return []



def _expand_env_value(value):
    """Expand environment variables in benchmark YAML values."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_env_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_value(item) for key, item in value.items()}
    return value

class BenchmarkRunner:
    def __init__(self, sim_app, config_file_path, crash_report_path, debug_print):
        self._sim_app = sim_app
        self._sim_app.set_setting("/app/scripting/ignoreWarningDialog", True)
        # Inputs
        self.config_file_path = config_file_path
        self.crash_report_path = crash_report_path
        self.debug_print = debug_print

        # Benchmark results
        self.results = []
        self.current_episode = None
        self.episode_start_time = None
        self.episode_success = False

        # Simulation components
        self._sim_manager = None
        self._setup_sim_sub = None
        self._setup_sim_succeed = False
        self._settings = None
        self._config = None
        self._benchmark_config = None

        # Robot tracking
        self._spawned_robot_prim_path = None
        self._robot_type = None  # Track robot type for position updates
        self._robot_position = None
        self._goal_position = None

        # Metrics tracking
        self._robot_path = []  # Track robot path for SPL calculation
        self._collision_detected = False
        # NOTE: we do not store collided object lists in results JSON (keep list only internally).
        self._episode_start_position = None

        # Simulation timing
        self._simulation_frame_count = 0
        self._episode_start_frame = 0
        self._navigation_timing = None
        self._physics_timing_sub = None
        
        # Physical collision (contact sensor)
        self._physx_contact_sub = None
        self._physical_collision_detected = False
        self._robot_contact_paths = set()       # prim paths we consider "robot bodies" for contact filtering
        self._contact_sensor_enabled = False
        self._contact_threshold = 10.0  # Minimum contact force to register as collision (N)
        self._contact_duration_threshold_frames = 100  # Require sustained contact before counting collision
        self._contact_consecutive_frames = 0
        self._max_contact_consecutive_frames = 0
        self._contact_sensor_prim_path = None  # Path to the contact sensor prim
        self._contact_sensor_interface = None  # Contact sensor interface
        self._contact_sensors = []  # List of contact sensor instances


    async def run_benchmark(self):
        """
        Legacy single-process run of all episodes (kept for completeness).
        In the new design, the parent process orchestrates episodes via subprocesses.
        """
        print("Starting benchmark run...")

        # Load benchmark configuration
        self._load_benchmark_config()

        # Enable extensions and setup
        self._enable_extensions()
        await self._sim_app.app.next_update_async()

        self._set_simulation_settings()
        await self._sim_app.app.next_update_async()

        # Initialize simulation manager
        from isaacsim.replicator.agent.core.simulation import SimulationManager
        self._sim_manager = SimulationManager()

        # Run each episode (note: this mode does NOT close SimulationApp between episodes)
        for episode_name, episode_config in self._benchmark_config.items():
            # Skip metadata keys that aren't episode configs (should be dicts)
            if episode_name == 'seed':
                continue
            if episode_name == 'navigation_method':
                continue
            if not isinstance(episode_config, dict):
                # Skip entries that aren't dictionaries
                continue

            print(f"\n{'=' * 50}")
            print(f"Running Episode: {episode_name}")
            print(f"{'=' * 50}")

            success = await self._run_episode(episode_name, episode_config)

            # Calculate metrics
            navigation_error = self._get_distance_to_goal()
            success_rate = 1.0 if success else 0.0
            # Use physical collision if detected, otherwise fall back to distance-based
            any_collision = self._physical_collision_detected or self._collision_detected
            collision = 1.0 if any_collision else 0.0
            human_collision = 1.0 if self._collision_detected else 0.0
            physical_collision = 1.0 if self._physical_collision_detected else 0.0
            spl = self._calculate_spl(success)

            # Calculate simulation duration in frames
            simulation_duration_frames = self._simulation_frame_count - self._episode_start_frame
            simulation_duration_seconds = simulation_duration_frames / 30.0  # Convert frames to seconds at 30 FPS

            # Record results
            result = {
                'episode': episode_name,
                'scene': episode_config['scene'],
                'start': episode_config['start'],
                'goal': episode_config['goal'],
                'instruction': episode_config['instruction'],
                'timeout': episode_config['timeout'],
                'robot_type': episode_config['robot_type'],
                'success': success,
                'duration': simulation_duration_seconds,
                'duration_frames': simulation_duration_frames,
                'navigation_error': navigation_error,
                'success_rate': success_rate,
                'collision': collision,
                'human_collision': human_collision,
                'physical_collision': physical_collision,
                'spl': spl,
                'path_length': self._calculate_path_length(),
                'physical_collision_detected': self._physical_collision_detected,
                'physical_collision_count': self._max_contact_consecutive_frames,
            }
            result.update(self._navigation_timing_result())
            self.results.append(result)

            if getattr(self, '_contact_fov_metrics', None):
                measured = self._contact_fov_metrics.result()
                result.update(measured)
                result['collision'] = float(measured['contact_collision'])
                result['human_collision'] = float(measured['contact_human_collision'])
                result['physical_collision'] = float(measured['contact_collision'])
                result['physical_collision_detected'] = measured['contact_collision']
                result['physical_collision_count'] = measured['contact_pairs']
                self._contact_fov_metrics.close()
            # Clean up contact sensor
            self._cleanup_contact_sensor()

            print(f"Episode {episode_name} completed: {'SUCCESS' if success else 'FAILED'}")
            print(f"Simulation Duration: {result['duration']:.2f}s ({result['duration_frames']} frames)")
            print(f"Navigation Error: {result['navigation_error']:.2f}m")
            print(f"Success weighted by Path Length: {result['spl']:.3f}")
            print(f"Path Length: {result['path_length']:.2f}m")
            # No per-contact physics logging in benchmark results.

        # Print final results
        self._print_final_results()
        return True

    async def run_single_episode_worker(self, episode_name: str):
        """
        Child-process entry: run exactly one episode under a fresh SimulationApp,
        then return a single result dict. The child process will close SimulationApp.
        """
        try:
            # Load and prep config
            self._load_benchmark_config()
            if episode_name not in self._benchmark_config:
                raise KeyError(f"Episode '{episode_name}' not found in config.")
        except Exception as e:
            print(f"[run_single_episode_worker] Error loading config for episode '{episode_name}': {e}")
            import traceback
            traceback.print_exc()
            raise

        episode_config = self._benchmark_config[episode_name]

        # Reset per-episode trackers
        self.results = []
        self.current_episode = None
        self.episode_start_time = None
        self.episode_success = False
        self._spawned_robot_prim_path = None
        self._robot_type = None  # Track robot type for position updates
        self._robot_position = None
        self._goal_position = None
        self._robot_path = []
        self._collision_detected = False
        self._episode_start_position = None
        self._simulation_frame_count = 0
        self._episode_start_frame = 0
        self._physical_collision_detected = False
        self._contact_consecutive_frames = 0
        self._max_contact_consecutive_frames = 0
        
        # Set goal threshold from episode config (fallback to global config, default to 1.5m)
        global_threshold = self._benchmark_config.get('success_threshold', 1.5)
        self._goal_threshold = episode_config.get('success_threshold', global_threshold)
        
        # Propagate episode parameters to behavior scripts via environment variables
        instruction = str(episode_config.get('instruction', ''))
        os.environ['TICVLA_INSTRUCTION'] = instruction
        os.environ['TICVLA_EPISODE_NAME'] = episode_name

        goal = episode_config.get('goal', None)
        if goal is not None and len(goal) >= 2:
            os.environ['TICVLA_GOAL_X'] = str(float(goal[0]))
            os.environ['TICVLA_GOAL_Y'] = str(float(goal[1]))
            os.environ['TICVLA_GOAL_Z'] = str(float(goal[2] if len(goal) > 2 else 0.0))

        # Enable + settings
        self._enable_extensions()
        await self._sim_app.app.next_update_async()
        self._set_simulation_settings()
        await self._sim_app.app.next_update_async()

        from isaacsim.replicator.agent.core.simulation import SimulationManager
        self._sim_manager = SimulationManager()
        
        # Run the episode
        try:
            success = await self._run_episode(episode_name, episode_config)
        except Exception as e:
            print(f"[run_single_episode_worker] Error running episode '{episode_name}': {e}")
            import traceback
            traceback.print_exc()
            raise

        # Metrics
        try:
            navigation_error = self._get_distance_to_goal()
            success_rate = 1.0 if success else 0.0
            # Use physical collision if detected, otherwise fall back to distance-based
            any_collision = self._physical_collision_detected or self._collision_detected
            collision = 1.0 if any_collision else 0.0
            human_collision = 1.0 if self._collision_detected else 0.0
            physical_collision = 1.0 if self._physical_collision_detected else 0.0
            spl = self._calculate_spl(success)
            simulation_duration_frames = self._simulation_frame_count - self._episode_start_frame
            simulation_duration_seconds = simulation_duration_frames / 30.0

            result = {
                'episode': episode_name,
                'scene': episode_config['scene'],
                'start': episode_config['start'],
                'goal': episode_config['goal'],
                'instruction': episode_config['instruction'],
                'timeout': episode_config['timeout'],
                'robot_type': episode_config['robot_type'],
                'success': success,
                'duration': simulation_duration_seconds,
                'duration_frames': simulation_duration_frames,
                'navigation_error': navigation_error,
                'success_rate': success_rate,
                'collision': collision,
                'human_collision': human_collision,
                'physical_collision': physical_collision,
                'spl': spl,
                'path_length': self._calculate_path_length(),
                'physical_collision_detected': self._physical_collision_detected,
                'physical_collision_count': self._max_contact_consecutive_frames,
            }

            result.update(self._navigation_timing_result())

            if getattr(self, '_contact_fov_metrics', None):
                measured = self._contact_fov_metrics.result()
                result.update(measured)
                result['collision'] = float(measured['contact_collision'])
                result['human_collision'] = float(measured['contact_human_collision'])
                result['physical_collision'] = float(measured['contact_collision'])
                result['physical_collision_detected'] = measured['contact_collision']
                result['physical_collision_count'] = measured['contact_pairs']
                self._contact_fov_metrics.close()
            # Clean up contact sensor
            self._cleanup_contact_sensor()

            # Cleanup temp config file if it exists
            if hasattr(self, '_temp_config_path') and os.path.exists(self._temp_config_path):
                try:
                    os.remove(self._temp_config_path)
                except Exception as cleanup_error:
                    print(f"Warning: Failed to cleanup temp config: {cleanup_error}")

            return result
        except Exception as e:
            print(f"[run_single_episode_worker] Error calculating metrics for episode '{episode_name}': {e}")
            import traceback
            traceback.print_exc()
            # Cleanup temp config file even on error
            if hasattr(self, '_temp_config_path') and os.path.exists(self._temp_config_path):
                try:
                    os.remove(self._temp_config_path)
                except Exception:
                    pass
            raise

    def _load_benchmark_config(self):
        """Load the benchmark configuration from YAML file."""
        with open(self.config_file_path, "r") as f:
            self._benchmark_config = _expand_env_value(yaml.safe_load(f))
        print(f"Loaded benchmark config with {len(self._benchmark_config) - 1 if 'seed' in self._benchmark_config else len(self._benchmark_config)} episodes")

    async def _run_episode(self, episode_name, episode_config):
        """Run a single benchmark episode."""
        self.current_episode = episode_config
        self.episode_name = episode_name
        self.episode_start_time = time.time()
        self.episode_success = False

        # Reset metrics tracking
        self._robot_path = []
        self._collision_detected = False
        self._episode_start_position = None
        self._simulation_frame_count = 0
        self._episode_start_frame = 0

        # Reset physical collision tracking
        self._physical_collision_detected = False
        self._contact_consecutive_frames = 0
        self._max_contact_consecutive_frames = 0

        # Reset error counters
        self._position_update_error_count = 0

        # Set up simulation for this episode
        await self._setup_episode_simulation(episode_config)
        print(f"Setup episode simulation succeed")

        # Spawn robot at start position
        robot_type = episode_config['robot_type'].replace('_', '_').title()
        if robot_type == 'Nova_Carter':
            robot_type = 'Nova_Carter'
        elif robot_type == 'Spot':
            robot_type = 'Spot'

        # Store robot type for position tracking
        self._robot_type = robot_type

        # Get start_yaw from config (default to 0 if not specified)
        start_yaw = episode_config.get('start_yaw', 0.0)
        self._spawned_robot_prim_path = await self._spawn_robot(robot_type, episode_config['start'], start_yaw)

        # Set up robot
        if robot_type == "Spot":
            await self._set_quadruped_robot(self._spawned_robot_prim_path)
        elif robot_type == "Nova_Carter":
            await self._set_carter_robot(self._spawned_robot_prim_path)
        print(f"Set up robot succeed")

        # Set up contact sensor for collision detection
        await self._setup_contact_sensor(self._spawned_robot_prim_path)
        print(f"Set up contact sensor succeed")
        self._contact_fov_metrics = None
        if episode_config.get('physical_human_metrics', False):
            from contact_fov_metrics import ContactFOVMetrics
            import omni.usd
            self._contact_fov_metrics = ContactFOVMetrics(omni.usd.get_context().get_stage(), self._spawned_robot_prim_path, expected_humans=episode_config["num_people"])


        # Set goal position
        self._goal_position = np.array(episode_config['goal'])

        # Set goal threshold from episode config (fallback to global config, default to 1.5m)
        global_threshold = self._benchmark_config.get('success_threshold', 1.5)
        self._goal_threshold = episode_config.get('success_threshold', global_threshold)

        # Record start position for path tracking
        self._episode_start_position = np.array(episode_config['start'])

        # Run navigation with timeout (in simulation frames)
        # Add TIMEOUT_EXTENSION_SECONDS to all timeouts
        timeout_seconds = episode_config['timeout'] + TIMEOUT_EXTENSION_SECONDS
        timeout_frames = int(timeout_seconds * 30)  # Convert seconds to frames (30 FPS, must be int)
        success = await self._run_navigation(timeout_frames)

        return success

    async def _setup_episode_simulation(self, episode_config):
        """Set up simulation for a specific episode."""
        # Generate commands for this episode
        await self._gen_random_commands()
        print(f"Generate commands succeed")

        # Create a temporary config for this episode
        temp_config = self._create_episode_config(episode_config)
        # Load the temporary config
        can_load_config = self._sim_manager.load_config_file(temp_config)
        if not can_load_config:
            raise Exception("Failed to load episode configuration")

        # Optional custom stationary-human pilot using native character loading.
        human = episode_config.get("stationary_human")
        humans = episode_config.get("scripted_humans") or episode_config.get("stationary_humans") or ([human] if human else [])
        if humans:
            if episode_config['num_people'] != len(humans):
                raise ValueError("stationary_humans must match num_people")
            from pxr import Gf
            import carb
            tmp_root = _benchmark_tmp_dir(self.config_file_path, os.environ.get('BENCHMARK_RUN_ID', 'default'))
            commands = []
            for i, spec in enumerate(humans):
                name = 'Character' if i == 0 else f'Character_{i:02d}'
                route = spec.get('walk_route', [])
                if route:
                    commands.append(f"{name} Idle {float(spec.get('initial_idle', 3))}\n")
                    for _ in range(20):
                        for point in route:
                            if len(point) != 3:
                                raise ValueError('walk_route points must have x,y,z')
                            commands.append(f"{name} GoTo {point[0]} {point[1]} {point[2]} _\n")
                else:
                    commands.append(f"{name} Idle 10000\n")
            (tmp_root / 'benchmark_character_commands.txt').write_text(''.join(commands))
            randomizer = self._sim_manager._character_randomizer
            original_position = randomizer.get_random_position
            original_spawn = self._sim_manager.spawn_character_by_idx
            randomizer.get_random_position = lambda idx, spawn_area=None: carb.Float3(*humans[idx]['position'])
            self._sim_manager.spawn_character_by_idx = lambda location, rotation, idx: original_spawn(location, humans[idx].get('yaw_degrees', 0), idx)
            try:
                await self._setup_sim()
            finally:
                randomizer.get_random_position = original_position
                self._sim_manager.spawn_character_by_idx = original_spawn
            print('VEGA_STATIONARY_HUMANS', json.dumps(humans), flush=True)
        else:
            await self._setup_sim()
        print(f"Setup sim succeed: {self._setup_sim_succeed}")

        if episode_config.get('target_object'):
            from scenario_target import add_target_object
            add_target_object(episode_config['target_object'])

        # Warmup simulation
        await self._warmup(frames=200)
        print(f"Warmup complete")

    def _create_episode_config(self, episode_config):
        """Create a temporary config file for this episode."""
        scene_url = episode_config['scene']

        # Get run_id from environment for unique temp paths (multi-instance support)
        run_id = os.environ.get('BENCHMARK_RUN_ID', 'default')
        tmp_root = _benchmark_tmp_dir(self.config_file_path, run_id)
        tmp_dir = str(tmp_root)

        config_template = {
            'isaacsim.replicator.agent': {
                'version': '0.7.19',
                'global': {
                    'seed': self._benchmark_config.get('seed', 42) if self._benchmark_config else 42,
                    'simulation_length': int((episode_config['timeout'] + TIMEOUT_EXTENSION_SECONDS) * 30)  # Convert seconds to frames (30 FPS), with timeout extension (must be int)
                },
                'scene': {
                    'asset_path': scene_url
                },
                'sensor': {
                    'camera_num': 0
                },
                'character': {
                    'command_file': str(tmp_root / "benchmark_character_commands.txt"),
                    'num': episode_config['num_people']
                },
                'robot': {
                    'command_file': str(tmp_root / "benchmark_robot_commands.txt"),
                    'nova_carter_num': 0,
                    'iw_hub_num': 0,
                    'write_data': False
                },
                'replicator': {
                    'writer': 'IRABasicWriter',
                    'parameters': {
                        'output_dir': str(tmp_root / "benchmark_output" / self.episode_name),
                        'object_info_bounding_box_2d_tight': False,
                        'object_info_bounding_box_2d_loose': False,
                        'object_info_bounding_box_3d': False,
                        'agent_info_skeleton_data': False,
                        'semantic_filter_predicate': 'class:character|robot;id:*',
                        'image_output_format': 'jpg',
                        'rgb': True,
                        'camera_params': True
                    }
                }
            }
        }

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        temp_config_path = f'{tmp_dir}/benchmark_config_{self.episode_name}.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config_template, f)

        # Store path for cleanup later
        self._temp_config_path = temp_config_path

        return temp_config_path

    def _start_navigation_timing(self, timeline):
        """Observe both clocks without changing the existing frame-based protocol."""
        self._navigation_timing = {
            'version': 'physics_clock_v1',
            'benchmark_duration_basis': 'duration_frames / 30.0',
            'window': 'navigation entry to terminal condition before timeline stop; setup and warmup excluded',
            'timeline_start_time_s': float(timeline.get_current_time()),
            'timeline_end_time_s': None,
            'timeline_duration_s': None,
            'physics_duration_s': 0.0,
            'physics_steps': 0,
            'physics_dt_min_s': None,
            'physics_dt_max_s': None,
        }
        self._physics_timing_active = True

        def on_physics_step(dt):
            if not self._physics_timing_active:
                return
            timing = self._navigation_timing
            dt = float(dt)
            timing['physics_duration_s'] += dt
            timing['physics_steps'] += 1
            old_min, old_max = timing['physics_dt_min_s'], timing['physics_dt_max_s']
            timing['physics_dt_min_s'] = dt if old_min is None else min(old_min, dt)
            timing['physics_dt_max_s'] = dt if old_max is None else max(old_max, dt)

        try:
            import omni.physx
            self._physics_timing_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(on_physics_step)
        except Exception as exc:
            # Missing telemetry must not alter navigation or existing metrics.
            self._navigation_timing['physics_clock_error'] = repr(exc)

    def _finish_navigation_timing(self, timeline):
        if not getattr(self, '_physics_timing_active', False):
            return
        self._physics_timing_active = False
        timing = self._navigation_timing
        timing['timeline_end_time_s'] = float(timeline.get_current_time())
        timing['timeline_duration_s'] = timing['timeline_end_time_s'] - timing['timeline_start_time_s']
        self._physics_timing_sub = None
        print('NAVIGATION_TIMING', json.dumps(timing), flush=True)

    def _navigation_timing_result(self):
        if self._navigation_timing is None:
            return {}
        timing = dict(self._navigation_timing)
        physical_duration = timing['physics_duration_s'] if timing['physics_steps'] else None
        return {'physical_duration_s': physical_duration, 'timing_telemetry': timing}

    async def _run_navigation(self, timeout_frames):
        import omni.timeline
        import omni.replicator.core as rep
        import omni.kit.app

        app = omni.kit.app.get_app()
        timeline = omni.timeline.get_timeline_interface()

        original_timecode = timeline.get_time_codes_per_second()
        timeline.set_time_codes_per_second(30)
        timeline.commit_silently()
        await app.next_update_async()

        # make sure we're playing
        if not timeline.is_playing():
            timeline.play()

        # enable capture and run for N frames while we poll status
        rep.orchestrator.set_capture_on_play(True)
        await app.next_update_async()

        self._episode_start_frame = self._simulation_frame_count
        if getattr(self, '_contact_fov_metrics', None):
            self._contact_fov_metrics.active = True
        self._start_navigation_timing(timeline)

        try:
            for _ in range(timeout_frames):
                # per-frame: update pos, check goal/collisions, count frames
                await self._update_robot_position()
                if getattr(self, '_contact_fov_metrics', None):
                    self._contact_fov_metrics.measure(self._simulation_frame_count)
                self._simulation_frame_count += 1
                # Avoid printing every frame (can flood logs and slow down Isaac Sim).
                if self.debug_print and (self._simulation_frame_count % 30 == 0):
                    print(f"Simulation frame count: {self._simulation_frame_count}")
                # Update environment variable so behavior scripts can access the frame count
                os.environ['BENCHMARK_SIMULATION_FRAME_COUNT'] = str(self._simulation_frame_count)

                # reached goal?
                if self._is_goal_reached():
                    self._finish_navigation_timing(timeline)
                    rep.orchestrator.set_capture_on_play(False)
                    timeline.stop()
                    # restore timecode
                    timeline.set_time_codes_per_second(original_timecode)
                    timeline.commit_silently()
                    await app.next_update_async()
                    return True

                # advance one frame
                await app.next_update_async()

        except Exception as e:
            print(f"Warning: Error during navigation: {e}")
            if getattr(self, "_contact_fov_metrics", None):raise
        finally:
            # timeout or error - ensure proper cleanup
            self._finish_navigation_timing(timeline)
            rep.orchestrator.set_capture_on_play(False)
            timeline.stop()
            timeline.set_time_codes_per_second(original_timecode)
            timeline.commit_silently()
            await app.next_update_async()

        return False

    async def _update_robot_position(self):
        """Update the current robot position."""
        if not self._spawned_robot_prim_path:
            return

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(self._spawned_robot_prim_path)
        
        if not robot_prim or not robot_prim.IsValid():
            return

        # Try to get position from robot-specific body link, with fallbacks
        position_prim = None
        
        if self._robot_type == "Spot":
            # Spot has a 'body' child
            body_prim = robot_prim.GetChild('body')
            if body_prim and body_prim.IsValid():
                position_prim = body_prim
        elif self._robot_type == "Nova_Carter":
            # Nova Carter uses 'chassis_link' as the base link
            chassis_prim = robot_prim.GetChild('chassis_link')
            if chassis_prim and chassis_prim.IsValid():
                position_prim = chassis_prim
            else:
                # Fallback: try other common names
                for base_name in ['base_link', 'base', 'chassis', 'body']:
                    base_prim = robot_prim.GetChild(base_name)
                    if base_prim and base_prim.IsValid():
                        position_prim = base_prim
                        break
        
        # Fallback: use robot root if no body link found
        if position_prim is None:
            position_prim = robot_prim

        if position_prim and position_prim.IsValid():
            try:
                # Use world transform for robust position
                world_mtx = omni.usd.get_world_transform_matrix(position_prim)
                position = world_mtx.ExtractTranslation()
                self._robot_position = np.array([position[0], position[1], position[2]])

                # Record path for SPL calculation
                self._robot_path.append(self._robot_position.copy())

                # Check for collisions (distance-based)
                await self._check_collisions()
                
                # Read contact sensor for physical collision detection
                self._read_contact_sensor()
            except Exception as e:
                # Log error but don't crash - position will remain None
                if not hasattr(self, '_position_error_logged'):
                    print(f"Warning: Failed to update robot position: {e}")
                    self._position_error_logged = True


    def _is_goal_reached(self):
        """Check if robot has reached the goal."""
        if self._robot_position is None or self._goal_position is None:
            return False

        distance = self._get_distance_to_goal()
        return distance < self._goal_threshold

    def _get_distance_to_goal(self):
        """Get distance to goal."""
        if self._robot_position is None or self._goal_position is None:
            return float('inf')

        # Only consider x,y coordinates for 2D navigation
        robot_2d = self._robot_position[:2]
        goal_2d = self._goal_position[:2]
        return np.linalg.norm(robot_2d - goal_2d)

    async def _check_collisions(self):
        """Check for collisions with humans/characters."""
        import omni.usd
        from pxr import UsdGeom

        if getattr(self, '_contact_fov_metrics', None):
            return  # Physical contact reports replace center-distance classification in this mode.
        stage = omni.usd.get_context().get_stage()

        # Get only root character prims (avoid checking nested geometry)
        character_root_path = "/World/Characters"
        character_root_prim = stage.GetPrimAtPath(character_root_path)
        
        if not character_root_prim or not character_root_prim.IsValid():
            return
            
        # Check only direct children of Characters root (e.g., Character_38, Character_39, etc.)
        # Exclude specific paths that should not be checked for collisions
        excluded_paths = [
            "/World/Characters/Biped_Setup/biped_demo_meters"
        ]
        
        for character_prim in character_root_prim.GetChildren():
            if character_prim.IsValid() and character_prim.IsActive():
                # Skip excluded paths
                prim_path = character_prim.GetPath().pathString
                if prim_path in excluded_paths:
                    continue
                
                # Find the SkelRoot (the actual animated character part)
                skelroot_prim = self._find_character_skelroot(character_prim)
                if not skelroot_prim:
                    # Fallback to root character if no SkelRoot found
                    skelroot_prim = character_prim
                
                # Also check if the SkelRoot path is excluded
                skelroot_path = skelroot_prim.GetPath().pathString
                if skelroot_path in excluded_paths:
                    continue
                
                # Use world transform of the SkelRoot (actual moving part)
                world_mtx = omni.usd.get_world_transform_matrix(skelroot_prim)
                char_position = world_mtx.ExtractTranslation()
                char_pos = np.array([char_position[0], char_position[1], char_position[2]])

                # Check 2D distance (ignore Z for ground-based navigation)
                robot_2d = self._robot_position[:2]
                char_2d = char_pos[:2]
                distance = np.linalg.norm(robot_2d - char_2d)

                # Collision threshold
                collision_threshold = 0.2  # meters

                if distance < collision_threshold:
                    self._collision_detected = True
                    print(f"Collision detected with {skelroot_prim.GetPath().pathString} at distance {distance:.2f}m")

    def _find_character_skelroot(self, character_root_prim):
        """Find the SkelRoot prim of a character (the actual animated part)."""
        from pxr import Usd
        
        # Look for SkelRoot prims within the character hierarchy
        for prim in Usd.PrimRange(character_root_prim):
            if prim.GetTypeName() == "SkelRoot":
                return prim
        
        return None

    def _calculate_spl(self, success):
        """Calculate Success weighted by Path Length (SPL)."""
        if not success or len(self._robot_path) < 2:
            return 0.0 

        # Calculate actual path length
        path_length = 0.0
        for i in range(1, len(self._robot_path)):
            # Only consider 2D distance for ground navigation
            pos1 = self._robot_path[i - 1][:2]
            pos2 = self._robot_path[i][:2]
            path_length += np.linalg.norm(pos2 - pos1)

        # Calculate optimal path length (straight line from start to goal)
        if self._episode_start_position is not None and self._goal_position is not None:
            start_2d = self._episode_start_position[:2]
            goal_2d = self._goal_position[:2]
            optimal_length = np.linalg.norm(goal_2d - start_2d)

            if optimal_length > 0:
                spl = optimal_length / max(path_length, optimal_length)
                return float(min(spl, 1.0))  # Cap at 1.0

        return 0.0

    def _calculate_path_length(self):
        """Calculate total path length traveled."""
        if len(self._robot_path) < 2:
            return 0.0

        path_length = 0.0
        for i in range(1, len(self._robot_path)):
            # Only consider 2D distance for ground navigation
            pos1 = self._robot_path[i - 1][:2]
            pos2 = self._robot_path[i][:2]
            path_length += np.linalg.norm(pos2 - pos1)

        return float(path_length)

    def _print_final_results(self):
        """Print final benchmark results using self.results."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        total_episodes = len(self.results)
        successful_episodes = sum(1 for r in self.results if r.get('success'))
        success_rate = (successful_episodes / total_episodes) * 100 if total_episodes > 0 else 0

        # Calculate aggregate metrics (use safe access with defaults)
        navigation_errors = [r.get('navigation_error', float('inf')) for r in self.results]
        collisions = [_binary_metric(r, 'collision', 'collision_rate') for r in self.results]
        human_collisions = [_binary_metric(r, 'human_collision', 'human_collision_rate') for r in self.results]
        physical_collisions = [_binary_metric(r, 'physical_collision', 'physical_collision_rate') for r in self.results]
        spl_scores = [r.get('spl', 0.0) for r in self.results]
        path_lengths = [r.get('path_length', 0.0) for r in self.results]
        durations = [r.get('duration', 0.0) for r in self.results]
        duration_frames = [r.get('duration_frames', 0.0) for r in self.results]

        # Filter out inf navigation errors for averaging
        navigation_errors_finite = [ne for ne in navigation_errors if np.isfinite(ne)]

        avg_navigation_error = np.mean(navigation_errors_finite) if navigation_errors_finite else 0.0
        avg_collision_rate = np.mean(collisions) * 100 if collisions else 0.0
        avg_human_collision_rate = np.mean(human_collisions) * 100 if human_collisions else 0.0
        avg_physical_collision_rate = np.mean(physical_collisions) * 100 if physical_collisions else 0.0
        avg_spl = np.mean(spl_scores) if spl_scores else 0.0
        avg_path_length = np.mean(path_lengths) if path_lengths else 0.0
        avg_duration = np.mean(durations) if durations else 0.0
        avg_duration_frames = np.mean(duration_frames) if duration_frames else 0.0

        print(f"Total Episodes: {total_episodes}")
        print(f"Successful: {successful_episodes}")
        print(f"Failed: {total_episodes - successful_episodes}")
        print(f"Success Rate (SR): {success_rate:.1f}%")
        print(f"Average Navigation Error (NE): {avg_navigation_error:.2f}m")
        print(f"Total Collision Rate (human OR physical): {avg_collision_rate:.1f}%")
        print(f"Human Collision Rate: {avg_human_collision_rate:.1f}%")
        print(f"Physical Contact Rate: {avg_physical_collision_rate:.1f}%")
        print(f"Average Success weighted by Path Length: {avg_spl:.3f}")
        print(f"Average Path Length: {avg_path_length:.2f}m")
        print(f"Average Simulation Duration: {avg_duration:.2f}s ({avg_duration_frames:.0f} frames)")
        print()

        # Detailed results
        for result in self.results:
            status = "✓ SUCCESS" if result.get('success') else "✗ FAILED"
            collision_status = "⚠ COLLISION" if _binary_metric(result, 'collision', 'collision_rate') > 0 else "✓ NO COLLISION"
            phys_collision = "⚠ PHYS" if result.get('physical_collision_detected', False) else ""
            dur = result.get('duration', 0.0)
            frames = result.get('duration_frames', 0)
            print(f"{result.get('episode','UNKNOWN')}: {status} | {collision_status} {phys_collision} ({dur:.1f}s, {int(frames)} frames)")
            print(f"  Scene: {str(result.get('scene','')).split('/')[-1] if result.get('scene') else 'N/A'}")
            print(f"  Robot: {result.get('robot_type','N/A')}")
            print(f"  Instruction: {result.get('instruction','')}")
            print(f"  Navigation Error: {result.get('navigation_error', 0.0):.2f}m")
            print(f"  Success Rate: {result.get('success_rate', 0.0):.3f}")
            print(f"  Success weighted by Path Length: {result.get('spl', 0.0):.3f}")
            print(f"  Path Length: {result.get('path_length', 0.0):.2f}m")
            print()

    def _save_results_to_file(self, config_file_path=None, incremental=False):
        """Save benchmark results to JSON and text files.
        
        Args:
            config_file_path: Path to config file
            incremental: If True, use consistent filename for incremental updates. If False, use timestamp.
        """
        if not self.results:
            return None

        # Get run_id from environment for unique results paths (multi-instance support)
        run_id = os.environ.get('BENCHMARK_RUN_ID', 'default')
        
        # Create results directory if it doesn't exist (with run_id subdirectory)
        results_dir = Path(f"./benchmark_results/{run_id}")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for filename (only if not incremental)
        if incremental:
            timestamp = "latest"  # Use consistent filename for incremental updates
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Extract config name if available
        config_name = "benchmark"
        if config_file_path:
            config_name = Path(config_file_path).stem

        # Calculate aggregate metrics
        total_episodes = len(self.results)
        successful_episodes = sum(1 for r in self.results if r.get('success'))
        success_rate = (successful_episodes / total_episodes) * 100 if total_episodes > 0 else 0

        navigation_errors = [r.get('navigation_error', float('inf')) for r in self.results]
        collisions = [_binary_metric(r, 'collision', 'collision_rate') for r in self.results]
        human_collisions = [_binary_metric(r, 'human_collision', 'human_collision_rate') for r in self.results]
        physical_collisions = [_binary_metric(r, 'physical_collision', 'physical_collision_rate') for r in self.results]
        spl_scores = [r.get('spl', 0.0) for r in self.results]
        path_lengths = [r.get('path_length', 0.0) for r in self.results]
        durations = [r.get('duration', 0.0) for r in self.results]
        duration_frames = [r.get('duration_frames', 0.0) for r in self.results]

        navigation_errors_finite = [ne for ne in navigation_errors if np.isfinite(ne)]
        avg_navigation_error = np.mean(navigation_errors_finite) if navigation_errors_finite else 0.0
        avg_collision_rate = np.mean(collisions) * 100 if collisions else 0.0
        avg_human_collision_rate = np.mean(human_collisions) * 100 if human_collisions else 0.0
        avg_physical_collision_rate = np.mean(physical_collisions) * 100 if physical_collisions else 0.0
        avg_spl = np.mean(spl_scores) if spl_scores else 0.0
        avg_path_length = np.mean(path_lengths) if path_lengths else 0.0
        avg_duration = np.mean(durations) if durations else 0.0
        avg_duration_frames = np.mean(duration_frames) if duration_frames else 0.0

        # Prepare summary data
        summary = {
            "timestamp": timestamp,
            "config_file": config_file_path if config_file_path else None,
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "failed_episodes": total_episodes - successful_episodes,
            "success_rate_percent": success_rate,
            "average_metrics": {
                "navigation_error_m": avg_navigation_error,
                "total_collision_rate_percent": avg_collision_rate,
                "human_collision_rate_percent": avg_human_collision_rate,
                "physical_contact_rate_percent": avg_physical_collision_rate,
                "spl": avg_spl,
                "path_length_m": avg_path_length,
                "duration_s": avg_duration,
                "duration_frames": avg_duration_frames,
            },
            "episodes": self.results
        }

        # Save JSON file
        json_filename = results_dir / f"{config_name}_results_{timestamp}.json"
        try:
            with open(json_filename, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Results saved to JSON: {json_filename}")
            sys.stdout.flush()  # Flush to ensure message appears immediately
        except Exception as e:
            print(f"Failed to save JSON file: {e}", file=sys.stderr)
            sys.stderr.flush()

        # Save human-readable text file
        text_filename = results_dir / f"{config_name}_results_{timestamp}.txt"
        try:
            with open(text_filename, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("BENCHMARK RESULTS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                if config_file_path:
                    f.write(f"Config File: {config_file_path}\n")
                f.write("\n")
                f.write("AGGREGATE METRICS\n")
                f.write("-" * 60 + "\n")
                f.write(f"Total Episodes: {total_episodes}\n")
                f.write(f"Successful: {successful_episodes}\n")
                f.write(f"Failed: {total_episodes - successful_episodes}\n")
                f.write(f"Success Rate (SR): {success_rate:.1f}%\n")
                f.write(f"Average Navigation Error (NE): {avg_navigation_error:.2f}m\n")
                f.write(f"Total Collision Rate (human OR physical): {avg_collision_rate:.1f}%\n")
                f.write(f"Human Collision Rate: {avg_human_collision_rate:.1f}%\n")
                f.write(f"Physical Contact Rate: {avg_physical_collision_rate:.1f}%\n")
                f.write(f"Average Success weighted by Path Length: {avg_spl:.3f}\n")
                f.write(f"Average Path Length: {avg_path_length:.2f}m\n")
                f.write(f"Average Simulation Duration: {avg_duration:.2f}s ({avg_duration_frames:.0f} frames)\n")
                f.write("\n")
                f.write("DETAILED EPISODE RESULTS\n")
                f.write("-" * 60 + "\n\n")

                for result in self.results:
                    status = "✓ SUCCESS" if result.get('success') else "✗ FAILED"
                    collision_status = "⚠ COLLISION" if _binary_metric(result, 'collision', 'collision_rate') > 0 else "✓ NO COLLISION"
                    phys_collision = "⚠ PHYS" if result.get('physical_collision_detected', False) else ""
                    dur = result.get('duration', 0.0)
                    frames = result.get('duration_frames', 0)
                    f.write(f"{result.get('episode','UNKNOWN')}: {status} | {collision_status} {phys_collision} ({dur:.1f}s, {int(frames)} frames)\n")
                    f.write(f"  Scene: {str(result.get('scene','')).split('/')[-1] if result.get('scene') else 'N/A'}\n")
                    f.write(f"  Robot: {result.get('robot_type','N/A')}\n")
                    f.write(f"  Instruction: {result.get('instruction','')}\n")
                    f.write(f"  Navigation Error: {result.get('navigation_error', 0.0):.2f}m\n")
                    f.write(f"  Success Rate: {result.get('success_rate', 0.0):.3f}\n")
                    f.write(f"  Success weighted by Path Length: {result.get('spl', 0.0):.3f}\n")
                    f.write(f"  Path Length: {result.get('path_length', 0.0):.2f}m\n")
                    f.write("\n")
            print(f"Results saved to text: {text_filename}")
            sys.stdout.flush()  # Flush to ensure message appears immediately
            return str(text_filename)
        except Exception as e:
            print(f"Failed to save text file: {e}", file=sys.stderr)
            sys.stderr.flush()
            return None

    # Reuse methods from collect_data.py
    def _enable_extensions(self):
        import omni.kit.app
        ext_manager = omni.kit.app.get_app().get_extension_manager()

        extensions = [
            "omni.kit.viewport.window", "omni.kit.manipulator.prim", "omni.kit.property.usd",
            "omni.kit.scripting", "omni.anim.timeline", "omni.anim.graph.core",
            "omni.anim.retarget.core", "omni.anim.navigation.core",
            "omni.anim.people", "omni.isaac.sensor", "isaacsim.replicator.agent.core",
            "omni.kit.mesh.raycast", "omni.physx", "omni.physx.tensors"
        ]

        for ext in extensions:
            ext_manager.set_extension_enabled_immediate(ext, True)

    def _set_simulation_settings(self):
        import carb
        import omni.replicator.core as rep

        rep.settings.carb_settings("/omni/replicator/backend/writeThreads", 16)
        self._settings = carb.settings.get_settings()

        # Enable Python script execution automatically (suppresses warning dialog)
        # This is needed for USD files that contain Python behavior scripts (like outdoor scenes)
        self._settings.set("/persistent/scripting/enablePythonScripting", True)
        self._settings.set("/persistent/scripting/allowPythonScripting", True)

        # Navigation settings
        self._settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
        self._settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
        self._settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentHeight", 180)
        self._settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentRadius", 40)
        self._settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMaxStepHeight", 20)
        self._settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMaxFloorSlope", 45.0)

        # Debug settings
        self._settings.set("/log/level", "info")
        self._settings.set("/exts/isaacsim.replicator.agent/debug_print", self.debug_print)

        # Crash reporter
        self._settings.set("/crashreporter/enabled", True)
        if self.crash_report_path:
            self._settings.set("/crashreporter/dumpDir", self.crash_report_path)

    async def _setup_sim(self):
        def done_callback(_e):
            self._setup_sim_succeed = True
            self._setup_sim_sub = None

        self._setup_sim_sub = self._sim_manager.register_set_up_simulation_done_callback(done_callback)
        self._sim_manager.set_up_simulation_from_config_file()

        while self._setup_sim_sub and not self._sim_app.is_exiting():
            await self._sim_app.app.next_update_async()

    async def _warmup(self, frames=200):
        for _ in range(frames):
            await self._sim_app.app.next_update_async()

    async def _spawn_robot(self, robot_type: str, start_position, start_yaw: float = 0.0) -> str:
        """Spawn robot at specified position and orientation."""
        import omni.usd
        from pxr import Gf, UsdGeom
        from isaacsim.core.utils.stage import add_reference_to_stage

        stage = omni.usd.get_context().get_stage()
        robots_root = "/World/Robots"

        if robot_type == "Spot":
            asset_url = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/BostonDynamics/spot/spot.usd"
            name = "Spot"
        elif robot_type == "Nova_Carter":
            asset_url = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Carter/nova_carter_sensors.usd"
            name = "Nova_Carter"
        else:
            raise ValueError(f"Unknown robot type '{robot_type}'")

        prim_path = f"{robots_root}/{name}"
        add_reference_to_stage(usd_path=asset_url, prim_path=prim_path)
        await self._sim_app.app.next_update_async()

        # Set robot position and orientation
        robot_prim = stage.GetPrimAtPath(prim_path)
        x = float(start_position[0])
        y = float(start_position[1])
        z = float(start_position[2]) if len(start_position) > 2 else 0.1

        if robot_type == "Spot":
            z += 0.8  # Spot needs to be higher
        else:
            z += 0.1  # Nova Carter can be lower

        # Get Xformable interface for setting transforms
        xformable = UsdGeom.Xformable(robot_prim)
        
        # Set translation
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        
        if translate_op is None:
            translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        
        translate_op.Set(Gf.Vec3d(x, y, z))

        # Set rotation (yaw around Z-axis)
        rotate_z_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                rotate_z_op = op
                break
        
        if rotate_z_op is None:
            rotate_z_op = xformable.AddRotateZOp(UsdGeom.XformOp.PrecisionFloat)
        
        rotate_z_op.Set(float(start_yaw))

        await self._sim_app.app.next_update_async()
        print(f"Spawned {robot_type} at ({x}, {y}, {z}) with yaw {start_yaw}°")

        return prim_path

    async def _set_carter_robot(self, prim_path: str):
        """Set up Nova Carter robot."""
        import omni.usd
        import carb
        from pxr import Sdf

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(prim_path)

        if not robot_prim or not robot_prim.IsValid():
            carb.log_warn(f"Robot prim not found: {prim_path}")
            return

        # Determine which TIC-VLA behavior script to use
        # Priority: command-line/env var > YAML config > default
        nav_method = None
        # First check environment variable (set from command-line)
        if os.getenv('NAVIGATION_METHOD'):
            nav_method = os.getenv('NAVIGATION_METHOD').lower()
        # Fall back to YAML config if not set via command-line
        elif hasattr(self, '_benchmark_config') and self._benchmark_config:
            global_nav_method = self._benchmark_config.get('navigation_method', None)
            if global_nav_method:
                nav_method = global_nav_method.lower()
        # Default fallback
        if not nav_method:
            nav_method = 'ticvla'
        
        if nav_method != 'ticvla':
            raise ValueError(
                f"Unsupported navigation method in the public TIC-VLA release: {nav_method}. "
                "Use 'ticvla'."
            )
        script_path = os.path.abspath("./behavior/nova_carter_test_ticvla.py")
        script_name = "TIC-VLA"
        print("Using TIC-VLA behavior script")

        # Attach behavioral script
        import omni.kit.commands
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(robot_prim.GetPrimPath())])
        attr = robot_prim.GetAttribute("omni:scripting:scripts")
        attr.Set([Sdf.AssetPath(script_path)])
        print(f"Set up Nova Carter behavior script ({script_name})")

    async def _set_quadruped_robot(self, prim_path: str):
        """Set up Spot robot."""
        import omni.usd
        import carb
        from pxr import Sdf
        import omni.kit.commands

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(prim_path)

        if not robot_prim or not robot_prim.IsValid():
            carb.log_warn(f"Robot prim not found: {prim_path}")
            return

        # Determine which TIC-VLA behavior script to use
        # Priority: command-line/env var > YAML config > default
        nav_method = None
        # First check environment variable (set from command-line)
        if os.getenv('NAVIGATION_METHOD'):
            nav_method = os.getenv('NAVIGATION_METHOD').lower()
        # Fall back to YAML config if not set via command-line
        elif hasattr(self, '_benchmark_config') and self._benchmark_config:
            global_nav_method = self._benchmark_config.get('navigation_method', None)
            if global_nav_method:
                nav_method = global_nav_method.lower()
        # Default fallback
        if not nav_method:
            nav_method = 'ticvla'
        
        if nav_method != 'ticvla':
            raise ValueError(
                f"Unsupported navigation method in the public TIC-VLA release: {nav_method}. "
                "Use 'ticvla'."
            )
        script_path = os.path.abspath("./behavior/spot_test_ticvla.py")
        script_name = "TIC-VLA"
        print("Using TIC-VLA behavior script for Spot")

        # Attach behavioral script
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(robot_prim.GetPrimPath())])
        attr = robot_prim.GetAttribute("omni:scripting:scripts")
        attr.Set([Sdf.AssetPath(script_path)])
        print(f"Set up Spot behavior script ({script_name})")

    async def _setup_contact_sensor(self, robot_prim_path: str):
        """
        Set up contact sensor for the robot using Isaac Sim's ContactSensor API.
        Reference: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/sensors/isaacsim_sensors_physics_contact.html
        """
        import omni.usd
        import omni.kit.commands
        from pxr import Gf, Usd
        import numpy as np
        
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        
        if not robot_prim or not robot_prim.IsValid():
            print(f"Warning: Robot prim not found for contact sensor setup: {robot_prim_path}")
            return
        
        # Clear previous contact tracking
        self._robot_contact_paths.clear()
        self._physical_collision_detected = False
        self._contact_consecutive_frames = 0
        self._max_contact_consecutive_frames = 0
        self._contact_sensors = []  # Store contact sensor instances
        
        # Find the main body/chassis prim to attach contact sensor
        # For Nova Carter: chassis_link, For Spot: body
        body_prim_path = None
        if self._robot_type == "Nova_Carter":
            chassis_path = f"{robot_prim_path}/chassis_link"
            chassis_prim = stage.GetPrimAtPath(chassis_path)
            if chassis_prim and chassis_prim.IsValid():
                body_prim_path = chassis_path
        elif self._robot_type == "Spot":
            body_path = f"{robot_prim_path}/body"
            body_prim = stage.GetPrimAtPath(body_path)
            if body_prim and body_prim.IsValid():
                body_prim_path = body_path
        
        # Fallback to robot root
        if body_prim_path is None:
            body_prim_path = robot_prim_path
        
        # Collect robot paths for self-collision filtering
        for prim in Usd.PrimRange(robot_prim):
            self._robot_contact_paths.add(prim.GetPath().pathString)
        
        # Create contact sensor using Isaac Sim command
        # This is the recommended way per Isaac Sim documentation
        sensor_path = f"{body_prim_path}/Contact_Sensor"
        
        try:
            # Create contact sensor using IsaacSensorCreateContactSensor command
            success, sensor_prim = omni.kit.commands.execute(
                "IsaacSensorCreateContactSensor",
                path="Contact_Sensor",
                parent=body_prim_path,
                sensor_period=-1,  # Update every physics step
                min_threshold=self._contact_threshold,  # Minimum force threshold in Newtons
                max_threshold=10000000,  # Maximum force threshold
                translation=Gf.Vec3d(0, 0, 0),
                radius=-1,  # Use parent's collision geometry
            )
            
            if success:
                print(f"Contact sensor created at {sensor_path}")
                self._contact_sensor_prim_path = sensor_path
            else:
                print(f"Warning: Failed to create contact sensor at {sensor_path}")
                self._contact_sensor_prim_path = None
                
        except Exception as e:
            print(f"Warning: Could not create contact sensor: {e}")
            self._contact_sensor_prim_path = None
        
        # Acquire the contact sensor interface for reading data
        try:
            from isaacsim.sensors.physics import _sensor
            self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
            print("Contact sensor interface acquired")
        except Exception as e:
            print(f"Warning: Could not acquire contact sensor interface: {e}")
            self._contact_sensor_interface = None
        
        self._contact_sensor_enabled = True
        print(f"Contact sensor setup complete for robot at {robot_prim_path}")

    def _read_contact_sensor(self):
        """
        Read contact sensor data using Isaac Sim's contact sensor interface.
        This should be called every frame during navigation.
        """
        if not self._contact_sensor_enabled:
            return
        
        if self._contact_sensor_interface is None or not hasattr(self, '_contact_sensor_prim_path'):
            return
        
        if self._contact_sensor_prim_path is None:
            return
        
        try:
            # Get sensor reading using the interface
            # use_latest_data=True gets the reading from the current physics step
            reading = self._contact_sensor_interface.get_sensor_reading(
                self._contact_sensor_prim_path, 
                use_latest_data=True
            )

            if os.environ.get('VEGA_TRACE_ROOT'):
                trace_directory = Path(os.environ['VEGA_TRACE_ROOT']) / os.environ['TICVLA_EPISODE_NAME']
                trace_directory.mkdir(parents=True, exist_ok=True)
                with (trace_directory / 'contacts.jsonl').open('a') as stream:
                    stream.write(json.dumps({'frame': self._simulation_frame_count, 'valid': bool(reading.is_valid), 'in_contact': bool(reading.in_contact), 'force_n': float(reading.value)}) + '\n')
#             print(f"Contact sensor reading: {reading.value}, {reading.is_valid}, {reading.in_contact}")
            
            # Count only sustained, meaningful contact. A one-frame scrape or low-force
            # physics jitter should not mark the whole episode as physical collision.
            if reading.is_valid and reading.in_contact and float(reading.value) >= self._contact_threshold:
                self._contact_consecutive_frames += 1
                self._max_contact_consecutive_frames = max(
                    self._max_contact_consecutive_frames,
                    self._contact_consecutive_frames,
                )

                if (
                    not self._physical_collision_detected
                    and self._contact_consecutive_frames >= self._contact_duration_threshold_frames
                ):
                    self._physical_collision_detected = True
                    print(
                        "Contact sensor detected sustained collision "
                        f"({self._contact_consecutive_frames} frames >= "
                        f"{self._contact_duration_threshold_frames}, "
                        f"force >= {self._contact_threshold:.1f} N)"
                    )
            else:
                self._contact_consecutive_frames = 0
                            
        except Exception as e:
            if not hasattr(self, '_contact_read_error_logged'):
                print(f"Warning: Error reading contact sensor: {e}")
                self._contact_read_error_logged = True

    def _cleanup_contact_sensor(self):
        """Clean up contact sensor."""
        # Remove contact sensor prim if it exists
        if hasattr(self, '_contact_sensor_prim_path') and self._contact_sensor_prim_path:
            try:
                import omni.usd
                stage = omni.usd.get_context().get_stage()
                if stage:
                    sensor_prim = stage.GetPrimAtPath(self._contact_sensor_prim_path)
                    if sensor_prim and sensor_prim.IsValid():
                        stage.RemovePrim(self._contact_sensor_prim_path)
            except Exception:
                pass
            self._contact_sensor_prim_path = None
        
        self._contact_sensor_interface = None
        self._contact_sensor_enabled = False
        self._robot_contact_paths.clear()

    async def _gen_random_commands(self):
        """
        Generate random commands for characters and robots based on the current episode configuration.
        """
        print("Generating random commands for benchmark...")

        # Detect scene type from the episode scene URL
        scene_url = self.current_episode.get('scene', '')
        scene_type = self._detect_scene_type(scene_url)

        if not scene_type:
            print(f"Unknown scene type from URL: {scene_url}")
            print("Supported scene types: warehouse, hospital, office, outdoor")
            return

        print(f"Detected scene type: {scene_type}")

        # Get episode configuration with sensible defaults
        num_people = self.current_episode.get('num_people', 3)
        robot_type = self.current_episode.get('robot_type', 'nova_carter')
        seed = self._benchmark_config.get('seed', 42) if self._benchmark_config else 42

        print(f"Episode config - People: {num_people}, Robot: {robot_type}, Seed: {seed}")

        # Generate character commands
        await self._generate_character_commands(scene_type, num_people, seed)

        # Generate robot commands
        await self._generate_robot_commands(scene_type, robot_type, seed)

    def _detect_scene_type(self, scene_url: str) -> str:
        """
        Detect scene type from the scene URL.
        Returns 'warehouse', 'hospital', 'office', 'outdoor' or None.
        """
        if not scene_url:
            return None

        scene_url_lower = scene_url.lower()

        if 'warehouse' in scene_url_lower:
            return 'warehouse'
        elif 'hospital' in scene_url_lower:
            return 'hospital'
        elif 'office' in scene_url_lower:
            return 'office'
        elif 'outdoor' in scene_url_lower:
            return 'outdoor'
        else:
            return None

    async def _generate_character_commands(self, scene_type: str, num_people: int, seed: int):
        """Generate character commands for the given scene type."""
        # Get run_id from environment for unique temp paths (multi-instance support)
        run_id = os.environ.get('BENCHMARK_RUN_ID', 'default')
        tmp_root = _benchmark_tmp_dir(self.config_file_path, run_id)
        tmp_root.mkdir(parents=True, exist_ok=True)
        character_filename = str(tmp_root / "benchmark_character_commands.txt")

        _ = generate_character_commands(
            targets=scene_type,
            num_characters=num_people,
            num_commands=10,  # 10 commands per character
            seed=seed,
            filename=character_filename,
            base_name="Character"
        )

        print(f"Generated character commands for {num_people} characters in {scene_type} scene")
        print(f"Commands saved to: {character_filename}")

    async def _generate_robot_commands(self, scene_type: str, robot_type: str, seed: int):
        """Generate robot commands for the given scene type and robot type."""
        # Normalize robot type name for the command generation
        robot_name = "Nova_Carter" if str(robot_type).lower() == "nova_carter" else "Spot"

        # Get run_id from environment for unique temp paths (multi-instance support)
        run_id = os.environ.get('BENCHMARK_RUN_ID', 'default')
        tmp_root = _benchmark_tmp_dir(self.config_file_path, run_id)
        tmp_root.mkdir(parents=True, exist_ok=True)
        robot_filename = str(tmp_root / "benchmark_robot_commands.txt")

        _ = generate_robot_commands(
            targets=scene_type,
            robot_type=robot_name,
            num_commands=1,  # 1 command for the robot
            seed=seed,
            filename=robot_filename
        )

        print(f"Generated robot commands for {robot_name} in {scene_type} scene")
        print(f"Commands saved to: {robot_filename}")


# ---------- Child process wrapper ----------

def _run_child_episode(config_file_path: str, episode_name: str, crash_report_path: str, debug_print: bool, result_json_path: str) -> int:
    """
    Child process entry point: create SimulationApp, run one episode, write results JSON, then close.
    Returns OS exit code (0 on success).
    """
    # Read navigation_method from command-line if provided (takes priority)
    args = get_args()
    if args.navigation_method:
        os.environ['NAVIGATION_METHOD'] = args.navigation_method.lower()
    
    # Late import inside child so parent doesn't load Isaac runtime
    from isaacsim import SimulationApp
    
    sim_app = SimulationApp(launch_config=APP_CONFIG)
    
    # Helper function to create error payload with episode config
    def _create_error_payload(error_type: str, error_msg: str = "") -> dict:
        """Create error payload with episode config data."""
        try:
            # Load config to get episode info
            runner_temp = BenchmarkRunner(None, os.path.abspath(config_file_path), None, False)
            runner_temp._load_benchmark_config()
            episode_config = runner_temp._benchmark_config.get(episode_name, {}) if hasattr(runner_temp, '_benchmark_config') and runner_temp._benchmark_config else {}
            return {
                "episode": episode_name,
                "scene": episode_config.get('scene'),
                "start": episode_config.get('start'),
                "goal": episode_config.get('goal'),
                "instruction": episode_config.get('instruction', ''),
                "timeout": episode_config.get('timeout', 0),
                "robot_type": episode_config.get('robot_type', ''),
                "success": False,
                "duration": 0.0,
                "duration_frames": 0,
                "navigation_error": float('inf'),
                "success_rate": 0.0,
                "collision": 0.0,
                "human_collision": 0.0,
                "physical_collision": 0.0,
                "spl": 0.0,
                "path_length": 0.0,
                "physical_collision_detected": False,
                "error": f"{error_type}:{error_msg}" if error_msg else error_type
            }
        except Exception as config_error:
            print(f"Failed to load config for error payload: {config_error}")
            return {"episode": episode_name, "success": False, "error": f"{error_type}:{error_msg}" if error_msg else error_type}

    try:
        runner = BenchmarkRunner(sim_app, os.path.abspath(config_file_path), crash_report_path, debug_print)

        from omni.kit.async_engine import run_coroutine
        task = run_coroutine(runner.run_single_episode_worker(episode_name))

        # Pump the app until the task completes
        ok = False
        result = None
        episode_error_msg = ""
        try:
            while not task.done() and not sim_app.is_exiting():
                sim_app.update()

            try:
                result = task.result()
                ok = True
            except asyncio.CancelledError:
                print("Episode was cancelled.")
                ok = False
                result = None
                episode_error_msg = "cancelled"
            except Exception as e:
                print(f"Episode error: {e}")
                import traceback
                traceback.print_exc()
                ok = False
                result = None
                episode_error_msg = str(e)

            # Persist single-episode result (always write something)
            if ok and result is not None:
                payload = result
            else:
                # Include episode config in error payload so parent can normalize properly
                payload = _create_error_payload("child_failed", episode_error_msg)
            
            with open(result_json_path, "w") as f:
                json.dump(payload, f)

        finally:
            sim_app.close()
    except Exception as e:
        print(f"Child bootstrap error: {e}")
        import traceback
        traceback.print_exc()
        # Ensure some JSON exists for parent to read
        try:
            error_payload = _create_error_payload("bootstrap", str(e))
            with open(result_json_path, "w") as f:
                json.dump(error_payload, f)
        except Exception:
            pass
        return 2

    return 0


# ---------- Parent process orchestration ----------

def _run_episode_in_subprocess(script_path: str, config_file: str, episode_name: str, crash_report_path: str, debug_print: bool, run_id: str, navigation_method: str = None) -> dict:
    """Spawn a child Python process that executes exactly one episode and returns the result dict."""
    with tempfile.TemporaryDirectory() as td:
        out_json = os.path.join(td, f"{episode_name}_result.json")

        cmd = [
            sys.executable,
            script_path,
            "-c", config_file,
            "--episode_name", episode_name,
            "--result_json", out_json,
            "--child",
            "--run_id", run_id,
        ]
        if crash_report_path:
            cmd += ["--crash_report_path", crash_report_path]
        if debug_print:
            cmd += ["--debug_print"]
        if navigation_method:
            cmd += ["--navigation_method", navigation_method]

        proc = subprocess.run(cmd)  # inherit stdio by default
        if proc.returncode != 0:
            print(f"[Parent] Child for episode '{episode_name}' exited with code {proc.returncode}")

        # Small delay to allow GPU memory cleanup and resource release
        import time
        time.sleep(0.5)

        # Load whatever JSON the child produced
        if os.path.isfile(out_json):
            with open(out_json, "r") as f:
                try:
                    return json.load(f)
                except Exception as e:
                    print(f"[Parent] Failed to parse child JSON for '{episode_name}': {e}")
                    return {"episode": episode_name, "success": False, "error": "json_parse_error"}
        else:
            return {"episode": episode_name, "success": False, "error": "no_result_file"}


def _normalize_result(raw: dict, episode_name: str, episode_config: dict) -> dict:
    """Ensure required keys exist so final printer never crashes."""
    # Ensure episode_config is a dict
    if not isinstance(episode_config, dict):
        episode_config = {}
    
    defaults = {
        'episode': episode_name,
        'scene': episode_config.get('scene'),
        'start': episode_config.get('start'),
        'goal': episode_config.get('goal'),
        'instruction': episode_config.get('instruction', ''),
        'timeout': episode_config.get('timeout', 0),
        'robot_type': episode_config.get('robot_type', ''),
        'success': False,
        'duration': 0.0,
        'duration_frames': 0,
        'navigation_error': float('inf'),
        'success_rate': 0.0,
        'collision': 0.0,
        'human_collision': 0.0,
        'physical_collision': 0.0,
        'spl': 0.0,
        'path_length': 0.0,
        'physical_collision_detected': False,
        'physical_collision_count': 0,
    }
    normalized = {**defaults, **(raw or {})}
    normalized['collision'] = _binary_metric(normalized, 'collision', 'collision_rate')
    normalized['human_collision'] = _binary_metric(normalized, 'human_collision', 'human_collision_rate')
    normalized['physical_collision'] = _binary_metric(normalized, 'physical_collision', 'physical_collision_rate')
    normalized.pop('collision_rate', None)
    normalized.pop('human_collision_rate', None)
    normalized.pop('physical_collision_rate', None)
    # Coerce numeric types for safety
    for k in ['duration', 'navigation_error', 'success_rate', 'collision', 'human_collision', 'physical_collision', 'spl', 'path_length']:
        try:
            normalized[k] = float(normalized.get(k, defaults[k]))
        except Exception:
            normalized[k] = float(defaults[k])
    try:
        normalized['duration_frames'] = int(normalized.get('duration_frames', defaults['duration_frames']))
    except Exception:
        normalized['duration_frames'] = 0
    try:
        normalized['physical_collision_count'] = int(normalized.get('physical_collision_count', 0))
    except Exception:
        normalized['physical_collision_count'] = 0
    normalized['success'] = bool(normalized.get('success', False))
    normalized['physical_collision_detected'] = bool(normalized.get('physical_collision_detected', False))
    return normalized


# ---------- CLI ----------

def get_args():
    parser = argparse.ArgumentParser("DynaNav Benchmark")
    parser.add_argument("-c", "--config_file", required=True, help="Path to benchmark config YAML file")

    parser.add_argument("--crash_report_path", required=False, default=None, help="Path to store crash reports")
    parser.add_argument("--debug_print", required=False, default=False, action="store_true", help="Enable debug print")
    
    # Navigation method (command-line takes priority over YAML config)
    parser.add_argument(
        "--navigation_method", "--method", 
        default=None,
        choices=['ticvla'],
        help="Navigation method to use. Public release supports ticvla."
    )

    # Run ID for multi-instance support (allows running multiple benchmarks simultaneously)
    parser.add_argument("--run_id", default=None, help="Unique run ID for this benchmark instance. If not provided, auto-generated from timestamp. Use different IDs to run multiple benchmarks in parallel.")

    # Child/parent flow
    parser.add_argument("--child", action="store_true", help="Run a single episode worker and exit")
    parser.add_argument("--episode_name", default=None, help="Episode name to run in child mode")
    parser.add_argument("--result_json", default=None, help="File path to write single-episode JSON result (child mode)")
    parser.add_argument("--start_from", default=None, help="Episode name to start from (inclusive) in parent mode")
    parser.add_argument("--reverse_episodes", action="store_true", help="Run episodes in reverse order")

    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    config_file_path = args.config_file
    crash_report_path = args.crash_report_path
    debug_print = args.debug_print

    # Generate or use provided run_id for multi-instance support
    if args.run_id:
        run_id = args.run_id
    else:
        # Auto-generate run_id from timestamp if not provided
        run_id = time.strftime("%Y%m%d_%H%M%S")
    
    # Set run_id in environment for behavior scripts to use
    os.environ['BENCHMARK_RUN_ID'] = run_id
    
    # Set navigation method from command-line (takes priority over YAML config)
    if args.navigation_method:
        os.environ['NAVIGATION_METHOD'] = args.navigation_method.lower()

    if args.child:
        # --- Child mode: run exactly one episode and exit ---
        if not args.episode_name or not args.result_json:
            print("Child mode requires --episode_name and --result_json", file=sys.stderr)
            sys.exit(2)
        print(f"[Child] Running episode '{args.episode_name}'")
        rc = _run_child_episode(config_file_path, args.episode_name, crash_report_path, debug_print, args.result_json)
        sys.exit(rc)

    # --- Parent mode: orchestrate all episodes in separate child processes ---
    print("Benchmark Config:", config_file_path)
    print("Crash Report Path:", crash_report_path)
    print("Debug Print:", debug_print)
    print("Run ID:", run_id)
    if args.navigation_method:
        print(f"Navigation Method: {args.navigation_method} (from command-line)")
    else:
        print("Navigation Method: Will use YAML config or default")
    if args.start_from:
        print("Start From:", args.start_from)
    if args.reverse_episodes:
        print("Episode Order: Reversed")

    # Check config file exists
    if not os.path.isfile(config_file_path):
        print("Invalid config file path. Exit.", file=sys.stderr)
        return

    # Load episodes list (parent does not touch SimulationApp)
    with open(config_file_path, "r") as f:
        benchmark_config = yaml.safe_load(f)

    # Collect per-episode results
    all_results = []
    start_from = args.start_from
    started = start_from is None
    items = list(benchmark_config.items())
    if args.reverse_episodes:
        items = list(reversed(items))
    for episode_name, episode_config in items:
        # Skip metadata keys that aren't episode configs (should be dicts)
        if episode_name == "seed":
            continue
        if episode_name == "navigation_method":
            continue
        if not isinstance(episode_config, dict):
            # Skip metadata entries that are not episode dictionaries.
            continue

        if not started:
            if episode_name == start_from:
                started = True
            else:
                continue

        print(f"\n{'=' * 50}\nLaunching episode (subprocess): {episode_name}\n{'=' * 50}")
        sys.stdout.flush()  # Ensure output is visible immediately
        raw = _run_episode_in_subprocess(
            script_path=os.path.abspath(__file__),
            config_file=config_file_path,
            episode_name=episode_name,
            crash_report_path=crash_report_path,
            debug_print=debug_print,
            run_id=run_id,
            navigation_method=args.navigation_method,
        )
        normalized = _normalize_result(raw, episode_name, episode_config)
        all_results.append(normalized)
        
        # Save results incrementally after each episode
        class _Printer:
            def __init__(self, results):
                self.results = results
            _save_results_to_file = BenchmarkRunner._save_results_to_file
        
        printer = _Printer(all_results)
        saved_file = printer._save_results_to_file(config_file_path=config_file_path, incremental=True)
        if saved_file:
            print(f"[REALTIME] Results updated after episode '{episode_name}': {saved_file}")
            sys.stdout.flush()  # Flush to ensure message appears immediately

    # Pretty print final aggregate (reuse the class printer without SimulationApp)
    class _Printer:
        def __init__(self, results):
            self.results = results

        _print_final_results = BenchmarkRunner._print_final_results
        _save_results_to_file = BenchmarkRunner._save_results_to_file

    printer = _Printer(all_results)
    printer._print_final_results()
    # Final save with timestamp (in addition to the latest incremental file)
    saved_file = printer._save_results_to_file(config_file_path=config_file_path, incremental=False)
    if saved_file:
        print(f"\n[FINAL] Results summary saved to: {saved_file}")
        sys.stdout.flush()

    print("BENCHMARK COMPLETE.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
