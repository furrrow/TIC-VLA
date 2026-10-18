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
from dataclasses import dataclass, field
from generate_commands import generate_character_commands, generate_robot_commands
import benchmark_reporting as report_utils
import benchmark_config as config_utils
import benchmark_runtime as runtime
from scripted_humans import generate_scripted_commands

SIMULATION_FPS = runtime.SIMULATION_FPS


def _print_collision_warning(collision_type, episode_name, details):
    """Print an immediately visible collision banner, even through buffered logs."""
    label = f"COLLISION WARNING: {collision_type}"
    context = f"episode={episode_name or 'UNKNOWN'} | {details}"
    width = max(78, len(label) + 8, len(context) + 8)
    border = "!" * width
    use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    emphasis = "\033[1;31m" if use_color else ""
    reset = "\033[0m" if use_color else ""
    print(
        f"\n{emphasis}{border}\n"
        f"!!! {label}\n"
        f"!!! {context}\n"
        f"{border}{reset}\n",
        file=sys.stderr,
        flush=True,
    )

@dataclass
class EpisodeState:
    """Tracking data that starts fresh for each episode."""

    name: str = ""
    config: dict | None = None
    start_time: float | None = None
    spawned_robot_prim_path: str | None = None
    robot_type: str | None = None
    robot_position: np.ndarray | None = None
    goal_position: np.ndarray | None = None
    failure_reason: str | None = None
    robot_path: list[np.ndarray] = field(default_factory=list)
    human_collision_detected: bool = False
    human_collision_count: int = 0
    active_human_collisions: set[str] = field(default_factory=set)
    start_position: np.ndarray | None = None
    simulation_frame_count: int = 0
    start_frame: int = 0
    physical_collision_detected: bool = False
    contact_consecutive_frames: int = 0
    max_contact_consecutive_frames: int = 0
    goal_threshold: float = 1.5
    position_error_logged: bool = False


class BenchmarkRunner:
    def __init__(self, sim_app, config_file_path, crash_report_path, debug_print, run_id, navigation_method, experiment_mode):
        self._sim_app = sim_app
        self._sim_app.set_setting("/app/scripting/ignoreWarningDialog", True)
        # Inputs
        self.config_file_path = config_file_path
        self.crash_report_path = crash_report_path
        self.debug_print = debug_print
        self.run_id = run_id
        self.navigation_method = navigation_method
        self.experiment_mode = experiment_mode
        self.behavior_context = None

        self.results = []
        self.state = EpisodeState()

        # Simulation components
        self._sim_manager = None
        self._setup_sim_sub = None
        self._setup_sim_succeed = False
        self._settings = None
        self._config = None
        self._benchmark_config = None

        # Physical collision (contact sensor)
        self._physx_contact_sub = None
        self._robot_contact_paths = set()       # prim paths we consider "robot bodies" for contact filtering
        self._contact_sensor_enabled = False
        self._contact_threshold = 10.0  # Minimum contact force to register as collision (N)
        self._contact_duration_threshold_frames = 5  # Require sustained contact before counting collision
        self.human_collision_threshold = 0.2  # meters
        self._contact_sensor_prim_path = None  # Path to the contact sensor prim
        self._contact_sensor_prim_paths = []  # Paths to all contact sensor prims
        self._contact_sensor_interface = None  # Contact sensor interface
        self._contact_sensors = []  # List of contact sensor instances

    # LEGACY, not used...
    # async def run_benchmark(self):
    #     """
    #     Legacy single-process run of all episodes (kept for completeness).
    #     In the new design, the parent process orchestrates episodes via subprocesses.
    #     """
    #     print("Starting benchmark run...")
    #
    #     # Load benchmark configuration
    #     self._load_benchmark_config()
    #
    #     # Enable extensions and setup
    #     self._enable_extensions()
    #     await self._sim_app.app.next_update_async()
    #
    #     self._set_simulation_settings()
    #     await self._sim_app.app.next_update_async()
    #
    #     # Initialize simulation manager
    #     from isaacsim.replicator.agent.core.simulation import SimulationManager
    #     self._sim_manager = SimulationManager()
    #
    #     # Run each episode (note: this mode does NOT close SimulationApp between episodes)
    #     for episode_name, episode_config in config_utils.select_episodes(self._benchmark_config):
    #
    #         print(f"\n{'=' * 50}")
    #         print(f"Running Episode: {episode_name}")
    #         print(f"{'=' * 50}")
    #
    #         success = await self._run_episode(episode_name, episode_config)
    #
    #         result = self.collect_and_build_episode_result(episode_name, episode_config, success)
    #         self.results.append(result)
    #
    #         # Clean up contact sensor
    #         self._cleanup_contact_sensor()
    #
    #         print(f"Episode {episode_name} completed: {'SUCCESS' if success else 'FAILED'}")
    #         print(f"Simulation Duration: {result['duration']:.2f}s ({result['duration_frames']} frames)")
    #         print(f"Navigation Error: {result['navigation_error']:.2f}m")
    #         print(f"Success weighted by Path Length: {result['spl']:.3f}")
    #         print(f"Path Length: {result['path_length']:.2f}m")
    #         # No per-contact physics logging in benchmark results.
    #
    #     # Print final results
    #     report_utils.print_final_results(self.results)
    #     return True

    async def run_single_episode_worker(self, episode_name: str):
        """
        Child-process entry: run exactly one episode under a fresh SimulationApp,
        then return a single result dict. The child process will close SimulationApp.
        """
        try:
            # Load and prep config
            self._load_benchmark_config()
            episode_config = config_utils.get_episode(self._benchmark_config, episode_name)
        except Exception as e:
            print(f"[run_single_episode_worker] Error loading config for episode '{episode_name}': {e}")
            import traceback
            traceback.print_exc()
            raise

        goal = episode_config['goal']
        self.behavior_context = runtime.BehaviorContext(
            run_id=self.run_id,
            episode_name=episode_name,
            goal=(float(goal[0]), float(goal[1]), float(goal[2] if len(goal) > 2 else 0.0)),
            instruction=episode_config['instruction'],
            log_dir=os.getenv('TICVLA_LOG_DIR', './logs'),
        )

        # Enable + settings
        self._enable_extensions()
        await self._sim_app.app.next_update_async()
        self._set_simulation_settings()
        await self._sim_app.app.next_update_async()

        from isaacsim.replicator.agent.core.simulation import SimulationManager
        self._sim_manager = SimulationManager()
        
        try:
            success = await self.run_episode(episode_name, episode_config)
            result = self.collect_and_build_episode_result(episode_name, episode_config, success)
            return result
        finally:
            self._cleanup_episode_resources()

    def _cleanup_episode_resources(self):
        """Release resources created for one episode on every exit path."""
        try:
            if self.behavior_context and self.behavior_context.ros_cleanup:
                self.behavior_context.ros_cleanup()
        finally:
            try:
                self._cleanup_contact_sensor()
            finally:
                runtime.unregister(self.state.spawned_robot_prim_path)
                self._setup_sim_sub = None
                path = getattr(self, '_temp_config_path', None)
                self._temp_config_path = None
                if path:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        print(f"Warning: Failed to cleanup temp config: {error}")

    def collect_and_build_episode_result(self, episode_name: str, episode_config: dict, success: bool) -> dict:
        metrics = report_utils.EpisodeMetrics(
            success=success,
            duration_frames=self.state.simulation_frame_count - self.state.start_frame,
            navigation_error=self._get_distance_to_goal(),
            spl=self._calculate_spl(success),
            path_length=self._calculate_path_length(),
            human_collision=self.state.human_collision_detected,
            human_collision_count=self.state.human_collision_count,
            physical_collision=self.state.physical_collision_detected,
            physical_collision_count=self.state.max_contact_consecutive_frames,
            failure_reason=self.state.failure_reason,
        )
        return report_utils.build_episode_result(
            episode_name, episode_config, metrics, experiment_mode=self.experiment_mode
        )

    def _load_benchmark_config(self):
        """Load the benchmark configuration from YAML file."""
        self._benchmark_config = config_utils.load_benchmark_config(self.config_file_path)
        print(f"Loaded benchmark config with {len(config_utils.select_episodes(self._benchmark_config))} episodes")

    async def run_episode(self, episode_name, episode_config):
        """Run a single benchmark episode."""
        self.state = EpisodeState(
            name=episode_name,
            config=episode_config,
            start_time=time.monotonic(),
            goal_threshold=episode_config['success_threshold'],
        )

        # Set up simulation for this episode
        await self._setup_episode_simulation(episode_config)
        print(f"Setup episode simulation succeed")

        # Spawn robot at start position
        robot_type = episode_config['robot_type'].replace('_', '_').title()

        # Store robot type for position tracking
        self.state.robot_type = robot_type

        # Get start_yaw from config (default to 0 if not specified)
        start_yaw = episode_config['start_yaw']
        self.state.spawned_robot_prim_path = await self._spawn_robot(robot_type, episode_config['start'], start_yaw)
        runtime.register(self.state.spawned_robot_prim_path, self.behavior_context)

        # Set up robot
        if robot_type == "Spot":
            await self._set_quadruped_robot(self.state.spawned_robot_prim_path)
        elif robot_type == "Nova_Carter":
            await self._set_carter_robot(self.state.spawned_robot_prim_path)
        await self._set_third_person_view(self.state.spawned_robot_prim_path)
        print(f"Set up robot succeed")

        # Set up contact sensor for collision detection
        await self._setup_contact_sensor(self.state.spawned_robot_prim_path)
        print(f"Set up contact sensor succeed")

        # Set goal position
        self.state.goal_position = np.array(episode_config['goal'])

        # Record start position for path tracking
        self.state.start_position = np.array(episode_config['start'])

        # Run navigation with timeout (in simulation frames)
        timeout_seconds = episode_config['timeout']
        timeout_frames = int(timeout_seconds * SIMULATION_FPS)
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

        # Set up simulation
        humans = episode_config.get('scripted_humans')
        if humans is None:
            await self._setup_sim()
        else:
            import carb

            command_path = config_utils.benchmark_tmp_dir(self.config_file_path, self.run_id) / "benchmark_character_commands.txt"
            command_path.parent.mkdir(parents=True, exist_ok=True)
            command_path.write_text(generate_scripted_commands(humans), encoding="utf-8")
            randomizer = self._sim_manager._character_randomizer
            original_position = randomizer.get_random_position
            original_spawn = self._sim_manager.spawn_character_by_idx
            next_human = 0

            def scripted_position(spawn_area=None):
                # Isaac Sim 0.7.28 passes a spawn area here, not a character index.
                # load_characters_from_config_file calls this once per character, in index order.
                nonlocal next_human
                if next_human >= len(humans):
                    raise RuntimeError("Isaac Sim requested more character positions than scripted_humans provides")
                position = carb.Float3(*humans[next_human]['position'])
                next_human += 1
                return position

            randomizer.get_random_position = scripted_position
            self._sim_manager.spawn_character_by_idx = (
                lambda location, rotation, idx: original_spawn(location, humans[idx]['yaw_degrees'], idx)
            )
            try:
                await self._setup_sim()
            finally:
                randomizer.get_random_position = original_position
                self._sim_manager.spawn_character_by_idx = original_spawn
        print(f"Setup sim succeed: {self._setup_sim_succeed}")

        # Warmup simulation
        await self._warmup(frames=200)
        print(f"Warmup complete")

    def _create_episode_config(self, episode_config):
        """Create a temporary config file for this episode."""
        scene_url = episode_config['scene']

        tmp_root = config_utils.benchmark_tmp_dir(self.config_file_path, self.run_id)
        tmp_dir = str(tmp_root)

        config_template = {
            'isaacsim.replicator.agent': {
                'version': '0.7.19',
                'global': {
                    'seed': self._benchmark_config['seed'],
                    'simulation_length': int(episode_config['timeout'] * SIMULATION_FPS)
                },
                'scene': {
                    'asset_path': scene_url
                },
                'sensor': {
                    'camera_num': 0
                },
                'character': {
                    'asset_path': config_utils.asset_path('DEFAULT_CHARACTER_ASSET_PATH'),
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
                        'output_dir': str(tmp_root / "benchmark_output" / self.state.name),
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
        temp_config_path = f'{tmp_dir}/benchmark_config_{self.state.name}.yaml'
        self._temp_config_path = temp_config_path
        with open(temp_config_path, 'w') as f:
            yaml.dump(config_template, f)

        return temp_config_path

    async def _run_navigation(self, timeout_frames):
        import omni.timeline
        import omni.replicator.core as rep
        import omni.kit.app

        app = omni.kit.app.get_app()
        timeline = omni.timeline.get_timeline_interface()

        original_timecode = timeline.get_time_codes_per_second()
        try:
            timeline.set_time_codes_per_second(SIMULATION_FPS)
            timeline.commit_silently()
            await app.next_update_async()

            if not timeline.is_playing():
                timeline.play()

            rep.orchestrator.set_capture_on_play(True)
            await app.next_update_async()

            self.state.start_frame = self.state.simulation_frame_count
            for _ in range(timeout_frames):
                # per-frame: update pos, check goal/collisions, count frames
                await self._update_robot_position()
                self.state.simulation_frame_count += 1
                # Avoid printing every frame (can flood logs and slow down Isaac Sim).
                if self.debug_print and (self.state.simulation_frame_count % SIMULATION_FPS == 0):
                    print(f"Simulation frame count: {self.state.simulation_frame_count}")
                self.behavior_context.frame_count = self.state.simulation_frame_count

                failure_reason = self.behavior_context.failure_reason
                if failure_reason:
                    self.state.failure_reason = failure_reason
                    print(f"Episode failed early: {failure_reason}")
                    return False

                # reached goal?
                if self._is_goal_reached():
                    return True

                # advance one frame
                await app.next_update_async()

            self.state.failure_reason = (
                f"Timed out after {timeout_frames / SIMULATION_FPS:.1f}s "
                f"({timeout_frames} frames) without reaching the goal"
            )
        except Exception as e:
            self.state.failure_reason = f"Navigation error: {e}"
            print(f"Warning: Error during navigation: {e}")
        finally:
            # Restore capture and timeline state after success, failure, or setup error.
            for action in (
                lambda: rep.orchestrator.set_capture_on_play(False),
                timeline.stop,
                lambda: timeline.set_time_codes_per_second(original_timecode),
                timeline.commit_silently,
            ):
                try:
                    action()
                except Exception as cleanup_error:
                    print(f"Warning: Failed to restore navigation state: {cleanup_error}")
            try:
                await app.next_update_async()
            except Exception as cleanup_error:
                print(f"Warning: Failed to update app after navigation cleanup: {cleanup_error}")

        return False

    async def _update_robot_position(self):
        """Update the current robot position."""
        if not self.state.spawned_robot_prim_path:
            return

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(self.state.spawned_robot_prim_path)
        
        if not robot_prim or not robot_prim.IsValid():
            return

        # Try to get position from robot-specific body link, with fallbacks
        position_prim = None
        
        if self.state.robot_type == "Spot":
            # Spot has a 'body' child
            body_prim = robot_prim.GetChild('body')
            if body_prim and body_prim.IsValid():
                position_prim = body_prim
        elif self.state.robot_type == "Nova_Carter":
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
                self.state.robot_position = np.array([position[0], position[1], position[2]])

                # Record path for SPL calculation
                self.state.robot_path.append(self.state.robot_position.copy())

                # Check for collisions (distance-based)
                await self._check_collisions()
                
                # Read contact sensor for physical collision detection
                self._read_contact_sensor()
            except Exception as e:
                # Log error but don't crash - position will remain None
                if not self.state.position_error_logged:
                    print(f"Warning: Failed to update robot position: {e}")
                    self.state.position_error_logged = True


    def _is_goal_reached(self):
        """Check if robot has reached the goal."""
        if self.state.robot_position is None or self.state.goal_position is None:
            return False

        distance = self._get_distance_to_goal()
        return distance < self.state.goal_threshold

    def _get_distance_to_goal(self):
        """Get distance to goal."""
        if self.state.robot_position is None or self.state.goal_position is None:
            return float('inf')

        # Only consider x,y coordinates for 2D navigation
        robot_2d = self.state.robot_position[:2]
        goal_2d = self.state.goal_position[:2]
        return np.linalg.norm(robot_2d - goal_2d)

    async def _check_collisions(self):
        """Check for collisions with humans/characters."""
        import omni.usd
        from pxr import UsdGeom

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
        colliding_now = set()
        
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
                robot_2d = self.state.robot_position[:2]
                char_2d = char_pos[:2]
                distance = np.linalg.norm(robot_2d - char_2d)
                if distance < (self.human_collision_threshold + 0.1):
                    print(f"distance: {distance:.2f} to {skelroot_prim.GetPath().pathString} ")
                if distance < self.human_collision_threshold:
                    colliding_now.add(skelroot_path)
                    if skelroot_path not in self.state.active_human_collisions:
                        self.state.human_collision_count += 1
                        _print_collision_warning(
                            "HUMAN",
                            self.state.name,
                            f"character={skelroot_path} | distance={distance:.2f} m "
                            f"(< {self.human_collision_threshold:.2f} m) | "
                            f"frame={self.state.simulation_frame_count} | "
                            f"collision_count={self.state.human_collision_count}",
                        )
        self.state.active_human_collisions = colliding_now
        self.state.human_collision_detected = self.state.human_collision_count > 0

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
        if not success or len(self.state.robot_path) < 2:
            return 0.0 

        # Calculate actual path length
        path_length = 0.0
        for i in range(1, len(self.state.robot_path)):
            # Only consider 2D distance for ground navigation
            pos1 = self.state.robot_path[i - 1][:2]
            pos2 = self.state.robot_path[i][:2]
            path_length += np.linalg.norm(pos2 - pos1)

        # Calculate optimal path length (straight line from start to goal)
        if self.state.start_position is not None and self.state.goal_position is not None:
            start_2d = self.state.start_position[:2]
            goal_2d = self.state.goal_position[:2]
            optimal_length = np.linalg.norm(goal_2d - start_2d)

            if optimal_length > 0:
                spl = optimal_length / max(path_length, optimal_length)
                return float(min(spl, 1.0))  # Cap at 1.0

        return 0.0

    def _calculate_path_length(self):
        """Calculate total path length traveled."""
        if len(self.state.robot_path) < 2:
            return 0.0

        path_length = 0.0
        for i in range(1, len(self.state.robot_path)):
            # Only consider 2D distance for ground navigation
            pos1 = self.state.robot_path[i - 1][:2]
            pos2 = self.state.robot_path[i][:2]
            path_length += np.linalg.norm(pos2 - pos1)

        return float(path_length)

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
        if self.navigation_method == 'ticvla_ros':
            # Isaac's bridge supplies Python 3.11-compatible rclpy and message types.
            ext_manager.set_extension_enabled_immediate('isaacsim.ros2.bridge', True)

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
        self._settings.set(
            "/exts/isaacsim.replicator.agent/asset_settings/default_biped_assets_path",
            config_utils.asset_path('DEFAULT_BIPED_ASSET_PATH'),
        )

        # Crash reporter
        self._settings.set("/crashreporter/enabled", True)
        if self.crash_report_path:
            self._settings.set("/crashreporter/dumpDir", self.crash_report_path)

    async def _setup_sim(self):
        def done_callback(_e):
            self._setup_sim_succeed = True
            self._setup_sim_sub = None

        self._setup_sim_succeed = False
        self._setup_sim_sub = self._sim_manager.register_set_up_simulation_done_callback(done_callback)
        try:
            self._sim_manager.set_up_simulation_from_config_file()

            while self._setup_sim_sub and not self._sim_app.is_exiting():
                await self._sim_app.app.next_update_async()
        finally:
            # The callback handle owns the subscription; dropping it releases it.
            self._setup_sim_sub = None

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
        # note, they are using the older version of nova-carter on purpose, it will break if we try nova-carter.use from v5.1
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

        nav_method = self.navigation_method
        
        if nav_method not in ('ticvla', 'ticvla_ros'):
            raise ValueError(
                f"Unsupported navigation method in the public TIC-VLA release: {nav_method}. "
                "Use 'ticvla'."
            )
        script_path = os.path.abspath(
            "./behavior/nova_carter_ros_bridge.py" if nav_method == 'ticvla_ros'
            else "./behavior/nova_carter_test_ticvla.py"
        )
        script_name = "ROS sensor bridge" if nav_method == 'ticvla_ros' else "TIC-VLA"
        print(f"Using {script_name} behavior script")

        # Attach behavioral script
        import omni.kit.commands
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(robot_prim.GetPrimPath())])
        attr = robot_prim.GetAttribute("omni:scripting:scripts")
        attr.Set([Sdf.AssetPath(script_path)])
        print(f"Set up Nova Carter behavior script ({script_name})")

    async def _set_third_person_view(self, prim_path: str):
        """Point the GUI viewport at the robot camera used for third-person logs."""
        camera_path = f"{prim_path}/chassis_link/third_person_view_cam"

        try:
            import carb
            from omni.kit.viewport.utility import get_active_viewport
            from pxr import Sdf

            viewport = get_active_viewport()
            if viewport is None:
                carb.log_warn("No active viewport available for robot camera view")
                return

            viewport.camera_path = Sdf.Path(camera_path)
            await self._sim_app.app.next_update_async()
            print(f"Viewport camera set to {camera_path}")
        except Exception as e:
            try:
                import carb
                carb.log_warn(f"Failed to set robot viewport camera: {e}")
            except Exception:
                print(f"Warning: Failed to set robot viewport camera: {e}")

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

        nav_method = self.navigation_method
        
        if nav_method != 'ticvla':
            raise ValueError(
                f"Navigation method {nav_method!r} is not supported for Spot; use 'ticvla'."
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
        from pxr import Gf, Usd, UsdPhysics
        import numpy as np
        
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        
        if not robot_prim or not robot_prim.IsValid():
            print(f"Warning: Robot prim not found for contact sensor setup: {robot_prim_path}")
            return
        
        # Clear previous contact tracking
        self._robot_contact_paths.clear()
        self.state.physical_collision_detected = False
        self.state.contact_consecutive_frames = 0
        self.state.max_contact_consecutive_frames = 0
        self._contact_sensors = []  # Store contact sensor instances
        self._contact_sensor_prim_path = None
        self._contact_sensor_prim_paths = []
        if hasattr(self, '_contact_read_error_logged'):
            delattr(self, '_contact_read_error_logged')
        
        # Find the main body/chassis prim to attach contact sensor
        # For Nova Carter: chassis_link, For Spot: body
        body_prim_path = None
        if self.state.robot_type == "Nova_Carter":
            chassis_path = f"{robot_prim_path}/chassis_link"
            chassis_prim = stage.GetPrimAtPath(chassis_path)
            if chassis_prim and chassis_prim.IsValid():
                body_prim_path = chassis_path
        elif self.state.robot_type == "Spot":
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

            # print(f"Contact sensor reading: {reading.value:.2f}, valid: {reading.is_valid}, in_contact:{reading.in_contact}")
            
            # Count only sustained, meaningful contact. A one-frame scrape or low-force
            # physics jitter should not mark the whole episode as physical collision.
            if reading.is_valid and reading.in_contact and float(reading.value) >= self._contact_threshold:
                self.state.contact_consecutive_frames += 1
                self.state.max_contact_consecutive_frames = max(
                    self.state.max_contact_consecutive_frames,
                    self.state.contact_consecutive_frames,
                )

                if (
                    not self.state.physical_collision_detected
                    and self.state.contact_consecutive_frames >= self._contact_duration_threshold_frames
                ):
                    self.state.physical_collision_detected = True
                    _print_collision_warning(
                        "PHYSICAL CONTACT",
                        self.state.name,
                        f"sustained_frames={self.state.contact_consecutive_frames} "
                        f"(threshold={self._contact_duration_threshold_frames}) | "
                        f"force={float(reading.value):.1f} N "
                        f"(threshold={self._contact_threshold:.1f} N) | "
                        f"frame={self.state.simulation_frame_count}",
                    )
            else:
                self.state.contact_consecutive_frames = 0
                            
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
        scene_url = self.state.config.get('scene', '')
        scene_type = self._detect_scene_type(scene_url)

        if not scene_type:
            print(f"Unknown scene type from URL: {scene_url}")
            print("Supported scene types: warehouse, hospital, office, outdoor")
            return

        print(f"Detected scene type: {scene_type}")

        # Get episode configuration with sensible defaults
        num_people = self.state.config['num_people']
        robot_type = self.state.config['robot_type']
        seed = self._benchmark_config['seed']

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
        tmp_root = config_utils.benchmark_tmp_dir(self.config_file_path, self.run_id)
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

        tmp_root = config_utils.benchmark_tmp_dir(self.config_file_path, self.run_id)
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

def run_child_episode(config_file_path: str, episode_name: str, crash_report_path: str, debug_print: bool, result_json_path: str, benchmark_config: dict, run_id: str, navigation_method: str, experiment_mode: str) -> int:
    """
    Child process entry point: create SimulationApp, run one episode, write results JSON, then close.
    Returns OS exit code (0 on success).
    """
    # Late import inside child so parent doesn't load Isaac runtime
    from isaacsim import SimulationApp
    
    sim_app = SimulationApp(launch_config=config_utils.APP_CONFIG)
    
    # Helper function to create error payload with episode config
    def _create_error_payload(error_type: str, error_msg: str = "") -> dict:
        """Create error payload with episode config data."""
        error = f"{error_type}:{error_msg}" if error_msg else error_type
        episode_config = config_utils.get_episode(benchmark_config, episode_name)
        return report_utils.build_episode_result(
            episode_name, episode_config, experiment_mode=experiment_mode, error=error
        )

    try:
        runner = BenchmarkRunner(
            sim_app, os.path.abspath(config_file_path), crash_report_path, debug_print,
            run_id, navigation_method, experiment_mode,
        )

        from omni.kit.async_engine import run_coroutine
        task = run_coroutine(runner.run_single_episode_worker(episode_name))

        # Pump the app until the task completes
        ok = False
        result = None
        episode_error_msg = ""
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
            payload = _create_error_payload("child_failed", episode_error_msg)

        with open(result_json_path, "w") as f:
            json.dump(payload, f)
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
    finally:
        sim_app.close()

    return 0


# ---------- Parent process orchestration ----------

def run_episode_in_subprocess(script_path: str, config_file: str, episode_name: str, crash_report_path: str, debug_print: bool, run_id: str, navigation_method: str = None, experiment_mode: str = "normal") -> dict:
    """Start benchmark.py --child; its main() dispatches to run_child_episode()."""
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
            "--experiment_mode", experiment_mode,
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


def _normalize_result(raw: dict, episode_name: str, episode_config: dict, experiment_mode: str = "normal") -> dict:
    """Ensure required keys exist so final printer never crashes."""
    # Ensure episode_config is a dict
    if not isinstance(episode_config, dict):
        episode_config = {}
    
    defaults = report_utils.build_episode_result(
        episode_name, episode_config, experiment_mode=experiment_mode
    )
    normalized = {**defaults, **(raw or {})}
    # Read legacy metric names before defaults fill in the current names.
    for key, legacy_key in (
        ('collision', 'collision_rate'),
        ('human_collision', 'human_collision_rate'),
        ('physical_collision', 'physical_collision_rate'),
    ):
        normalized[key] = report_utils.binary_metric(raw or {}, key, legacy_key)
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
    try:
        normalized['human_collision_count'] = int(normalized.get('human_collision_count', 0))
    except Exception:
        normalized['human_collision_count'] = 0
    normalized['success'] = bool(normalized.get('success', False))
    normalized['physical_collision_detected'] = bool(normalized.get('physical_collision_detected', False))
    return normalized


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser("DynaNav Benchmark")
    parser.add_argument("-c", "--config_file", required=True, help="Path to benchmark config YAML file")

    parser.add_argument("--crash_report_path", required=False, default=None, help="Path to store crash reports")
    parser.add_argument("--debug_print", required=False, default=False, action="store_true", help="Enable debug print")
    
    # Command-line selection overrides the method in the benchmark YAML.
    parser.add_argument(
        "--navigation_method", "--method", 
        default=None,
        choices=['ticvla', 'ticvla_ros'],
        help="Navigation method; ticvla_ros publishes Nova Carter sensor topics."
    )

    # Run ID for multi-instance support (allows running multiple benchmarks simultaneously)
    parser.add_argument("--run_id", default=None, help="Unique run ID for this benchmark instance. If not provided, auto-generated from timestamp. Use different IDs to run multiple benchmarks in parallel.")
    parser.add_argument(
        "--experiment_mode", "--mode", choices=["normal", "steered"], default="normal",
        help="Controller mode to record in benchmark results (default: normal).",
    )

    # Child/parent flow
    parser.add_argument("--child", action="store_true", help="Run a single episode worker and exit")
    parser.add_argument("--episode_name", default=None, help="Episode name to run in child mode")
    parser.add_argument("--result_json", default=None, help="File path to write single-episode JSON result (child mode)")
    parser.add_argument("--start_from", default=None, help="Episode name to start from (inclusive) in parent mode")
    parser.add_argument("--reverse_episodes", action="store_true", help="Run episodes in reverse order")

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    config_file_path = args.config_file
    crash_report_path = args.crash_report_path
    debug_print = args.debug_print
    experiment_mode = args.experiment_mode

    # Generate or use provided run_id for multi-instance support
    if args.run_id:
        run_id = args.run_id
    else:
        # Auto-generate run_id from timestamp if not provided
        run_id = time.strftime("%Y%m%d_%H%M%S")
    
    if not os.path.isfile(config_file_path):
        print("Invalid config file path. Exit.", file=sys.stderr)
        return
    try:
        benchmark_config = config_utils.load_benchmark_config(config_file_path)
        navigation_method = args.navigation_method or benchmark_config['navigation_method']
        if args.navigation_method and args.navigation_method.lower() not in ('ticvla', 'ticvla_ros'):
            raise ValueError("navigation_method must be 'ticvla' or 'ticvla_ros'")
        if args.child:
            if not args.episode_name or not args.result_json:
                raise ValueError("Child mode requires --episode_name and --result_json")
            config_utils.get_episode(benchmark_config, args.episode_name)
        else:
            episodes = config_utils.select_episodes(
                benchmark_config, start_from=args.start_from, reverse=args.reverse_episodes
            )
    except (OSError, yaml.YAMLError, ValueError) as exc:
        print(f"Invalid benchmark config or episode selection: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.child:
        # --- Child mode: run exactly one episode and exit ---
        if not args.episode_name or not args.result_json:
            print("Child mode requires --episode_name and --result_json", file=sys.stderr)
            sys.exit(2)
        print(f"[Child] Running episode '{args.episode_name}'")
        rc = run_child_episode(
            config_file_path, args.episode_name, crash_report_path, debug_print,
            args.result_json, benchmark_config, run_id, navigation_method, experiment_mode,
        )
        sys.exit(rc)

    # --- Parent mode: orchestrate all episodes in separate child processes ---
    print("Benchmark Config:", config_file_path)
    print("Crash Report Path:", crash_report_path)
    print("Debug Print:", debug_print)
    print("Run ID:", run_id)
    print(f"Navigation Method: {navigation_method} (resolved from command-line or config)")
    print(f"Experiment Mode: {experiment_mode}")
    if args.start_from:
        print("Start From:", args.start_from)
    if args.reverse_episodes:
        print("Episode Order: Reversed")

    # Collect per-episode results
    all_results = []
    for episode_name, episode_config in episodes:

        print(f"\n{'=' * 50}\nLaunching episode (subprocess): {episode_name}\n{'=' * 50}")
        sys.stdout.flush()  # Ensure output is visible immediately
        raw = run_episode_in_subprocess(
            script_path=os.path.abspath(__file__),
            config_file=config_file_path,
            episode_name=episode_name,
            crash_report_path=crash_report_path,
            debug_print=debug_print,
            run_id=run_id,
            navigation_method=navigation_method,
            experiment_mode=experiment_mode,
        )
        normalized = _normalize_result(raw, episode_name, episode_config, experiment_mode)
        all_results.append(normalized)
        
        # Save results incrementally after each episode
        saved_file = report_utils.save_results_to_file(
            all_results, config_file_path=config_file_path,
            incremental=True, run_id=run_id,
        )
        if saved_file:
            print(f"[REALTIME] Results updated after episode {episode_name!r}: {saved_file}")
            sys.stdout.flush()

    report_utils.print_final_results(all_results)
    saved_file = report_utils.save_results_to_file(
        all_results, config_file_path=config_file_path,
        incremental=False, run_id=run_id,
    )
    if saved_file:
        print(f"\n[FINAL] Results summary saved to: {saved_file}")
        sys.stdout.flush()

    print("BENCHMARK COMPLETE.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
