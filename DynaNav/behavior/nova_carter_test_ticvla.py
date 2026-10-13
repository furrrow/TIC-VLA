"""
TIC-VLA-based autonomous navigation for the Isaac Nova Carter robot.
Non-blocking TIC-VLA inference: runs model predict on a background thread
and uses the latest completed prediction in the main simulation loop.

This version consumes the full trajectory returned by TIC-VLA (T=10 waypoints),
where each waypoint corresponds to 0.1s of integrated motion. The trajectory
is placed into an action queue and consumed over time in on_update().
"""

import os
import cv2
import sys
import json
import time
import math
import threading
from typing import Optional, Dict, Any, List

import numpy as np
import torch

# Plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import carb
import omni
import omni.usd
from omni.kit.scripting import BehaviorScript
from omni.anim.people.settings import PeopleSettings, AgentEvent
from omni.timeline import get_timeline_interface

# Isaac wheeled robot components
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.rotations import quat_to_rot_matrix

# Navigation helper for publishing position
from robot_navigation_manager import RobotNavigationManager

# Camera and rotation utilities (align with Spot sensor behavior)
from pxr import Usd, UsdGeom, Gf, UsdPhysics
from pxr import PhysxSchema
import omni.replicator.core as rep

# TIC-VLA model components
TICVLA = None
try:
    # Add parent directory to path to import from the local ticvla.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # DynaNav directory
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from ticvla import TICVLA
except ImportError as e:
    print(f"Warning: TIC-VLA components not found: {e}")
    TICVLA = None

import shutil
from pathlib import Path

def _empty_dir(path: str, *, make: bool = True, keep_root: bool = True):
    """
    Delete all contents under `path` (files + subfolders), but keep the folder itself.
    Safe for repeated calls.
    """
    p = Path(path)
    try:
        if not p.exists():
            if make:
                p.mkdir(parents=True, exist_ok=True)
            return
        if not p.is_dir():
            return

        for child in p.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except Exception:
                # best-effort cleanup; don't crash the episode
                pass

        if make:
            p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


class NovaCarterTICVLA(BehaviorScript):
    """TIC-VLA-based autonomous navigation behavior for Nova Carter robot with async inference."""

    # === Robot-specific defaults (Nova Carter) ===
    _wheel_joints = ["joint_wheel_left", "joint_wheel_right"]
    _wheel_radius = 0.152
    _wheel_base = 0.413

    # === TIC-VLA Configuration ===
    _model_path = None  # Will be set via configuration
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _inference_mode = True
    # === Checkpoint Configuration ===
    _checkpoint_path = os.getenv("TICVLA_CHECKPOINT_PATH", "checkpoints/ticvla.ckpt")
    
    
    # === Action smoothing parameters ===
    _max_accel_lin = 2.0   # m/s^2
    _max_decel_lin = 2.5   # m/s^2
    _max_accel_ang = 3.0   # rad/s^2
    _max_decel_ang = 3.5   # rad/s^2
    _lin_deadband = 0.001   # m/s (small threshold to filter noise)
    _ang_deadband = 0.0005  # rad/s (small threshold to filter noise)

    # === Navigation parameters ===
    _goal_threshold = 2  # meters
    _max_linear_velocity = 1.5  # m/s
    _max_angular_velocity = 1.2  # rad/s

    def on_init(self):
        """Initialize the TIC-VLA behavior script."""
        carb.log_info(f"[{self.get_agent_name()}] Initializing NovaCarterTICVLA...")

        # Create wheeled robot & controller
        self._wheeled_robot: WheeledRobot = WheeledRobot(
            prim_path=str(self.prim_path),
            name=self.get_agent_name(),
            wheel_dof_names=self._wheel_joints,
            create_robot=False,
        )
        self._controller = WheelBasePoseController(
            name=self.get_agent_name() + "_wheeled_base_pose_controller",
            open_loop_wheel_controller=DifferentialController(
                name=self.get_agent_name() + "_differential_controller",
                wheel_radius=self._wheel_radius,
                wheel_base=self._wheel_base,
            ),
            is_holonomic=False,
        )
        self._diff = self._controller._open_loop_wheel_controller

        # TIC-VLA model components
        self._ticvla_model: Optional[TICVLA] = None

        # Command state (smoothed)
        self._cmd_v = 0.0
        self._cmd_w = 0.0

        # Robot state tracking
        self._robot_position = np.array([0.0, 0.0, 0.0])
        self._robot_orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self._robot_velocity = np.array([0.0, 0.0, 0.0])
        self._robot_yaw_speed = 0.0
        self._goal_position = None
        self._instruction = ""
        self._current_action = np.array([0.0, 0.0])  # [linear_vel, angular_vel]
        
        # Previous state for velocity calculation
        self._prev_position = None
        self._prev_orientation = None
        self._prev_delta_time = 0.0
        self._prev_heading = None  # For computing yaw speed from motion heading
        self._current_heading = 0.0  # Current heading angle for plotting
        
        # Offsets for model context (frame-based, same as benchmark):
        # Compute 1s displacement (t-1 -> t) expressed in the (t-1) frame at 1Hz.
        # We use sim steps (frames @ 30Hz): step 0 -> (0,0,0), step 30 -> delta(0->30), step 60 -> delta(30->60), ...
        # Same timing as benchmark: time = frames / 30.0 (30 FPS)
        self._delta_1s_sampled_1s: List[tuple[int, List[float]]] = []  # (step, [dx,dy,dz]) in (t-1) frame (FLU)
        self._last_recorded_frame: int = -1  # Track last frame we recorded a waypoint to avoid duplicates
        self._prev_position_for_offset: np.ndarray | None = None
        self._prev_quaternion_for_offset: np.ndarray | None = None
        self._offset_update_frame_count = 0
        
        # Track VLM generation start pose and frame for dx, dy calculation in robot state
        # Keep only the 2 most recent VLM generation start poses
        # Note: VLM generation runs asynchronously, so we track when VLM actually starts, not when policy inference starts
        # Uses frame count (like benchmark) - assumes 30 FPS: time = frames / 30.0
        self._vlm_generation_start_positions: List[np.ndarray] = []  # List of positions when VLM generation started
        self._vlm_generation_start_quaternions: List[np.ndarray] = []  # List of quaternions [w, x, y, z] when VLM generation started
        self._vlm_generation_start_frames: List[int] = []  # List of frame numbers when VLM generation started (30 FPS)
        self._first_inference_complete: bool = False  # Flag to track if first inference has completed

        # Camera and sensor setup (replicator annotators)
        self._camera_front_prim_path = None
        self._camera_tp_prim_path = None
        self._rp_front = None
        self._rp_tp = None
        self._rgb_annotator_front = None
        self._rgb_annotator_tp = None
        self._camera_image_front = None
        self._camera_image_tp = None

        # Image history buffering for model
        self._image_history = []  # Keep last N frames for better prediction
        self._max_history_frames = 4  # Maximum number of historical frames to keep
        self._sensor_update_frequency = 10  # Hz
        # Keep 9 seconds of history based on sensor frequency
        self._max_history_frames = int(19 * self._sensor_update_frequency)
        self._sensor_frame_count = -1
        self._image_buffer_dir = None  # Directory to temporarily store camera frames

        # Settings and navigation
        self.setting = carb.settings.get_settings()
        self._navmeshEnabled = False
        self._avoidanceOn = False
        self._navigation_manager: Optional[RobotNavigationManager] = None

        # Internal flags
        self._initialized = False
        self._model_loaded = False
        self._navigation_active = False

        # Data streaming and logging
        self._stream_data = True
        self._log_interval_frames = 10
        self._last_log_frame = 0
        # Get run_id from environment for unique log paths (multi-instance support)
        self._run_id = os.getenv('BENCHMARK_RUN_ID', 'default')
        self.save_log_dir = os.getenv('TICVLA_LOG_DIR', './logs')
        # self._log_directory = f"./logs/{self._run_id}/carter_ticvla_data"
        self._log_directory = f"{self.save_log_dir}/{self._run_id}/carter_ticvla_data"
        self._frame_count = 0
        self._create_log_directory()
        
        # Trajectory tracking for plotting
        self._trajectory_history = []  # List of [x, y] positions
        self._robot_state_history = []  # List of [vx, vy, vz, yaw_speed] for each frame
        self._heading_history = []  # List of heading angles (yaw) for each frame
        self._trajectory_plotted = False  # Flag to ensure plot is saved only once

        # Event system
        self._bus = None
        self._agent_registered_event = None

        # Inference synchronization (for _current_action updates)
        self._inference_lock = threading.Lock()
        self._inference_timeout_sec = 3.0   # consider stale if older than this
        self._base_link_path = "/World/Robots/Nova_Carter/chassis_link"  # adjust if different
        self._prev_R = None

        # Stuck detection and recovery
        self._stuck_position_threshold = 0.1  # meters - consider stuck if moved less than this
        self._stuck_time_threshold_frames = 150  # frames - stuck if not moving for this many frames (3 seconds at 30 FPS)
        self._backup_duration_frames = 300  # Deprecated: stuck now ends the episode as a failure.
        self._backup_speed = -0.5  # Deprecated: kept for compatibility with existing state reset paths.
        self._last_movement_position = None  # Last position where significant movement was detected
        self._last_movement_frame = None  # Frame when last movement was detected (don't start until first inference completes)
        self._is_backing_up = False  # Flag indicating if currently backing up
        self._backup_start_frame = None  # Frame when backup started
        self._backup_completion_frame = None  # Frame when backup completed (used to ensure we wait for inference that starts after this)
        self._waiting_for_inference_after_backup = False  # Flag to wait for next inference after backup completes

        # Load TIC-VLA model
        self._load_ticvla_model()
        
        # Log initialization - note that _frame_count may not be set yet during on_init
        print("[NovaCarterTIC-VLA] Initialization complete.")


    def _create_log_directory(self):
        try:
            if not os.path.exists(self._log_directory):
                os.makedirs(self._log_directory)
                carb.log_info(f"Created log directory: {self._log_directory}")
        except Exception as e:
            carb.log_warn(f"Failed to create log directory: {e}")

    def on_play(self):
        """Called when simulation starts playing."""
        # Stop robot immediately to prevent any movement from previous episode
        self._cmd_v = 0.0
        self._cmd_w = 0.0
        with self._inference_lock:
            self._current_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # Apply stop command to wheels immediately
        if self._wheeled_robot and self._wheeled_robot.is_valid():
            try:
                action = self._diff.forward(command=np.array([0.0, 0.0], dtype=np.float32))
                self._wheeled_robot.apply_wheel_actions(action)
            except Exception as e:
                carb.log_warn(f"Failed to stop robot during reset: {e}")
        
        # CRITICAL: Reset goal and instruction BEFORE reading from environment
        # This ensures no stale values persist if environment variables are missing
        self._goal_position = None
        self._instruction = ""
        
        # Reset trajectory history for new episode
        self._trajectory_history = []
        self._robot_state_history = []
        self._heading_history = []
        self._trajectory_plotted = False
        
        self._delta_1s_sampled_1s = []
        self._prev_position_for_offset = None
        self._last_recorded_frame = -1
        self._prev_quaternion_for_offset = None
        self._offset_update_frame_count = 0
        
        # Reset VLM generation start tracking
        self._vlm_generation_start_positions = []
        self._vlm_generation_start_quaternions = []
        self._vlm_generation_start_frames = []
        self._first_inference_complete = False
        
        # --- Episode filesystem cleanup ---
        _empty_dir(self._log_directory)
        
        # Reset model's internal state (critical for clean episode transitions)
        # MUST happen before any new inference starts to prevent stale state
        if self._ticvla_model is not None:
            try:
                # Ensure any pending inference completes or is cancelled
                with self._inference_lock:
                    # Clear current action to prevent stale commands
                    self._current_action = np.array([0.0, 0.0], dtype=np.float32)
                
                # Reset model state (this will shutdown/recreate executor and clear model state)
                self._ticvla_model.reset_episode_state()
                carb.log_info("[NovaCarterTIC-VLA] Reset model state for new episode")
                
                # Increased delay to ensure executor cleanup and async inference completion
                # This is critical for sequential episodes to prevent state leakage
                time.sleep(0.2)  # Increased from 0.05 to 0.2 seconds
            except Exception as e:
                carb.log_warn(f"Failed to reset model state: {e}")
                import traceback
                carb.log_warn(traceback.format_exc())
        
        # Reset filtered states
        if hasattr(self, "_psi_des_filt"):
            delattr(self, "_psi_des_filt")
        
        # Reset previous states
        self._prev_position = None
        self._prev_orientation = None
        self._prev_R = None
        self._prev_heading = None
        
        # Reset stuck detection state
        self._last_movement_position = None
        self._last_movement_frame = None  # Will be initialized after first inference completes
        self._is_backing_up = False
        self._backup_start_frame = None
        self._backup_completion_frame = None
        self._waiting_for_inference_after_backup = False
        
        # Reset frame counter and image history for new episode
        old_image_count = len(self._image_history)
        self._frame_count = 0
        self._sensor_frame_count = -1
        self._last_log_frame = 0
        
        # Clear old images from previous episode
        if self._image_history:
            for img_path in self._image_history:
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except Exception as e:
                        carb.log_warn(f"Failed to remove image {img_path}: {e}")
            self._image_history.clear()
        print(f"[on_play] Reset episode state: frame_count=0, cleared {old_image_count} images")
        
        self._setup_for_play()
        try:
            if not getattr(self._wheeled_robot, "_initialized", False):
                self._wheeled_robot.initialize()
                carb.log_info(f"[{self.get_agent_name()}] WheeledRobot initialized in on_play()")
        except Exception as e:
            carb.log_warn(f"WheeledRobot.initialize failed in on_play: {e}")

        self._register_to_agent_manager()
        self._init_camera_capture()

        # Note: No longer using executor - calling _inference_task synchronously
        # The VLM generation still runs asynchronously internally via predict_async

    def on_stop(self):
        """Called when simulation stops."""
        print("[on_stop] Called")
        self._finalize_and_flush()
        print("[on_stop] Completed successfully")

    def on_destroy(self):
        """Called when script is destroyed."""
        print(f"[on_destroy] Called at frame {getattr(self, '_frame_count', 'unknown')}")
        # Call finalize if not already done
        self._finalize_and_flush()
        print("[on_destroy] Completed successfully")

    def _load_ticvla_model(self):
        if TICVLA is None:
            raise ImportError(
                "TICVLA is None. Import from ticvla failed. "
                "Check printed warning above for the real ImportError cause."
            )
        # Try to find model path from environment or configuration
        default_path = os.getenv('TICVLA_BASE_MODEL_PATH', os.getenv('TICVLA_MODEL_PATH', 'OpenGVLab/InternVL3-1B'))
        model_path = default_path

        # Load model using TIC-VLA constructor
        self._ticvla_model = TICVLA(model_path=model_path, device=self._device)

        # Try to load checkpoint if present
        ckpt_path = self._checkpoint_path
        state_dict = torch.load(ckpt_path, map_location='cpu')["state_dict"]
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
        self._ticvla_model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded successfully")
        self._ticvla_model.eval()

        # Use the same log directory for image buffering
        self._image_buffer_dir = self._log_directory
        print(f"Using log directory for image buffer: {self._image_buffer_dir}")
        self._model_loaded = True
        print("TIC-VLA model loaded successfully")

    def _alive(self) -> bool:
        # Check if timeline is playing
        tl = get_timeline_interface()
        if not tl or not tl.is_playing():
            return False
        # Stage must exist
        try:
            ctx = omni.usd.get_context()
            if not ctx or not ctx.get_stage():
                return False
        except Exception as e:
            # Stage has been destroyed (expected during shutdown)
            carb.log_warn(f"Exception checking stage (expected during shutdown): {e}")
            return False
        # Robot wrapper must still be valid
        return bool(self._wheeled_robot and self._wheeled_robot.is_valid())

    def _setup_for_play(self):
        """Set up navigation and settings for play mode."""
        self._navmeshEnabled = self.setting.get(PeopleSettings.NAVMESH_ENABLED)
        self._avoidanceOn = self.setting.get(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED)

        self._navigation_manager = RobotNavigationManager(
            str(self.prim_path), self._wheeled_robot, navmesh_enabled=False, dynamic_avoidance_enabled=self._avoidanceOn
        )

        # Event bus for agent registration
        self._agent_registered_event = carb.events.type_from_string(AgentEvent.AgentRegistered)
        self._bus = omni.kit.app.get_app().get_message_bus_event_stream()

        self._navigation_active = True

        # Configure from environment variables set by benchmark
        self._configure_from_environment()

        carb.log_info("[NovaCarterTIC-VLA] TIC-VLA navigation initialized")

    def _register_to_agent_manager(self):
        """Register this agent with the agent manager."""
        if self._bus and self._agent_registered_event:
            agent_name = self.get_agent_name()
            info = {"agent_name": str(agent_name), "prim_path": str(self.prim_path)}
            self._bus.push(self._agent_registered_event, payload=info)

    def _init_camera_capture(self):
        # Front and third-person camera handles
        self._camera_front_prim_path = None
        self._camera_tp_prim_path = None
        self._rp_front = None
        self._rp_tp = None
        self._rgb_annotator_front = None
        self._rgb_annotator_tp = None
        self._camera_image_front = None
        self._camera_image_tp = None

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        robot_prim = stage.GetPrimAtPath(str(self.prim_path))
        if not robot_prim or not robot_prim.IsValid():
            carb.log_warn("Robot prim not found for camera discovery")
            return

        self._camera_front_prim_path = "/World/Robots/Nova_Carter/chassis_link/front_hawk/left/camera_left"
        self._camera_tp_prim_path = "/World/Robots/Nova_Carter/chassis_link/third_person_view_cam"

        # Create render products and attach annotators for any that exist
        if self._camera_front_prim_path is not None:
            try:
                self._rp_front = rep.create.render_product(self._camera_front_prim_path, resolution=(1920, 1080))
                self._rgb_annotator_front = rep.AnnotatorRegistry.get_annotator("rgb")
                self._rgb_annotator_front.attach([self._rp_front])
                carb.log_info(f"Attached RGB annotator to front camera: {self._camera_front_prim_path}")
            except Exception as e:
                carb.log_warn(f"Failed attaching front RGB annotator: {e}")

        if self._camera_tp_prim_path is not None:
            try:
                self._rp_tp = rep.create.render_product(self._camera_tp_prim_path, resolution=(1920, 1080))
                self._rgb_annotator_tp = rep.AnnotatorRegistry.get_annotator("rgb")
                self._rgb_annotator_tp.attach([self._rp_tp])
                carb.log_info(f"Attached RGB annotator to third-person camera: {self._camera_tp_prim_path}")
            except Exception as e:
                carb.log_warn(f"Failed attaching TP RGB annotator: {e}")

    def _teardown_camera_capture(self):
        try:
            if getattr(self, "_rgb_annotator_front", None) is not None:
                try:
                    self._rgb_annotator_front.detach()
                except Exception as e:
                    carb.log_warn(f"Failed to detach front RGB annotator: {e}")
            if getattr(self, "_rgb_annotator_tp", None) is not None:
                try:
                    self._rgb_annotator_tp.detach()
                except Exception as e:
                    carb.log_warn(f"Failed to detach TP RGB annotator: {e}")
            self._rgb_annotator_front = None
            self._rgb_annotator_tp = None
            self._rp_front = None
            self._rp_tp = None
            self._camera_front_prim_path = None
            self._camera_tp_prim_path = None
            self._camera_image_front = None
            self._camera_image_tp = None
        except Exception as e:
            carb.log_warn(f"Camera capture teardown failed: {e}")

    def _capture_camera_image(self):
        # Front camera
        if getattr(self, "_rgb_annotator_front", None) is not None:
            try:
                data_front = self._rgb_annotator_front.get_data()
                if data_front is None or not hasattr(data_front, "shape") or data_front.size == 0:
                    self._camera_image_front = None
                else:
                    self._camera_image_front = data_front
            except Exception as e:
                carb.log_warn(f"Failed to get front camera data: {e}")
                self._camera_image_front = None

        # Third person camera
        if getattr(self, "_rgb_annotator_tp", None) is not None:
            try:
                data_tp = self._rgb_annotator_tp.get_data()
                if data_tp is None or not hasattr(data_tp, "shape") or data_tp.size == 0:
                    self._camera_image_tp = None
                else:
                    self._camera_image_tp = data_tp
            except Exception as e:
                carb.log_warn(f"Failed to get TP camera data: {e}")
                self._camera_image_tp = None

    def _configure_from_environment(self):
        """Configure TIC-VLA behavior from environment variables set by benchmark."""
        try:
            # Read goal position from environment variables
            goal_x = os.getenv('TICVLA_GOAL_X')
            goal_y = os.getenv('TICVLA_GOAL_Y')
            goal_z = os.getenv('TICVLA_GOAL_Z', '0.0')

            if goal_x is not None and goal_y is not None:
                goal_position = np.array([float(goal_x), float(goal_y), float(goal_z)])
                self._goal_position = goal_position
                carb.log_info(f"Goal position configured from environment: {goal_position}")

            # Read instruction from environment variables
            instruction = os.getenv('TICVLA_INSTRUCTION')
            if instruction is not None:
                self._instruction = instruction
                carb.log_info(f"Instruction configured from environment: {instruction}")

        except Exception as e:
            carb.log_warn(f"Failed to configure from environment: {e}")

    def _robot_ready(self):
        stage = omni.usd.get_context().get_stage()
        p = stage.GetPrimAtPath(str(self.prim_path))
        return p and p.IsValid()

    def _finalize_and_flush(self):
        if getattr(self, "_did_finalize", False):
            return
        self._did_finalize = True
        try:
            self._navigation_active = False
            # No executor to shut down - inference runs synchronously now
            # Tear down cameras after saving
            self._teardown_camera_capture()
            self._cleanup_images()
        except Exception as e:
            carb.log_warn(f"Finalize failed: {e}")

    def on_update(self, current_time: float, delta_time: float):
        """Main update loop for robot behavior."""
        # Skip if navigation is inactive (during teardown)
        if not self._navigation_active:
            return
        
        benchmark_frame_count_str = os.getenv('BENCHMARK_SIMULATION_FRAME_COUNT')
        if benchmark_frame_count_str is not None:
            try:
                self._frame_count = int(benchmark_frame_count_str)
            except (ValueError, TypeError):
                pass

        try:
            app = omni.kit.app.get_app()
            tl = get_timeline_interface()
            ctx = omni.usd.get_context()
            stage = ctx.get_stage() if ctx else None

            # If any teardown condition is true, finalize once and then return.
            if (app and getattr(app, "is_exiting", None) and app.is_exiting()) or not (tl and tl.is_playing()) or stage is None:
                self._finalize_and_flush()
                return
        except Exception as e:
            carb.log_warn(f"Exception during teardown check (expected during shutdown): {e}")
            self._finalize_and_flush()
            return
        
        if not self._initialized:
            try:
                self._wheeled_robot.initialize()
            except Exception as e:
                carb.log_error(f"Failed to initialize wheeled robot: {e}")
                import traceback
                carb.log_error(traceback.format_exc())
            self._initialized = True

        # Update robot kinematic state
        self._update_robot_state(delta_time)
        
        frame_idx = int(self._frame_count)
        if frame_idx % 30 == 0 and frame_idx != self._last_recorded_frame:
            if self._robot_position is not None:
                try:
                    p_cur, R_cur, quat_wxyz = self._get_pose_R_quat()
                    
                    # First tick: no previous frame yet
                    if self._prev_position_for_offset is None:
                        self._prev_position_for_offset = p_cur.copy()
                        self._prev_quaternion_for_offset = quat_wxyz.copy()
                        self._delta_1s_sampled_1s.append((frame_idx, [0.0, 0.0, 0.0]))
                        self._last_recorded_frame = frame_idx
                    else:
                        delta_world = p_cur - self._prev_position_for_offset
                        R_prev = quat_to_rot_matrix(self._prev_quaternion_for_offset)
                        vec_body = R_prev.T @ delta_world
                        vec_flu = vec_body.copy()
                        self._delta_1s_sampled_1s.append((frame_idx, vec_flu.tolist()))
                        self._prev_position_for_offset = p_cur.copy()
                        self._prev_quaternion_for_offset = quat_wxyz.copy()
                        self._last_recorded_frame = frame_idx
                except Exception as e:
                    carb.log_warn(f"Failed to update 1s delta: {e}")

        # Capture images from existing cameras
        self._capture_camera_image()

        # Stuck detection and recovery logic (run BEFORE inference to prevent inference during backup)
        # Only start stuck detection AFTER first inference completes (don't count the initial stationary period)
        
        # Check if currently backing up
        if self._is_backing_up:
            # Check if backup duration has elapsed using frame count
            if self._backup_start_frame is not None:
                backup_elapsed_frames = self._frame_count - self._backup_start_frame
                
                # Log progress every 30 frames (approximately every second at 30 FPS)
                if backup_elapsed_frames > 0 and backup_elapsed_frames % 30 == 0 and backup_elapsed_frames < self._backup_duration_frames:
                    remaining_frames = self._backup_duration_frames - backup_elapsed_frames
                    remaining_time = remaining_frames / 30.0
                    carb.log_info(f"[NovaCarterTIC-VLA] Backing up progress: {backup_elapsed_frames}/{self._backup_duration_frames} frames ({remaining_time:.1f}s remaining, frame={self._frame_count}, start_frame={self._backup_start_frame})")
                
                # Check if backup duration has elapsed - do this check every frame
                if backup_elapsed_frames >= self._backup_duration_frames:
                    # Backup complete - stop backing up and wait for next inference before resuming
                    self._backup_completion_frame = self._frame_count  # Remember when backup completed
                    backup_start_frame_saved = self._backup_start_frame  # Save for logging
                    self._is_backing_up = False
                    self._backup_start_frame = None
                    self._waiting_for_inference_after_backup = True  # Wait for fresh inference before resuming
                    # Clear stale commands - will use zero commands until next inference
                    with self._inference_lock:
                        self._current_action = np.array([0.0, 0.0], dtype=np.float32)
                    # Reset movement tracking after backup completes - but DON'T initialize it yet
                    # We'll initialize it when fresh inference arrives and normal operation resumes
                    # This prevents stuck detection from triggering during the wait period
                    self._last_movement_position = None
                    self._last_movement_frame = None
                    carb.log_warn(f"[NovaCarterTIC-VLA] *** Backup recovery complete after {backup_elapsed_frames} frames ({self._backup_duration_frames} frames requested, completed at frame={self._frame_count}, backup started at frame={backup_start_frame_saved}), waiting for next inference that starts AFTER frame {self._backup_completion_frame} before resuming normal operation ***")
            else:
                # This shouldn't happen, but log it for debugging
                carb.log_warn(f"[NovaCarterTIC-VLA] WARNING: _is_backing_up=True but _backup_start_frame is None at frame {self._frame_count}")
        
        # Only do stuck detection if first inference has completed (don't count initial stationary period)
        # Also skip stuck detection while waiting for inference after backup (robot is intentionally stationary)
        if self._first_inference_complete and not self._is_backing_up and not self._waiting_for_inference_after_backup:
            # Check if robot has moved significantly
            if self._robot_position is not None:
                if self._last_movement_position is None or self._last_movement_frame is None:
                    # Initialize tracking on first frame after inference completes
                    self._last_movement_position = self._robot_position.copy()
                    self._last_movement_frame = self._frame_count
                else:
                    # Calculate 2D distance (ignore Z)
                    position_2d = self._robot_position[:2]
                    last_position_2d = self._last_movement_position[:2]
                    distance_moved = np.linalg.norm(position_2d - last_position_2d)
                    
                    if distance_moved >= self._stuck_position_threshold:
                        # Robot has moved significantly - reset stuck tracking
                        self._last_movement_position = self._robot_position.copy()
                        self._last_movement_frame = self._frame_count
                        # If we were backing up and now moving, stop backup (shouldn't happen here since we check not backing_up above)
                        if self._is_backing_up:
                            backup_frames = self._frame_count - self._backup_start_frame if self._backup_start_frame is not None else 0
                            self._is_backing_up = False
                            self._backup_start_frame = None
                            carb.log_info(f"[NovaCarterTIC-VLA] Robot unstuck after {backup_frames} frames of backup, stopping backup recovery")
            
            # Detect if stuck (not moved for threshold frames)
            if self._last_movement_frame is not None:
                frames_since_movement = self._frame_count - self._last_movement_frame
                if frames_since_movement >= self._stuck_time_threshold_frames:
                    stuck_time_seconds = frames_since_movement / 30.0
                    failure_reason = (
                        f"robot_stuck:not moved for {frames_since_movement} frames "
                        f"/ {stuck_time_seconds:.2f}s"
                    )
                    carb.log_warn(f"[NovaCarterTIC-VLA] Robot stuck detected ({failure_reason}); ending episode as failure")
                    os.environ["TICVLA_EPISODE_FAILURE_REASON"] = failure_reason
                    self._navigation_active = False
                    self._is_backing_up = False
                    self._waiting_for_inference_after_backup = False
                    self._backup_start_frame = None
                    self._backup_completion_frame = None
                    self._stop_robot()
                    return

        # Sensor/update cadence for model
        self._sensor_frame_count += 1
        if self._sensor_frame_count % max(1, (30 // self._sensor_update_frequency)) == 0:
            # Buffer the latest front image to disk for the model (main thread does file write)
            if self._image_buffer_dir is not None and self._camera_image_front is not None:

                img = self._camera_image_front
                # Check if image is valid (not empty)
                if img is None or img.size == 0:
                    carb.log_warn("Camera image is empty, skipping save")
                else:
                    if img.ndim == 3 and img.shape[2] == 4:
                        img = img[:, :, :3]
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    filename_front = f"{self._image_buffer_dir}/front_frame_{self._frame_count:06d}.jpg"
                    # Ensure directory exists before saving
                    os.makedirs(self._image_buffer_dir, exist_ok=True)
                    # Only append to history after image is successfully saved
                    if cv2.imwrite(filename_front, img_bgr):
                        self._image_history.append(filename_front)
                    else:
                        carb.log_warn(f"Failed to save image: {filename_front} (directory: {self._image_buffer_dir}, exists: {os.path.exists(self._image_buffer_dir)})")
                    if len(self._image_history) > self._max_history_frames:
                        oldest = self._image_history.pop(0)
                        if os.path.exists(oldest):
                            try:
                                os.remove(oldest)
                            except Exception as e:
                                carb.log_warn(f"Failed to remove old image {oldest}: {e}")

            # Save third-person view camera image to logs
            if self._image_buffer_dir is not None and self._camera_image_tp is not None:
                img_tp = self._camera_image_tp
                if img_tp is not None and img_tp.size > 0:
                    if img_tp.ndim == 3 and img_tp.shape[2] == 4:
                        img_tp = img_tp[:, :, :3]
                    img_tp_bgr = cv2.cvtColor(img_tp, cv2.COLOR_RGB2BGR)
                    filename_tp = f"{self._image_buffer_dir}/tp_frame_{self._frame_count:06d}.jpg"
                    os.makedirs(self._image_buffer_dir, exist_ok=True)
                    if not cv2.imwrite(filename_tp, img_tp_bgr):
                        carb.log_warn(f"Failed to save third-person view image: {filename_tp}")

            # Decide whether to run (or schedule) TIC-VLA inference
            # Skip inference during backup recovery, but allow it when waiting for inference after backup
            have_images = len(self._image_history) > 0 and self._model_loaded and (self._ticvla_model is not None) and not self._is_backing_up
            if have_images:
                # Get current robot pose for dx, dy calculation and waypoint transformation
                p_cur, R_cur, quat_wxyz = self._get_pose_R_quat()
                robot_pose = {
                    'position': p_cur.tolist(),
                    'quaternion': quat_wxyz.tolist(),
                    'rotation_matrix': R_cur.tolist()
                }
                
                # _get_sampled_image_paths() returns [oldest (-9s), -6s, -3s, current] (newest at last)
                sampled_paths = self._get_sampled_image_paths()
                # Validate that all image paths exist before calling inference
            
                if sampled_paths:
                    if len(self._vlm_generation_start_frames) > 0:
                        ref_position = self._vlm_generation_start_positions[0]
                        ref_quaternion = self._vlm_generation_start_quaternions[0]
                        ref_vlm_generation_start_frame = self._vlm_generation_start_frames[0]
                        
                        # Compute displacement in world frame
                        delta_world = p_cur - ref_position
                        # Transform to VLM generation start frame (delayed ego FLU)
                        R_start = quat_to_rot_matrix(ref_quaternion)
                        vec_body = R_start.T @ delta_world
                        vec_flu = vec_body.copy()
                        dx = float(vec_flu[0])
                        dy = float(vec_flu[1])
                        
                        # Calculate delay from frame difference (same as benchmark: frames / 30.0)
                        step_delay = self._frame_count - ref_vlm_generation_start_frame
                        delay_time = float(step_delay) / 30.0
                    else:
                        dx, dy = 0.0, 0.0
                        delay_time = 0.0
                    
                    # Robot state: send all available state [vx, vy, vz, yaw_speed, dx, dy]
                    # Model will select what it needs: [vx, vy, vz, yaw_speed] for waypoint version
                    robot_state = (
                        float(self._robot_velocity[0]),
                        float(self._robot_velocity[1]),
                        float(self._robot_velocity[2]) if len(self._robot_velocity) > 2 else 0.0,
                        float(self._robot_yaw_speed),
                        dx,
                        dy,
                    )
                    
                    # Validate inputs before calling inference
                    if not sampled_paths:
                        raise ValueError("[NovaCarterTIC-VLA] ERROR: No image paths provided for inference!")
                    if not self._instruction or len(self._instruction.strip()) == 0:
                        raise ValueError("[NovaCarterTIC-VLA] ERROR: Empty or missing instruction!")
                    
                    # Prepare waypoints: filter out initial (0,0,0) and convert to (relative_time, waypoint) format
                    elapsed_time = float(self._frame_count) / 30.0
                    raw_waypoints = list(self._delta_1s_sampled_1s)
                    # Filter out initial (0,0,0) waypoint and convert to (relative_time, waypoint) format
                    previous_waypoints = []
                    for step, waypoint in raw_waypoints:
                        # Skip initial (0,0,0) waypoint
                        if abs(waypoint[0]) < 1e-6 and abs(waypoint[1]) < 1e-6 and abs(waypoint[2]) < 1e-6:
                            continue
                        relative_time = float(step) / 30.0
                        previous_waypoints.append((relative_time, waypoint))
                    
                    previous_waypoints_text = self._format_previous_waypoints_text(previous_waypoints, elapsed_time)
                    
                    # Validate previous waypoints text format
                    if not previous_waypoints_text or len(previous_waypoints_text.strip()) == 0:
                        raise ValueError("[NovaCarterTIC-VLA] ERROR: Empty previous waypoints text!")
                    
                    # Call inference synchronously (like old script) - this blocks until inference completes
                    # This ensures inference runs at exactly 10Hz with fresh actions every cycle
                    # The VLM generation still runs asynchronously internally via predict_async
                    # Double-check backup status right before calling (might have changed during validation)
                    if self._is_backing_up:
                        carb.log_info("[NovaCarterTIC-VLA] Skipping inference - backup in progress")
                    else:
                        try:
                            vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose = self._inference_task(
                                sampled_paths,
                                self._instruction,
                                robot_state,
                                self._frame_count,
                                robot_pose,
                                previous_waypoints_text,
                                delay_time,
                                "wheeled robot"
                            )
                            
                            # If we were waiting for inference after backup, check BEFORE adding new generation to list
                            # This way we check the generation that actually produced these waypoints
                            if self._waiting_for_inference_after_backup and self._backup_completion_frame is not None:
                                backup_completion_frame = self._backup_completion_frame
                                carb.log_warn(f"[NovaCarterTIC-VLA] *** WAITING CHECK: frame={self._frame_count}, vlm_gen_start_step={vlm_generation_start_step}, tracking_list={self._vlm_generation_start_frames}, backup_completed_at={backup_completion_frame} ***")
                                
                                # The waypoints we just received are from the most recently COMPLETED inference
                                # If a new generation just started, these waypoints are from the CURRENT most recent entry (before we add the new one)
                                # If no new generation started, these waypoints are from the most recent tracked generation
                                if vlm_generation_start_step is not None:
                                    # New generation started - waypoints are from the CURRENT most recent tracked generation (before adding new one)
                                    if len(self._vlm_generation_start_frames) > 0:
                                        completed_gen_start = self._vlm_generation_start_frames[-1]
                                        carb.log_warn(f"[NovaCarterTIC-VLA] *** CHECKING RESUME: completed_gen_start={completed_gen_start}, backup_completion_frame={backup_completion_frame}, comparison={completed_gen_start >= backup_completion_frame} (frame={self._frame_count}, new_gen={vlm_generation_start_step}) ***")
                                        if completed_gen_start >= backup_completion_frame:
                                            # The completed generation started after backup - these waypoints are fresh!
                                            self._waiting_for_inference_after_backup = False
                                            self._backup_completion_frame = None
                                            # Re-initialize movement tracking for fresh navigation (prevents stuck detection from triggering immediately)
                                            if self._robot_position is not None:
                                                self._last_movement_position = self._robot_position.copy()
                                                self._last_movement_frame = self._frame_count
                                            carb.log_warn(f"[NovaCarterTIC-VLA] *** ✅✅✅ Fresh waypoints from generation that started at {completed_gen_start} (>= backup completion {backup_completion_frame}). New gen started at {vlm_generation_start_step}. RESUMING NORMAL OPERATION! (frame={self._frame_count}) ✅✅✅ ***")
                                        else:
                                            carb.log_warn(f"[NovaCarterTIC-VLA] *** DISCARDING waypoints: Generation that started at {completed_gen_start} (before backup at {backup_completion_frame}) just completed. New gen started at {vlm_generation_start_step}. Still waiting... (frame={self._frame_count}) ***")
                                    else:
                                        carb.log_warn(f"[NovaCarterTIC-VLA] *** DISCARDING waypoints: New gen started at {vlm_generation_start_step} but no tracked generations yet (list is empty). Still waiting... (frame={self._frame_count}) ***")
                                else:
                                    # No new generation started - waypoints are from the most recent tracked generation
                                    if len(self._vlm_generation_start_frames) > 0:
                                        most_recent_gen_start = self._vlm_generation_start_frames[-1]
                                        carb.log_warn(f"[NovaCarterTIC-VLA] *** CHECKING RESUME (no new gen): most_recent_gen_start={most_recent_gen_start}, backup_completion_frame={backup_completion_frame}, comparison={most_recent_gen_start >= backup_completion_frame} (frame={self._frame_count}) ***")
                                        if most_recent_gen_start >= backup_completion_frame:
                                            self._waiting_for_inference_after_backup = False
                                            self._backup_completion_frame = None
                                            # Re-initialize movement tracking for fresh navigation (prevents stuck detection from triggering immediately)
                                            if self._robot_position is not None:
                                                self._last_movement_position = self._robot_position.copy()
                                                self._last_movement_frame = self._frame_count
                                            carb.log_warn(f"[NovaCarterTIC-VLA] *** ✅✅✅ Fresh waypoints from generation that started at {most_recent_gen_start} (>= backup completion {backup_completion_frame}). RESUMING NORMAL OPERATION! (frame={self._frame_count}) ✅✅✅ ***")
                                        else:
                                            carb.log_warn(f"[NovaCarterTIC-VLA] *** DISCARDING waypoints: Generation that started at {most_recent_gen_start} (before backup at {backup_completion_frame}). Still waiting... (frame={self._frame_count}) ***")
                                    else:
                                        carb.log_warn(f"[NovaCarterTIC-VLA] *** DISCARDING waypoints: No tracked generations, likely from before backup (completed at {backup_completion_frame}). Still waiting... (frame={self._frame_count}) ***")
                            
                            # If a new generation started, store the pose that was captured in the model
                            # (captured in _start_kv_cache_generation when generation actually starts)
                            # Do this AFTER the waiting check so we can check the generation that produced the waypoints
                            if vlm_generation_start_step is not None and vlm_generation_start_pose is not None:
                                # Store the pose that was captured when generation actually started (matches the generation step)
                                pos = vlm_generation_start_pose.get('position')
                                quat = vlm_generation_start_pose.get('quaternion')
                                if pos is not None and quat is not None:
                                    # Convert to numpy arrays if they're not already
                                    if isinstance(pos, list):
                                        pos = np.array(pos)
                                    if isinstance(quat, list):
                                        quat = np.array(quat)
                                    self._vlm_generation_start_positions.append(pos.copy())
                                    self._vlm_generation_start_quaternions.append(quat.copy())
                                    self._vlm_generation_start_frames.append(vlm_generation_start_step)
                                    
                                    # Debug: log when adding generation to tracking list
                                    if self._waiting_for_inference_after_backup:
                                        carb.log_warn(f"[NovaCarterTIC-VLA] *** TRACKING: Added new generation {vlm_generation_start_step} to list. Current list: {self._vlm_generation_start_frames}, backup_completed_at={self._backup_completion_frame} (frame={self._frame_count}) ***")
                                    
                                    # Keep only the 2 most recent entries
                                    if len(self._vlm_generation_start_frames) > 2:
                                        self._vlm_generation_start_frames.pop(0)
                                        self._vlm_generation_start_positions.pop(0)
                                        self._vlm_generation_start_quaternions.pop(0)
                            
                            if not self._first_inference_complete and kv_cache_available:
                                self._first_inference_complete = True
                                # Initialize stuck detection tracking now that inference has completed
                                if self._robot_position is not None:
                                    self._last_movement_position = self._robot_position.copy()
                                    self._last_movement_frame = self._frame_count
                                carb.log_info("[NovaCarterTIC-VLA] First VLM generation completed, model state available, starting movement and stuck detection")
                        except Exception as e:
                            carb.log_warn(f"Inference task failed: {e}")
                            import traceback
                            carb.log_warn(traceback.format_exc())

        
        # Command selection priority: Backup > Waiting for inference after backup > First inference check > Model commands
        if self._is_backing_up:
            # Backup recovery takes priority - force backup commands regardless of inference state
            # Override commands with backup motion: straight backward, no rotation
            # Force velocities immediately (no slew during backup) to ensure immediate backward motion
            # Ignore any model commands - force backup behavior
            self._cmd_v = self._backup_speed
            self._cmd_w = 0.0
            # Log every 10 frames during backup (reduce log spam but still verify it's working)
            elapsed = self._frame_count - self._backup_start_frame if self._backup_start_frame is not None else -1
            if self._frame_count % 10 == 0:
                carb.log_info(f"[NovaCarterTIC-VLA] *** BACKING UP ***: frame={self._frame_count}, cmd_v={self._cmd_v:.3f}, cmd_w={self._cmd_w:.3f}, backup_start_frame={self._backup_start_frame}, elapsed_frames={elapsed}/{self._backup_duration_frames}")
        elif self._waiting_for_inference_after_backup:
            # After backup completes, wait for next inference before resuming - keep robot stationary
            # This ensures we use fresh navigation commands, not stale ones from before backup
            target_v = 0.0
            target_w = 0.0
            self._cmd_v = self._slew(self._cmd_v, target_v, self._max_accel_lin, self._max_decel_lin, delta_time, self._lin_deadband)
            self._cmd_w = self._slew(self._cmd_w, target_w, self._max_accel_ang, self._max_decel_ang, delta_time, self._ang_deadband)
            if self._frame_count % 30 == 0:  # Log once per second
                carb.log_warn(f"[NovaCarterTIC-VLA] *** WAITING FOR FRESH INFERENCE AFTER BACKUP ***: frame={self._frame_count}, backup_completion_frame={self._backup_completion_frame} - robot stationary")
        elif not self._first_inference_complete:
            # Keep robot stationary until first inference completes (but not if backing up - handled above)
            target_v = 0.0
            target_w = 0.0
            self._cmd_v = self._slew(self._cmd_v, target_v, self._max_accel_lin, self._max_decel_lin, delta_time, self._lin_deadband)
            self._cmd_w = self._slew(self._cmd_w, target_w, self._max_accel_ang, self._max_decel_ang, delta_time, self._ang_deadband)
        else:
            # Convert model action to smoothed wheel commands (protected by lock)
            with self._inference_lock:
                target_v = float(self._current_action[0])
                target_w = float(self._current_action[1])
            
            # Apply smoothing only when NOT backing up
            self._cmd_v = self._slew(self._cmd_v, target_v, self._max_accel_lin, self._max_decel_lin, delta_time, self._lin_deadband)
            self._cmd_w = self._slew(self._cmd_w, target_w, self._max_accel_ang, self._max_decel_ang, delta_time, self._ang_deadband)

        # Final safety check: if backing up, ensure commands are correct (CRITICAL - do this right before applying)
        if self._is_backing_up:
            self._cmd_v = self._backup_speed
            self._cmd_w = 0.0
            # Also clear any model actions to prevent them from being used next frame
            with self._inference_lock:
                self._current_action = np.array([self._backup_speed, 0.0], dtype=np.float32)
            carb.log_info(f"[NovaCarterTIC-VLA] Final backup check: frame={self._frame_count}, forcing cmd_v={self._cmd_v:.3f}, cmd_w={self._cmd_w:.3f}")
        
        # Additional safety: check if we should still be backing up (in case completion check was missed)
        if self._backup_start_frame is not None:
            backup_elapsed = self._frame_count - self._backup_start_frame
            if backup_elapsed >= self._backup_duration_frames and self._is_backing_up:
                # Backup should have completed but didn't - force completion now
                carb.log_warn(f"[NovaCarterTIC-VLA] *** SAFETY: Backup should have completed! frame={self._frame_count}, elapsed={backup_elapsed}, duration={self._backup_duration_frames}. Forcing completion now. ***")
                self._backup_completion_frame = self._frame_count
                backup_start_saved = self._backup_start_frame
                self._is_backing_up = False
                self._backup_start_frame = None
                self._waiting_for_inference_after_backup = True
                with self._inference_lock:
                    self._current_action = np.array([0.0, 0.0], dtype=np.float32)
                if self._robot_position is not None:
                    self._last_movement_position = self._robot_position.copy()
                    self._last_movement_frame = self._frame_count
        
        # Push command to wheels
        if self._wheeled_robot and self._wheeled_robot.is_valid():
            try:
                # One final check before sending to hardware - backup and waiting always win
                if self._is_backing_up:
                    final_v = self._backup_speed
                    final_w = 0.0
                    carb.log_info(f"[NovaCarterTIC-VLA] Sending backup command to wheels: v={final_v:.3f}, w={final_w:.3f}, frame={self._frame_count}")
                elif self._waiting_for_inference_after_backup:
                    final_v = 0.0
                    final_w = 0.0
                    if self._frame_count % 30 == 0:
                        carb.log_warn(f"[NovaCarterTIC-VLA] *** WAITING FOR INFERENCE - forcing zero commands: v={final_v:.3f}, w={final_w:.3f}, frame={self._frame_count} ***")
                else:
                    final_v = self._cmd_v
                    final_w = self._cmd_w
                
                action = self._diff.forward(command=np.array([final_v, final_w], dtype=np.float32))
                self._wheeled_robot.apply_wheel_actions(action)
            except Exception as e:
                carb.log_error(f"Failed to apply wheel actions: {e}")
                import traceback
                carb.log_error(traceback.format_exc())
                return

        # Stream data and log
        self._stream_robot_data()

        # Publish position for avoidance if enabled
        if self._avoidanceOn and self._navigation_manager is not None:
            try:
                self._navigation_manager.publish_robot_position(delta_time, radius=2.0)
            except Exception as e:
                carb.log_warn(f"Failed to publish robot position (possibly during teardown): {e}")

    def _get_pose_R_quat(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._base_link_path)
        M = omni.usd.get_world_transform_matrix(prim)
        p = np.array(M.ExtractTranslation(), dtype=np.float64)
        R_native = np.array(M, dtype=np.float64)[:3, :3]
        rot = M.ExtractRotation()
        q = rot.GetQuat()
        quat_wxyz = np.array([q.GetReal(), *q.GetImaginary()], dtype=np.float64)
        return p, R_native, quat_wxyz

    def _update_robot_state(self, delta_time: float):
        stage = omni.usd.get_context().get_stage()
        if stage is None: return
        prim = stage.GetPrimAtPath(self._base_link_path)
        if not (prim and prim.IsValid()): return

        p_cur, R_cur_native, quat_wxyz = self._get_pose_R_quat()
        self._robot_position = p_cur
        self._robot_quaternion = quat_wxyz

        # Get velocities directly from Isaac Sim API (world frame)
        lin_vel_I = self._wheeled_robot.get_linear_velocity()
        ang_vel_I = self._wheeled_robot.get_angular_velocity()
        
        # If velocities are None (first frame or not yet available), zero them
        if lin_vel_I is None: lin_vel_I = np.zeros(3, dtype=np.float64)
        if ang_vel_I is None: ang_vel_I = np.zeros(3, dtype=np.float64)
        
        # Ensure velocities are numpy arrays with correct dtype
        lin_vel_I = np.asarray(lin_vel_I, dtype=np.float64)
        ang_vel_I = np.asarray(ang_vel_I, dtype=np.float64)
        
        pos_IB, q_IB = self._wheeled_robot.get_world_pose()
        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)

        lin_vel_b[1] = -lin_vel_b[1]

        # Assign body-frame velocities to robot state
        self._robot_velocity = lin_vel_b.astype(np.float64)
        self._robot_yaw_speed = float(ang_vel_b[2])

        # Heading for plotting
        R_cur_flu = R_cur_native
        fwd = R_cur_flu[:, 0]
        self._current_heading = float(np.arctan2(fwd[1], fwd[0]))

        self._prev_R = R_cur_native
        self._prev_position = p_cur

            
    # --- Data streaming -------------------------------------------------------
    def _stream_robot_data(self):
        if not self._stream_data:
            return

        robot_state = {
            "frame_count": self._frame_count,
            "position": self._robot_position.tolist(),
            "quaternion": self._robot_quaternion.tolist(),
            "velocity": self._robot_velocity.tolist(),
            "yaw_speed": float(self._robot_yaw_speed),
            "commands": {
                "v": float(self._cmd_v),
                "w": float(self._cmd_w),
            },
        }

        if self._frame_count - self._last_log_frame >= self._log_interval_frames:
            self._log_robot_state(robot_state)
            self._last_log_frame = self._frame_count
        
        # Update trajectory history
        if self._robot_position is not None and len(self._robot_position) >= 2:
            self._trajectory_history.append([self._robot_position[0], self._robot_position[1]])
        
        # Update robot state history
        self._robot_state_history.append([
            float(self._robot_velocity[0]),
            float(self._robot_velocity[1]),
            float(self._robot_velocity[2]),
            float(self._robot_yaw_speed)
        ])
        
        # Update heading history
        self._heading_history.append(self._current_heading)

    def _log_robot_state(self, robot_state: Dict[str, Any]):
        filename = f"{self._log_directory}/robot_state_frame_{self._frame_count:06d}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(robot_state, f, indent=2)
        except Exception as e:
            carb.log_warn(f"Failed to write robot state log: {e}")
    
    def _cleanup_images(self):
        """Clean up TIC-VLA images after episode ends."""
        if self._image_history:
            try:
                removed_count = 0
                for img_path in self._image_history:
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                            removed_count += 1
                        except Exception as e:
                            carb.log_warn(f"Failed to remove image {img_path}: {e}")
                carb.log_info(f"Cleaned up {removed_count} TIC-VLA images")
                self._image_history.clear()
            except Exception as e:
                carb.log_warn(f"Failed to cleanup images: {e}")

    # --- Image sampling -------------------------------------------------------
    def _get_sampled_image_paths(self):
        """
        Return up to 4 image paths sampled at 3-second intervals: current, -3s, -6s, -9s.
        Each image is 3 seconds (30 frames at 10 Hz) apart.
        Returns in order: [oldest (-9s), -6s, -3s, current] (newest at last).
        
        Raises ValueError if images are missing or invalid.
        """
        hist = self._image_history
        n = len(hist)
        if n == 0:
            raise ValueError("[NovaCarterTIC-VLA] ERROR: No images in history! Cannot perform inference without images.")
        
        # At 10 Hz sensor frequency, images are saved every 0.1s (every 3 frames at 30 Hz)
        # To get 3-second intervals, we need 3 seconds = 30 sensor updates = 30 entries in history
        # So we sample every 30 entries in _image_history to get 3-second intervals
        step_entries = 30  # 30 entries = 3 seconds at 10 Hz (each entry is 0.1s apart)
        
        def pick(offset_entries: int):
            """Pick image offset_entries back from the most recent entry in history."""
            idx = max(0, n - 1 - offset_entries)
            return hist[idx]
        
        # Sample at 3-second intervals: current (0), -3s (30 entries), -6s (60 entries), -9s (90 entries)
        # Return in order: [oldest (-9s), -6s, -3s, current] (newest at last)
        # Only sample images that are actually 30 entries apart (3 seconds)
        sampled = []
        offsets = [step_entries * 3, step_entries * 2, step_entries * 1, 0]  # -9s, -6s, -3s, current
        
        # Sample at 3-second intervals (every 30 entries)
        # Only include images if we have enough history to actually get 3-second intervals
        for offset in offsets:
            # Only sample if we have enough history for this offset
            if n - 1 - offset >= 0:  # We have enough history for this offset
                idx = n - 1 - offset
                picked = hist[idx]
                sampled.append(picked)
            elif offset == 0:  # Always include current (offset 0)
                idx = n - 1
                picked = hist[idx]
                sampled.append(picked)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sampled = []
        for img in sampled:
            if img not in seen:
                seen.add(img)
                unique_sampled.append(img)
        
        # Validate all images exist
        for img_path in unique_sampled:
            if not os.path.exists(img_path):
                raise ValueError(f"[NovaCarterTIC-VLA] ERROR: Image file does not exist: {img_path}")
            if not os.path.isfile(img_path):
                raise ValueError(f"[NovaCarterTIC-VLA] ERROR: Image path is not a file: {img_path}")
        
        if len(unique_sampled) == 0:
            raise ValueError("[NovaCarterTIC-VLA] ERROR: No valid images sampled! Cannot perform inference.")
        
        return unique_sampled

    # --- Previous waypoints formatting ---
    def _format_previous_waypoints_text(self, previous_waypoints: List[tuple[float, List[float]]], elapsed_time: float) -> str:
        """Format previous waypoints into text describing execution times and points."""
        lines = []
        
        if previous_waypoints:
            waypoint_strs = []
            for relative_time, waypoint in previous_waypoints:
                x, y, z = waypoint
                waypoint_strs.append(f"({x:.2f}, {y:.2f}, {z:.2f})")
            waypoint_list = ", ".join(waypoint_strs)
            
            lines.append(f"From 0.0s to current timestamp time is {elapsed_time:.1f}s. (a list of waypoints 1s in between): {waypoint_list}")
            lines.append("Each waypoint (x, y, z) is the displacement over the previous 1.0s. x is forward, y is left, z is up.")
        else:
            lines.append(f"From 0.0s to current timestamp time is {elapsed_time:.1f}s. No waypoints available.")
        
        return "\n".join(lines)
    
    def _inference_task(self, image_paths, instruction, robot_state_tuple, current_step: int, robot_pose: dict | None = None, previous_waypoints_text: str = "", delay_time: float = 0.0, robot_type: str = "wheeled robot"):
        """
        Inference task: runs TICVLA.predict_async synchronously and returns waypoints and VLM generation start step.
        Called directly from main thread (synchronous execution) to ensure 10Hz timing.
        The VLM generation still runs asynchronously internally via predict_async.
        
        Args:
            image_paths: List of image file paths
            instruction: Navigation instruction
            robot_state_tuple: Robot state (vx, vy, vz, yaw_speed, dx, dy) - all available state, model selects what it needs
            current_step: Current simulation step/frame number (used to calculate delay in steps)
            robot_pose: Current robot pose {'position': [x,y,z], 'quaternion': [w,x,y,z], 'rotation_matrix': 3x3}
                       Used to transform waypoint from generation frame to current frame.
            previous_waypoints_text: Formatted text describing previous waypoints at 1s intervals (matches training format).
            delay_time: Time delay between current frame and second-to-last VLM generation start (in seconds).
            robot_type: Type of robot (e.g., "wheeled robot", "legged robot")
        
        Returns:
            (vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose):
            - vlm_generation_start_step: Frame number when VLM generation started (None if no new generation started)
            - kv_cache_available: True if model state is available (at least one generation has completed), False otherwise
            - vlm_generation_start_pose: Pose (position, quaternion) captured in model when generation actually started, or None
                                       This pose matches the vlm_generation_start_step exactly (captured in _start_kv_cache_generation)
        """
        # Early exit if backup is in progress - don't run inference during backup
        if self._is_backing_up:
            carb.log_info(f"[NovaCarterTIC-VLA] Inference task aborted - backup in progress (frame={current_step})")
            return None, False, None
        
        if self._ticvla_model is None:
            return None, False, None

        # Note: Pose is now captured in the model when generation actually starts
        # (in _start_kv_cache_generation), so we don't need to capture it here

        # Convert robot_state_tuple back to tensor on worker thread
        # robot_state_tuple is (vx, vy, vz, yaw_speed, dx, dy) - send all 6 values, model will select what it needs
        rs = torch.tensor([robot_state_tuple[0], robot_state_tuple[1], robot_state_tuple[2], robot_state_tuple[3], robot_state_tuple[4], robot_state_tuple[5]], dtype=torch.float32)

        # Validate inputs before model call
        if len(image_paths) == 0:
            raise ValueError(f"[NovaCarterTIC-VLA] ERROR: Empty image_paths list!")
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise ValueError(f"[NovaCarterTIC-VLA] ERROR: Image file does not exist: {img_path}")
            if not os.path.isfile(img_path):
                raise ValueError(f"[NovaCarterTIC-VLA] ERROR: Image path is not a file: {img_path}")
        
        if not instruction or len(instruction.strip()) == 0:
            raise ValueError(f"[NovaCarterTIC-VLA] ERROR: Empty or missing instruction!")
        
        if not previous_waypoints_text or len(previous_waypoints_text.strip()) == 0:
            raise ValueError(f"[NovaCarterTIC-VLA] ERROR: Empty previous_waypoints_text!")
        
        # Validate robot state tensor
        # robot_state format: [vx, vy, vz, yaw_speed, dx, dy] (6 values)
        #   vx: forward velocity in body frame (m/s) - positive = forward
        #   vy: lateral velocity in body frame (m/s) - positive = left
        #   vz: vertical velocity in body frame (m/s) - should be ~0 for ground robots
        #   yaw_speed: angular velocity around z-axis (rad/s) - positive = counterclockwise
        #   dx: goal displacement in x (m) - displacement from delayed frame to current
        #   dy: goal displacement in y (m) - displacement from delayed frame to current
        if rs is None:
            raise ValueError(f"[NovaCarterTIC-VLA] ERROR: robot_state tensor is None!")
        if not torch.is_tensor(rs):
            raise ValueError(f"[NovaCarterTIC-VLA] ERROR: robot_state is not a tensor: {type(rs)}")
        if rs.shape[0] != 6:
            raise ValueError(f"[NovaCarterTIC-VLA] ERROR: robot_state must have 6 values [vx, vy, vz, yaw_speed, dx, dy], got {rs.shape[0]} values: {rs}")
        if not torch.all(torch.isfinite(rs)):
            raise ValueError(f"[NovaCarterTIC-VLA] ERROR: robot_state contains non-finite values: {rs}")
        
        # Log robot state values for debugging (only if values look unusual)
        vx, vy, vz, yaw_speed, dx, dy = rs[0].item(), rs[1].item(), rs[2].item(), rs[3].item(), rs[4].item(), rs[5].item()
        # Check for suspicious values
        if abs(vx) > 3.0 or abs(vy) > 1.0 or abs(vz) > 0.5 or abs(yaw_speed) > 3.0:
            carb.log_warn(f"[NovaCarterTIC-VLA] Robot state values may be unusual: vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}, yaw_speed={yaw_speed:.3f}, dx={dx:.3f}, dy={dy:.3f} (frame={current_step})")
        
        # Check again if backup started during validation/preparation (before heavy model call)
        if self._is_backing_up:
            carb.log_info(f"[NovaCarterTIC-VLA] Inference task aborted before model call - backup started (frame={current_step})")
            return None, False, None
        
        # Run model predict (this is the heavy call)
        # Returns: (response, waypoints, vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose)
        with torch.no_grad():
            response, waypoints, vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose = self._ticvla_model.predict_async(
                image_paths=image_paths,
                instruction=instruction,
                robot_state=rs,
                current_step=current_step,
                current_robot_pose=robot_pose,
                previous_waypoints_text=previous_waypoints_text,
                time_delay=delay_time,
                robot_type=robot_type
            )
        
        # Check AGAIN if backup started during model inference (model call takes time)
        if self._is_backing_up:
            carb.log_warn(f"[NovaCarterTIC-VLA] Inference completed but backup started during inference (frame={current_step}) - discarding results and setting backup commands")
            # Set backup commands immediately and return - do NOT use model outputs
            with self._inference_lock:
                self._current_action = np.array([self._backup_speed, 0.0], dtype=np.float32)
            # Don't print anything - just return
            return None, False, None

        wps = waypoints[0].float().cpu().numpy()  # (T,2) body/FLU
        T = len(wps)

        # Tunable parameters
        L_des = 1        # Lookahead distance (1 before
        k_angular = 0.8      # Angular gain (balanced between 0.4 and 0.8)
        alpha_filter = 0.35  # Yaw filter smoothing (increased from 0.2 for less lag)
        eps = 1e-3

        # --- build arc-length s along the waypoint polyline ---
        inc = np.diff(wps, axis=0)                          # (T-1,2)
        seg = np.hypot(inc[:, 0], inc[:, 1])                # segment lengths
        s = np.concatenate([[0.0], np.cumsum(seg)])         # (T,) arc-length

        # pick first index whose arc-length >= L_des
        j = int(np.searchsorted(s, L_des, side="left"))
        j = int(np.clip(j, 2, T - 3))

        xL, yL = float(wps[j, 0]), float(wps[j, 1])
        L = float(np.hypot(xL, yL))

        v_max = float(self._max_linear_velocity)
        w_max = float(self._max_angular_velocity)

        if L < eps:
            v_cmd, w_cmd = 0.0, 0.0
        else:
            # Heading error to lookahead point
            yaw_err = math.atan2(yL, xL)

            # Smooth yaw error filter (prevents oscillation)
            if not hasattr(self, "_yaw_err_filt"):
                self._yaw_err_filt = yaw_err
            e = math.atan2(math.sin(yaw_err - self._yaw_err_filt),
                          math.cos(yaw_err - self._yaw_err_filt))
            self._yaw_err_filt += alpha_filter * e

            # Pure pursuit curvature (for feedforward)
            kappa = 2.0 * yL / (L * L)

            # Speed: slow down for sharp turns
            v_kappa = w_max / (abs(kappa) + eps)
            v_cmd = float(np.clip(min(v_max, v_kappa), 0.0, v_max))

            # Angular: reduced feedforward + moderate feedback
            w_ff = 0.5 * v_cmd * kappa              # Reduced feedforward (was 1.0)
            w_fb = k_angular * self._yaw_err_filt   # Moderate feedback
            w_cmd = float(np.clip(w_ff + w_fb, -w_max, w_max))
        

        # CRITICAL: Check if we're waiting for inference after backup
        # IMPORTANT: We do NOT return early here - we must return vlm_generation_start_step
        # so the main loop can track new generations and detect when fresh inference completes
        # We just skip updating commands if waypoints are stale
        should_skip_commands = False
        if self._waiting_for_inference_after_backup and self._backup_completion_frame is not None:
            # Check which generation produced these waypoints by looking at the tracking list
            # The waypoints we just received are from the most recently COMPLETED generation
            waypoint_gen_start = None
            if vlm_generation_start_step is not None and len(self._vlm_generation_start_frames) > 0:
                # New generation started - waypoints are from the most recent tracked generation
                waypoint_gen_start = self._vlm_generation_start_frames[-1]
            elif len(self._vlm_generation_start_frames) > 0:
                # No new generation - waypoints are from the most recent tracked generation
                waypoint_gen_start = self._vlm_generation_start_frames[-1]
            
            if waypoint_gen_start is not None and waypoint_gen_start < self._backup_completion_frame:
                # Waypoints are from a generation that started before backup - skip using for commands
                should_skip_commands = True
                carb.log_warn(f"[NovaCarterTIC-VLA] *** DISCARDING waypoints for commands: Generation that started at {waypoint_gen_start} (before backup at {self._backup_completion_frame}). Still waiting (frame={current_step}) ***")
                # Don't update commands - keep using zero (but still return generation info for tracking)
                with self._inference_lock:
                    self._current_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # CRITICAL: Check one more time before updating commands (backup might have started during command calculation)
        # This is the last chance to abort before commands are set
        if self._is_backing_up:
            carb.log_warn(f"[NovaCarterTIC-VLA] Backup detected after command calculation - discarding model commands and setting backup (frame={current_step})")
            with self._inference_lock:
                self._current_action = np.array([self._backup_speed, 0.0], dtype=np.float32)
            # Don't print anything - backup is active, don't show model commands
            return None, False, None

        # Final check right before updating - if backup started, use backup commands
        if self._is_backing_up:
            with self._inference_lock:
                self._current_action = np.array([self._backup_speed, 0.0], dtype=np.float32)
            # Don't print - backup is active
            return None, False, None

        # Only update _current_action if NOT backing up and NOT skipping commands (waiting or stale waypoints)
        # Double-check one more time (backup might have just started)
        if not self._is_backing_up and not should_skip_commands:
            with self._inference_lock:
                self._current_action = np.array([v_cmd, w_cmd], dtype=np.float32)
            # Only print if backup is still not active
            if not self._is_backing_up:
                p_cur, R_cur, quat_wxyz = self._get_pose_R_quat()
                # print(f"v_cmd={v_cmd:.3f}, w_cmd={w_cmd:.3f} p_cur={p_cur}")
        else:
            # Backup or waiting - set appropriate commands
            if self._is_backing_up:
                with self._inference_lock:
                    self._current_action = np.array([self._backup_speed, 0.0], dtype=np.float32)
            elif self._waiting_for_inference_after_backup:
                with self._inference_lock:
                    self._current_action = np.array([0.0, 0.0], dtype=np.float32)

        
        # Return VLM generation start step, model-state availability, and pose captured at generation start
        # (pose is captured in model when generation actually starts, matching the generation step)
        return vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose

       
    # --- Velocity helpers -----------------------------------------------------
    def _smooth_stop(self, dt: float):
        self._cmd_v = self._slew(self._cmd_v, 0.0, self._max_accel_lin, self._max_decel_lin, dt, self._lin_deadband)
        self._cmd_w = self._slew(self._cmd_w, 0.0, self._max_accel_ang, self._max_decel_ang, dt, self._ang_deadband)

    @staticmethod
    def _slew(cur: float, tgt: float, accel: float, decel: float, dt: float, deadband: float) -> float:
        dt = max(1e-4, float(dt))
        dv = tgt - cur
        limit = accel if dv > 0.0 else decel
        max_step = limit * dt
        if dv > max_step:
            cur += max_step
        elif dv < -max_step:
            cur -= max_step
        else:
            cur = tgt
        if abs(cur) < deadband and abs(tgt) < deadband:
            cur = 0.0
        return cur

    def _is_goal_reached(self) -> bool:
        """Check if the robot has reached the goal position."""
        if self._robot_position is None or self._goal_position is None:
            return False

        # Calculate 2D distance (ignore Z coordinate for ground navigation)
        robot_2d = self._robot_position[:2]
        goal_2d = self._goal_position[:2]
        distance = np.linalg.norm(robot_2d - goal_2d)

        return distance < self._goal_threshold

    def _stop_robot(self):
        """Stop the robot by setting velocities to zero."""
        with self._inference_lock:
            self._current_action = np.array([0.0, 0.0])
        self._cmd_v = 0.0
        self._cmd_w = 0.0
        try:
            action = self._diff.forward(command=np.array([0.0, 0.0], dtype=np.float32))
            self._wheeled_robot.apply_wheel_actions(action)
        except Exception as e:
            carb.log_warn(f"Failed to stop robot (possibly during teardown): {e}")

    def set_goal_position(self, goal_position: np.ndarray):
        """Set the goal position for navigation."""
        self._goal_position = goal_position.copy()
        carb.log_info(f"Goal set to: {self._goal_position}")

    def set_instruction(self, instruction: str):
        """Set the navigation instruction.""" 
        self._instruction = instruction
        carb.log_info(f"Instruction set: {self._instruction}")

    def set_model_path(self, model_path: str):
        """Set the TIC-VLA model path."""
        self._model_path = model_path
        carb.log_info(f"Model path set to: {self._model_path}")

    def get_agent_name(self):
        """Get the agent name from prim path."""
        return str(self.prim_path).split("/")[-1]

    def is_navigation_active(self) -> bool:
        """Check if navigation is currently active."""
        return self._navigation_active

    def get_robot_position(self) -> Optional[np.ndarray]:
        """Get current robot position."""
        return self._robot_position.copy() if self._robot_position is not None else None

    def get_goal_position(self) -> Optional[np.ndarray]:
        """Get current goal position."""
        return self._goal_position.copy() if self._goal_position is not None else None
