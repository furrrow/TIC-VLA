"""
TIC-VLA-based autonomous navigation for the Isaac Spot robot.
Non-blocking TIC-VLA inference: runs model predict on a background thread
and uses the latest completed prediction in the main simulation loop.

This version consumes the action returned by TIC-VLA and converts it
to Spot's base velocity commands [vx, vy, wz] for quadruped locomotion.
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

# Spot policy (quadruped control)
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from isaacsim.core.utils.rotations import quat_to_rot_matrix, euler_to_rot_matrix

# Navigation helper for publishing position
from robot_navigation_manager import RobotNavigationManager

# Camera and rotation utilities
from pxr import Usd, UsdGeom, Gf, UsdPhysics, Sdf
from pxr import PhysxSchema
import omni.replicator.core as rep

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

# training bounds (from env.yaml)
VX_MIN, VX_MAX = -2.0, 3.0
VY_MIN, VY_MAX = -1.5, 1.5
WZ_MIN, WZ_MAX = -2.0, 2.0

# safety bounds (your preference)
VX_SAFE = 1.2
VY_SAFE = 0.8
WZ_SAFE = 1.0

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

import benchmark_runtime as runtime


class SpotTICVLA(BehaviorScript):
    """TIC-VLA-based autonomous navigation behavior for Spot robot with async inference."""

    # === TIC-VLA Configuration ===
    _model_path = None  # Will be set via configuration
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _inference_mode = True
    # === Checkpoint Configuration ===
    _checkpoint_path = os.getenv("TICVLA_CHECKPOINT_PATH", "checkpoints/ticvla.ckpt")
    
    # === Action smoothing parameters (for Spot base velocity) ===
    _max_accel_vx = 2.0   # m/s^2
    _max_decel_vx = 2.5   # m/s^2
    _max_accel_vy = 2.0   # m/s^2
    _max_decel_vy = 2.5   # m/s^2
    _max_accel_wz = 3.0   # rad/s^2
    _max_decel_wz = 3.5   # rad/s^2
    _deadband_v = 0.001   # m/s (small threshold to filter noise)
    _deadband_w = 0.0005  # rad/s (small threshold to filter noise)

    # === Navigation parameters ===
    _goal_threshold = 2  # meters
    _max_linear_velocity = 1.5  # m/s
    _max_angular_velocity = 1.0  # rad/s

    def on_init(self):
        """Initialize the TIC-VLA behavior script."""
        carb.log_info(f"[{self.get_agent_name()}] Initializing SpotTICVLA...")
        self._benchmark_context = runtime.get(str(self.prim_path))

        # Create Spot policy controller
        self._policy: SpotFlatTerrainPolicy = SpotFlatTerrainPolicy(
            prim_path=str(self.prim_path),
            name=self.get_agent_name(),
        )
        self._policy._decimation = 1

        # TIC-VLA model components
        self._ticvla_model: Optional[TICVLA] = None

        # Command state (smoothed) - Spot uses [vx, vy, wz]
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        self._cmd_wz = 0.0
        self._base_command = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Robot state tracking
        self._robot_position = np.array([0.0, 0.0, 0.0])
        self._robot_orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self._robot_velocity = np.array([0.0, 0.0, 0.0])
        self._robot_yaw_speed = 0.0
        self._goal_position = None
        self._instruction = ""
        self._current_action = np.array([0.0, 0.0, 0.0])  # [v_x, v_y, v_theta] from TIC-VLA
        
        # Previous state for velocity calculation
        self._prev_position = None
        self._prev_orientation = None
        self._prev_delta_time = 0.0
        self._prev_heading = None  # For computing yaw speed from motion heading
        self._current_heading = 0.0  # Current heading angle for plotting
        
        # Position offset tracking for previous waypoints (matches training)
        # Each offset is relative to the PREVIOUS frame at 1Hz (1s intervals)
        self._position_offset_history: List[tuple[int, List[float]]] = []  # List of (step, [x, y, z]) offsets at 1Hz (in FLU frame)
        self._prev_position_for_offset: np.ndarray | None = None  # Previous position when offset was computed (1s ago)
        self._prev_quaternion_for_offset: np.ndarray | None = None  # Previous quaternion [w, x, y, z] when offset was computed (1s ago)
        # No limit on waypoint history - keep all waypoints to match elapsed time
        self._offset_update_frame_count = 0  # Track frames for 1Hz updates
        
        # Track VLM generation start pose for dx, dy calculation in robot state
        # Keep only the 2 most recent VLM generation start poses
        # Note: VLM generation runs asynchronously, so we track when VLM actually starts, not when policy inference starts
        self._vlm_generation_start_positions: List[np.ndarray] = []  # List of positions when VLM generation started
        self._vlm_generation_start_quaternions: List[np.ndarray] = []  # List of quaternions [w, x, y, z] when VLM generation started
        self._vlm_generation_start_frames: List[int] = []  # List of frame numbers when VLM generation started
        self._first_inference_complete: bool = False  # Flag to track if first inference has completed

        # Camera and sensor setup (replicator annotators)
        self._camera_head_prim_path = None
        self._camera_tp_prim_path = None
        self._rp_head = None
        self._rp_tp = None
        self._rgb_annotator_head = None
        self._rgb_annotator_tp = None
        self._camera_image_head = None
        self._camera_image_tp = None

        # Image history buffering for model
        self._image_history = []  # Keep last N frames for better prediction
        self._sensor_update_frequency = 10  # Hz
        # Retain enough sensor frames for the history samples.
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
        self._run_id = self._benchmark_context.run_id
        self.save_log_dir = self._benchmark_context.log_dir
        self._log_directory = f"{self.save_log_dir}/{self._run_id}/spot_ticvla_data"

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
        self._base_link_path = f"{str(self.prim_path)}/body"  # Spot's body link
        self._prev_R = None
        self._robot_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        
        # Stuck detection and recovery
        self._stuck_position_threshold = 0.1  # meters - consider stuck if moved less than this
        self._stuck_time_threshold_frames = 150  # Five seconds at the simulation frame rate
        self._backup_duration_frames = 300  # frames - backup for this many frames when stuck (straight back, no rotation)
        self._backup_speed = -0.5  # m/s - backward speed during recovery
        self._last_movement_position = None  # Last position where significant movement was detected
        self._last_movement_frame = None  # Frame when last movement was detected (don't start until first inference completes)
        self._is_backing_up = False  # Flag indicating if currently backing up
        self._backup_start_frame = None  # Frame when backup started
        self._backup_completion_frame = None  # Frame when backup completed (used to ensure we wait for inference that starts after this)
        self._waiting_for_inference_after_backup = False  # Flag to wait for next inference after backup completes

        

        # Load TIC-VLA model
        self._load_ticvla_model()
        print("[SpotTIC-VLA] Initialization complete.")


    def _create_log_directory(self):
        try:
            if not os.path.exists(self._log_directory):
                os.makedirs(self._log_directory)
                carb.log_info(f"Created log directory: {self._log_directory}")
        except Exception as e:
            carb.log_warn(f"Failed to create log directory: {e}")

    def on_play(self):
        """Called when simulation starts playing."""
        # CRITICAL: Reset goal and instruction BEFORE reading from environment
        # This ensures no stale values persist if environment variables are missing
        self._goal_position = None
        self._instruction = ""
        
        # Reset trajectory history for new episode
        self._trajectory_history = []
        self._robot_state_history = []
        self._heading_history = []
        self._trajectory_plotted = False
        
        # Reset position offset tracking
        self._position_offset_history = []
        self._prev_position_for_offset = None
        self._prev_quaternion_for_offset = None
        self._offset_update_frame_count = 0
        self._last_recorded_frame = -1
        
        # Reset VLM generation start tracking
        self._vlm_generation_start_positions = []
        self._vlm_generation_start_quaternions = []
        self._vlm_generation_start_frames = []
        self._first_inference_complete = False
        
        # --- Episode filesystem cleanup ---
        _empty_dir(self._log_directory)
        
        # Reset command velocities and actions
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        self._cmd_wz = 0.0
        self._base_command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        with self._inference_lock:
            self._current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Remove settling period - match Nova Carter's immediate response
        
        # Reset model's internal state (critical for clean episode transitions)
        # MUST happen before any new inference starts to prevent stale state
        if self._ticvla_model is not None:
            try:
                # Ensure any pending inference completes or is cancelled
                with self._inference_lock:
                    # Clear current action to prevent stale commands
                    self._current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                
                # Reset model state (this will shutdown/recreate executor and clear model state)
                self._ticvla_model.reset_episode_state()
                carb.log_info("[SpotTIC-VLA] Reset model state for new episode")
                
                # Increased delay to ensure executor cleanup and async inference completion
                # This is critical for sequential episodes to prevent state leakage
                time.sleep(0.2)  # Increased from 0.05 to 0.2 seconds
            except Exception as e:
                carb.log_warn(f"Failed to reset model state: {e}")
                import traceback
                carb.log_warn(traceback.format_exc())
        
        # Reset filtered states
        if hasattr(self, "_yaw_err_filt"):
            delattr(self, "_yaw_err_filt")
        
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
            try:
                for img_path in self._image_history:
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    except Exception as e:
                        carb.log_warn(f"Failed to remove image {img_path}: {e}")
            except Exception as e:
                carb.log_warn(f"Failed to cleanup image history: {e}")
            self._image_history.clear()
        print(f"[on_play] Reset episode state: frame_count=0, cleared {old_image_count} images")
        
        self._setup_for_play()
        try:
            # Initialize policy if it exposes an initialize() method
            init_fn = getattr(self._policy, "initialize", None)
            if callable(init_fn):
                self._policy.initialize()
                carb.log_info(f"[{self.get_agent_name()}] Spot policy initialized in on_play()")
        except Exception as e:
            carb.log_warn(f"Spot policy initialize failed in on_play: {e}")

        self._register_to_agent_manager()
        self._init_camera_capture()
        
        # Note: Inference is now synchronous (like Nova Carter) - no executor needed

    def on_stop(self):
        """Called when simulation stops."""
        print("[on_stop] Called")
        self._finalize_and_flush()
        print("[on_stop] Completed successfully")

    def on_destroy(self):
        """Called when script is destroyed."""
        print("[on_destroy] Called")
        # Call finalize if not already done
        self._finalize_and_flush()
        print("[on_destroy] Completed successfully")

    def _load_ticvla_model(self):
        """Load the TIC-VLA model for inference."""
        if TICVLA is None:
            carb.log_error("TIC-VLA components not available. Navigation will be disabled.")
            return

        # Try to find model path from environment or configuration
        default_path = os.getenv('TICVLA_BASE_MODEL_PATH', os.getenv('TICVLA_MODEL_PATH', 'OpenGVLab/InternVL3-1B'))
        model_path = default_path

        if not os.path.exists(model_path):
            carb.log_warn(f"TIC-VLA model path not found: {model_path}")
            return

        carb.log_info(f"Loading TIC-VLA model from: {model_path}")

        # Load model using TIC-VLA constructor
        self._ticvla_model = TICVLA(model_path=model_path, device=self._device)

        # Try to load checkpoint if present
        ckpt_path = self._checkpoint_path
        if os.path.exists(ckpt_path):
            try:
                carb.log_info(f"Attempting to load checkpoint: {ckpt_path}")
                state_dict = torch.load(ckpt_path, map_location='cpu')["state_dict"]
                state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
                self._ticvla_model.load_state_dict(state_dict, strict=False)
                carb.log_info("Checkpoint loaded successfully")
            except Exception as ckpt_error:
                carb.log_warn(f"Failed to load checkpoint (architecture mismatch?): {ckpt_error}")
                carb.log_warn("Continuing with base model weights (untrained)")
        else:
            carb.log_info(f"Checkpoint not found at {ckpt_path}, using base model")
        self._ticvla_model.eval()

        # Use the same log directory for image buffering
        self._image_buffer_dir = self._log_directory
        carb.log_info(f"Using log directory for image buffer: {self._image_buffer_dir}")

        self._model_loaded = True
        carb.log_info("TIC-VLA model loaded successfully")

    def _alive(self) -> bool:
        # Check if timeline is playing
        tl = get_timeline_interface()
        if not tl or not tl.is_playing():
            return False
        # Stage must exist
        ctx = omni.usd.get_context()
        if not ctx or not ctx.get_stage():
            return False
        # Policy must still be valid
        return bool(self._policy)

    def _setup_for_play(self):
        """Set up navigation and settings for play mode."""
        self._navmeshEnabled = self.setting.get(PeopleSettings.NAVMESH_ENABLED)
        self._avoidanceOn = self.setting.get(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED)

        # Get robot handle from policy for nav manager
        spot_robot = getattr(self._policy, "robot", None)
        self._navigation_manager = RobotNavigationManager(
            str(self.prim_path), spot_robot, navmesh_enabled=False, dynamic_avoidance_enabled=self._avoidanceOn
        )

        # Event bus for agent registration
        self._agent_registered_event = carb.events.type_from_string(AgentEvent.AgentRegistered)
        self._bus = omni.kit.app.get_app().get_message_bus_event_stream()

        self._navigation_active = True

        # Configure from environment variables set by benchmark
        self._configure_from_benchmark()

        carb.log_info("[SpotTIC-VLA] TIC-VLA navigation initialized")

    def _register_to_agent_manager(self):
        """Register this agent with the agent manager."""
        if self._bus and self._agent_registered_event:
            agent_name = self.get_agent_name()
            info = {"agent_name": str(agent_name), "prim_path": str(self.prim_path)}
            self._bus.push(self._agent_registered_event, payload=info)

    def _init_camera_capture(self):
        """Set up head camera and third person view camera (creates them if needed)."""
        import math
        try:
            # Isaac Sim 5+
            from isaacsim.sensors.camera import Camera
        except Exception:
            # Legacy (deprecated)
            from omni.isaac.sensor import Camera
        
        # Head and third-person camera handles
        self._camera_head_prim_path = None
        self._camera_tp_prim_path = None
        self._head_camera = None
        self._third_person_camera = None
        self._rp_head = None
        self._rp_tp = None
        self._rgb_annotator_head = None
        self._rgb_annotator_tp = None
        self._camera_image_head = None
        self._camera_image_tp = None

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        robot_prim = stage.GetPrimAtPath(str(self.prim_path))
        if not robot_prim or not robot_prim.IsValid():
            carb.log_warn("Robot prim not found for camera setup")
            return

        # Set up head camera (mounted on robot head)
        self._setup_head_camera(stage, robot_prim)
        
        # Set up third person camera (external view for logging)
        self._setup_third_person_camera(stage, robot_prim)

    def _setup_head_camera(self, stage, robot_prim):
        """Set up head-mounted RGB camera for navigation."""
        import math
        try:
            # Isaac Sim 5+
            from isaacsim.sensors.camera import Camera
        except Exception:
            # Legacy (deprecated)
            from omni.isaac.sensor import Camera
        
        try:
            # Resolve body link
            body_path = f"{str(self.prim_path)}/body"
            body_prim = stage.GetPrimAtPath(body_path)
            if not (body_prim and body_prim.IsValid()):
                carb.log_warn(f"Body link not found at {body_path}; mounting on robot root.")
                body_path = str(self.prim_path)
                body_prim = robot_prim

            # Make sure we can author a child under the body
            try:
                if body_prim.IsInstanceable():
                    body_prim.SetInstanceable(False)
            except Exception as e:
                carb.log_warn(f"Failed to set body prim instanceable: {e}")

            # Create Camera sensor under /body
            cam_name = "Head_Camera"
            cam_prim_path = f"{body_path}/{cam_name}"
            self._camera_head_prim_path = cam_prim_path

            # Pose: near head (forward +X, left +Y, up +Z) relative to body
            cam_translation = np.array([0.44, 0.075, 0.01], dtype=float)

            # Orientation: replicate CameraUtil alignment so yaw=0 points +X
            cam_yaw_deg = 90.0       # look straight forward (+X)
            cam_pitch_deg = -2.0   # slight look-down
            roll_fix_deg = -90.0      # rotate image back by +90°
            np_mat_yaw     = euler_to_rot_matrix(np.array([0, cam_yaw_deg, 0]), degrees=True, extrinsic=False)
            np_mat_pitch   = euler_to_rot_matrix(np.array([-cam_pitch_deg, 0, 0]), degrees=True, extrinsic=False)
            np_mat_default = euler_to_rot_matrix(np.array([90, -90, 0]), degrees=True, extrinsic=False)
            np_mat_rollfix = euler_to_rot_matrix(np.array([roll_fix_deg, 0, 0]), degrees=True, extrinsic=False)

            rot_matrix = (
                Gf.Matrix3d(np_mat_rollfix.T.tolist())
                * Gf.Matrix3d(np_mat_pitch.T.tolist())
                * Gf.Matrix3d(np_mat_yaw.T.tolist())
                * Gf.Matrix3d(np_mat_default.T.tolist())
            )
            quat = rot_matrix.ExtractRotation().GetQuat()
            cam_orientation = np.array([quat.GetReal(), *list(quat.GetImaginary())], dtype=float)  # [w, x, y, z]

            # Create the sensor
            self._head_camera = Camera(
                prim_path=cam_prim_path,
                name=cam_name,
                frequency=runtime.SIMULATION_FPS,
                resolution=(1920, 1080),
                translation=cam_translation,
                orientation=cam_orientation,
            )
            self._head_camera.initialize()
            carb.log_info(f"Camera sensor created at {cam_prim_path}")

            # Set intrinsics to match 90° horizontal FOV
            width_px, height_px = 1920, 1080
            film_w_mm = 20.955 * (width_px / 1920.0)
            film_h_mm = film_w_mm * (height_px / float(width_px))
            focal_mm = (0.5 * film_w_mm) / math.tan(math.radians(90.0) * 0.5)

            usd_cam = UsdGeom.Camera.Get(stage, Sdf.Path(cam_prim_path))
            usd_cam.CreateHorizontalApertureAttr(film_w_mm)
            usd_cam.CreateVerticalApertureAttr(film_h_mm)
            usd_cam.CreateFocalLengthAttr(float(focal_mm))
            usd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

            # Create Replicator render product for data capture
            self._rp_head = rep.create.render_product(cam_prim_path, resolution=(1920, 1080))
            self._rgb_annotator_head = rep.AnnotatorRegistry.get_annotator("rgb")
            self._rgb_annotator_head.attach([self._rp_head])
            carb.log_info(f"Replicator render product created for {cam_prim_path}")
            
        except Exception as e:
            carb.log_error(f"Failed to setup head camera: {e}")
            import traceback
            carb.log_error(traceback.format_exc())

    def _setup_third_person_camera(self, stage, robot_prim):
        """Set up third person view camera for logging."""
        import math
        try:
            # Isaac Sim 5+
            from isaacsim.sensors.camera import Camera
        except Exception:
            # Legacy (deprecated)
            from omni.isaac.sensor import Camera
        
        try:
            # Resolve body link
            body_path = f"{str(self.prim_path)}/body"
            body_prim = stage.GetPrimAtPath(body_path)
            if not (body_prim and body_prim.IsValid()):
                carb.log_warn(f"Body link not found at {body_path}; mounting on robot root.")
                body_path = str(self.prim_path)
                body_prim = robot_prim

            # Create Third Person Camera sensor
            tp_cam_name = "ThirdPerson_Camera"
            tp_cam_prim_path = f"{body_path}/{tp_cam_name}"
            self._camera_tp_prim_path = tp_cam_prim_path

            # Position camera behind and above robot
            tp_cam_translation = np.array([-3.0, 0.0, 1.0], dtype=float)

            # Orientation: look at robot
            tp_cam_yaw_deg = 90.0
            tp_cam_pitch_deg = 0.0
            tp_roll_fix_deg = -90.0
            tp_np_mat_yaw     = euler_to_rot_matrix(np.array([0, tp_cam_yaw_deg, 0]), degrees=True, extrinsic=False)
            tp_np_mat_pitch   = euler_to_rot_matrix(np.array([-tp_cam_pitch_deg, 0, 0]), degrees=True, extrinsic=False)
            tp_np_mat_default = euler_to_rot_matrix(np.array([90, -90, 0]), degrees=True, extrinsic=False)
            tp_np_mat_rollfix = euler_to_rot_matrix(np.array([tp_roll_fix_deg, 0, 0]), degrees=True, extrinsic=False)

            tp_rot_matrix = (
                Gf.Matrix3d(tp_np_mat_rollfix.T.tolist())
                * Gf.Matrix3d(tp_np_mat_pitch.T.tolist())
                * Gf.Matrix3d(tp_np_mat_yaw.T.tolist())
                * Gf.Matrix3d(tp_np_mat_default.T.tolist())
            )
            tp_quat = tp_rot_matrix.ExtractRotation().GetQuat()
            tp_cam_orientation = np.array([tp_quat.GetReal(), *list(tp_quat.GetImaginary())], dtype=float)

            # Create the sensor
            self._third_person_camera = Camera(
                prim_path=tp_cam_prim_path,
                name=tp_cam_name,
                frequency=runtime.SIMULATION_FPS,
                resolution=(1920, 1080),
                translation=tp_cam_translation,
                orientation=tp_cam_orientation,
            )
            self._third_person_camera.initialize()
            carb.log_info(f"Third person camera sensor created at {tp_cam_prim_path}")

            # Set intrinsics
            width_px, height_px = 1920, 1080
            film_w_mm = 20.955 * (width_px / 1920.0)
            film_h_mm = film_w_mm * (height_px / float(width_px))
            focal_mm = (0.5 * film_w_mm) / math.tan(math.radians(90.0) * 0.5)

            usd_cam = UsdGeom.Camera.Get(stage, Sdf.Path(tp_cam_prim_path))
            usd_cam.CreateHorizontalApertureAttr(film_w_mm)
            usd_cam.CreateVerticalApertureAttr(film_h_mm)
            usd_cam.CreateFocalLengthAttr(float(focal_mm))
            usd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

            # Create Replicator render product for logging
            self._rp_tp = rep.create.render_product(tp_cam_prim_path, resolution=(1920, 1080))
            self._rgb_annotator_tp = rep.AnnotatorRegistry.get_annotator("rgb")
            self._rgb_annotator_tp.attach([self._rp_tp])
            carb.log_info(f"Replicator render product created for {tp_cam_prim_path}")
            
        except Exception as e:
            carb.log_error(f"Failed to setup third person camera: {e}")
            import traceback
            carb.log_error(traceback.format_exc())

    def _teardown_camera_capture(self):
        try:
            if getattr(self, "_head_camera", None) is not None:
                try:
                    self._head_camera = None
                except Exception as e:
                    carb.log_warn(f"Failed to cleanup head camera: {e}")
            if getattr(self, "_third_person_camera", None) is not None:
                try:
                    self._third_person_camera = None
                except Exception as e:
                    carb.log_warn(f"Failed to cleanup third person camera: {e}")
            if getattr(self, "_rgb_annotator_head", None) is not None:
                try:
                    self._rgb_annotator_head.detach()
                except Exception as e:
                    carb.log_warn(f"Failed to detach head RGB annotator: {e}")
            if getattr(self, "_rgb_annotator_tp", None) is not None:
                try:
                    self._rgb_annotator_tp.detach()
                except Exception as e:
                    carb.log_warn(f"Failed to detach TP RGB annotator: {e}")
            self._head_camera = None
            self._third_person_camera = None
            self._rgb_annotator_head = None
            self._rgb_annotator_tp = None
            self._rp_head = None
            self._rp_tp = None
            self._camera_head_prim_path = None
            self._camera_tp_prim_path = None
            self._camera_image_head = None
            self._camera_image_tp = None
        except Exception as e:
            carb.log_warn(f"Camera capture teardown failed: {e}")

    def _capture_camera_image(self):
        # Head camera
        if getattr(self, "_rgb_annotator_head", None) is not None:
            try:
                data_head = self._rgb_annotator_head.get_data()
                if data_head is None or not hasattr(data_head, "shape") or data_head.size == 0:
                    self._camera_image_head = None
                else:
                    self._camera_image_head = data_head
            except Exception as e:
                carb.log_warn(f"Failed to get head camera data: {e}")
                self._camera_image_head = None

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

    def _configure_from_benchmark(self):
        self._goal_position = np.array(self._benchmark_context.goal)
        self._instruction = self._benchmark_context.instruction

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
            # Note: No executor to shutdown since inference is synchronous
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
        
        self._frame_count = self._benchmark_context.frame_count

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
                init_fn = getattr(self._policy, "initialize", None)
                if callable(init_fn):
                    self._policy.initialize()
            except Exception as e:
                carb.log_error(f"Failed to initialize Spot policy: {e}")
                import traceback
                carb.log_error(traceback.format_exc())
            self._initialized = True

        # Update robot kinematic state
        self._update_robot_state(delta_time)
        
        frame_idx = int(self._frame_count)
        if frame_idx % runtime.SIMULATION_FPS == 0 and frame_idx != getattr(self, '_last_recorded_frame', -1):
            if self._robot_position is not None:
                try:
                    p_cur, R_cur, quat_wxyz = self._get_pose_R_quat()
                    
                    if self._prev_position_for_offset is None:
                        self._position_offset_history.append((frame_idx, [0.0, 0.0, 0.0]))
                        self._prev_position_for_offset = p_cur.copy()
                        self._prev_quaternion_for_offset = quat_wxyz.copy()
                        self._last_recorded_frame = frame_idx
                    else:
                        delta_world = p_cur - self._prev_position_for_offset
                        R_prev = quat_to_rot_matrix(self._prev_quaternion_for_offset)
                        vec_body = R_prev.T @ delta_world
                        vec_flu = vec_body.copy()
                        self._position_offset_history.append((frame_idx, vec_flu.tolist()))
                        self._prev_position_for_offset = p_cur.copy()
                        self._prev_quaternion_for_offset = quat_wxyz.copy()
                        self._last_recorded_frame = frame_idx
                        # Keep all waypoints - no limit to match elapsed time naturally
                except Exception as e:
                    carb.log_warn(f"Failed to update position offset history: {e}")

        # Capture images from existing cameras
        self._capture_camera_image()

        # Stuck detection and recovery logic (run BEFORE inference to prevent inference during backup)
        # Only start stuck detection AFTER first inference completes (don't count the initial stationary period)
        
        # Check if currently backing up
        if self._is_backing_up:
            # Check if backup duration has elapsed using frame count
            if self._backup_start_frame is not None:
                backup_elapsed_frames = self._frame_count - self._backup_start_frame
                
                # Log backup progress once per simulation second.
                if backup_elapsed_frames > 0 and backup_elapsed_frames % runtime.SIMULATION_FPS == 0 and backup_elapsed_frames < self._backup_duration_frames:
                    remaining_frames = self._backup_duration_frames - backup_elapsed_frames
                    remaining_time = remaining_frames / runtime.SIMULATION_FPS
                    carb.log_info(f"[{self.get_agent_name()}] Backing up progress: {backup_elapsed_frames}/{self._backup_duration_frames} frames ({remaining_time:.1f}s remaining, frame={self._frame_count}, start_frame={self._backup_start_frame})")
                
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
                        self._current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    # Reset movement tracking after backup completes - but DON'T initialize it yet
                    # We'll initialize it when fresh inference arrives and normal operation resumes
                    # This prevents stuck detection from triggering during the wait period
                    self._last_movement_position = None
                    self._last_movement_frame = None
                    carb.log_warn(f"[{self.get_agent_name()}] *** Backup recovery complete after {backup_elapsed_frames} frames ({self._backup_duration_frames} frames requested, completed at frame={self._frame_count}, backup started at frame={backup_start_frame_saved}), waiting for next inference that starts AFTER frame {self._backup_completion_frame} before resuming normal operation ***")
            else:
                # This shouldn't happen, but log it for debugging
                carb.log_warn(f"[{self.get_agent_name()}] WARNING: _is_backing_up=True but _backup_start_frame is None at frame {self._frame_count}")
        
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
                            carb.log_info(f"[{self.get_agent_name()}] Robot unstuck after {backup_frames} frames of backup, stopping backup recovery")
            
            # Detect if stuck (not moved for threshold frames)
            if self._last_movement_frame is not None:
                frames_since_movement = self._frame_count - self._last_movement_frame
                if frames_since_movement >= self._stuck_time_threshold_frames:
                    # Robot is stuck - start backup recovery
                    self._is_backing_up = True
                    self._backup_start_frame = self._frame_count
                    # Clear model actions to prevent stale commands from being used
                    with self._inference_lock:
                        self._current_action = np.array([self._backup_speed, 0.0, 0.0], dtype=np.float32)
                    # Immediately reset velocities to backup mode (no smoothing)
                    self._cmd_vx = self._backup_speed
                    self._cmd_vy = 0.0
                    self._cmd_wz = 0.0
                    stuck_time_seconds = frames_since_movement / runtime.SIMULATION_FPS
                    backup_duration_seconds = self._backup_duration_frames / runtime.SIMULATION_FPS
                    carb.log_warn(f"[{self.get_agent_name()}] Robot stuck detected (not moved for {frames_since_movement} frames / {stuck_time_seconds:.2f}s), starting backup recovery for {self._backup_duration_frames} frames ({backup_duration_seconds:.1f}s)")
                    carb.log_warn(f"[{self.get_agent_name()}] Backup start: frame={self._backup_start_frame}, cmd_vx={self._cmd_vx:.3f}, cmd_wz={self._cmd_wz:.3f}, cleared _current_action")

        # Sensor/update cadence for model

        self._sensor_frame_count += 1
        if self._sensor_frame_count % max(1, (runtime.SIMULATION_FPS // self._sensor_update_frequency)) == 0:
            # Buffer the latest head image to disk for the model (main thread does file write)
            if self._image_buffer_dir is not None and self._camera_image_head is not None:

                img = self._camera_image_head
                # Check if image is valid (not empty)
                if img is None or img.size == 0:
                    carb.log_warn("Camera image is empty, skipping save")
                else:
                    if img.ndim == 3 and img.shape[2] == 4:
                        img = img[:, :, :3]
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    filename_head = f"{self._image_buffer_dir}/head_frame_{self._frame_count:06d}.jpg"
                    # Ensure directory exists before saving
                    os.makedirs(self._image_buffer_dir, exist_ok=True)
                    # Only append to history after image is successfully saved
                    if cv2.imwrite(filename_head, img_bgr):
                        self._image_history.append(filename_head)
                    else:
                        carb.log_warn(f"Failed to save image: {filename_head} (directory: {self._image_buffer_dir}, exists: {os.path.exists(self._image_buffer_dir)})")
                    if len(self._image_history) > self._max_history_frames:
                        oldest = self._image_history.pop(0)
                        try:
                            if os.path.exists(oldest):
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

            # Decide whether to run TIC-VLA inference (synchronously, like Nova Carter)
            have_images = len(self._image_history) > 0 and self._model_loaded and (self._ticvla_model is not None)
            if have_images:

                p_cur, R_cur, quat_wxyz = self._get_pose_R_quat()
                robot_pose = {
                    'position': p_cur.tolist(),
                    'quaternion': quat_wxyz.tolist(),
                    'rotation_matrix': R_cur.tolist()
                }
                
                sampled_paths = self._get_sampled_image_paths()
                valid_paths = []
                for path in sampled_paths:
                    if path and os.path.exists(path):
                        valid_paths.append(path)
                    else:
                        carb.log_warn(f"Image path does not exist or is invalid: {path}")
                if valid_paths:
                    if len(self._vlm_generation_start_frames) > 0:
                        ref_position = self._vlm_generation_start_positions[0]
                        ref_quaternion = self._vlm_generation_start_quaternions[0]
                        ref_vlm_generation_start_frame = self._vlm_generation_start_frames[0]
                        
                        delta_world = p_cur - ref_position
                        R_start = quat_to_rot_matrix(ref_quaternion)
                        vec_body = R_start.T @ delta_world
                        vec_flu = vec_body.copy()
                        dx = float(vec_flu[0])
                        dy = float(vec_flu[1])
                        
                        step_delay = self._frame_count - ref_vlm_generation_start_frame
                        delay_time = float(step_delay) / runtime.SIMULATION_FPS
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
                    
                    # Prepare waypoints: filter out initial (0,0,0) and convert to (relative_time, waypoint) format
                    elapsed_time = float(self._frame_count) / runtime.SIMULATION_FPS
                    raw_waypoints = list(self._position_offset_history)
                    # Filter out initial (0,0,0) waypoint and convert to (relative_time, waypoint) format
                    previous_waypoints = []
                    for step, waypoint in raw_waypoints:
                        # Skip initial (0,0,0) waypoint
                        if abs(waypoint[0]) < 1e-6 and abs(waypoint[1]) < 1e-6 and abs(waypoint[2]) < 1e-6:
                            continue
                        relative_time = float(step) / runtime.SIMULATION_FPS
                        previous_waypoints.append((relative_time, waypoint))
                    
                    previous_waypoints_text = self._format_previous_waypoints_text(previous_waypoints, elapsed_time)
                    
                    # Call inference synchronously (like Nova Carter) - this blocks until inference completes
                    # This ensures inference runs at exactly 10Hz with fresh actions every cycle
                    # The VLM generation still runs asynchronously internally via predict_async
                    try:
                        vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose = self._inference_task(
                            valid_paths,
                            self._instruction,
                            robot_state,
                            self._frame_count,
                            robot_pose,
                            previous_waypoints_text,
                            delay_time,
                            "legged robot"
                        )
                        
                        # If a new generation started, store the pose that was captured in the model
                        # (captured in _start_kv_cache_generation when generation actually starts)
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
                                
                                # Keep only the 2 most recent entries
                                if len(self._vlm_generation_start_frames) > 2:
                                    self._vlm_generation_start_frames.pop(0)
                                    self._vlm_generation_start_positions.pop(0)
                                    self._vlm_generation_start_quaternions.pop(0)
                        
                        if not self._first_inference_complete and kv_cache_available:
                            self._first_inference_complete = True
                            carb.log_info("[SpotTIC-VLA] First VLM generation completed, model state available, starting movement")
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
            self._cmd_vx = self._backup_speed
            self._cmd_vy = 0.0
            self._cmd_wz = 0.0
            # Log every 10 frames during backup (reduce log spam but still verify it's working)
            elapsed = self._frame_count - self._backup_start_frame if self._backup_start_frame is not None else -1
            if self._frame_count % 10 == 0:
                carb.log_info(f"[{self.get_agent_name()}] *** BACKING UP ***: frame={self._frame_count}, cmd_vx={self._cmd_vx:.3f}, cmd_wz={self._cmd_wz:.3f}, backup_start_frame={self._backup_start_frame}, elapsed_frames={elapsed}/{self._backup_duration_frames}")
        elif self._waiting_for_inference_after_backup:
            # After backup completes, wait for next inference before resuming - keep robot stationary
            # This ensures we use fresh navigation commands, not stale ones from before backup
            target_vx = 0.0
            target_vy = 0.0
            target_wz = 0.0
            self._cmd_vx = self._slew(self._cmd_vx, target_vx, self._max_accel_vx, self._max_decel_vx, delta_time, self._deadband_v)
            self._cmd_vy = self._slew(self._cmd_vy, target_vy, self._max_accel_vy, self._max_decel_vy, delta_time, self._deadband_v)
            self._cmd_wz = self._slew(self._cmd_wz, target_wz, self._max_accel_wz, self._max_decel_wz, delta_time, self._deadband_w)
            if self._frame_count % runtime.SIMULATION_FPS == 0:  # Log once per second
                carb.log_warn(f"[{self.get_agent_name()}] *** WAITING FOR FRESH INFERENCE AFTER BACKUP ***: frame={self._frame_count}, backup_completion_frame={self._backup_completion_frame} - robot stationary")
        elif not self._first_inference_complete:
            # Keep robot stationary until first inference completes (but not if backing up - handled above)
            target_vx = 0.0
            target_vy = 0.0
            target_wz = 0.0
            self._cmd_vx = self._slew(self._cmd_vx, target_vx, self._max_accel_vx, self._max_decel_vx, delta_time, self._deadband_v)
            self._cmd_vy = self._slew(self._cmd_vy, target_vy, self._max_accel_vy, self._max_decel_vy, delta_time, self._deadband_v)
            self._cmd_wz = self._slew(self._cmd_wz, target_wz, self._max_accel_wz, self._max_decel_wz, delta_time, self._deadband_w)
        else:
            # Convert model action to smoothed commands (protected by lock)
            with self._inference_lock:
                target_vx = float(self._current_action[0])
                target_vy = float(self._current_action[1])
                target_wz = float(self._current_action[2])
            
            # Apply smoothing only when NOT backing up
            # Force vy = 0.0 here as a final safeguard even if model/action had it
            target_vy = 0.0
            self._cmd_vx = self._slew(self._cmd_vx, target_vx, self._max_accel_vx, self._max_decel_vx, delta_time, self._deadband_v)
            self._cmd_vy = self._slew(self._cmd_vy, target_vy, self._max_accel_vy, self._max_decel_vy, delta_time, self._deadband_v)
            self._cmd_wz = self._slew(self._cmd_wz, target_wz, self._max_accel_wz, self._max_decel_wz, delta_time, self._deadband_w)

        # Final safety check: if backing up, ensure commands are correct (CRITICAL - do this right before applying)
        if self._is_backing_up:
            self._cmd_vx = self._backup_speed
            self._cmd_vy = 0.0
            self._cmd_wz = 0.0
            # Also clear any model actions to prevent them from being used next frame
            with self._inference_lock:
                self._current_action = np.array([self._backup_speed, 0.0, 0.0], dtype=np.float32)
            carb.log_info(f"[{self.get_agent_name()}] Final backup check: frame={self._frame_count}, forcing cmd_vx={self._cmd_vx:.3f}, cmd_wz={self._cmd_wz:.3f}")
        
        # Additional safety: check if we should still be backing up (in case completion check was missed)
        if self._backup_start_frame is not None:
            backup_elapsed = self._frame_count - self._backup_start_frame
            if backup_elapsed >= self._backup_duration_frames and self._is_backing_up:
                # Backup should have completed but didn't - force completion now
                carb.log_warn(f"[{self.get_agent_name()}] *** SAFETY: Backup should have completed! frame={self._frame_count}, elapsed={backup_elapsed}, duration={self._backup_duration_frames}. Forcing completion now. ***")
                self._backup_completion_frame = self._frame_count
                backup_start_saved = self._backup_start_frame
                self._is_backing_up = False
                self._backup_start_frame = None
                self._waiting_for_inference_after_backup = True
                with self._inference_lock:
                    self._current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                if self._robot_position is not None:
                    self._last_movement_position = self._robot_position.copy()
                    self._last_movement_frame = self._frame_count

        # Push command to Spot policy
        self._base_command[0] = self._cmd_vx
        self._base_command[1] = self._cmd_vy
        self._base_command[2] = self._cmd_wz
        
        try:
            if self._policy:
                self._policy.forward(delta_time, self._base_command)
            else:
                carb.log_error("[TIC-VLA] Policy is None, cannot send commands!")
        except Exception as e:
            carb.log_error(f"[TIC-VLA] Error calling policy.forward: {e}")
            import traceback
            carb.log_error(traceback.format_exc())

        # Stream data and log
        self._stream_robot_data()
        
        # Publish position for avoidance if enabled
        if self._avoidanceOn and self._navigation_manager is not None:
            self._navigation_manager.publish_robot_position(delta_time, radius=2.0)

    def _get_pose_R_quat(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._base_link_path)
        if not prim or not prim.IsValid():
            # Fallback to robot root
            prim = stage.GetPrimAtPath(str(self.prim_path))
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
        
        p_cur, R_cur_native, quat_wxyz = self._get_pose_R_quat()
        self._robot_position = p_cur
        self._robot_quaternion = quat_wxyz

        # Get velocities from policy's robot if available, or compute from pose
        spot_robot = getattr(self._policy, "robot", None)
        if spot_robot is not None:
            pos_IB, q_IB = spot_robot.get_world_pose()
            R_IB = quat_to_rot_matrix(q_IB)
            R_BI = R_IB.transpose()
            
            # Try to get velocities from robot
            lin_vel_I = spot_robot.get_linear_velocity()
            ang_vel_I = spot_robot.get_angular_velocity()
            
            if lin_vel_I is None:
                lin_vel_I = np.zeros(3, dtype=np.float64)
            if ang_vel_I is None:
                ang_vel_I = np.zeros(3, dtype=np.float64)
                
            lin_vel_I = np.asarray(lin_vel_I, dtype=np.float64)
            ang_vel_I = np.asarray(ang_vel_I, dtype=np.float64)
            
            lin_vel_b = np.matmul(R_BI, lin_vel_I)
            ang_vel_b = np.matmul(R_BI, ang_vel_I)
            
            # # Flip sign of vy to match expected convention
            # lin_vel_b[1] = -lin_vel_b[1]
            
            self._robot_velocity = lin_vel_b.astype(np.float64)
            self._robot_yaw_speed = float(ang_vel_b[2])

        # Compute orientation from rotation matrix
        euler = self._rotation_matrix_to_euler(R_cur_native)
        self._robot_orientation = euler

        # Heading for plotting
        R_cur_flu = R_cur_native
        fwd = R_cur_flu[:, 0]
        self._current_heading = float(np.arctan2(fwd[1], fwd[0]))

        self._prev_R = R_cur_native
        self._prev_position = p_cur
        self._prev_orientation = self._robot_orientation
        self._prev_delta_time = delta_time

    def _rotation_matrix_to_euler(self, R):
        """Convert rotation matrix to euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])
            
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
                "vx": float(self._cmd_vx),
                "vy": float(self._cmd_vy),
                "wz": float(self._cmd_wz),
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
                    try:
                        if os.path.exists(img_path):
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
        Falls back to earliest available if history isn't long enough yet.
        Returns in order: [oldest (-9s), -6s, -3s, current] (newest at last).
        """
        hist = self._image_history
        n = len(hist)
        if n == 0:
            return []
        
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
        
        return unique_sampled

    # --- Previous waypoints formatting (matches training) ---
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
    
    def _inference_task(self, image_paths, instruction, robot_state_tuple, current_step: int, robot_pose: dict | None = None, previous_waypoints_text: str = "", delay_time: float = 0.0, robot_type: str = "legged robot"):
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
        if self._ticvla_model is None:
            return None, False, None

        # Note: Pose is now captured in the model when generation actually starts
        # (in _start_kv_cache_generation), so we don't need to capture it here

        # Convert robot_state_tuple back to tensor on worker thread
        # robot_state_tuple is now (vx, vy, vz, yaw_speed, dx, dy)
        rs = torch.tensor([robot_state_tuple[0], robot_state_tuple[1], robot_state_tuple[2], robot_state_tuple[3], robot_state_tuple[4], robot_state_tuple[5]], dtype=torch.float32)

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

        wps = waypoints[0].float().cpu().numpy()  # (T,2)
        T = len(wps)

        # Tunable parameters
        L_des = 1.0        # Lookahead distance
        k_angular = 0.8      # Angular gain
        alpha_filter = 0.35  # Yaw filter smoothing
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
            v_x, v_y, w_z = 0.0, 0.0, 0.0
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
            v_x = float(np.clip(min(v_max, v_kappa), 0.0, v_max))
            
            # FORCE vy = 0 for consistency with non-holonomic behavior
            v_y = 0.0

            # Angular: reduced feedforward + moderate feedback
            w_ff = 0.5 * v_x * kappa              # Reduced feedforward
            w_fb = k_angular * self._yaw_err_filt   # Moderate feedback
            w_z = float(np.clip(w_ff + w_fb, -w_max, w_max))

        # CRITICAL: Check if we're waiting for inference after backup
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
                carb.log_warn(f"[{self.get_agent_name()}] *** DISCARDING waypoints for commands: Generation that started at {waypoint_gen_start} (before backup at {self._backup_completion_frame}). Still waiting (frame={current_step}) ***")
                # Don't update commands - keep using zero (but still return generation info for tracking)
                with self._inference_lock:
                    self._current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # CRITICAL: Check one more time before updating commands (backup might have started during command calculation)
        if self._is_backing_up:
            carb.log_warn(f"[{self.get_agent_name()}] Backup detected after command calculation - discarding model commands and setting backup (frame={current_step})")
            with self._inference_lock:
                self._current_action = np.array([self._backup_speed, 0.0, 0.0], dtype=np.float32)
            # Don't print anything - backup is active, don't show model commands
            return None, False, None

        # Only update _current_action if NOT backing up and NOT skipping commands (waiting or stale waypoints)
        if not self._is_backing_up and not should_skip_commands:
            with self._inference_lock:
                self._current_action = np.array([v_x, v_y, w_z], dtype=np.float32)
            # Only print if backup is still not active
            if not self._is_backing_up:
                print(f"v_x={v_x:.3f}, v_y={v_y:.3f}, w_z={w_z:.3f}")
        else:
            # Backup or waiting - set appropriate commands
            if self._is_backing_up:
                with self._inference_lock:
                    self._current_action = np.array([self._backup_speed, 0.0, 0.0], dtype=np.float32)
            elif self._waiting_for_inference_after_backup:
                with self._inference_lock:
                    self._current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        print(
            f"v_x={v_x:.3f} v_y={v_y:.3f} w_z={w_z:.3f}"
        )

        
        p_cur, R_cur, quat_wxyz = self._get_pose_R_quat()
        print(f"p_cur={p_cur}")
        
        # Return VLM generation start step, model-state availability, and pose captured at generation start
        # (pose is captured in model when generation actually starts, matching the generation step)
        return vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose
        
    # --- Velocity helpers -----------------------------------------------------
    def _smooth_stop(self, dt: float):
        self._cmd_vx = self._slew(self._cmd_vx, 0.0, self._max_accel_vx, self._max_decel_vx, dt, self._deadband_v)
        self._cmd_vy = self._slew(self._cmd_vy, 0.0, self._max_accel_vy, self._max_decel_vy, dt, self._deadband_v)
        self._cmd_wz = self._slew(self._cmd_wz, 0.0, self._max_accel_wz, self._max_decel_wz, dt, self._deadband_w)

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
        # Apply deadband filter with small threshold to filter noise
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
            self._current_action = np.array([0.0, 0.0, 0.0])  # [v_x, v_y, v_theta]
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        self._cmd_wz = 0.0
        self._base_command[:] = 0.0
        self._policy.forward(0.0, self._base_command)

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
