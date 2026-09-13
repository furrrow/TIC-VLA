from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from ticvla import TICVLA


# ============================================================
# Waypoint -> velocity controller
# ============================================================

class WaypointController:
    def __init__(
        self,
        lookahead: float = 1.0,
        k_angular: float = 0.8,
        alpha_filter: float = 0.35,
        v_max: float = 1.5,
        w_max: float = 1.2,
    ):
        self.lookahead = lookahead
        self.k_angular = k_angular
        self.alpha_filter = alpha_filter
        self.v_max = v_max
        self.w_max = w_max

        self.yaw_err_filt = None

    def reset(self):
        self.yaw_err_filt = None

    def __call__(self, waypoints: np.ndarray) -> tuple[float, float]:
        """
        Convert TIC-VLA trajectory to:
            linear velocity v [m/s]
            angular velocity omega [rad/s]

        waypoints shape:
            (T, 2)
        """

        wps = np.asarray(waypoints, dtype=np.float32)

        if wps.ndim != 2 or wps.shape[1] != 2:
            raise ValueError(
                f"Expected waypoints shape (T, 2), got {wps.shape}"
            )

        if len(wps) < 5:
            return 0.0, 0.0

        eps = 1e-3

        # ----------------------------------------------------
        # Compute distance along predicted trajectory
        # ----------------------------------------------------

        delta = np.diff(wps, axis=0)

        segment_lengths = np.hypot(
            delta[:, 0],
            delta[:, 1],
        )

        cumulative_distance = np.concatenate(
            [[0.0], np.cumsum(segment_lengths)]
        )

        # ----------------------------------------------------
        # Pick lookahead waypoint
        # ----------------------------------------------------

        j = int(
            np.searchsorted(
                cumulative_distance,
                self.lookahead,
                side="left",
            )
        )

        j = int(
            np.clip(
                j,
                2,
                len(wps) - 3,
            )
        )

        xL = float(wps[j, 0])
        yL = float(wps[j, 1])

        L = float(
            np.hypot(
                xL,
                yL,
            )
        )

        if L < eps:
            return 0.0, 0.0

        # ----------------------------------------------------
        # Heading error
        # ----------------------------------------------------

        yaw_err = math.atan2(
            yL,
            xL,
        )

        if self.yaw_err_filt is None:
            self.yaw_err_filt = yaw_err

        angular_difference = math.atan2(
            math.sin(
                yaw_err - self.yaw_err_filt
            ),
            math.cos(
                yaw_err - self.yaw_err_filt
            ),
        )

        self.yaw_err_filt += (
            self.alpha_filter
            * angular_difference
        )

        # ----------------------------------------------------
        # Pure pursuit curvature
        # ----------------------------------------------------

        curvature = (
            2.0
            * yL
            / (L * L)
        )

        # Slow down for high curvature
        v_curvature_limit = (
            self.w_max
            / (abs(curvature) + eps)
        )

        v_cmd = float(
            np.clip(
                min(
                    self.v_max,
                    v_curvature_limit,
                ),
                0.0,
                self.v_max,
            )
        )

        # ----------------------------------------------------
        # Angular velocity
        # ----------------------------------------------------

        omega_feedforward = (
            0.5
            * v_cmd
            * curvature
        )

        omega_feedback = (
            self.k_angular
            * self.yaw_err_filt
        )

        w_cmd = float(
            np.clip(
                omega_feedforward
                + omega_feedback,
                -self.w_max,
                self.w_max,
            )
        )

        return v_cmd, w_cmd


# ============================================================
# TIC-VLA loading
# ============================================================

def load_ticvla(
    checkpoint_path: str,
    base_model: str,
    device: str,
) -> TICVLA:

    print()
    print("========================================")
    print("Loading TIC-VLA")
    print("========================================")

    print("Base model:")
    print(base_model)

    model = TICVLA(
        model_path=base_model,
        device=device,
        num_action_chunks=30,
    )

    print()
    print("Loading checkpoint:")
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = checkpoint["state_dict"]

    state_dict = {
        k[len("model."):]: v
        for k, v in state_dict.items()
    }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print()
    print("TIC-VLA loaded successfully.")
    print()

    return model


# ============================================================
# Save OpenCV frame temporarily
# ============================================================

def save_temp_frame(
    frame_bgr: np.ndarray,
    path: Path,
) -> None:

    ok = cv2.imwrite(
        str(path),
        frame_bgr,
    )

    if not ok:
        raise RuntimeError(
            f"Failed to save frame to {path}"
        )


# ============================================================
# Draw simple trajectory visualization
# ============================================================

def draw_waypoints_simple(
    frame: np.ndarray,
    waypoints: np.ndarray,
) -> np.ndarray:
    """
    Simple bird's-eye-like visualization drawn in the lower center
    of the image.

    This is NOT camera calibration projection.

    It is only useful for quickly seeing predicted trajectory shape.
    """

    output = frame.copy()

    h, w = output.shape[:2]

    origin_x = w // 2
    origin_y = h - 40

    pixels_per_meter = 75.0

    previous_point = None

    for wp in waypoints:

        forward = float(wp[0])
        lateral = float(wp[1])

        px = int(
            origin_x
            - lateral * pixels_per_meter
        )

        py = int(
            origin_y
            - forward * pixels_per_meter
        )

        if (
            px < 0
            or px >= w
            or py < 0
            or py >= h
        ):
            continue

        cv2.circle(
            output,
            (px, py),
            3,
            (0, 255, 255),
            -1,
        )

        if previous_point is not None:

            cv2.line(
                output,
                previous_point,
                (px, py),
                (0, 255, 255),
                2,
            )

        previous_point = (
            px,
            py,
        )

    return output


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Run TIC-VLA inference over a video."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path.",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/TIC-VLA-model.ckpt",
    )

    parser.add_argument(
        "--base-model",
        default="OpenGVLab/InternVL3-1B",
    )

    parser.add_argument(
        "--instruction",
        default="Move forward safely and avoid obstacles.",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=10,
        help=(
            "Run inference every N video frames. "
            "Example: --skip 10."
        ),
    )

    parser.add_argument(
        "--output",
        default="ticvla_output.mp4",
    )

    parser.add_argument(
        "--v-max",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--w-max",
        type=float,
        default=1.2,
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> int:

    args = parse_args()

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    video_path = Path(args.video)

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video does not exist: "
            f"{video_path.resolve()}"
        )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{checkpoint_path.resolve()}"
        )

    # --------------------------------------------------------
    # Load TIC-VLA
    # --------------------------------------------------------

    model = load_ticvla(
        checkpoint_path=str(checkpoint_path),
        base_model=args.base_model,
        device=args.device,
    )

    controller = WaypointController(
        v_max=args.v_max,
        w_max=args.w_max,
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    input_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"Video FPS: {input_fps:.2f}"
    )

    print(
        f"Video frames: {frame_count}"
    )

    print(
        f"Resolution: "
        f"{frame_width}x{frame_height}"
    )

    # --------------------------------------------------------
    # Output video
    # --------------------------------------------------------

    output_fps = (
        input_fps
        / args.skip
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        args.output,
        fourcc,
        output_fps,
        (
            frame_width,
            frame_height,
        ),
    )

    # --------------------------------------------------------
    # Frame history
    #
    # TIC-VLA wants temporal visual context.
    #
    # We keep the latest four inference frames.
    # --------------------------------------------------------

    history = deque(
        maxlen=4
    )

    # --------------------------------------------------------
    # Temporary frame directory
    #
    # TIC-VLA predict() expects image paths.
    # --------------------------------------------------------

    temp_dir = Path(
        "DynaNav/tmp_video_frames"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_idx = 0
    inference_idx = 0

    peak_memory_gb = 0.0

    previous_time = time.perf_counter()

    try:

        while True:

            ret, frame_bgr = cap.read()

            if not ret:
                break

            frame_idx += 1

            # ----------------------------------------------
            # Skip frames
            # ----------------------------------------------

            if (
                frame_idx % args.skip
                != 0
            ):
                continue

            inference_idx += 1

            # ----------------------------------------------
            # Save current frame
            # ----------------------------------------------

            frame_path = (
                temp_dir
                / f"frame_{inference_idx:06d}.jpg"
            )

            save_temp_frame(
                frame_bgr,
                frame_path,
            )

            history.append(
                str(frame_path)
            )

            # ----------------------------------------------
            # Fill startup history
            #
            # Until we have four distinct frames,
            # repeat the oldest/current frame.
            # ----------------------------------------------

            if len(history) < 4:

                image_paths = (
                    [history[0]]
                    * (4 - len(history))
                    + list(history)
                )

            else:

                image_paths = list(
                    history
                )

            # ----------------------------------------------
            # Dummy robot state
            #
            # Offline video has no odometry.
            # ----------------------------------------------

            robot_state = torch.tensor(
                [
                    0.0,   # vx
                    0.0,   # vy
                    0.0,   # vz
                    0.0,   # yaw rate
                    0.0,   # dx
                    0.0,   # dy
                ],
                dtype=torch.float32,
            )

            # ----------------------------------------------
            # Run inference
            # ----------------------------------------------

            start = time.perf_counter()

            with torch.inference_mode():

                (
                    response,
                    waypoint_tensor,
                    _generation_start_step,
                    kv_cache_available,
                    _generation_start_pose,
                ) = model.predict(
                    image_paths=image_paths,
                    delayed_image_paths=image_paths,
                    instruction=args.instruction,
                    robot_state=robot_state,
                    time_delay=0.0,
                    robot_type="wheeled robot",
                )

            inference_time = (
                time.perf_counter()
                - start
            )

            # ----------------------------------------------
            # Extract trajectory
            # ----------------------------------------------

            waypoints = (
                waypoint_tensor[0]
                .float()
                .cpu()
                .numpy()
            )

            # ----------------------------------------------
            # Waypoint -> velocity
            # ----------------------------------------------

            v_cmd, w_cmd = controller(
                waypoints
            )

            # ----------------------------------------------
            # Print results
            # ----------------------------------------------

            print()
            print(
                "========================================"
            )
            print(
                f"Inference {inference_idx}"
            )
            print(
                "========================================"
            )

            print(
                f"Video frame: {frame_idx}"
            )

            print(
                f"Inference time: "
                f"{inference_time:.3f} s"
            )

            print(
                f"Inference rate: "
                f"{1 / inference_time:.2f} Hz"
            )

            print(
                f"Waypoint shape: "
                f"{waypoints.shape}"
            )

            print(
                f"v = {v_cmd:.3f} m/s"
            )

            print(
                f"omega = "
                f"{w_cmd:.3f} rad/s"
            )

            print(
                f"KV cache: "
                f"{kv_cache_available}"
            )

            print()
            print(
                "VLM response:"
            )
            print(
                response
            )

            # ----------------------------------------------
            # Visualization
            # ----------------------------------------------

            output_frame = (
                draw_waypoints_simple(
                    frame_bgr,
                    waypoints,
                )
            )

            cv2.putText(
                output_frame,
                f"v: {v_cmd:.2f} m/s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                output_frame,
                f"omega: {w_cmd:.2f} rad/s",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                output_frame,
                (
                    f"inference: "
                    f"{inference_time:.2f}s"
                ),
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            writer.write(
                output_frame
            )

            # ----------------------------------------------
            # Optional display
            # ----------------------------------------------

            if args.show:

                cv2.imshow(
                    "TIC-VLA",
                    output_frame,
                )

                key = cv2.waitKey(1)

                if (
                    key == ord("q")
                    or key == 27
                ):
                    break

            # ----------------------------------------------
            # GPU memory
            # ----------------------------------------------

            if torch.cuda.is_available():

                current_memory = (
                    torch.cuda.max_memory_allocated()
                    / 1024**3
                )

                peak_memory_gb = max(
                    peak_memory_gb,
                    current_memory,
                )

    except KeyboardInterrupt:

        print()
        print(
            "KeyboardInterrupt received."
        )

    finally:

        cap.release()

        writer.release()

        cv2.destroyAllWindows()

    print()
    print(
        "========================================"
    )
    print(
        "Finished"
    )
    print(
        "========================================"
    )

    print(
        f"Processed video frames: "
        f"{frame_idx}"
    )

    print(
        f"TIC-VLA inferences: "
        f"{inference_idx}"
    )

    print(
        f"Peak GPU memory: "
        f"{peak_memory_gb:.2f} GB"
    )

    print(
        f"Output saved to: "
        f"{args.output}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())