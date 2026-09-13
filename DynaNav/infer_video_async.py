"""Offline video inference using DynaNav's polling-based async API.

Run from the repository root with this file in DynaNav/. The first API call
blocks for VLM warmup; later calls reuse cached VLM state while generation runs
in the background. Output contains sampled frames at input_fps / skip.
Robot state is zero because the video has no odometry. History retains the
original four sampled-frame behavior, rather than calibrated 3-second spacing.
"""
from __future__ import annotations

import argparse
import math
import importlib.util
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

def async_model_class():
    """Load the DynaNav implementation explicitly, avoiding package name collision."""
    directory = Path(__file__).resolve().parent
    if not (directory / "ticvla.py").is_file():
        raise FileNotFoundError("Place this script in the repository's DynaNav directory.")
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("_ticvla_video_async_model", directory / "ticvla.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TICVLA


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

    model = async_model_class()(
        model_path=base_model,
        device=device,
        num_action_chunks=30,
    )

    print()
    print("Loading checkpoint:")
    print(checkpoint_path)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        state_dict = checkpoint["state_dict"]

        state_dict = {
            k.removeprefix("model."): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(state_dict, strict=True)
        model.eval()
    except BaseException:
        model.cleanup()
        raise

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

    parser.add_argument("--max-inferences", type=int, default=None,
                        help="Stop after N sampled frames (useful for a smoke run).")
    args = parser.parse_args()
    if args.skip < 1 or (args.max_inferences is not None and args.max_inferences < 1):
        parser.error("--skip and --max-inferences must be positive")
    if not all(math.isfinite(x) and x > 0 for x in (args.v_max, args.w_max)):
        parser.error("--v-max and --w-max must be finite and positive")
    return args


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_args()
    video_path = Path(args.video).resolve()
    output_path = Path(args.output).resolve()
    checkpoint_path = Path(args.checkpoint)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if output_path == video_path:
        raise ValueError("Output must differ from input video")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path.resolve()}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate a CUDA-capable environment and driver.")

    cap = cv2.VideoCapture(str(video_path))
    writer = None
    model = None
    frame_idx = inference_idx = 0
    controller = WaypointController(v_max=args.v_max, w_max=args.w_max)
    generation_starts = deque(maxlen=2)
    # Unique immutable files remain available until the background worker has stopped.
    with tempfile.TemporaryDirectory(prefix="ticvla-video-") as temp_dir:
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
                raise RuntimeError("Video has invalid FPS or dimensions")
            print(f"Video: {width}x{height}, {fps:.3f} FPS")
            model = load_ticvla(str(checkpoint_path), args.base_model, args.device)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / args.skip, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output_path}")
            history = deque(maxlen=4)
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % args.skip:
                    continue
                frame_path = Path(temp_dir) / f"frame_{frame_idx:09d}.jpg"
                save_temp_frame(frame_bgr, frame_path)
                history.append(str(frame_path))
                image_paths = [history[0]] * (4 - len(history)) + list(history)
                # Follow DynaNav's second-to-last generation-start convention.
                # Video timestamps, not wall-clock inference latency, define delay.
                reference = generation_starts[0] if generation_starts else frame_idx
                delay = (frame_idx - reference) / fps
                start = time.perf_counter()
                with torch.inference_mode():
                    response, waypoint_tensor, generation_step, cache_available, _ = model.predict_async(
                        image_paths=image_paths,
                        delayed_image_paths=list(image_paths),
                        instruction=args.instruction,
                        robot_state=torch.zeros(6, dtype=torch.float32),
                        current_step=frame_idx,
                        time_delay=delay,
                        robot_type="wheeled robot",
                    )
                if generation_step is not None:
                    generation_starts.append(generation_step)
                # Copy to CPU also waits for the returned CUDA waypoint computation.
                waypoints = waypoint_tensor.detach().float().cpu().numpy()
                if waypoints.ndim != 3 or waypoints.shape[0] != 1 or waypoints.shape[2] not in (2, 3):
                    raise ValueError(f"Expected (1,T,2) or (1,T,3) waypoints, got {waypoints.shape}")
                waypoints = waypoints[0, :, :2]  # controller uses forward/left; ignore heading
                if not np.isfinite(waypoints).all():
                    raise ValueError("Model returned non-finite waypoints")
                elapsed = time.perf_counter() - start
                v_cmd, w_cmd = controller(waypoints)
                inference_idx += 1
                print(f"Inference {inference_idx}, frame {frame_idx}: {elapsed:.3f}s, "
                      f"delay={delay:.3f}s, v={v_cmd:.3f} m/s, omega={w_cmd:.3f} rad/s, "
                      f"KV cache={cache_available}")
                if response:
                    print(f"VLM response: {response}")
                output_frame = draw_waypoints_simple(frame_bgr, waypoints)
                for y, text in ((40, f"v: {v_cmd:.2f} m/s"),
                                (75, f"omega: {w_cmd:.2f} rad/s"),
                                (110, f"inference: {elapsed:.2f}s")):
                    cv2.putText(output_frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                writer.write(output_frame)
                if args.show:
                    cv2.imshow("TIC-VLA", output_frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if args.max_inferences and inference_idx >= args.max_inferences:
                    break
        except KeyboardInterrupt:
            print("Interrupted; finalizing output.")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            try:
                if model is not None:
                    model.cleanup()
            finally:
                if args.show:
                    cv2.destroyAllWindows()
    if not inference_idx:
        raise RuntimeError("No frames processed; try a smaller --skip")
    print(f"Finished: {inference_idx} inferences, {frame_idx} input frames. Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
