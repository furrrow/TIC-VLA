#!/usr/bin/env python3
"""Run TIC-VLA waypoint inference over video with async VLM state updates."""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from infer_once import (
    _ensure_custom_utils_on_path,
    configure_runtime_cache,
)


DEFAULT_INSTRUCTION = "Move forward safely and avoid obstacles."
DEFAULT_BASE_MODEL = "models/InternVL3-1B"
DEFAULT_CHECKPOINT = "checkpoints/TIC-VLA-model.ckpt"
DEFAULT_HISTORY_LEN = 4
DEFAULT_CACHE_DIR = "./tmp/ticvla_infer_once/cache"

# DEFAULT_VIDEO = "/home/jim/Projects/steernav/assets/corridoor_omni_ft_2_left.mp4"
DEFAULT_VIDEO = "/home/gamma-nav/Documents/Projects/git_repos/steernav/assets/Cars_and_Gasstation.mp4"
# DEFAULT_CAMERA_MATRIX = "/home/jim/Projects/steernav/steernav/cam_matrix.json"
DEFAULT_CAMERA_MATRIX = "/home/gamma-nav/Documents/Projects/git_repos/steernav/steernav/cam_matrix.json"


def async_model_class() -> Any:
    """Load DynaNav's async TICVLA implementation without package collision."""
    root = Path(__file__).resolve().parent
    dynanav_ticvla = root / "DynaNav" / "ticvla.py"
    if not dynanav_ticvla.is_file():
        raise FileNotFoundError(f"Missing async model implementation: {dynanav_ticvla}")
    sys.path.insert(0, str(dynanav_ticvla.parent))
    spec = importlib.util.spec_from_file_location("_ticvla_async_model", dynanav_ticvla)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {dynanav_ticvla}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TICVLA


def load_ticvla(checkpoint_path: str, base_model: str, device: str) -> Any:
    configure_runtime_cache()
    print(f"Loading async TIC-VLA base model: {base_model}")
    model = async_model_class()(model_path=base_model, device=device, num_action_chunks=30)
    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint (missing/unexpected: {len(missing)}/{len(unexpected)})")
        model.eval()
    except BaseException:
        model.cleanup()
        raise
    print("Async TIC-VLA loaded successfully.")
    return model


def make_robot_state(args: argparse.Namespace) -> torch.Tensor:
    # Async DynaNav path accepts [vx, vy, vz, yaw_speed, dx, dy].
    return torch.tensor(
        [args.vx, args.vy, 0.0, args.yaw_rate, args.dx, args.dy],
        dtype=torch.float32,
    )


def save_frame(frame_bgr: np.ndarray, path: Path) -> None:
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Failed to save frame: {path}")


def sample_history_paths(
    history: deque[tuple[int, str]],
    history_len: int,
    interval_frames: int,
) -> list[str]:
    """Match Spot's oldest-to-current temporal sampling from saved frame history."""
    if not history:
        return []

    newest_index = history[-1][0]
    selected: list[str] = []
    seen: set[str] = set()
    offsets = [interval_frames * i for i in range(history_len - 1, -1, -1)]

    for offset in offsets:
        target_index = newest_index - offset
        if target_index > newest_index:
            continue
        if offset and target_index < history[0][0]:
            continue

        best_path = history[-1][1]
        for frame_index, path in reversed(history):
            if frame_index <= target_index:
                best_path = path
                break

        if best_path not in seen:
            selected.append(best_path)
            seen.add(best_path)

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run async TIC-VLA waypoint inference over video frames.",
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip", type=int, default=10)
    parser.add_argument("--history-len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument(
        "--history-interval-seconds",
        type=float,
        default=3.0,
        help="Seconds between VLM context frames, matching Spot's -9s/-6s/-3s/current pattern.",
    )
    parser.add_argument("--max-inferences", type=int, default=0)
    parser.add_argument("--quiet-response", action="store_true", default=True)
    parser.add_argument("--print-response", action="store_false", dest="quiet_response")
    parser.add_argument("--no-waypoint-print", action="store_true")
    parser.add_argument("--metric-waypoint-spacing", type=float, default=1.0)
    parser.add_argument("--camera-matrix", default=DEFAULT_CAMERA_MATRIX)
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--window-name", default="ticvla_async_overlay")
    parser.add_argument("--video-out", help="Optional MP4 path for overlay frames.")
    parser.add_argument("--robot-type", default="legged robot")

    state_group = parser.add_argument_group("robot state")
    state_group.add_argument("--vx", type=float, default=0.0)
    state_group.add_argument("--vy", type=float, default=0.0)
    state_group.add_argument("--yaw-rate", type=float, default=0.0)
    state_group.add_argument("--dx", type=float, default=0.0)
    state_group.add_argument("--dy", type=float, default=0.0)
    state_group.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Override async delay metadata. By default it is inferred from VLM start frames.",
    )
    args = parser.parse_args()

    if args.skip < 1:
        parser.error("--skip must be >= 1")
    if args.history_len < 1:
        parser.error("--history-len must be >= 1")
    if not math.isfinite(args.history_interval_seconds) or args.history_interval_seconds <= 0:
        parser.error("--history-interval-seconds must be finite and > 0")
    if args.max_inferences < 0:
        parser.error("--max-inferences must be >= 0")
    if args.metric_waypoint_spacing <= 0:
        parser.error("--metric-waypoint-spacing must be > 0")
    return args


def main() -> int:
    args = parse_args()
    configure_runtime_cache()
    _ensure_custom_utils_on_path()
    from custom_utils.io_utils import load_calibration, overlay_path

    video_path = Path(args.video)
    checkpoint_path = Path(args.checkpoint)
    base_model_path = Path(args.base_model)
    camera_matrix_path = Path(args.camera_matrix)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path.resolve()}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path.resolve()}")
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_model_path.resolve()}")
    if not camera_matrix_path.is_file():
        raise FileNotFoundError(f"Camera matrix file does not exist: {camera_matrix_path.resolve()}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")

    cam_matrix, _, T_base_from_cam = load_calibration(str(camera_matrix_path))
    T_cam_from_base = np.linalg.inv(T_base_from_cam)
    model = load_ticvla(str(checkpoint_path), str(base_model_path), args.device)
    robot_state = make_robot_state(args)

    cap = cv2.VideoCapture(str(video_path))
    writer = None
    if not cap.isOpened():
        model.cleanup()
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
        model.cleanup()
        cap.release()
        raise RuntimeError("Video has invalid FPS or dimensions.")

    if args.video_out:
        video_out = Path(args.video_out)
        if video_out.resolve() == video_path.resolve():
            model.cleanup()
            cap.release()
            raise ValueError("--video-out must differ from --video")
        video_out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / args.skip,
            (width, height),
        )
        if not writer.isOpened():
            model.cleanup()
            cap.release()
            raise RuntimeError(f"Could not create video output: {video_out}")

    if args.display:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, 720, 720)

    print(f"Opening video: {video_path}")
    interval_frames = max(1, int(round(args.history_interval_seconds * fps)))
    saved_interval_steps = max(1, int(round(interval_frames / args.skip)))
    history_capacity = saved_interval_steps * max(0, args.history_len - 1) + 1
    print(
        f"Sampling every {args.skip} frame(s); history_len={args.history_len}; "
        f"context interval={args.history_interval_seconds:.2f}s"
    )
    print("Async mode: first call warms VLM cache; later calls reuse latest cache.")
    if args.delay is None and (args.dx or args.dy):
        print("Note: --dx/--dy are static for offline video; Spot computes them from pose at runtime.")

    frame_index = -1
    inference_count = 0
    decoded_frame_count = 0
    peak_memory_gb = 0.0
    generation_starts: deque[int] = deque(maxlen=2)
    sampled_paths: deque[tuple[int, str]] = deque(maxlen=history_capacity)
    started_at = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="ticvla-async-video-") as temp_dir:
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                decoded_frame_count += 1
                frame_index += 1
                if decoded_frame_count % args.skip:
                    continue

                frame_path = Path(temp_dir) / f"frame_{frame_index:09d}.jpg"
                save_frame(frame_bgr, frame_path)
                sampled_paths.append((frame_index, str(frame_path)))
                image_paths = sample_history_paths(
                    sampled_paths,
                    args.history_len,
                    interval_frames,
                )
                if not image_paths:
                    raise RuntimeError("No sampled frame paths are available for inference.")

                reference = generation_starts[0] if generation_starts else frame_index
                time_delay = args.delay if args.delay is not None else (frame_index - reference) / fps

                t0 = time.perf_counter()
                with torch.inference_mode():
                    response, waypoint_tensor, generation_step, cache_available, _ = model.predict_async(
                        image_paths=image_paths,
                        delayed_image_paths=image_paths,
                        instruction=args.instruction,
                        robot_state=robot_state,
                        current_step=frame_index,
                        time_delay=time_delay,
                        robot_type=args.robot_type,
                    )
                if generation_step is not None:
                    generation_starts.append(generation_step)

                waypoints = waypoint_tensor.detach().float().cpu().numpy()
                if waypoints.ndim != 3 or waypoints.shape[0] != 1 or waypoints.shape[2] < 2:
                    raise ValueError(f"Expected waypoints shaped (1,T,2+), got {waypoints.shape}")
                waypoints_xy = waypoints[0, :, :2]
                if not np.isfinite(waypoints_xy).all():
                    raise ValueError("Model returned non-finite waypoints.")

                inference_seconds = time.perf_counter() - t0
                inference_count += 1
                path_xy = waypoints_xy * args.metric_waypoint_spacing

                if torch.cuda.is_available():
                    peak_memory_gb = max(
                        peak_memory_gb,
                        torch.cuda.max_memory_allocated() / 1024**3,
                    )

                print(
                    f"\nInference {inference_count} | frame={frame_index} | "
                    f"{inference_seconds:.2f}s | delay={time_delay:.2f}s | "
                    f"KV cache={cache_available}"
                )
                if response and not args.quiet_response:
                    print(response)
                if not args.no_waypoint_print:
                    for step, waypoint in enumerate(path_xy):
                        print(f"{step:02d}: x={waypoint[0]: .4f}, y={waypoint[1]: .4f}")

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                overlay_rgb = overlay_path(path_xy, frame_rgb, cam_matrix, T_cam_from_base)
                if overlay_rgb is None:
                    overlay_rgb = frame_rgb.copy()
                fps_text = f"async inference {1.0 / max(inference_seconds, 1e-6):.2f} Hz"
                cv2.putText(
                    overlay_rgb,
                    fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                if writer:
                    writer.write(overlay_bgr)
                if args.display:
                    cv2.imshow(args.window_name, overlay_bgr)
                    if cv2.waitKey(1) in (ord("q"), 27):
                        break

                if args.max_inferences and inference_count >= args.max_inferences:
                    break
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, stopping.")
        finally:
            cap.release()
            if writer:
                writer.release()
            try:
                model.cleanup()
            finally:
                if args.display:
                    cv2.destroyAllWindows()

    if not inference_count:
        raise RuntimeError("No frames processed; try a smaller --skip.")

    elapsed = time.perf_counter() - started_at
    print(
        f"\nFinished decoded_frames={decoded_frame_count}, "
        f"inferences={inference_count}, elapsed={elapsed:.2f}s"
    )
    print(f"Average wall time per inference: {elapsed / inference_count:.2f}s")
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
    if args.video_out:
        print(f"Wrote overlay video: {args.video_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
