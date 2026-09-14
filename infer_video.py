#!/usr/bin/env python3
"""Run TIC-VLA waypoint inference over a video stream.

This is the runtime-oriented companion to infer_once.py: it reads frames
directly from video, keeps a small in-memory history, and prints/saves waypoint
predictions without writing intermediate images. It can also project the
predicted path onto the video frame using the same overlay helper as
video_inference.py.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from infer_once import (
    InferenceImages,
    _ensure_custom_utils_on_path,
    configure_runtime_cache,
    infer_once,
    load_ticvla,
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


def make_robot_state(args: argparse.Namespace) -> torch.Tensor:
    # TIC-VLA expects [vx, vy, yaw_speed, dx, dy]. time_delay is passed separately.
    return torch.tensor(
        [args.vx, args.vy, args.yaw_rate, args.dx, args.dy],
        dtype=torch.float32,
    )


def pad_history(
    frames: list[Any],
    refs: list[str],
    history_len: int,
) -> tuple[list[Any], list[str]]:
    if not frames:
        raise ValueError("Cannot pad an empty frame history.")
    if len(frames) >= history_len:
        return frames, refs
    pad_count = history_len - len(frames)
    return [frames[0]] * pad_count + frames, [refs[0]] * pad_count + refs


def write_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "inference_index",
                "frame_index",
                "step",
                "x_forward_m",
                "y_left_m",
                "inference_seconds",
            ]
        )


def append_csv(
    path: Path,
    inference_index: int,
    frame_index: int,
    waypoints: Any,
    inference_seconds: float,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for step, waypoint in enumerate(waypoints):
            writer.writerow(
                [
                    inference_index,
                    frame_index,
                    step,
                    float(waypoint[0]),
                    float(waypoint[1]),
                    inference_seconds,
                ]
            )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TIC-VLA waypoint inference over video frames.",
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--skip",
        type=int,
        default=10,
        help="Run inference every N decoded video frames.",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=DEFAULT_HISTORY_LEN,
        help="Number of sampled frames to keep for delayed VLM context.",
    )
    parser.add_argument(
        "--max-inferences",
        type=int,
        default=0,
        help="Stop after this many inferences. 0 means process until EOS.",
    )
    parser.add_argument(
        "--quiet-response",
        action="store_true",
        default=True,
        help="Do not print VLM reasoning text.",
    )
    parser.add_argument(
        "--print-response",
        action="store_false",
        dest="quiet_response",
        help="Print VLM reasoning text for each inference.",
    )
    parser.add_argument(
        "--no-waypoint-print",
        action="store_true",
        help="Suppress per-inference waypoint arrays on stdout.",
    )
    parser.add_argument(
        "--metric-waypoint-spacing",
        type=float,
        default=1.0,
        help="Scale waypoint coordinates before overlay. TIC-VLA default here is 1 meter.",
    )
    parser.add_argument(
        "--camera-matrix",
        default=DEFAULT_CAMERA_MATRIX,
        help="Calibration JSON passed to custom_utils.io_utils.load_calibration.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the video frame with projected waypoint overlay.",
    )
    parser.add_argument(
        "--window-name",
        default="ticvla_overlay",
        help="OpenCV window name when --display is enabled.",
    )

    state_group = parser.add_argument_group("robot state")
    state_group.add_argument("--vx", type=float, default=0.0)
    state_group.add_argument("--vy", type=float, default=0.0)
    state_group.add_argument("--yaw-rate", type=float, default=0.0)
    state_group.add_argument("--dx", type=float, default=0.0)
    state_group.add_argument("--dy", type=float, default=0.0)
    state_group.add_argument("--delay", type=float, default=0.0)

    output_group = parser.add_argument_group("outputs")
    output_group.add_argument("--csv-out", help="Optional waypoint CSV output path.")
    output_group.add_argument("--jsonl-out", help="Optional JSONL output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_runtime_cache()

    if args.skip < 1:
        raise ValueError("--skip must be >= 1.")
    if args.history_len < 1:
        raise ValueError("--history-len must be >= 1.")
    if args.max_inferences < 0:
        raise ValueError("--max-inferences must be >= 0.")
    if args.metric_waypoint_spacing <= 0:
        raise ValueError("--metric-waypoint-spacing must be > 0.")

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

    _ensure_custom_utils_on_path()
    from custom_utils.io_utils import load_calibration, overlay_path
    from custom_utils.stream_handler import FrameStatus, InputStreamHandler

    cam_matrix, _, T_base_from_cam = load_calibration(str(camera_matrix_path))
    T_cam_from_base = np.linalg.inv(T_base_from_cam)

    model = load_ticvla(
        checkpoint_path=str(checkpoint_path),
        base_model=str(base_model_path),
        device=args.device,
    )
    robot_state = make_robot_state(args)

    csv_path = Path(args.csv_out) if args.csv_out else None
    jsonl_path = Path(args.jsonl_out) if args.jsonl_out else None
    if csv_path:
        write_csv_header(csv_path)
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")

    src = InputStreamHandler(
        kind="video",
        video_path=str(video_path),
        skip_n_fr=args.skip,
    )
    print(f"Opening video: {video_path}")
    print(f"Sampling every {args.skip} frame(s); history_len={args.history_len}")
    print(f"Overlay metric_waypoint_spacing={args.metric_waypoint_spacing}")

    if args.display:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, 720, 720)

    sampled_frames: deque[Any] = deque(maxlen=args.history_len)
    sampled_refs: deque[str] = deque(maxlen=args.history_len)
    decoded_frame_count = 0
    inference_count = 0
    peak_memory_gb = 0.0
    started_at = time.perf_counter()

    src.open()
    try:
        while True:
            frame_read = src.read()
            if frame_read.status == FrameStatus.EOS:
                break

            # For video mode, NO_FRAME means InputStreamHandler consumed a frame
            # that was skipped by skip_n_fr.
            decoded_frame_count += 1
            if frame_read.status == FrameStatus.NO_FRAME:
                continue
            if frame_read.status != FrameStatus.OK or frame_read.frame is None:
                raise RuntimeError(f"Failed to decode frame {decoded_frame_count - 1}.")

            frame_index = decoded_frame_count - 1
            frame_ref = f"{video_path}#frame={frame_index}"
            sampled_frames.append(frame_read.frame)
            sampled_refs.append(frame_ref)

            delayed_images, delayed_refs = pad_history(
                list(sampled_frames),
                list(sampled_refs),
                args.history_len,
            )
            images = InferenceImages(
                delayed=delayed_images,
                current=frame_read.frame,
                delayed_refs=delayed_refs,
                current_ref=frame_ref,
                source="video-buffer",
            )

            t0 = time.perf_counter()
            response, waypoints, _ = infer_once(
                model=model,
                images=images,
                instruction=args.instruction,
                robot_state=robot_state,
                time_delay=args.delay,
            )
            inference_seconds = time.perf_counter() - t0
            inference_count += 1
            path_xy = waypoints[:, :2] * args.metric_waypoint_spacing

            if torch.cuda.is_available():
                peak_memory_gb = max(
                    peak_memory_gb,
                    torch.cuda.max_memory_allocated() / 1024**3,
                )

            print(
                f"\nInference {inference_count} | frame={frame_index} | "
                f"{inference_seconds:.2f}s"
            )
            if not args.quiet_response:
                print(response)
            if not args.no_waypoint_print:
                for step, waypoint in enumerate(path_xy):
                    print(f"{step:02d}: x={waypoint[0]: .4f}, y={waypoint[1]: .4f}")

            overlay_img = overlay_path(
                trajectories=path_xy,
                img=frame_read.frame,
                cam_matrix=cam_matrix,
                T_cam_from_base=T_cam_from_base,
            )
            if overlay_img is None:
                overlay_img = frame_read.frame.copy()

            fps_text = f"inference {1.0 / max(inference_seconds, 1e-6):.2f} Hz"
            cv2.putText(
                overlay_img,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            if args.display:
                cv2.imshow(args.window_name, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) in (ord("q"), 27):
                    break

            if csv_path:
                append_csv(
                    csv_path,
                    inference_count,
                    frame_index,
                    waypoints,
                    inference_seconds,
                )
            if jsonl_path:
                append_jsonl(
                    jsonl_path,
                    {
                        "inference_index": inference_count,
                        "frame_index": frame_index,
                        "current_image": frame_ref,
                        "delayed_images": delayed_refs,
                        "inference_seconds": inference_seconds,
                        "response": response,
                        "waypoints": [
                            {
                                "step": step,
                                "x_forward_m": float(waypoint[0]),
                                "y_left_m": float(waypoint[1]),
                            }
                            for step, waypoint in enumerate(waypoints)
                        ],
                    },
                )

            if args.max_inferences and inference_count >= args.max_inferences:
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping.")
    finally:
        src.close()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started_at
    print(
        f"\nFinished decoded_frames={decoded_frame_count}, "
        f"inferences={inference_count}, elapsed={elapsed:.2f}s"
    )
    if inference_count:
        print(f"Average wall time per inference: {elapsed / inference_count:.2f}s")
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
    if csv_path:
        print(f"Wrote CSV: {csv_path}")
    if jsonl_path:
        print(f"Wrote JSONL: {jsonl_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
