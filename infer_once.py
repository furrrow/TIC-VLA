#!/usr/bin/env python3
"""Run one TIC-VLA waypoint prediction from image files or a video frame buffer.

This script is intentionally small and controller-friendly: it loads a TIC-VLA
checkpoint, runs a single inference call, prints the predicted relative
waypoints, and can also save them as JSON or CSV.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any
import torch

DEFAULT_INSTRUCTION = "Move forward safely and avoid obstacles."
DEFAULT_BASE_MODEL = "models/InternVL3-1B"
DEFAULT_CHECKPOINT = "checkpoints/TIC-VLA-model.ckpt"
DEFAULT_HISTORY_LEN = 4
DEFAULT_VIDEO = "/home/jim/Projects/steernav/assets/corridoor_omni_ft_2_left.mp4"
DEFAULT_CACHE_DIR = "/tmp/ticvla_infer_once/cache"


class InferenceImages:
    def __init__(
        self,
        delayed: list[Any],
        current: Any,
        delayed_refs: list[str],
        current_ref: str,
        source: str,
    ) -> None:
        self.delayed = delayed
        self.current = current
        self.delayed_refs = delayed_refs
        self.current_ref = current_ref
        self.source = source


def configure_runtime_cache() -> None:
    """Keep model/import caches in a writable location for local one-shot runs."""
    cache_root = Path(os.environ.get("TICVLA_RUNTIME_CACHE", DEFAULT_CACHE_DIR))
    cache_root.mkdir(parents=True, exist_ok=True)
    for env_name, fallback in (
        ("HF_HOME", cache_root / "huggingface"),
        ("HF_MODULES_CACHE", cache_root / "hf_modules"),
        ("TRANSFORMERS_CACHE", cache_root / "transformers"),
        ("MPLCONFIGDIR", cache_root / "matplotlib"),
    ):
        current = os.environ.get(env_name)
        if current:
            current_path = Path(current)
            try:
                current_path.mkdir(parents=True, exist_ok=True)
                test_path = current_path / ".ticvla_write_test"
                test_path.touch(exist_ok=True)
                test_path.unlink(missing_ok=True)
                continue
            except OSError:
                pass
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(fallback)


def _extract_submodule_state_dict(
    state_dict: dict[str, Any],
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                result[key.removeprefix(prefix)] = value
                break
    return result


def load_ticvla(checkpoint_path: str, base_model: str, device: str) -> Any:
    """Load TIC-VLA and its action checkpoint."""
    configure_runtime_cache()
    from ticvla import TICVLA

    print(f"Loading base model: {base_model}")
    model = TICVLA(
        model_path=base_model,
        action_horizon_steps=30,
        action_num_layers=3,
        train_vlm=False,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    vlm_state = _extract_submodule_state_dict(
        state_dict,
        prefixes=("model.vlm.", "model.vlm_model.vlm.", "vlm."),
    )
    action_state = _extract_submodule_state_dict(
        state_dict,
        prefixes=("model.action_expert.", "action_expert."),
    )

    if vlm_state and action_state:
        vlm_missing, vlm_unexpected = model.vlm.load_state_dict(vlm_state, strict=False)
        action_missing, action_unexpected = model.action_expert.load_state_dict(
            action_state,
            strict=False,
        )
        print(
            "Loaded submodule checkpoint "
            f"(VLM missing/unexpected: {len(vlm_missing)}/{len(vlm_unexpected)}, "
            f"action missing/unexpected: {len(action_missing)}/{len(action_unexpected)})"
        )
    else:
        normalized_state = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(normalized_state, strict=False)
        print(
            "Loaded full-model checkpoint "
            f"(missing/unexpected: {len(missing)}/{len(unexpected)})"
        )

    model = model.to(device)
    model.eval()
    print("TIC-VLA loaded successfully.")
    return model


def _ensure_custom_utils_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    custom_utils_root = repo_root / "custom_utils"
    if custom_utils_root.is_dir():
        sys.path.insert(0, str(custom_utils_root))


def read_video_frame_buffer(
    video_path: str,
    frame_index: int,
    history_len: int,
) -> InferenceImages:
    """Decode video once and return an in-memory history ending at frame_index."""
    configure_runtime_cache()

    _ensure_custom_utils_on_path()
    from custom_utils.stream_handler import FrameStatus, InputStreamHandler

    if frame_index < 0:
        raise ValueError("--frame-index must be >= 0.")

    frame_buffer: deque[Any] = deque(maxlen=history_len)
    frame_refs: deque[str] = deque(maxlen=history_len)
    src = InputStreamHandler(kind="video", video_path=video_path, skip_n_fr=1)
    src.open()
    try:
        read_index = 0
        while read_index <= frame_index:
            frame_read = src.read()
            if frame_read.status == FrameStatus.NO_FRAME:
                continue
            if frame_read.status == FrameStatus.EOS:
                raise RuntimeError(
                    f"Video ended before frame {frame_index}: {video_path}"
                )
            if frame_read.status != FrameStatus.OK or frame_read.frame is None:
                raise RuntimeError(f"Could not read frame {read_index} from {video_path}")

            frame_buffer.append(frame_read.frame)
            frame_refs.append(f"{video_path}#frame={read_index}")
            read_index += 1
    finally:
        src.close()

    delayed_images = list(frame_buffer)
    delayed_refs = list(frame_refs)
    if not delayed_images:
        raise RuntimeError(f"No frames were decoded from video: {video_path}")

    current_image = delayed_images[-1]
    current_ref = delayed_refs[-1]
    if len(delayed_images) < history_len:
        pad_count = history_len - len(delayed_images)
        delayed_images = [delayed_images[0]] * pad_count + delayed_images
        delayed_refs = [delayed_refs[0]] * pad_count + delayed_refs

    return InferenceImages(
        delayed=delayed_images,
        current=current_image,
        delayed_refs=delayed_refs,
        current_ref=current_ref,
        source="video-buffer",
    )


def resolve_image_inputs(args: argparse.Namespace) -> InferenceImages:
    """Return decoded frames or file paths from supported CLI forms."""
    if args.video and not args.current_image and not args.images:
        images = read_video_frame_buffer(
            video_path=args.video,
            frame_index=args.frame_index,
            history_len=args.history_len if args.pad_history else 1,
        )
        print(f"Decoded video frame buffer ending at frame {args.frame_index}.")
        return images
    elif args.current_image:
        current_image = args.current_image
        delayed_images = args.delayed_images or args.images or [current_image]
    elif args.images:
        current_image = args.images[-1]
        delayed_images = args.images
    else:
        raise ValueError("Provide either --current-image or --images.")

    if not delayed_images:
        delayed_images = [current_image]

    delayed_images = [str(Path(path)) for path in delayed_images]
    current_image = str(Path(current_image))

    if args.pad_history and len(delayed_images) < args.history_len:
        delayed_images = [delayed_images[0]] * (args.history_len - len(delayed_images)) + delayed_images

    missing_paths = [
        path
        for path in [current_image, *delayed_images]
        if not Path(path).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError("Missing image file(s): " + ", ".join(missing_paths))

    return InferenceImages(
        delayed=delayed_images,
        current=current_image,
        delayed_refs=delayed_images,
        current_ref=current_image,
        source="image-paths",
    )


def make_robot_state(args: argparse.Namespace) -> Any:
    # TIC-VLA expects [vx, vy, yaw_speed, dx, dy]. time_delay is passed separately.
    import torch

    return torch.tensor(
        [args.vx, args.vy, args.yaw_rate, args.dx, args.dy],
        dtype=torch.float32,
    )


def _image_to_pixel_values(image: Any, device: Any, input_size: int = 448, max_num: int = 1) -> Any:
    """Preprocess an image path, PIL image, or RGB ndarray for InternVL."""
    import numpy as np
    import torch
    from PIL import Image
    from ticvla.utils.vision import build_transform, dynamic_preprocess, load_image

    if isinstance(image, (str, Path)):
        return load_image(str(image), input_size=input_size, max_num=max_num).to(torch.bfloat16).to(device)

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB ndarray with shape HxWx3, got {image.shape}")
        pil_image = Image.fromarray(image.astype("uint8"), mode="RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(
        pil_image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    pixel_values = [transform(tile) for tile in tiles]
    return torch.stack(pixel_values).to(torch.bfloat16).to(device)


def _load_images_from_memory_or_paths(
    images: list[Any],
    device: Any,
    input_size: int = 448,
    max_num: int = 1,
) -> tuple[list[Any], list[int]]:
    pixel_values_list = []
    num_patches_list = []
    for image in images:
        pixel_values = _image_to_pixel_values(
            image,
            device=device,
            input_size=input_size,
            max_num=max_num,
        )
        if pixel_values is not None and pixel_values.numel() > 0:
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
    if not pixel_values_list:
        raise RuntimeError("No valid frames were available for TIC-VLA inference.")
    return pixel_values_list, num_patches_list


def _normalize_past_key_values(past_key_values: Any) -> Any:
    if past_key_values is not None and hasattr(past_key_values, "layers"):
        return tuple((layer.keys, layer.values) for layer in past_key_values.layers)
    if past_key_values is None:
        return tuple()
    return past_key_values


def _infer_once_in_memory(
    model: Any,
    images: InferenceImages,
    instruction: str,
    robot_state: Any,
    time_delay: float,
) -> tuple[str, Any, str]:
    """Run TIC-VLA from already decoded frames without writing image files."""
    import torch

    device = model.device
    instruction = instruction or DEFAULT_INSTRUCTION

    delayed_pixel_values_list, delayed_num_patches_list = _load_images_from_memory_or_paths(
        images.delayed,
        device=device,
        input_size=448,
        max_num=1,
    )

    system_text = (
        "You are a physical mobile robot assigned to perform navigation tasks.\n"
        "You are provided with a video consisting of visual observations, including historical and current frames.\n"
    )
    model.vlm.system_message = system_text

    user_text = (
        f"The navigation instruction is: {instruction}"
        "\nUse reasoning to predict the future target waypoints. "
        "First describe the relevant visual/navigation evidence, then return the future target waypoints "
        "for the next 3s, 6s, and 9s in format: (x, y, theta). "
        "Each waypoint represents the cumulative offset from the current position (total displacement over 3s, 6s, or 9s),"
        "where x is positive for forward, y is positive for left, and theta is the heading angle in radians."
    )

    generation_prompt = (
        "".join(f"Frame {i}: <image>\n" for i in range(len(delayed_num_patches_list)))
        + user_text
    )
    full_prompt_text = f"SYSTEM:\n{system_text}\n\nUSER:\n{user_text}"

    generation_config = dict(max_new_tokens=200, do_sample=True, temperature=0.7)
    delayed_pixel_values = torch.cat(delayed_pixel_values_list, dim=0)
    generated_response = model.vlm.chat(
        model.tokenizer,
        delayed_pixel_values,
        generation_prompt,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=delayed_num_patches_list,
    )

    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ref}
                for ref in images.delayed_refs
            ]
            + [{"type": "text", "text": user_text}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": generated_response}]},
    ]

    text_batch = model.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if isinstance(text_batch, list):
        text_batch = text_batch[0] if text_batch else ""

    img_start_token, img_end_token, img_context_token = "<img>", "</img>", "<IMG_CONTEXT>"
    num_image_token = 256
    query = text_batch
    if "<image>" in query:
        for tiles_for_image in delayed_num_patches_list:
            if tiles_for_image > 0:
                image_tokens = (
                    img_start_token
                    + (img_context_token * (num_image_token * tiles_for_image))
                    + img_end_token
                )
                query = query.replace("<image>", image_tokens, 1)
    else:
        prepend = []
        for tiles_for_image in delayed_num_patches_list:
            if tiles_for_image > 0:
                prepend.append(
                    img_start_token
                    + (img_context_token * (num_image_token * tiles_for_image))
                    + img_end_token
                )
        if prepend:
            query = "\n".join(prepend) + "\n" + query

    model.tokenizer.padding_side = "left"
    tokenized = model.tokenizer([query], return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    image_flags = torch.ones(delayed_pixel_values.shape[0], 1, dtype=torch.long, device=device)

    vlm_outputs = model.vlm(
        pixel_values=delayed_pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_flags=image_flags,
        return_dict=True,
        use_cache=True,
    )
    past_key_values = _normalize_past_key_values(vlm_outputs.past_key_values)

    current_pixel_values = _image_to_pixel_values(
        images.current,
        device=device,
        input_size=448,
        max_num=1,
    )
    image_embeds = model.vlm.extract_feature(current_pixel_values)
    image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1]).unsqueeze(0)

    if robot_state is None:
        robot_state_tensor = torch.zeros(5, device=device, dtype=torch.bfloat16)
    else:
        if not torch.is_tensor(robot_state):
            robot_state = torch.tensor(robot_state)
        robot_state_tensor = robot_state.to(device=device, dtype=torch.bfloat16).view(-1)

    time_delay_tensor = torch.tensor([time_delay], device=device, dtype=torch.bfloat16)
    state = torch.cat([robot_state_tensor, time_delay_tensor], dim=0).unsqueeze(0).unsqueeze(-1)

    waypoints = model.action_expert(image_embeds, state, kv_cache=past_key_values)
    return generated_response, waypoints, full_prompt_text


def infer_once(
    model: Any,
    images: InferenceImages,
    instruction: str,
    robot_state: Any,
    time_delay: float,
) -> tuple[str, Any, str]:
    import torch

    with torch.inference_mode():
        if images.source == "image-paths":
            response, waypoint_tensor, prompt = model.predict(
                delayed_image_paths=images.delayed,
                current_image_path=images.current,
                instruction=instruction,
                robot_state=robot_state,
                time_delay=float(time_delay),
            )
        else:
            response, waypoint_tensor, prompt = _infer_once_in_memory(
                model=model,
                images=images,
                instruction=instruction,
                robot_state=robot_state,
                time_delay=float(time_delay),
            )
    waypoints = waypoint_tensor[0].detach().float().cpu().numpy()
    return response, waypoints, prompt


def write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_csv(path: str, waypoints: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x_forward_m", "y_left_m"])
        for step, waypoint in enumerate(waypoints):
            writer.writerow([step, float(waypoint[0]), float(waypoint[1])])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one TIC-VLA inference for a robot navigation frame.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default="cuda:0")

    video_group = parser.add_argument_group("video input")
    video_group.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help="Video path used when no image paths are provided.",
    )
    video_group.add_argument(
        "--frame-index",
        type=int,
        default=10,
        help="Zero-based video frame to use as the current buffered frame.",
    )
    video_group.add_argument(
        "--extract-only",
        action="store_true",
        help="Decode/resolve inputs and exit before model inference.",
    )

    image_group = parser.add_argument_group("images")
    image_group.add_argument(
        "--current-image",
        help="Current camera frame used by the action head.",
    )
    image_group.add_argument(
        "--delayed-images",
        nargs="+",
        help="Historical/delayed frames in chronological order for VLM context.",
    )
    image_group.add_argument(
        "--images",
        nargs="+",
        help="Shortcut: chronological frames; the last image is used as current.",
    )
    image_group.add_argument(
        "--history-len",
        type=int,
        default=DEFAULT_HISTORY_LEN,
        help="Minimum delayed-frame count when --pad-history is enabled.",
    )
    image_group.add_argument(
        "--pad-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pad short image histories by repeating the first frame.",
    )

    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)

    state_group = parser.add_argument_group("robot state")
    state_group.add_argument("--vx", type=float, default=0.0, help="Forward velocity in m/s.")
    state_group.add_argument("--vy", type=float, default=0.0, help="Leftward velocity in m/s.")
    state_group.add_argument("--yaw-rate", type=float, default=0.0, help="Yaw rate in rad/s.")
    state_group.add_argument("--dx", type=float, default=0.0, help="Delayed-to-current forward displacement in m.")
    state_group.add_argument("--dy", type=float, default=0.0, help="Delayed-to-current leftward displacement in m.")
    state_group.add_argument("--delay", type=float, default=0.0, help="Visual/state delay in seconds.")

    output_group = parser.add_argument_group("outputs")
    output_group.add_argument("--json-out", help="Optional JSON output path.")
    output_group.add_argument("--csv-out", help="Optional CSV output path.")
    output_group.add_argument(
        "--quiet-response",
        action="store_true",
        help="Do not print the VLM text response.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.history_len < 1:
        raise ValueError("--history-len must be >= 1.")

    images = resolve_image_inputs(args)
    if args.extract_only:
        print(f"Input source: {images.source}")
        print(f"Current image: {images.current_ref}")
        print("Delayed images:")
        for ref in images.delayed_refs:
            print(f"  {ref}")
        return 0

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")

    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if not Path(args.base_model).exists():
        raise FileNotFoundError(f"Base model path does not exist: {args.base_model}")

    robot_state = make_robot_state(args)

    model = load_ticvla(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        device=args.device,
    )

    print("\nRunning inference...")
    response, waypoints, _ = infer_once(
        model=model,
        images=images,
        instruction=args.instruction,
        robot_state=robot_state,
        time_delay=args.delay,
    )

    if not args.quiet_response:
        print("\n================ VLM RESPONSE ================")
        print(response)

    print("\n================ ACTION WAYPOINTS ================")
    print("frame convention: x=forward meters, y=left meters")
    for step, waypoint in enumerate(waypoints):
        print(f"{step:02d}: x={waypoint[0]: .4f}, y={waypoint[1]: .4f}")

    payload = {
        "instruction": args.instruction,
        "image_source": images.source,
        "current_image": images.current_ref,
        "delayed_images": images.delayed_refs,
        "robot_state": {
            "vx": args.vx,
            "vy": args.vy,
            "yaw_rate": args.yaw_rate,
            "dx": args.dx,
            "dy": args.dy,
            "delay": args.delay,
        },
        "response": response,
        "waypoints": [
            {"step": step, "x_forward_m": float(wp[0]), "y_left_m": float(wp[1])}
            for step, wp in enumerate(waypoints)
        ],
    }

    if args.json_out:
        write_json(args.json_out, payload)
        print(f"\nWrote JSON: {args.json_out}")
    if args.csv_out:
        write_csv(args.csv_out, waypoints)
        print(f"Wrote CSV: {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
