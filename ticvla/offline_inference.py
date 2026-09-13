#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch

from ticvla.models.ticvla import TICVLA


def _extract_submodule_state_dict(state_dict, prefixes):
    """
    Pull one submodule out of a Lightning-style checkpoint.

    Examples:
        model.vlm.foo          -> foo
        model.action_expert.x -> x
    """
    result = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                result[key[len(prefix):]] = value
                break
    return result


def load_model(base_model_path: str,checkpoint_path: str,device: str = "cuda",):
    print(f"Loading base model: {base_model_path}")

    model = TICVLA(
        model_path=base_model_path,
        action_horizon_steps=30,
        action_num_layers=3,
        train_vlm=False,
    )

    print(f"Loading TIC-VLA checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path,map_location="cpu",)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # VLM
    vlm_state = _extract_submodule_state_dict(state_dict,
                                              prefixes=("model.vlm.", "vlm.",),)

    if not vlm_state:
        raise RuntimeError("Could not find VLM weights in checkpoint.")

    missing, unexpected = model.vlm.load_state_dict(vlm_state,strict=False,)

    print(f"VLM loaded: {len(missing)} missing, {len(unexpected)} unexpected")

    # Action expert
    action_state = _extract_submodule_state_dict(state_dict,
                                                 prefixes=("model.action_expert.",
                                                           "action_expert.",),)

    if not action_state:
        raise RuntimeError("Could not find action_expert weights in checkpoint.")

    missing, unexpected = model.action_expert.load_state_dict(action_state,strict=False,)

    print(f"Action Expert loaded: {len(missing)} missing, {len(unexpected)} unexpected")
    model = model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def infer(
    model,
    delayed_images,
    current_image,
    instruction,
    vx=0.0,
    vy=0.0,
    yaw_rate=0.0,
    dx=0.0,
    dy=0.0,
    time_delay=0.0,
):

    device = next(model.parameters()).device

    # TIC-VLA expects:
    #
    # [vx, vy, yaw_speed, delayed->current dx, delayed->current dy]
    #
    robot_state = torch.tensor(
        [vx, vy, yaw_rate, dx, dy],
        dtype=torch.float32,
        device=device,
    )

    response, waypoints, prompt = model.predict(
        delayed_image_paths=[
            str(Path(p))
            for p in delayed_images
        ],
        current_image_path=str(current_image),
        instruction=instruction,
        robot_state=robot_state,
        history=None,
        current_timestamp=None,
        time_delay=float(time_delay),
    )

    # (1, 30, 2) -> (30, 2)
    waypoints = (
        waypoints[0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    return response, waypoints


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--base-model", default="models/InternVL3-1B")
    parser.add_argument("--checkpoint", default="weights/TIC-VLA-model.ckpt")

    parser.add_argument(
        "--current-image",
        required=True,
    )

    parser.add_argument(
        "--delayed-images",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--instruction",
        default="Move forward safely and avoid obstacles.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    model = load_model(
        args.base_model,
        args.checkpoint,
        args.device,
    )

    response, waypoints = infer(
        model=model,

        delayed_images=args.delayed_images,
        current_image=args.current_image,

        instruction=args.instruction,

        # For an initial smoke test:
        vx=0.0,
        vy=0.0,
        yaw_rate=0.0,

        dx=0.0,
        dy=0.0,

        time_delay=0.0,
    )

    np.set_printoptions(
        precision=3,
        suppress=True,
    )

    print("\nVLM response:")
    print(response)

    print("\nDense TIC-VLA waypoints:")
    print(waypoints)

    print("\nshape:", waypoints.shape)


if __name__ == "__main__":
    main()