import argparse
import math

import numpy as np
import torch

# Because this script lives in DynaNav/, this imports DynaNav/ticvla.py
from ticvla import TICVLA


class WaypointController:
    """
    Standalone version of the waypoint -> (v, omega) controller
    used by the TIC-VLA Nova Carter DynaNav behavior.
    """

    def __init__(
        self,
        lookahead=1.0,
        k_angular=0.8,
        alpha_filter=0.35,
        v_max=1.5,
        w_max=1.2,
    ):
        self.lookahead = lookahead
        self.k_angular = k_angular
        self.alpha_filter = alpha_filter
        self.v_max = v_max
        self.w_max = w_max

        self.yaw_err_filt = None

    def __call__(self, waypoints):
        """
        waypoints: numpy array of shape (T, 2)

        Returns:
            v_cmd: linear velocity [m/s]
            w_cmd: angular velocity [rad/s]
        """

        wps = np.asarray(waypoints, dtype=np.float32)

        if wps.ndim != 2 or wps.shape[1] != 2:
            raise ValueError(
                f"Expected waypoints of shape (T, 2), got {wps.shape}"
            )

        T = len(wps)

        if T < 5:
            raise ValueError("Need at least 5 waypoints for this controller.")

        eps = 1e-3

        # Arc length along predicted waypoint trajectory.
        inc = np.diff(wps, axis=0)
        seg = np.hypot(inc[:, 0], inc[:, 1])
        s = np.concatenate([[0.0], np.cumsum(seg)])

        # Pick waypoint approximately lookahead meters along trajectory.
        j = int(np.searchsorted(s, self.lookahead, side="left"))
        j = int(np.clip(j, 2, T - 3))

        xL = float(wps[j, 0])
        yL = float(wps[j, 1])

        L = float(np.hypot(xL, yL))

        if L < eps:
            return 0.0, 0.0

        # Heading toward lookahead point.
        yaw_err = math.atan2(yL, xL)

        # Filter heading error.
        if self.yaw_err_filt is None:
            self.yaw_err_filt = yaw_err

        e = math.atan2(
            math.sin(yaw_err - self.yaw_err_filt),
            math.cos(yaw_err - self.yaw_err_filt),
        )

        self.yaw_err_filt += self.alpha_filter * e

        # Pure-pursuit curvature.
        kappa = 2.0 * yL / (L * L)

        # Slow down on sharp turns.
        v_kappa = self.w_max / (abs(kappa) + eps)

        v_cmd = float(
            np.clip(
                min(self.v_max, v_kappa),
                0.0,
                self.v_max,
            )
        )

        # Feedforward curvature + heading feedback.
        w_ff = 0.5 * v_cmd * kappa
        w_fb = self.k_angular * self.yaw_err_filt

        w_cmd = float(
            np.clip(
                w_ff + w_fb,
                -self.w_max,
                self.w_max,
            )
        )

        return v_cmd, w_cmd


def load_ticvla(checkpoint_path, base_model, device):
    print(f"Loading base model: {base_model}")

    model = TICVLA(
        model_path=base_model,
        device=device,
        num_action_chunks=30,
    )

    print(f"Loading TIC-VLA checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = checkpoint["state_dict"]

    # Matches the checkpoint loading performed by the repository's
    # Nova Carter behavior.
    state_dict = {
        k[len("model."):]: v
        for k, v in state_dict.items()
    }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print("TIC-VLA loaded successfully.")

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--base-model",
        default="OpenGVLab/InternVL3-1B",
    )

    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help=(
            "Images in chronological order. Ideally: "
            "t-9s, t-6s, t-3s, current."
        ),
    )

    parser.add_argument(
        "--instruction",
        default="Move forward safely and efficiently.",
    )

    # Robot state
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--delay", type=float, default=0.0)

    parser.add_argument(
        "--device",
        default="cuda:0",
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

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available.")

    # For a basic smoke test with only one image, repeat it four times.
    # This is only for verifying that inference works.
    if len(args.images) == 1:
        image_paths = args.images * 4
    else:
        image_paths = args.images

    model = load_ticvla(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        device=args.device,
    )

    # DynaNav accepts:
    #
    # [vx, vy, vz, yaw_speed, dx, dy]
    #
    # and internally reduces this to the five state values used by
    # the trained action model.
    robot_state = torch.tensor(
        [
            args.vx,
            args.vy,
            0.0,            # vz
            args.yaw_rate,
            args.dx,
            args.dy,
        ],
        dtype=torch.float32,
    )

    print("\nRunning inference...")

    with torch.inference_mode():
        (
            response,
            waypoints,
            _generation_start_step,
            kv_cache_available,
            _generation_start_pose,
        ) = model.predict(
            image_paths=image_paths,
            delayed_image_paths=image_paths,
            instruction=args.instruction,
            robot_state=robot_state,
            time_delay=args.delay,
            robot_type="wheeled robot",
        )

    # Shape: (1, 30, 2)
    wps = waypoints[0].float().cpu().numpy()

    print("\n================ VLM RESPONSE ================")
    print(response)

    print("\n================ ACTION WAYPOINTS ================")
    print("Shape:", wps.shape)
    print(wps)

    print("\nKV cache available:", kv_cache_available)

    controller = WaypointController(
        v_max=args.v_max,
        w_max=args.w_max,
    )

    v_cmd, w_cmd = controller(wps)

    print("\n================ CONTROL COMMAND ================")
    print(f"linear velocity v : {v_cmd:.4f} m/s")
    print(f"angular velocity ω: {w_cmd:.4f} rad/s")


if __name__ == "__main__":
    main()