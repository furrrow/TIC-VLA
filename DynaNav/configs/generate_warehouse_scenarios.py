#!/usr/bin/env python3
"""Generate controlled benchmark YAML files from the Warehouse template."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from generate_hospital_scenarios import (
    delay_combinations,
    parse_float_schedule,
    parse_people,
    write_config,
)


SCENE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
    "Isaac/5.1/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
)
START = [-10.0, 1.0, 0.01]
GOAL = [-8.2, 8.2, 0.01]
INSTRUCTION = "go to the person wearing a blue shirt and dark trousers."

TARGET = {
    "position": GOAL,
    "yaw_degrees": 270.0,
}

MOVERS = (
    (
        "head_on",
        {
            "position": [-10.7, 4.9, 0.01],
            "walk_route": [[-9.8, 0.1, 0.01]],
            "yaw_degrees": -79.38,
        },
    ),
    (
        "crosser_a",
        {
            "position": [-12.5, 5.2, 0.01],
            "walk_route": [[-6.0, 5.2, 0.01]],
            "yaw_degrees": 0.0,
        },
    ),
    (
        "crosser_b",
        {
            "position": [-7.5, 6.2, 0.01],
            "walk_route": [[-12.5, 6.2, 0.01]],
            "yaw_degrees": 180.0,
        },
    ),
)


def make_humans(num_people: int, delays: tuple[float, ...]) -> list[dict]:
    active_movers = MOVERS[: num_people - 1]
    movers = []
    for (_, template), delay in zip(active_movers, delays, strict=True):
        mover = copy.deepcopy(template)
        mover["initial_idle"] = delay
        movers.append(mover)
    if not movers:
        return [copy.deepcopy(TARGET)]
    return [movers[0], copy.deepcopy(TARGET), *movers[1:]]


def make_config(
    *,
    num_people: int,
    delays: list[tuple[float, ...]],
    seed: int,
    success_threshold: float,
    timeout: float,
    episode_prefix: str,
) -> dict:
    config: dict = {
        "seed": seed,
        "navigation_method": "ticvla_ros",
        "success_threshold": success_threshold,
    }
    width = max(2, len(str(len(delays))))
    for index, delay_values in enumerate(delays, start=1):
        name = f"{episode_prefix}_p{num_people}_{index:0{width}d}"
        config[name] = {
            "scene": SCENE,
            "start": copy.deepcopy(START),
            "start_yaw": 90.0,
            "goal": copy.deepcopy(GOAL),
            "instruction": INSTRUCTION,
            "timeout": timeout,
            "num_people": num_people,
            "robot_type": "nova_carter",
            "scripted_humans": make_humans(num_people, delay_values),
            "physical_human_metrics": True,
        }
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate benchmark YAMLs from the Warehouse Scenario 4 template."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DynaNav/configs/generated_warehouse"),
    )
    parser.add_argument("--people", type=parse_people, default=parse_people("2,3,4"))
    parser.add_argument(
        "--head-on-delays",
        type=parse_float_schedule,
        default=parse_float_schedule("0.2:1.5:20"),
    )
    parser.add_argument(
        "--crosser-a-delays",
        type=parse_float_schedule,
        default=parse_float_schedule("0.2:1.5:20"),
    )
    parser.add_argument(
        "--crosser-b-delays",
        type=parse_float_schedule,
        default=parse_float_schedule("0.5:2.5:20"),
    )
    parser.add_argument(
        "--combination-mode",
        choices=("product", "paired", "stratified"),
        default="stratified",
    )
    parser.add_argument("--difficulty-names", action="store_true")
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--success-threshold", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--episode-prefix", default="episode_warehouse")
    parser.add_argument("--filename-prefix", default="scenario4_warehouse")
    parser.add_argument("--output-name", help="exact filename for a single --people value")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seed < 0:
        raise SystemExit("--seed must be nonnegative")
    if not math.isfinite(args.success_threshold) or args.success_threshold <= 0:
        raise SystemExit("--success-threshold must be positive and finite")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout must be positive and finite")

    schedules = [args.head_on_delays, args.crosser_a_delays, args.crosser_b_delays]
    if args.output_name and len(args.people) != 1:
        raise SystemExit("--output-name requires exactly one --people value")
    for num_people in args.people:
        try:
            delays = list(
                delay_combinations(
                    schedules[: num_people - 1], args.combination_mode, seed=args.seed
                )
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        config = make_config(
            num_people=num_people,
            delays=delays,
            seed=args.seed,
            success_threshold=args.success_threshold,
            timeout=args.timeout,
            episode_prefix=args.episode_prefix,
        )
        difficulty = {1: "target_only", 2: "easy", 3: "medium", 4: "hard"}[num_people]
        label = difficulty if args.difficulty_names else f"people{num_people}"
        filename = args.output_name or f"{args.filename_prefix}_{label}_{len(delays)}trials.yaml"
        output = args.output_dir / filename
        try:
            write_config(output, config, force=args.force)
        except FileExistsError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"wrote {output} ({len(delays)} episodes, {num_people} people)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
