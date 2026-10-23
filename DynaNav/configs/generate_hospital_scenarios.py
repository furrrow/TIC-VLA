#!/usr/bin/env python3
"""Generate controlled Scenario 3 benchmark YAML files.

The stationary target is always present. Increasing ``num_people`` adds the
head-on pedestrian, the first lateral crosser, and then the second lateral
crosser. Delay schedules may be combined as a Cartesian grid, paired by index,
or independently shuffled as reproducible timing strata.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import random
from pathlib import Path
from typing import Iterable

import yaml


SCENE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
    "Isaac/5.1/Isaac/Environments/Hospital/hospital.usd"
)
START = [-26.0, 11.0, 0.01]
GOAL = [-19.44101607487867, 11.0, 0.01]
TARGET_POSITION = [-18.8, 11.0, 0.01]
INSTRUCTION = "go to the security guard at the end of the hallway."

TARGET = {
    "position": TARGET_POSITION,
    "yaw_degrees": 180.0,
}

WHITE_COAT_GOAL = [-19.44101607487867, 9.85, 0.01]
WHITE_COAT_INSTRUCTION = "go to the person wearing a long white coat and light trousers."
WHITE_COAT_TARGET = {
    "position": WHITE_COAT_GOAL,
    "yaw_degrees": 180.0,
}

MOVERS = (
    (
        "head_on",
        {
            "position": [-21.8, 11.0, 0.01],
            "walk_route": [
                [-26.8, 11.0, 0.01],
                [-21.8, 11.0, 0.01],
            ],
            "yaw_degrees": 180.0,
        },
    ),
    (
        "crosser_a",
        {
            "position": [-23.4, 9.55, 0.01],
            "walk_route": [
                [-23.4, 11.85, 0.01],
                [-23.4, 9.55, 0.01],
            ],
            "yaw_degrees": 90.0,
        },
    ),
    (
        "crosser_b",
        {
            "position": [-22.3, 11.85, 0.01],
            "walk_route": [
                [-22.3, 9.55, 0.01],
                [-22.3, 11.85, 0.01],
            ],
            "yaw_degrees": -90.0,
        },
    ),
)

SOUTHBOUND_MOVERS = (
    MOVERS[0],
    (
        "southbound_a",
        {
            "position": [-21.8, 11.85, 0.01],
            "walk_route": [[-26.8, 11.85, 0.01]],
            "yaw_degrees": 180.0,
        },
    ),
    (
        "southbound_b",
        {
            "position": [-21.8, 9.55, 0.01],
            "walk_route": [[-26.8, 9.55, 0.01]],
            "yaw_degrees": 180.0,
        },
    ),
)


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):  # noqa: ANN001 - PyYAML override
        return True


def _parse_float_schedule(
    value: str, *, label: str, nonnegative: bool
) -> list[float]:
    """Parse ``a,b,c`` or an inclusive ``start:stop:count`` schedule."""
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError(f"{label} schedule cannot be empty")

    try:
        if ":" in value:
            parts = value.split(":")
            if len(parts) != 3:
                raise ValueError("range must be start:stop:count")
            start, stop = map(float, parts[:2])
            count = int(parts[2])
            if count < 1:
                raise ValueError("range count must be at least 1")
            if count == 1:
                values = [start]
            else:
                step = (stop - start) / (count - 1)
                values = [start + index * step for index in range(count)]
        else:
            values = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid {label} schedule {value!r}: {exc}"
        ) from exc

    if not all(math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError(f"{label} values must be finite")
    if nonnegative and any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(f"{label} values must be nonnegative")
    return [round(item, 6) for item in values]


def parse_float_schedule(value: str) -> list[float]:
    return _parse_float_schedule(value, label="delay", nonnegative=True)


def parse_yaw_schedule(value: str) -> list[float]:
    return _parse_float_schedule(value, label="yaw", nonnegative=False)


def parse_people(value: str) -> list[int]:
    try:
        people = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("people must be comma-separated integers") from exc
    if not people or any(item < 1 or item > 4 for item in people):
        raise argparse.ArgumentTypeError("people values must be between 1 and 4")
    if len(set(people)) != len(people):
        raise argparse.ArgumentTypeError("people values must not contain duplicates")
    return people


def paired_delays(schedules: list[list[float]]) -> Iterable[tuple[float, ...]]:
    """Pair schedules by index, broadcasting schedules containing one value."""
    if not schedules:
        return [()]
    length = max(map(len, schedules))
    invalid = [len(schedule) for schedule in schedules if len(schedule) not in (1, length)]
    if invalid:
        raise ValueError(
            "paired schedules must have one value or share the same length; "
            f"received lengths {[len(schedule) for schedule in schedules]}"
        )
    return zip(*(schedule * length if len(schedule) == 1 else schedule for schedule in schedules))


def stratified_delays(
    schedules: list[list[float]], *, seed: int
) -> Iterable[tuple[float, ...]]:
    """Pair delay strata while independently shuffling later movers."""
    if not schedules:
        return [()]
    length = max(map(len, schedules))
    invalid = [len(schedule) for schedule in schedules if len(schedule) not in (1, length)]
    if invalid:
        raise ValueError(
            "stratified schedules must have one value or share the same length; "
            f"received lengths {[len(schedule) for schedule in schedules]}"
        )

    columns = []
    for index, schedule in enumerate(schedules):
        values = schedule * length if len(schedule) == 1 else list(schedule)
        if index > 0 and len(schedule) > 1:
            random.Random(seed + index * 1_000_003).shuffle(values)
        columns.append(values)
    return zip(*columns)


def delay_combinations(
    schedules: list[list[float]], mode: str, *, seed: int
) -> Iterable[tuple[float, ...]]:
    if not schedules:
        return [()]
    if mode == "product":
        return itertools.product(*schedules)
    if mode == "stratified":
        return stratified_delays(schedules, seed=seed)
    return paired_delays(schedules)


def make_humans(
    num_people: int,
    delays: tuple[float, ...],
    movers: tuple[tuple[str, dict], ...] = MOVERS,
    target: dict = TARGET,
) -> list[dict]:
    active_movers = movers[: num_people - 1]
    movers = []
    for (_, template), delay in zip(active_movers, delays, strict=True):
        mover = copy.deepcopy(template)
        mover["initial_idle"] = delay
        movers.append(mover)

    if not movers:
        return [copy.deepcopy(target)]
    return [movers[0], copy.deepcopy(target), *movers[1:]]


def make_config(
    *,
    num_people: int,
    delays: list[tuple[float, ...]],
    seed: int,
    success_threshold: float,
    timeout: float,
    episode_prefix: str,
    movers: tuple[tuple[str, dict], ...] = MOVERS,
    goal: list[float] = GOAL,
    instruction: str = INSTRUCTION,
    target: dict = TARGET,
    start_yaws: tuple[float, ...] | list[float] = (0.0,),
) -> dict:
    config: dict = {
        "seed": seed,
        "navigation_method": "ticvla_ros",
        "success_threshold": success_threshold,
    }
    if len(delays) % len(start_yaws) == 0:
        episode_yaws = list(start_yaws) * (len(delays) // len(start_yaws))
    else:
        raise ValueError(
            "start yaw schedule length must divide the episode count; "
            f"received {len(start_yaws)} values for {len(delays)} episodes"
        )
    width = max(2, len(str(len(delays))))
    for index, (delay_values, start_yaw) in enumerate(
        zip(delays, episode_yaws, strict=True), start=1
    ):
        name = f"{episode_prefix}_p{num_people}_{index:0{width}d}"
        config[name] = {
            "scene": SCENE,
            "start": copy.deepcopy(START),
            "start_yaw": start_yaw,
            "goal": copy.deepcopy(goal),
            "instruction": instruction,
            "timeout": timeout,
            "num_people": num_people,
            "robot_type": "nova_carter",
            "scripted_humans": make_humans(num_people, delay_values, movers, target),
            "physical_human_metrics": True,
        }
    return config


def write_config(path: Path, config: dict, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(
            config,
            stream,
            Dumper=_NoAliasSafeDumper,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate controlled benchmark YAMLs based on ICRA v4 Scenario 3."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DynaNav/configs/generated_scenario3"),
        help="directory for generated YAML files",
    )
    parser.add_argument(
        "--people",
        type=parse_people,
        default=parse_people("2,3,4"),
        help="comma-separated total people counts (1-4); the target is included",
    )
    parser.add_argument(
        "--head-on-delays",
        type=parse_float_schedule,
        default=parse_float_schedule("0.2:1.5:5"),
        help="comma list or inclusive start:stop:count schedule",
    )
    parser.add_argument(
        "--crosser-a-delays",
        type=parse_float_schedule,
        default=parse_float_schedule("4.302"),
        help="delay schedule for the first lateral crosser",
    )
    parser.add_argument(
        "--crosser-b-delays",
        type=parse_float_schedule,
        default=parse_float_schedule("5.349"),
        help="delay schedule for the second lateral crosser",
    )
    parser.add_argument(
        "--combination-mode",
        choices=("product", "paired", "stratified"),
        default="product",
        help="use every combination, pair by index, or shuffle paired delay strata",
    )
    parser.add_argument(
        "--lateral-motion",
        choices=("crossing", "southbound"),
        default="crossing",
        help=(
            "use the original back-and-forth lateral routes or one-way "
            "southbound walkers on separate fixed-y lanes"
        ),
    )
    parser.add_argument(
        "--start-yaw-degrees",
        type=parse_yaw_schedule,
        default=parse_yaw_schedule("0.0"),
        help=(
            "robot start yaw values in degrees; shorter schedules repeat when "
            "their length divides the episode count"
        ),
    )
    parser.add_argument(
        "--target-profile",
        choices=("security-guard", "white-coat"),
        default="security-guard",
        help="select the centerline security guard or offset white-coat target",
    )
    parser.add_argument(
        "--difficulty-names",
        action="store_true",
        help="name 2/3/4-person files easy/medium/hard",
    )
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--success-threshold", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--episode-prefix", default="episode_s3")
    parser.add_argument(
        "--filename-prefix",
        default="scenario3",
        help="prefix used for generated YAML filenames",
    )
    parser.add_argument(
        "--output-name",
        help="exact filename for a single --people value",
    )
    parser.add_argument("--force", action="store_true", help="replace existing output files")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seed < 0:
        raise SystemExit("--seed must be nonnegative")
    if not math.isfinite(args.success_threshold) or args.success_threshold <= 0:
        raise SystemExit("--success-threshold must be positive and finite")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout must be positive and finite")

    all_schedules = [args.head_on_delays, args.crosser_a_delays, args.crosser_b_delays]
    movers = SOUTHBOUND_MOVERS if args.lateral_motion == "southbound" else MOVERS
    if args.target_profile == "white-coat":
        goal = WHITE_COAT_GOAL
        instruction = WHITE_COAT_INSTRUCTION
        target = WHITE_COAT_TARGET
    else:
        goal = GOAL
        instruction = INSTRUCTION
        target = TARGET
    if args.output_name and len(args.people) != 1:
        raise SystemExit("--output-name requires exactly one --people value")
    for num_people in args.people:
        active_schedules = all_schedules[: num_people - 1]
        try:
            delays = list(
                delay_combinations(active_schedules, args.combination_mode, seed=args.seed)
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if len(delays) % len(args.start_yaw_degrees) != 0:
            raise SystemExit(
                "--start-yaw-degrees schedule length must divide the episode "
                f"count ({len(delays)} for this configuration)"
            )
        config = make_config(
            num_people=num_people,
            delays=delays,
            seed=args.seed,
            success_threshold=args.success_threshold,
            timeout=args.timeout,
            episode_prefix=args.episode_prefix,
            movers=movers,
            goal=goal,
            instruction=instruction,
            target=target,
            start_yaws=args.start_yaw_degrees,
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
