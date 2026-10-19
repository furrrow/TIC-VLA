"""Load and validate the benchmark YAML without importing Isaac Sim."""

import math
import os
import re
from pathlib import Path

import yaml


GLOBAL_KEYS = {
    "seed", "navigation_method", "inference_method", "success_threshold",
    "moving_away_distance_buffer_m", "moving_away_grace_seconds",
    "moving_away_min_progress_m",
}
EPISODE_KEYS = {
    "scene", "start", "start_yaw", "goal", "instruction", "timeout",
    "num_people", "robot_type", "success_threshold",
    "scripted_humans", "physical_human_metrics",
    "moving_away_distance_buffer_m", "moving_away_grace_seconds",
    "moving_away_min_progress_m",
}
_VARIABLE = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z_0-9]*")

APP_CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 640, "height": 480}


def benchmark_tmp_dir(config_file_path, run_id):
    """Keep generated commands and YAML beside the source config, independent of cwd."""
    return Path(config_file_path).resolve().parent / "tmp" / str(run_id)


def asset_path(environment_name):
    """Resolve an asset path supplied by the benchmark environment."""
    return os.path.expanduser(os.path.expandvars(os.environ[environment_name]))


def _expand(value):
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if _VARIABLE.search(expanded):
            raise ValueError(f"unresolved environment variable in {value!r}")
        return expanded
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _number(value, field, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return value


def _integer(value, field, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _position(value, field):
    if not isinstance(value, list) or len(value) not in (2, 3):
        raise ValueError(f"{field} must contain two or three coordinates")
    for coordinate in value:
        _number(coordinate, field)
    return value


def load_benchmark_config(path):
    """Return a validated mapping with defaults applied to every episode."""
    with Path(path).open(encoding="utf-8") as stream:
        config = _expand(yaml.safe_load(stream))
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a YAML mapping")

    seed = _integer(config.get("seed", 42), "seed")
    threshold = _number(config.get("success_threshold", 1.5), "success_threshold", positive=True)
    method = config.get("navigation_method", "ticvla_ros")
    if not isinstance(method, str) or method.lower() not in ("ticvla", "ticvla_ros"):
        raise ValueError("navigation_method must be 'ticvla' or 'ticvla_ros'")
    inference_method = config.get("inference_method", "TIC-VLA")
    if not isinstance(inference_method, str) or not inference_method.strip():
        raise ValueError("inference_method must be a nonempty string")
    validated = {
        "seed": seed,
        "navigation_method": method.lower(),
        "inference_method": inference_method.strip(),
        "success_threshold": threshold,
        "moving_away_distance_buffer_m": _number(
            config.get("moving_away_distance_buffer_m", 0.5),
            "moving_away_distance_buffer_m", positive=True,
        ),
        "moving_away_grace_seconds": _number(
            config.get("moving_away_grace_seconds", 1.0),
            "moving_away_grace_seconds", positive=True,
        ),
        "moving_away_min_progress_m": _number(
            config.get("moving_away_min_progress_m", 0.5),
            "moving_away_min_progress_m", positive=True,
        ),
    }

    for name, episode in config.items():
        if name in GLOBAL_KEYS:
            continue
        if not isinstance(name, str) or not name:
            raise ValueError("episode names must be nonempty strings")
        if not isinstance(episode, dict):
            raise ValueError(f"episode {name!r} must be a mapping")
        unknown = set(episode) - EPISODE_KEYS
        if unknown:
            raise ValueError(f"episode {name!r} has unknown fields: {', '.join(sorted(map(str, unknown)))}")
        for field in ("scene", "start", "goal", "timeout", "robot_type"):
            if field not in episode:
                raise ValueError(f"episode {name!r} is missing {field}")
        entry = dict(episode)
        for field in ("scene", "robot_type"):
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise ValueError(f"episode {name!r} {field} must be a nonempty string")
        if entry["robot_type"].lower() not in ("spot", "nova_carter"):
            raise ValueError(f"episode {name!r} robot_type must be 'spot' or 'nova_carter'")
        for field in ("start", "goal"):
            _position(entry[field], f"episode {name!r} {field}")
        _number(entry["timeout"], f"episode {name!r} timeout", positive=True)
        entry["start_yaw"] = _number(entry.get("start_yaw", 0.0), f"episode {name!r} start_yaw")
        entry["num_people"] = _integer(entry.get("num_people", 3), f"episode {name!r} num_people")
        humans = entry.get("scripted_humans")
        if humans is not None:
            if not isinstance(humans, list) or len(humans) != entry["num_people"]:
                raise ValueError(f"episode {name!r} scripted_humans must contain num_people entries")
            for index, human in enumerate(humans):
                label = f"episode {name!r} scripted_humans[{index}]"
                if not isinstance(human, dict) or "position" not in human:
                    raise ValueError(f"{label} must contain position")
                if set(human) - {"position", "yaw_degrees", "initial_idle", "walk_route"}:
                    raise ValueError(f"{label} has unknown fields")
                if len(_position(human["position"], f"{label} position")) != 3:
                    raise ValueError(f"{label} position must have x, y, z")
                human["yaw_degrees"] = _number(human.get("yaw_degrees", 0.0), f"{label} yaw_degrees")
                human["initial_idle"] = _number(human.get("initial_idle", 3.0), f"{label} initial_idle")
                if human["initial_idle"] < 0:
                    raise ValueError(f"{label} initial_idle must be nonnegative")
                route = human.get("walk_route", [])
                if not isinstance(route, list):
                    raise ValueError(f"{label} walk_route must be a list")
                for point in route:
                    if len(_position(point, f"{label} walk_route point")) != 3:
                        raise ValueError(f"{label} walk_route points must have x, y, z")
        metrics = entry.get("physical_human_metrics", False)
        if not isinstance(metrics, bool):
            raise ValueError(f"episode {name!r} physical_human_metrics must be boolean")
        entry["success_threshold"] = _number(
            entry.get("success_threshold", threshold), f"episode {name!r} success_threshold", positive=True
        )
        for field in (
            "moving_away_distance_buffer_m", "moving_away_grace_seconds",
            "moving_away_min_progress_m",
        ):
            entry[field] = _number(
                entry.get(field, validated[field]), f"episode {name!r} {field}", positive=True
            )
        entry["instruction"] = entry.get("instruction", "")
        if not isinstance(entry["instruction"], str):
            raise ValueError(f"episode {name!r} instruction must be a string")
        validated[name] = entry

    if len(validated) == len(GLOBAL_KEYS):
        raise ValueError("benchmark config contains no episodes")
    return validated


def select_episodes(config, *, start_from=None, reverse=False):
    """Return selected (name, config) pairs in execution order."""
    episodes = [(name, value) for name, value in config.items() if name not in GLOBAL_KEYS]
    if reverse:
        episodes.reverse()
    if start_from is not None:
        names = [name for name, _ in episodes]
        if start_from not in names:
            raise ValueError(f"start episode {start_from!r} not found in benchmark config")
        episodes = episodes[names.index(start_from):]
    return episodes


def get_episode(config, name):
    for episode_name, episode in select_episodes(config):
        if episode_name == name:
            return episode
    raise ValueError(f"episode {name!r} not found in benchmark config")
