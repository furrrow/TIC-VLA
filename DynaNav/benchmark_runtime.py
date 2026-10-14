"""In-process episode context shared with Isaac-created behavior scripts."""

from dataclasses import dataclass


SIMULATION_FPS = 30


@dataclass
class BehaviorContext:
    run_id: str
    goal: tuple[float, float, float]
    instruction: str
    log_dir: str
    frame_count: int = 0
    failure_reason: str | None = None


_contexts: dict[str, BehaviorContext] = {}


def register(prim_path: str, context: BehaviorContext) -> None:
    _contexts[str(prim_path)] = context


def get(prim_path: str) -> BehaviorContext:
    return _contexts[str(prim_path)]


def unregister(prim_path: str | None) -> None:
    if prim_path is not None:
        _contexts.pop(str(prim_path), None)
