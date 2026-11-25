"""In-process episode context shared with Isaac-created behavior scripts."""

from dataclasses import dataclass
from typing import Callable


SIMULATION_FPS = 30


@dataclass
class BehaviorContext:
    run_id: str
    episode_name: str
    goal: tuple[float, float, float]
    instruction: str
    log_dir: str
    frame_count: int = 0
    failure_reason: str | None = None
    ros_cleanup: Callable[[], None] | None = None


_contexts: dict[str, BehaviorContext] = {}


def register(prim_path: str, context: BehaviorContext) -> None:
    _contexts[str(prim_path)] = context


def get(prim_path: str) -> BehaviorContext:
    return _contexts[str(prim_path)]


def unregister(prim_path: str | None) -> None:
    if prim_path is not None:
        _contexts.pop(str(prim_path), None)


class SimulationElapsed:
    """Elapsed simulator seconds; paused updates do not consume a budget."""

    def __init__(self):
        self.start = None
        self.last = None
        self.elapsed = 0.0

    def update(self, now):
        import math
        now = float(now)
        if not math.isfinite(now):
            raise ValueError('Nonfinite simulation time')
        if self.last is not None and now < self.last - 1e-9:
            raise ValueError('Simulation clock moved backward during navigation')
        if self.start is None:
            self.start = now
        self.last = now
        self.elapsed = max(0.0, now - self.start)
        return self.elapsed


class SustainedCondition:
    """Track a continuous predicate in simulation seconds, clearing on reset."""

    def __init__(self):
        self.since = None
        self.last = None

    def update(self, now, active):
        import math
        now = float(now)
        if not math.isfinite(now):
            raise ValueError('Nonfinite simulation time')
        if self.last is not None and now < self.last:
            self.since = None
        self.last = now
        if not active:
            self.since = None
            return 0.0
        if self.since is None:
            self.since = now
        return max(0.0, now - self.since)
