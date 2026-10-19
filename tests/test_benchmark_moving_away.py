import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile


DYNANAV_DIR = Path(__file__).resolve().parents[1] / "DynaNav"
sys.path.insert(0, str(DYNANAV_DIR))

import benchmark
import benchmark_config
import benchmark_reporting


class MovingAwayTrackingTests(unittest.TestCase):
    def make_runner(self):
        runner = benchmark.BenchmarkRunner.__new__(benchmark.BenchmarkRunner)
        runner.state = benchmark.EpisodeState(
            moving_away_distance_buffer_m=1.0,
            moving_away_grace_frames=3,
            moving_away_min_progress_m=0.5,
        )
        return runner

    def test_sustained_regression_terminates_after_grace_period(self):
        runner = self.make_runner()
        distances = [10.0, 9.4, 8.0, 9.2, 9.3, 9.4]
        decisions = [runner._update_goal_distance_tracking(value) for value in distances]

        self.assertEqual(decisions, [False, False, False, False, False, True])
        self.assertEqual(runner.state.closest_goal_distance_m, 8.0)

    def test_recovery_inside_buffer_resets_grace_period(self):
        runner = self.make_runner()
        distances = [10.0, 8.0, 9.2, 9.3, 8.5, 9.2, 9.3]
        decisions = [runner._update_goal_distance_tracking(value) for value in distances]

        self.assertFalse(any(decisions))
        self.assertEqual(runner.state.moving_away_consecutive_frames, 2)

    def test_detector_does_not_arm_before_minimum_progress(self):
        runner = self.make_runner()
        distances = [10.0, 9.8, 11.1, 11.2, 11.3]

        self.assertFalse(any(runner._update_goal_distance_tracking(value) for value in distances))


class MovingAwayConfigAndReportingTests(unittest.TestCase):
    def test_config_defaults_and_reporting(self):
        config_text = """
seed: 1
episode_1:
  scene: scene.usd
  start: [0, 0, 0]
  goal: [10, 0, 0]
  timeout: 30
  robot_type: spot
"""
        with NamedTemporaryFile("w", suffix=".yaml") as stream:
            stream.write(config_text)
            stream.flush()
            config = benchmark_config.load_benchmark_config(stream.name)

        episode = config["episode_1"]
        self.assertEqual(episode["moving_away_distance_buffer_m"], 1.0)
        self.assertEqual(episode["moving_away_grace_seconds"], 2.0)
        self.assertEqual(episode["moving_away_min_progress_m"], 0.5)

        result = benchmark_reporting.build_episode_result(
            "episode_1", episode,
            benchmark_reporting.EpisodeMetrics(closest_goal_distance_m=2.25),
        )
        summary = benchmark_reporting.summarize_results([result])
        self.assertEqual(result["closest_goal_distance_m"], 2.25)
        self.assertEqual(summary["average_metrics"]["closest_goal_distance_m"], 2.25)


if __name__ == "__main__":
    unittest.main()
