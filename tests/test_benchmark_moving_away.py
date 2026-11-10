import sys
import types
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import AsyncMock, Mock, patch


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

    def test_success_is_latched_without_terminating_distance_tracking(self):
        runner = self.make_runner()
        runner.state.goal_threshold = 1.5

        self.assertFalse(runner._update_success_criteria(1.5))
        self.assertTrue(runner._update_success_criteria(1.49))
        self.assertTrue(runner.state.success_criteria_met)
        self.assertFalse(runner._update_success_criteria(3.0))
        self.assertTrue(runner.state.success_criteria_met)


class GoalCollisionPrecedenceTests(unittest.IsolatedAsyncioTestCase):
    async def run_frame(self, distance):
        runner = benchmark.BenchmarkRunner.__new__(benchmark.BenchmarkRunner)
        runner.state = benchmark.EpisodeState(name="episode_1", goal_threshold=2.0)
        runner.behavior_context = types.SimpleNamespace(failure_reason=None, frame_count=0)
        runner.terminate_on_human_collision = True
        runner.terminate_when_moving_away = False
        runner.debug_print = False
        runner._update_robot_position = AsyncMock(
            side_effect=lambda: setattr(runner.state, "human_collision_detected", True)
        )
        runner._get_distance_to_goal = Mock(return_value=distance)
        runner._update_goal_distance_tracking = Mock(return_value=False)

        timeline = types.SimpleNamespace(
            get_time_codes_per_second=Mock(return_value=60),
            set_time_codes_per_second=Mock(),
            commit_silently=Mock(),
            is_playing=Mock(return_value=True),
            stop=Mock(),
        )
        app = types.SimpleNamespace(next_update_async=AsyncMock())
        modules = {
            "omni": types.ModuleType("omni"),
            "omni.timeline": types.SimpleNamespace(
                get_timeline_interface=Mock(return_value=timeline)
            ),
            "omni.replicator": types.ModuleType("omni.replicator"),
            "omni.replicator.core": types.SimpleNamespace(
                orchestrator=types.SimpleNamespace(set_capture_on_play=Mock())
            ),
            "omni.kit": types.ModuleType("omni.kit"),
            "omni.kit.app": types.SimpleNamespace(
                get_app=Mock(return_value=app)
            ),
        }
        modules["omni"].timeline = modules["omni.timeline"]
        modules["omni"].replicator = modules["omni.replicator"]
        modules["omni"].kit = modules["omni.kit"]
        modules["omni.replicator"].core = modules["omni.replicator.core"]
        modules["omni.kit"].app = modules["omni.kit.app"]

        with patch.dict(sys.modules, modules), redirect_stderr(StringIO()):
            result = await runner._run_navigation(timeout_frames=1)
        return runner, result

    async def test_same_frame_goal_and_collision_is_success(self):
        runner, result = await self.run_frame(0.4)

        self.assertTrue(result)
        self.assertEqual(runner.state.termination_condition, "goal_reached")

    async def test_collision_outside_goal_threshold_is_failure(self):
        runner, result = await self.run_frame(3.0)

        self.assertFalse(result)
        self.assertEqual(runner.state.termination_condition, "human_collision")


class MovingAwayConfigAndReportingTests(unittest.TestCase):
    def test_termination_banner_is_prominent_and_uses_stderr(self):
        output = StringIO()
        with redirect_stderr(output):
            benchmark._print_termination_banner(
                "episode_1", True, "timeout", "Success criterion was met", source="Parent"
            )

        banner = output.getvalue()
        self.assertIn("EPISODE TERMINATED [Parent]", banner)
        self.assertIn(">>> Episode: episode_1", banner)
        self.assertIn(">>> Status: SUCCESS", banner)
        self.assertIn(">>> Condition: timeout", banner)
        self.assertIn(">>> Reason: Success criterion was met", banner)

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
        self.assertEqual(episode["moving_away_distance_buffer_m"], 0.5)
        self.assertEqual(episode["moving_away_grace_seconds"], 1.0)
        self.assertEqual(episode["moving_away_min_progress_m"], 0.5)

        result = benchmark_reporting.build_episode_result(
            "episode_1", episode,
            benchmark_reporting.EpisodeMetrics(closest_goal_distance_m=2.25),
        )
        summary = benchmark_reporting.summarize_results([result])
        self.assertEqual(result["closest_goal_distance_m"], 2.25)
        self.assertEqual(summary["median_metrics"]["closest_goal_distance_m"], 2.25)

    def test_aggregate_reporting_uses_medians_and_reports_success_rate(self):
        results = [
            {
                "success": success,
                "navigation_error": navigation_error,
                "closest_goal_distance_m": closest_distance,
                "path_length": path_length,
                "human_collision": human_collision,
                "physical_collision": physical_collision,
            }
            for (
                success, navigation_error, closest_distance, path_length,
                human_collision, physical_collision,
            ) in [
                (True, 1.0, 2.0, 10.0, 0.0, 0.0),
                (False, 3.0, 4.0, 20.0, 1.0, 0.0),
                (True, 100.0, 50.0, 1000.0, 0.0, 1.0),
            ]
        ]

        summary = benchmark_reporting.summarize_results(results)
        metrics = summary["median_metrics"]
        self.assertEqual(metrics["navigation_error_m"], 3.0)
        self.assertEqual(metrics["closest_goal_distance_m"], 4.0)
        self.assertEqual(metrics["path_length_m"], 20.0)
        self.assertAlmostEqual(summary["success_rate_percent"], 200 / 3)

        aggregate_lines = benchmark_reporting._aggregate_lines(summary)
        self.assertTrue(any("Success Rate (SR): 66.7%" in line for line in aggregate_lines))
        human_index = next(i for i, line in enumerate(aggregate_lines) if line.startswith("Human Collision Rate"))
        self.assertTrue(aggregate_lines[human_index + 1].startswith("Physical Collision Rate"))
        self.assertTrue(aggregate_lines[human_index + 2].startswith("Median Closest Goal Distance"))

    def test_successful_termination_is_printed_and_serialized(self):
        result = benchmark_reporting.build_episode_result(
            "episode_1", {},
            benchmark_reporting.EpisodeMetrics(
                success=True,
                termination_condition="moving_away",
                termination_reason="Terminated after moving away from the goal",
            ),
        )

        self.assertEqual(result["termination_condition"], "moving_away")
        self.assertEqual(
            result["termination_reason"],
            "Terminated after moving away from the goal",
        )
        lines = benchmark_reporting._episode_lines(result)
        self.assertIn("  Termination Condition: moving_away", lines)
        self.assertIn(
            "  Termination Reason: Terminated after moving away from the goal",
            lines,
        )

        result["child_log"] = "/tmp/episode_1_isaac.log"
        self.assertIn(
            "  Isaac Sim Log: /tmp/episode_1_isaac.log",
            benchmark_reporting._episode_lines(result),
        )


if __name__ == "__main__":
    unittest.main()
