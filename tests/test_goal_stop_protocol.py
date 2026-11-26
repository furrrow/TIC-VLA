"""Close-approach stopping reuses historical success latching and move-away logic."""
import json
import sys
import types
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DynaNav'))
import benchmark
import benchmark_config
import benchmark_reporting


def load_config(tmp_path,margin=.1,episode_collision=.4):
    p=tmp_path/'scenario.yaml'
    p.write_text(yaml.safe_dump(dict(success_threshold=1.,human_collision_threshold_m=.4,
        goal_stop_collision_margin_m=margin,episode_one=dict(scene='scene.usd',start=[0,0,0],goal=[10,0,0],
        timeout=40,robot_type='nova_carter',human_collision_threshold_m=episode_collision))))
    return benchmark_config.load_benchmark_config(p)


def test_radius_follows_per_episode_collision_threshold(tmp_path):
    config=load_config(tmp_path,episode_collision=.6)
    assert benchmark_config.goal_stop_radius_m(config['episode_one'])==pytest.approx(.7)
    assert config['episode_one']['success_threshold']==1.
    config=load_config(tmp_path,margin=None)
    assert benchmark_config.goal_stop_radius_m(config['episode_one'])==1.


@pytest.mark.parametrize('margin',[0,-.1,float('nan'),float('inf'),True,'0.1'])
def test_invalid_margin_is_rejected(tmp_path,margin):
    with pytest.raises(ValueError):load_config(tmp_path,margin=margin)


class NavigationProtocolTests(__import__('unittest').IsolatedAsyncioTestCase):
    async def navigate(self,distances,collision_frame=None,failure_frame=None,clock_step=1/30,timeout_seconds=None,version=2,threshold=1.,physical_frame=None):
        runner=benchmark.BenchmarkRunner.__new__(benchmark.BenchmarkRunner)
        runner.state=benchmark.EpisodeState(name='episode_one',protocol_version=version,goal_threshold=threshold,goal_stop_radius_m=.5,
            goal_stop_collision_margin_m=.1,moving_away_grace_frames=3,
            moving_away_grace_seconds=2/30)
        runner.human_collision_threshold=.4
        runner.behavior_context=types.SimpleNamespace(failure_reason=None,frame_count=0)
        runner.terminate_on_human_collision=True;runner.terminate_when_moving_away=True;runner.debug_print=False
        index=-1
        async def advance():
            nonlocal index
            index+=1
            runner.state.human_collision_detected=(index==collision_frame)
            runner.state.physical_collision_detected=(index==physical_frame)
            if index==failure_frame:runner.behavior_context.failure_reason='Zero controller command'
        runner._update_robot_position=AsyncMock(side_effect=advance)
        runner._get_distance_to_goal=Mock(side_effect=lambda: distances[index])
        timeline=types.SimpleNamespace(get_time_codes_per_second=Mock(return_value=30),set_time_codes_per_second=Mock(),
            commit_silently=Mock(),is_playing=Mock(return_value=True),stop=Mock(),
            get_current_time=lambda: max(0,index+1)*clock_step)
        app=types.SimpleNamespace(next_update_async=AsyncMock())
        modules={'omni':types.ModuleType('omni'),'omni.timeline':types.SimpleNamespace(get_timeline_interface=lambda:timeline),
            'omni.replicator':types.ModuleType('omni.replicator'),
            'omni.replicator.core':types.SimpleNamespace(orchestrator=types.SimpleNamespace(set_capture_on_play=Mock())),
            'omni.kit':types.ModuleType('omni.kit'),'omni.kit.app':types.SimpleNamespace(get_app=lambda:app)}
        modules['omni'].timeline=modules['omni.timeline'];modules['omni'].replicator=modules['omni.replicator']
        modules['omni'].kit=modules['omni.kit'];modules['omni.kit'].app=modules['omni.kit.app']
        modules['omni.replicator'].core=modules['omni.replicator.core']
        with patch.dict(sys.modules,modules),redirect_stderr(StringIO()):
            result=await runner._run_navigation(timeout_seconds if timeout_seconds is not None else len(distances)*clock_step)
        return runner,result

    async def test_v3_success_at_one_point_five_continues_to_stop(self):
        runner,result=await self.navigate([2.,1.5,.8,.5],version=3,threshold=1.5)
        self.assertTrue(result)
        self.assertEqual(runner.state.simulation_frame_count,4)
        self.assertEqual(runner.state.termination_condition,'goal_reached')

    async def test_v3_physical_contact_overrides_latched_success(self):
        runner,result=await self.navigate([2.,1.5,.5],version=3,threshold=1.5,physical_frame=2)
        self.assertFalse(result)
        self.assertTrue(runner.state.success_criteria_met)
        self.assertEqual(runner.state.termination_condition,'physical_collision')

    async def test_passing_one_metre_does_not_stop_before_half_metre(self):
        runner,result=await self.navigate([2.,.9,.7,.5])
        self.assertTrue(result);self.assertEqual(runner.state.simulation_frame_count,4)
        self.assertEqual(runner.state.termination_condition,'goal_reached')
        self.assertEqual(runner.state.closest_goal_distance_m,.5)

    async def test_move_away_preserves_prior_scoring_success(self):
        runner,result=await self.navigate([2.,.8,1.31,1.32,1.33])
        self.assertTrue(result);self.assertEqual(runner.state.termination_condition,'moving_away')
        self.assertEqual(runner.state.closest_goal_distance_m,.8)

    async def test_move_away_without_one_metre_is_failure(self):
        runner,result=await self.navigate([3.,1.1,1.61,1.62,1.63])
        self.assertFalse(result);self.assertEqual(runner.state.termination_condition,'moving_away')

    async def test_step_crossing_collision_and_stop_radii_is_collision(self):
        runner,result=await self.navigate([2.,.8,.35],collision_frame=2)
        self.assertFalse(result);self.assertEqual(runner.state.termination_condition,'human_collision')
        self.assertTrue(runner.state.success_criteria_met)

    async def test_stall_remains_failure_even_after_reaching_scoring_threshold(self):
        runner,result=await self.navigate([2.,.8,.75],failure_frame=2)
        self.assertFalse(result);self.assertEqual(runner.state.termination_condition,'behavior_failure')
        self.assertEqual(runner.state.closest_goal_distance_m,.75)

    async def test_timeout_reports_closest_approach_and_latched_success(self):
        runner,result=await self.navigate([2.,.8,.9])
        self.assertTrue(result);self.assertEqual(runner.state.termination_condition,'timeout')
        self.assertEqual(runner.state.closest_goal_distance_m,.8)

    async def test_nonfinite_distances_do_not_trigger_close_stop(self):
        runner,result=await self.navigate([float('inf'),float('nan'),1.2,.49])
        self.assertTrue(result);self.assertEqual(runner.state.closest_goal_distance_m,.49)


def test_result_records_scoring_and_termination_protocol(tmp_path):
    episode=load_config(tmp_path)['episode_one']
    result=benchmark_reporting.build_episode_result('episode_one',episode,
        benchmark_reporting.EpisodeMetrics(success=True,success_criteria_met=True,closest_goal_distance_m=.8,
            termination_condition='moving_away',termination_reason='Sustained moving away'))
    assert result['goal_stop_radius_m']==.5
    assert result['success_threshold_m']==1.
    assert result['success_criteria_met'] is True
    assert result['goal_stop_protocol']=='collision_margin'
    json.dumps(result)


def test_close_scenarios_change_only_stop_policy():
    base=Path(__file__).resolve().parents[1]/'DynaNav/configs/steernav_scenarios'
    for stem in ['scenario3_crossover_hard_20trials','trials','odom_alignment_episode12']:
        original=yaml.safe_load((base/(stem+'.yaml')).read_text())
        new=yaml.safe_load((base/(stem+'_close_approach.yaml')).read_text())
        assert new.pop('goal_stop_collision_margin_m')==.1
        assert new==original


def test_generator_only_adds_close_stop_when_requested():
    import importlib.util
    source=Path(__file__).resolve().parents[1]/'DynaNav/configs/generate_hospital_scenarios.py'
    spec=importlib.util.spec_from_file_location('hospital_generator',source)
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    kwargs=dict(num_people=2,delays=[(.2,)],seed=36,success_threshold=1.,
        human_collision_threshold=.4,timeout=40.,episode_prefix='episode_s3')
    original=generator.make_config(**kwargs)
    extended=generator.make_config(**kwargs,goal_stop_collision_margin=.1)
    assert extended.pop('goal_stop_collision_margin_m')==.1
    assert extended==original
