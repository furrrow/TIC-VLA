import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import protocol_v2 as protocol


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_only_approved_scenario_fields_change(level):
    info=protocol.check_scenario(ROOT/f'DynaNav/configs/crossover_scenarios_v2/scenario3_crossover_{level}_20trials.yaml')
    assert len(set(info['episode_ids']))==20


def test_duplicate_effective_cleanup_is_impossible():
    source=Path('/home/jim/Projects/steernav/steernav/ros_inference.py').read_text()
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SteeringNode')
    methods=[m for m in node.body if isinstance(m,ast.FunctionDef) and m.name=='destroy_node']
    assert len(methods)==1
    class Base:
        def destroy_node(self):
            self.super_calls+=1
            return True
    test_class=ast.ClassDef(name='SteeringNode',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[test_class],type_ignores=[]))
    scope={'Base':Base};exec(compile(module,'cleanup-test','exec'),scope)
    instance=scope['SteeringNode']();instance.super_calls=0
    instance.timer=Mock();instance.video_writer=Mock();instance.receiver_timing=Mock()
    instance.receiver_timing.close.return_value=None
    resources=[instance.timer,instance.video_writer,instance.receiver_timing]
    assert instance.destroy_node() is True
    assert instance.destroy_node() is True
    assert instance.super_calls==1
    for item,method in zip(resources,['cancel','release','close']):getattr(item,method).assert_called_once()
    assert all(getattr(instance,k) is None for k in ['timer','video_writer','receiver_timing'])


@pytest.mark.parametrize('repo',['TIC-VLA','NaVILA','OmniVLA'])
@pytest.mark.parametrize('mode',['normal','no-velocity','steered'])
def test_launchers_are_equal_without_ros(repo,mode):
    command=['bash',str(ROOT.parent/repo/'nova_ros.bash'),mode,'--validate-only']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    flags=result.stdout.split('SteerNav flags: ',1)[1].strip()
    if mode=='normal':assert not flags
    else:
        for pair in ['--time-look-ahead 1.5','--velocity-window 5','--odom-alignment extrapolate',
                     '--odom-pose-max-extrapolation 0.25','--image-queue-depth 1',
                     '--robot-radius 0.5','--additional-clearance 0.5']:
            assert pair in flags
        assert '--velocity-reference-log' not in flags
        assert '--taper-pedestrian-sweep' not in flags


def test_all_outer_drivers_dry_run_without_launching(tmp_path):
    # Fail if dry-run unexpectedly reaches a process launcher or SSH.
    for cmd in ['ssh','scp','tmux','uv','ros2']:
        p=tmp_path/cmd;p.write_text('#!/bin/sh\necho UNEXPECTED_LAUNCH >&2\nexit 99\n');p.chmod(0o755)
    import os
    env=dict(os.environ,PATH=str(tmp_path)+':'+os.environ['PATH'])
    for repo in ['TIC-VLA','NaVILA','OmniVLA']:
        result=subprocess.run(['bash',str(ROOT.parent/repo/'run_nova_scenarios.bash'),'--dry-run',
            *[f'scenario3_crossover_{v}_20trials' for v in ['easy','medium','hard']]],env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert result.stdout.count('"no_launch": true')==3
        clearance = '0.25' if repo == 'NaVILA' else '0.5'
        assert result.stdout.count(f'"STEERNAV_ADDITIONAL_CLEARANCE": "{clearance}"') == 3


def test_navila_campaign_rejects_old_clearance_override(monkeypatch):
    import argparse
    monkeypatch.setenv('STEERNAV_ADDITIONAL_CLEARANCE', '0.5')
    args = argparse.Namespace(
        scenario=ROOT/'DynaNav/configs/crossover_scenarios_v2/scenario3_crossover_easy_20trials.yaml',
        policy='NaVILA', mode='steered', dt='0.75', fix_end='false', dry_run=True)
    with pytest.raises(ValueError, match='Frozen setting mismatch: STEERNAV_ADDITIONAL_CLEARANCE'):
        protocol.prepare(args)


def test_retry_never_repeats_navigation_outcomes(tmp_path):
    first=tmp_path/'attempt1';first.mkdir()
    (first/'benchmark_results.json').write_text(json.dumps(dict(episodes=[dict(episode='a',termination_condition='human_collision')])))
    with pytest.raises(ValueError):protocol.check_retry(tmp_path/'attempt2')
    with pytest.raises(FileExistsError):protocol.check_retry(first)
    (first/'benchmark_results.json').write_text(json.dumps(dict(episodes=[dict(episode='a',error='child crashed')])))
    protocol.check_retry(tmp_path/'attempt2')


def test_episode_aggregation_does_not_weight_long_episodes():
    from compare_pedestrian_velocities import episode_aggregation
    rows=[dict(event='visual_velocity',episode_id=ep,estimated_velocity_camera=[1,0,0],projection_applied=True) for ep in ['a']*100+['b']]
    records=[dict(episode_id=ep,comparison_status='compared',vector_error_mps=error,pedestrian_id='p') for ep,error in [('a',0.)]*100+[('b',2.)]]
    result=episode_aggregation(rows,records,['a','b','missing'])
    assert result['condition_metrics']['median_vector_error_mps']['median']==1.
    assert result['episodes'][-1]['observations']==0
    assert result['condition_metrics']['median_vector_error_mps']['episodes_with_data']==2


def test_effective_cleanup_drains_actual_writer(tmp_path):
    module_path=Path('/home/jim/Projects/steernav/steernav/receiver_timing.py')
    spec=importlib.util.spec_from_file_location('writer_for_cleanup',module_path)
    writer_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer_module)
    source=module_path.with_name('ros_inference.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='SteeringNode')
    method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='destroy_node')
    class Base:
        def destroy_node(self): self.super_calls+=1;return True
    scope={'Base':Base}
    cls=ast.ClassDef(name='SteeringNode',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'real-writer-cleanup','exec'),scope)
    obj=scope['SteeringNode']();obj.super_calls=0;obj.timer=Mock();obj.video_writer=Mock();obj.get_logger=Mock(return_value=Mock())
    obj.timer.cancel.side_effect=RuntimeError('timer already cancelled')
    path=tmp_path/'receiver_timing.jsonl';obj.receiver_timing=writer_module.ReceiverTiming(path,capacity=8)
    for i in range(1000):obj.receiver_timing.emit('record',index=i)
    obj.destroy_node();obj.destroy_node()
    summary=json.loads(path.read_text().splitlines()[-1])
    assert summary['event']=='writer_summary' and summary['clean_shutdown'] and summary['queue_remaining']==0
    assert summary['written_records']+summary['dropped_records']==summary['attempted_records']==1000
    assert obj.super_calls==1
    original=path.read_bytes()
    with pytest.raises(FileExistsError):writer_module.ReceiverTiming(path)
    assert path.read_bytes()==original


@pytest.mark.parametrize('version',[2,3])
@pytest.mark.parametrize('invalid',[False,True])
def test_synthetic_twenty_episode_acceptance(tmp_path,invalid,version):
    import argparse,shutil
    sys.path.insert(0,str(ROOT/'DynaNav'))
    from benchmark_config import load_benchmark_config,select_episodes
    from benchmark_reporting import build_episode_result,EpisodeMetrics
    source=ROOT/f'DynaNav/configs/crossover_scenarios_v{version}/scenario3_crossover_easy_20trials.yaml'
    shutil.copy2(source,tmp_path/'scenario.yaml')
    protocol.write_new(tmp_path/'protocol_manifest.json',dict(scenario=protocol.check_scenario(source)))
    rows=[]
    for name,config in select_episodes(load_benchmark_config(source)):
        metrics=EpisodeMetrics(success=False,closest_goal_distance_m=.35,goal_human_collision=True,
            termination_condition='goal_human_collision',termination_reason='Goal person contact',
            min_goal_human_distance_m=.35,min_non_goal_human_distance_m=2.)
        rows.append(build_episode_result(name,config,metrics,inference_method='TIC-VLA'))
    if invalid:rows[0]['success']=True
    (tmp_path/'benchmark_results.json').write_text(json.dumps(dict(run_id='synthetic',experiment_mode='normal',inference_method='TIC-VLA',episodes=rows)))
    args=argparse.Namespace(capture=str(tmp_path),mode='normal',policy='TIC-VLA',run_id='synthetic')
    if invalid:
        with pytest.raises(SystemExit):protocol.finish(args)
    else:protocol.finish(args)
    status=json.loads((tmp_path/'capture_status.json').read_text())
    assert status['valid']==(not invalid)
    assert not status['retryable_infrastructure_failure']
    assert (tmp_path/'episode_outcomes.csv').exists() and (tmp_path/'termination_summary.json').exists()


def test_goal_contact_overrides_latched_success_and_stop(monkeypatch):
    import asyncio
    path=ROOT/'tests/test_goal_stop_protocol.py'
    spec=importlib.util.spec_from_file_location('navigation_test_harness',path)
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    original=harness.benchmark.BenchmarkRunner._update_success_criteria
    def update(self,distance):
        result=original(self,distance)
        if distance < .4:self.state.goal_human_collision=True
        return result
    monkeypatch.setattr(harness.benchmark.BenchmarkRunner,'_update_success_criteria',update)
    obj=harness.NavigationProtocolTests()
    runner,result=asyncio.run(obj.navigate([2.,1.,.35]))
    assert runner.state.success_criteria_met
    assert result is False and runner.state.termination_condition=='goal_human_collision'
    assert not runner.state.human_collision_detected


def test_collision_identity_is_explicit_and_hc_excludes_goal(monkeypatch):
    import asyncio,types
    import numpy as np
    sys.path.insert(0,str(ROOT/'DynaNav'))
    import benchmark
    def prim(path,position):
        obj=Mock();obj.IsValid.return_value=True;obj.IsActive.return_value=True
        obj.GetPath.return_value=types.SimpleNamespace(pathString=path)
        obj.position=position
        return obj
    goal=prim('/World/Characters/Character_01',[.3,0,0])
    walker=prim('/World/Characters/Character',[2,0,0])
    characters=Mock();characters.IsValid.return_value=True;characters.GetChildren.return_value=[walker,goal]
    stage=Mock();stage.GetPrimAtPath.return_value=characters
    omni=types.ModuleType('omni')
    omni.usd=types.SimpleNamespace(get_context=lambda:types.SimpleNamespace(get_stage=lambda:stage),
        get_world_transform_matrix=lambda obj:types.SimpleNamespace(ExtractTranslation=lambda:obj.position))
    monkeypatch.setitem(sys.modules,'pxr',types.SimpleNamespace(UsdGeom=Mock()))
    monkeypatch.setitem(sys.modules,'omni',omni);monkeypatch.setitem(sys.modules,'omni.usd',omni.usd)
    runner=benchmark.BenchmarkRunner.__new__(benchmark.BenchmarkRunner)
    runner.state=benchmark.EpisodeState(name='synthetic',robot_position=np.array([0,0,0]),goal_human_path='/World/Characters/Character_01')
    runner.human_collision_threshold=.4;runner._find_character_skelroot=lambda obj:obj
    asyncio.run(runner._check_collisions())
    assert runner.state.goal_human_collision and not runner.state.human_collision_detected
    assert runner.state.min_goal_human_distance_m==.3
    assert runner.state.min_non_goal_human_distance_m==2.
    # A walker near the goal remains a non-goal pedestrian regardless of position.
    walker.position=[.2,0,0]
    asyncio.run(runner._check_collisions())
    assert runner.state.human_collision_detected
    assert {r['goal_human'] for r in runner.state.collision_identities}=={True,False}


def test_stale_live_owner_metadata_never_signals(tmp_path,monkeypatch):
    import os
    path=Path('/home/jim/Projects/steernav/steernav/receiver_timing.py')
    spec=importlib.util.spec_from_file_location('stale_writer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    log=tmp_path/'receiver_timing.jsonl';log.touch()
    owner=dict(pid=1234567,start_ticks='stale',boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),log=str(log.resolve()))
    Path(str(log)+'.owner.json').write_text(json.dumps(owner))
    monkeypatch.setattr(module,'process_identity',lambda pid:'another-owner')
    kill=Mock();monkeypatch.setattr(module.os,'kill',kill)
    result=module.finalize(log,timeout=.01)
    assert 'Stale writer owner' in result['error'];kill.assert_not_called()
    metadata=Path(str(log)+'.shutdown.json').read_bytes()
    assert module.finalize(log)==result
    assert Path(str(log)+'.shutdown.json').read_bytes()==metadata
