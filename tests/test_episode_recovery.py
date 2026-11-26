import argparse
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'DynaNav'))
import protocol_v2 as protocol
import recover_crossover_episodes as recovery
from benchmark_config import load_benchmark_config,select_episodes
from benchmark_reporting import EpisodeMetrics,build_episode_result


def source_fixture(tmp_path,missing=False):
    scenario=ROOT/'DynaNav/configs/crossover_scenarios_v2/scenario3_crossover_easy_20trials.yaml'
    source=tmp_path/'source';source.mkdir();shutil.copy2(scenario,source/'scenario.yaml')
    manifest=dict(run_id='original',policy='TIC-VLA',mode='normal',frozen=protocol.FROZEN,scenario=protocol.check_scenario(scenario))
    protocol.write_new(source/'protocol_manifest.json',manifest)
    rows=[]
    for name,config in select_episodes(load_benchmark_config(scenario)):
        rows.append(build_episode_result(name,config,EpisodeMetrics(success=False,closest_goal_distance_m=2.,
            termination_condition='human_collision',termination_reason='Non-goal collision',human_collision=True,
            min_goal_human_distance_m=2.,min_non_goal_human_distance_m=.3),inference_method='TIC-VLA'))
    if missing:rows=[r for r in rows if r['episode']!='episode_s3_p2_07']
    else:rows[6].update(error='no_result_file',isaac_sim_errors=['[crash] A crash has occurred'])
    protocol.write_new(source/'benchmark_results.json',dict(run_id='original',inference_method='TIC-VLA',experiment_mode='normal',episodes=rows))
    return source,scenario,manifest


@pytest.mark.parametrize('missing',[False,True])
def test_only_missing_or_verified_crashes_are_selectable(tmp_path,missing):
    source,scenario,manifest=source_fixture(tmp_path,missing)
    selected=recovery.read_selection(scenario,'TIC-VLA','normal','episode_s3_p2_07',source)
    assert selected['episode_ids']==['episode_s3_p2_07']
    with pytest.raises(ValueError):recovery.read_selection(scenario,'TIC-VLA','normal','episode_s3_p2_01',source)
    with pytest.raises(ValueError):recovery.read_selection(scenario,'TIC-VLA','steered','episode_s3_p2_07',source)
    with pytest.raises(ValueError):recovery.read_selection(scenario,'NaVILA','normal','episode_s3_p2_07',source)
    (source/'scenario.yaml').write_text('tampered')
    with pytest.raises(ValueError):recovery.read_selection(scenario,'TIC-VLA','normal','episode_s3_p2_07',source)


def test_recovery_wrap_and_strict_single_episode_acceptance(tmp_path):
    source,scenario,manifest=source_fixture(tmp_path)
    ep='episode_s3_p2_07';selection=recovery.read_selection(scenario,'TIC-VLA','normal',ep,source)
    dest=tmp_path/'attempt2';dest.mkdir();shutil.copy2(scenario,dest/'scenario.yaml')
    config=load_benchmark_config(scenario)[ep]
    raw=build_episode_result(ep,config,EpisodeMetrics(success=False,closest_goal_distance_m=1.5,
        human_collision=True,termination_condition='human_collision',termination_reason='Ordinary collision'),inference_method='TIC-VLA')
    (dest/'raw_episode_result.json').write_text(json.dumps(raw))
    manifest.update(run_id='retry',selected_episode_ids=[ep],recovery_source=selection)
    protocol.write_new(dest/'protocol_manifest.json',manifest)
    import yaml
    full=yaml.safe_load(scenario.read_text())
    (dest/'episode_selection.yaml').write_text(yaml.safe_dump({k:v for k,v in full.items() if not k.startswith('episode_') or k==ep}))
    recovery.wrap(dest)
    with pytest.raises(FileExistsError):recovery.wrap(dest)
    protocol.finish(argparse.Namespace(capture=str(dest),mode='normal',policy='TIC-VLA',run_id='retry'))
    assert json.loads((dest/'capture_status.json').read_text())['valid']
    assert not json.loads((dest/'benchmark_results.json').read_text())['episodes'][0]['success']
    assert json.loads((dest/'condition_metrics.json').read_text())['metrics']['success']['sample_std'] is None


def test_retry_guard_only_allows_unfinished_selected_episode(tmp_path):
    first=tmp_path/'attempt1';first.mkdir()
    protocol.write_new(first/'benchmark_results.json',dict(episodes=[dict(episode='episode_a',termination_condition='human_collision'),dict(episode='episode_b',error='no_result_file')]))
    protocol.check_retry(tmp_path/'attempt2', ['episode_b'])
    with pytest.raises(ValueError):protocol.check_retry(tmp_path/'attempt2',['episode_a'])
    with pytest.raises(ValueError):protocol.check_retry(tmp_path/'attempt2')


@pytest.mark.parametrize('accepted',[True,False])
def test_merge_requires_accepted_recovery_and_preserves_originals(tmp_path,accepted):
    source,scenario,manifest=source_fixture(tmp_path)
    data=json.loads((source/'benchmark_results.json').read_text())
    run='TIC-VLA_crossover_v2_test_ticvla_scenario3_crossover_easy_20trials_attempt1_normal'
    data['run_id']=run
    original_bytes=json.dumps([dict(run_id=run,result=data)]).encode()
    folder=tmp_path/'plan';folder.mkdir();(folder/'original_remote_snapshot.json').write_bytes(original_bytes)
    capture=tmp_path/'recovery';capture.mkdir()
    row=dict(data['episodes'][0],episode='episode_s3_p2_07')
    protocol.write_new(capture/'capture_status.json',dict(valid=accepted))
    protocol.write_new(capture/'benchmark_results.json',dict(episodes=[row]))
    protocol.write_new(folder/'recovery_plan.json',dict(jobs=[dict(capture=str(capture),episode=row['episode'],source_run_id=run)]))
    if accepted:
        recovery.merge(folder)
        merged=json.loads((folder/'merged'/run/'benchmark_results.json').read_text())
        assert len(merged['episodes'])==20 and not any(r.get('error') for r in merged['episodes'])
        assert merged['episodes'][0]==data['episodes'][0]
        assert not next(r for r in merged['episodes'] if r['episode']==row['episode'])['success']
        with pytest.raises(FileExistsError):recovery.merge(folder)
    else:
        with pytest.raises(ValueError):recovery.merge(folder)
    assert (folder/'original_remote_snapshot.json').read_bytes()==original_bytes
