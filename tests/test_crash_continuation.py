import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'DynaNav'))
import protocol_v2 as protocol
from benchmark_config import load_benchmark_config,select_episodes
from benchmark_reporting import EpisodeMetrics,build_episode_result


def fixture(tmp_path):
    source=ROOT/'DynaNav/configs/crossover_scenarios_v2/scenario3_crossover_easy_20trials.yaml'
    shutil.copy2(source,tmp_path/'scenario.yaml')
    protocol.write_new(tmp_path/'protocol_manifest.json',dict(scenario=protocol.check_scenario(source)))
    rows=[]
    for name,config in select_episodes(load_benchmark_config(source)):
        rows.append(build_episode_result(name,config,EpisodeMetrics(success=False,closest_goal_distance_m=2.,
            termination_condition='human_collision',termination_reason='Non-goal collision',human_collision=True,
            min_goal_human_distance_m=2.,min_non_goal_human_distance_m=.3),inference_method='TIC-VLA'))
    rows[6].update(error='no_result_file',termination_condition=None,termination_reason=None,
        closest_goal_distance_m=None,normalized_progress=None,isaac_sim_errors=['[crash] A crash has occurred.'])
    return rows


@pytest.mark.parametrize('problem,expected',[('isolated',10),('bad_metadata',1),('no_crash_evidence',1),('missing_id',1),('bad_good_metric',1),('many_crashes',1),('missing_telemetry',1)])
def test_finish_records_crash_without_accepting_capture(tmp_path,problem,expected):
    rows=fixture(tmp_path)
    if problem=='bad_metadata':rows[0]['goal_stop_radius_m']=1.
    if problem=='no_crash_evidence':rows[6]['isaac_sim_errors']=[]
    if problem=='missing_id':rows.pop()
    if problem=='bad_good_metric':rows[0]['closest_goal_distance_m']=None
    if problem=='many_crashes':
        for i in [7,8]:rows[i]=dict(rows[6],episode=rows[i]['episode'])
    mode='steered' if problem=='missing_telemetry' else 'normal'
    for row in rows:row['experiment_mode']=mode
    data=dict(run_id='synthetic',inference_method='TIC-VLA',experiment_mode=mode,episodes=rows)
    (tmp_path/'benchmark_results.json').write_text(json.dumps(data))
    original=(tmp_path/'benchmark_results.json').read_bytes()
    args=argparse.Namespace(capture=str(tmp_path),mode=mode,policy='TIC-VLA',run_id='synthetic')
    with pytest.raises(SystemExit) as exception:protocol.finish(args)
    assert exception.value.code==expected
    status=json.loads((tmp_path/'capture_status.json').read_text())
    assert status['valid'] is False
    assert status['continuation']['continuation_allowed']==(expected==10)
    assert (tmp_path/'benchmark_results.json').read_bytes()==original
    if expected==10:
        queue=json.loads((tmp_path/'recovery_needed.json').read_text())
        assert [r['episode'] for r in queue['deferred_episodes']]==['episode_s3_p2_07']
        assert len(status['completed_navigation_episode_ids'])==19


@pytest.mark.parametrize('exit_status,opt_in,expected',[(10,'true',0),(10,'false',10),(1,'true',1),(0,'true',0)])
def test_shell_only_continues_explicit_deferred_status(tmp_path,exit_status,opt_in,expected):
    code=f'''
SCENARIO=not-v2.yaml
BATCH_ID=test
RUN_ATTEMPT_ID=attempt1
source "{ROOT}/scripts/protocol_v2_run.bash"
PROTOCOL_V2=true
REMOTE_REPO=/remote
REMOTE_HOST=fake
INFERENCE_METHOD=TIC-VLA
run_id=test
run_log_dir="{tmp_path}"
CONTINUE_ON_EPISODE_CRASH={opt_in}
scp() {{ return 0; }}
python3() {{ return {exit_status}; }}
finish_run_capture normal
'''
    result=subprocess.run(['bash','-c',code],capture_output=True,text=True)
    assert result.returncode==expected,result.stderr


def test_resume_plan_selects_only_unattempted_conditions(tmp_path):
    from plan_crossover_continuation import plan,save_plan
    campaign='unit_campaign'
    batch=f'{campaign}_omnivla_scenario3_crossover_easy_20trials'
    folder=tmp_path/'OmniVLA/logs/orchestration'/batch/f'{batch}_steered/attempt1'
    folder.mkdir(parents=True)
    (folder/'capture_status.json').write_text(json.dumps(dict(valid=True)))
    data=plan(campaign,tmp_path)
    assert not data['blockers'] and len(data['jobs'])==26
    assert not any(j['policy']=='OmniVLA' and 'easy' in j['scenario'] and j['mode']=='steered' for j in data['jobs'])
    output=tmp_path/'plan';save_plan(data,output)
    subprocess.run(['bash','-n',str(output/'resume.bash')],check=True)
    with pytest.raises(FileExistsError):save_plan(data,output)
