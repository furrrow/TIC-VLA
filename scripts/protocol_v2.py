#!/usr/bin/env python3
"""Protocol v2 preparation/acceptance helpers; never launches ROS or Isaac."""
import argparse
import base64
import hashlib
import json
import os
import numpy as np
from pathlib import Path
import shutil
import subprocess
import sys
import yaml
from source_snapshot import snapshot
from save_benchmark_outcomes import save_outcomes
from summarize_receiver_timing import read_summary

ROOT = Path(__file__).resolve().parents[1]
FROZEN = dict(OBJECT_VELOCITY_SOURCE='estimated', STEERNAV_TAPER_PEDESTRIAN_SWEEP='false',
    STEERNAV_ODOM_ALIGNMENT='extrapolate', STEERNAV_ODOM_POSE_MAX_EXTRAPOLATION='0.25',
    STEERNAV_TIME_LOOK_AHEAD='1.5', STEERNAV_VELOCITY_WINDOW='5', STEERNAV_IMAGE_QUEUE_DEPTH='1',
    STEERNAV_PROCESS_REPEATED_IMAGES='false', STEERNAV_ROBOT_RADIUS='0.5', STEERNAV_ADDITIONAL_CLEARANCE='0.5')
APPROVED_GLOBAL = {'protocol_version','success_threshold','goal_stop_collision_margin_m',
    'moving_away_min_progress_m','moving_away_distance_buffer_m','moving_away_grace_seconds'}


def frozen_settings(policy):
    settings = dict(FROZEN)
    if policy == 'NaVILA':
        settings['STEERNAV_ADDITIONAL_CLEARANCE'] = '0.25'
    return settings


def check_scenario(path):
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    old = yaml.safe_load((ROOT/'DynaNav/configs/crossover_scenarios'/path.name).read_text())
    episodes = [k for k in data if k.startswith('episode_')]
    if len(episodes) != 20 or set(episodes) != {k for k in old if k.startswith('episode_')}:
        raise ValueError('Require exactly 20 original unique episode IDs')
    if {k:v for k,v in data.items() if k not in APPROVED_GLOBAL and k not in episodes} != {k:v for k,v in old.items() if k not in APPROVED_GLOBAL and k not in episodes}:
        raise ValueError('Unapproved global scenario changes')
    version = data.get("protocol_version")
    if version not in (2, 3): raise ValueError("Require protocol version 2 or 3")
    expected = dict(protocol_version=version, success_threshold=1.5 if version == 3 else 1.0, human_collision_threshold_m=.4,
        goal_stop_collision_margin_m=.1, moving_away_min_progress_m=.5,
        moving_away_distance_buffer_m=.5, moving_away_grace_seconds=1.)
    for k,v in expected.items():
        if data.get(k) != v: raise ValueError(f'Incorrect {k}')
    for name in episodes:
        ep = data[name]
        if {k:v for k,v in ep.items() if k != 'goal_human_index'} != old[name]:
            raise ValueError(f'Unapproved trial change: {name}')
        index = ep.get('goal_human_index')
        if type(index) is not int or index < 0 or index >= len(ep['scripted_humans']):
            raise ValueError('Missing stable goal identity')
        human = ep['scripted_humans'][index]
        if human.get('walk_route') or human['position'] != ep['goal'] or ep['timeout'] != 40:
            raise ValueError('Invalid goal person/timeout')
    # Exercise the same strict loader used by the simulator.
    sys.path.insert(0,str(ROOT/'DynaNav'))
    from benchmark_config import load_benchmark_config
    load_benchmark_config(path)
    return dict(protocol_version=version, success_radius_m=expected["success_threshold"], sha256=hashlib.sha256(path.read_bytes()).hexdigest(), episode_ids=episodes)


def write_new(path, data):
    with Path(path).open('x') as stream: json.dump(data,stream,indent=2);stream.write('\n')


def save_snapshot(folder, name, value):
    archive=base64.b64decode(value.pop('dirty_sources_tar_gz_base64'))
    diff=base64.b64decode(value.pop('diff_base64'))
    (folder/f'{name}.dirty_sources.tar.gz').write_bytes(archive)
    (folder/f'{name}.diff.patch').write_bytes(diff)
    write_new(folder/f'{name}.json',value)


def check_retry(folder, selected_ids=None):
    """Conservatively refuse repeating any already observed navigation trial."""
    folder=Path(folder)
    if folder.exists(): raise FileExistsError(f'Existing capture directory: {folder}')
    for previous in folder.parent.glob('*/benchmark_results.json'):
        for row in json.loads(previous.read_text()).get('episodes',[]):
            if selected_ids and row.get('episode') not in selected_ids:continue
            infrastructure = bool(row.get('error')) or row.get('termination_condition') in ('infrastructure_error','child_crash','child_timeout')
            if not infrastructure:
                raise ValueError('Retry would repeat navigation outcomes; mixed captures require episode-selective recovery, not a full condition retry')


def prepare(args):
    frozen = frozen_settings(args.policy)
    scenario=Path(args.scenario)
    info=check_scenario(scenario)
    from recover_crossover_episodes import read_selection
    recovery=read_selection(scenario,args.policy,args.mode,os.environ.get('PROTOCOL_V2_RECOVERY_EPISODE'),os.environ.get('PROTOCOL_V2_RECOVERY_SOURCE'))
    for key,value in frozen.items():
        if os.environ.get(key,value)!=value: raise ValueError(f'Frozen setting mismatch: {key}')
    if args.dt!='0.75' or args.fix_end!='false': raise ValueError('Require dt=.75 and free final waypoint')
    if args.dry_run:
        print(json.dumps(dict(scenario=info, frozen=frozen, mode=args.mode, policy=args.policy, no_launch=True),indent=2));return
    if os.environ.get('PROTOCOL_V2_APPROVED')!='true':
        raise ValueError('Future simulator execution requires explicit user approval and PROTOCOL_V2_APPROVED=true')
    folder=Path(args.capture)
    check_retry(folder,recovery["episode_ids"] if recovery else None)
    folder.mkdir(parents=True,exist_ok=False)
    write_new(folder/'capture_status.json',dict(valid=False,state='preflight',retryable_infrastructure_failure=True))
    shutil.copy2(scenario,folder/'scenario.yaml')
    if recovery:
        config=yaml.safe_load(scenario.read_text())
        subset={k:v for k,v in config.items() if not k.startswith('episode_') or k in recovery['episode_ids']}
        (folder/'episode_selection.yaml').write_text(yaml.safe_dump(subset,sort_keys=False))
        write_new(folder/'recovery_selection.json',recovery)
    provenance=folder/'provenance';provenance.mkdir()
    for name in ('TIC-VLA','steernav','NaVILA','OmniVLA','Steernav-ICRA26'):
        repo=ROOT.parent/name
        value=snapshot(repo)
        expected={'TIC-VLA':'sim','steernav':'sim-ground-truth','NaVILA':'custom'}.get(name)
        if expected and value['branch']!=expected:raise ValueError(f'Wrong branch for {name}')
        save_snapshot(provenance,name,value)
        if name=='TIC-VLA': local=value
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',args.host,'python3','-',args.remote_repo,args.run_id]
    remote=json.loads(subprocess.check_output(command,input=(ROOT/'scripts/source_snapshot.py').read_bytes(),timeout=60))
    save_snapshot(provenance,'remote-TIC-VLA',remote)
    if remote.get('capture_exists'):raise FileExistsError('Remote attempt results already exist; refusing reuse')
    required=['DynaNav/benchmark.py','DynaNav/benchmark_config.py','DynaNav/benchmark_reporting.py','DynaNav/scripted_humans.py',str(scenario)]
    for name in required:
        if local['source_hashes'].get(name) is None or local['source_hashes'][name]!=remote['source_hashes'].get(name):
            raise ValueError(f'Remote source mismatch (deployment not performed by this task): {name}')
    write_new(folder/'protocol_manifest.json',dict(protocol_version=info['protocol_version'], policy=args.policy,mode=args.mode,
        run_id=args.run_id,scenario=info,selected_episode_ids=recovery["episode_ids"] if recovery else info["episode_ids"],
        recovery_source=recovery,frozen=frozen,controller_dt=.75,fix_end=False,timeout=40,
        success_radius_m=info["success_radius_m"],stop_radius_m=.5,non_goal_hc_radius_m=.4,
        reference_logging=args.mode=='steered',reference_role='shadow only',independent_bag=False,
        aggregation='episode; bootstrap whole episodes, never frames'))


def crash_continuation(outcomes, errors):
    """Continue scheduling only for proven, isolated Isaac startup crashes."""
    crashes=[]
    for row in outcomes:
        if (row.get('error')=='no_result_file' and not row.get('termination_condition')
                and row.get('duration_frames')==0
                and any('crash' in str(line).lower() for line in row.get('isaac_sim_errors',[]))):
            crashes.append(row)
    allowed_errors=set()
    for row in crashes:
        ep=row['episode']
        allowed_errors.update((f'{ep}: missing/unknown termination condition',f'{ep}: missing termination reason',
                               f'{ep}: infrastructure failure: no_result_file'))
    good=[r for r in outcomes if r not in crashes]
    # Aggregate incompleteness is tolerable only when caused entirely by crashed rows.
    for key in ('success','human_collision','goal_human_collision','physical_collision','closest_goal_distance_m','normalized_progress','path_length','duration'):
        values=[r.get(key) for r in good]
        try:finite=all(v is not None and np.isfinite(float(v)) for v in values)
        except (TypeError,ValueError):finite=False
        if finite:allowed_errors.add(f'Incomplete or nonfinite episode metric: {key}')
    blocking=[error for error in errors if error not in allowed_errors]
    can_continue=bool(crashes) and len(crashes)<=2 and len(outcomes)==20 and bool(good) and not blocking
    return dict(continuation_allowed=can_continue,blocking_errors=blocking,
        deferred_episodes=[dict(episode=r['episode'],reason='isaac_crash_no_result',child_log=r.get('child_log')) for r in crashes],
        crash_count=len(crashes),max_isolated_crashes_per_condition=2)


def finish(args):
    folder=Path(args.capture);errors=[]
    try:
        manifest=json.loads((folder/'protocol_manifest.json').read_text())
        if hashlib.sha256((folder/'scenario.yaml').read_bytes()).hexdigest()!=manifest['scenario']['sha256']:raise ValueError('Scenario hash mismatch')
        expected_n=len(manifest['selected_episode_ids']) if manifest.get('recovery_source') else 20
        if manifest.get('recovery_source'):
            from recover_crossover_episodes import read_selection
            selection=manifest['recovery_source']
            checked=read_selection(folder/'scenario.yaml',args.policy,args.mode,manifest['selected_episode_ids'][0],selection['source_capture'])
            if checked!=selection:raise ValueError('Recovery selection evidence changed')
        summary=save_outcomes(folder/'benchmark_results.json',folder/('episode_selection.yaml' if manifest.get('recovery_source') else 'scenario.yaml'),args.mode,args.run_id)
        errors.extend(summary['infrastructure_or_validation_errors'])
        data=json.loads((folder/'benchmark_results.json').read_text())
        if data.get('run_id')!=args.run_id: errors.append('Run ID mismatch')
        for row in data['episodes']:
            ep=yaml.safe_load((folder/'scenario.yaml').read_text())[row['episode']]
            scenario_config=yaml.safe_load((folder/'scenario.yaml').read_text())
            required=dict(protocol_version=scenario_config['protocol_version'],success_threshold_m=scenario_config['success_threshold'],goal_human_index=ep['goal_human_index'],human_collision_threshold_m=.4,
                moving_away_min_progress_m=.5,moving_away_distance_buffer_m=.5,moving_away_grace_seconds=1.)
            for k,v in required.items():
                if row.get(k)!=v: errors.append(f"{row['episode']}: invalid {k}")
            for key in ('closest_goal_distance_m','normalized_progress','goal_human_collision','min_goal_human_distance_m','min_non_goal_human_distance_m','collision_identities','physical_collision','path_length','duration'):
                if key not in row:errors.append(f"{row['episode']}: missing {key}")
            if row.get('goal_human_collision') and (row.get('success') or row.get('human_collision') or row.get('termination_condition')!='goal_human_collision'):
                errors.append('Goal contact incorrectly scored')
        if (folder/'.remote.status').exists() and (folder/'.remote.status').read_text().strip()!='0':errors.append('Remote infrastructure exit status nonzero')
        if data.get('inference_method')!=args.policy:errors.append('Policy mismatch')
        rng=np.random.default_rng(36)
        condition={}
        for key in ('success','human_collision','goal_human_collision','physical_collision','closest_goal_distance_m','normalized_progress','path_length','duration'):
            values=np.asarray([r[key] for r in data['episodes'] if r.get(key) is not None],dtype=float)
            if not np.isfinite(values).all() or len(values)!=expected_n:
                errors.append(f'Incomplete or nonfinite episode metric: {key}');continue
            samples=np.mean(rng.choice(values,size=(10000,len(values)),replace=True),axis=1)
            condition[key]=dict(episodes=expected_n,mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)) if expected_n>1 else None,
                ci95=[float(v) for v in np.percentile(samples,[2.5,97.5])])
        write_new(folder/'condition_metrics.json',dict(metrics=condition,confidence_interval='95% percentile bootstrap of whole episodes, 10000 resamples, seed 36'))
        if args.mode=='steered':
            receiver=read_summary(folder/'receiver_timing.jsonl');write_new(folder/'receiver_summary.json',receiver)
            summaries=receiver['writer_summaries']
            if len(summaries)!=1 or not summaries[0].get('clean_shutdown') or summaries[0].get('queue_remaining')!=0 or any(k not in summaries[0] for k in ('attempted_records','written_records','dropped_records')):
                errors.append('Invalid writer summary')
            if summaries:
                last=summaries[-1]
                counts=[last.get(k) for k in ('attempted_records','written_records','dropped_records')]
                if any(type(v) is not int or v<0 for v in counts) or counts[0]!=counts[1]+counts[2]:errors.append('Inconsistent writer loss counters')
            raw=[json.loads(line) for line in (folder/'receiver_timing.jsonl').read_text().splitlines()]
            settings=[r.get('arguments',{}) for r in raw if r.get('event')=='settings']
            expected=dict(object_velocity_source='estimated',velocity_reference_log=True,time_look_ahead=1.5,velocity_window=5,
                odom_alignment='extrapolate',odom_pose_max_extrapolation=.25,image_queue_depth=1,process_repeated_images=False,
                taper_pedestrian_sweep=False,robot_radius=.5,
                additional_clearance=float(manifest.get('frozen', FROZEN)['STEERNAV_ADDITIONAL_CLEARANCE']),fix_end=False)
            if len(settings)!=1 or any(settings[0].get(k)!=v for k,v in expected.items()):errors.append('Effective receiver settings do not match frozen configuration')
            shutdown=json.loads((folder/'receiver_timing.jsonl.shutdown.json').read_text())
            if shutdown.get('error') or not shutdown.get('summary_present'):errors.append('Writer finalization failed')
            subprocess.run([sys.executable,str(ROOT/'scripts/compare_pedestrian_velocities.py'),str(folder/'receiver_timing.jsonl')],check=True)
    except Exception as error: errors.append(str(error))
    outcomes=json.loads((folder/'benchmark_results.json').read_text()).get('episodes',[]) if (folder/'benchmark_results.json').exists() else []
    navigation_ids=[r.get('episode') for r in outcomes if not r.get('error') and r.get('termination_condition') not in ('infrastructure_error','child_crash','child_timeout')]
    if not (folder/'termination_summary.json').exists():
        write_new(folder/'termination_summary.json',dict(complete=False,recorded_episodes=len(outcomes),infrastructure_or_validation_errors=errors))
    decision=crash_continuation(outcomes,errors)
    if decision['deferred_episodes']:
        write_new(folder/'recovery_needed.json',dict(run_id=args.run_id,policy=args.policy,mode=args.mode,
            capture=str(folder.resolve()),episode_only_retry_required=True,
            deferred_episodes=decision['deferred_episodes']))
    status=dict(continuation=decision, completed_navigation_episode_ids=navigation_ids, valid=not errors,state='complete' if not errors else 'infrastructure_or_validation_failure',
        validation_errors=errors,retryable_infrastructure_failure=bool(errors) and not navigation_ids,
        navigation_outcomes_are_never_retryable=True,run_id=args.run_id)
    # Status is the one mutable lifecycle artifact. Raw results and prior attempts remain intact.
    (folder/'capture_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))
    if errors:raise SystemExit(10 if decision["continuation_allowed"] else 1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','finish'))
    p.add_argument('--scenario');p.add_argument('--capture');p.add_argument('--mode',required=True)
    p.add_argument('--policy',required=True);p.add_argument('--run-id',default='dry-run')
    p.add_argument('--host',default='isaac');p.add_argument('--remote-repo',default='/home/isaac/Projects/TIC-VLA')
    p.add_argument('--dt',default='0.75');p.add_argument('--fix-end',default='false');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    (prepare if args.action=='prepare' else finish)(args)
