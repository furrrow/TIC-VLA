#!/usr/bin/env python3
"""Prepare episode-only recovery through existing drivers; never launch on import/plan."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import yaml

ROOT=Path(__file__).resolve().parents[1]


def read_selection(scenario,policy,mode,episode=None,source=None):
    if not episode and not source:return None
    if not episode or not source:raise ValueError('Recovery requires both episode and source capture')
    source=Path(source).resolve()
    manifest=json.loads((source/'protocol_manifest.json').read_text())
    data=json.loads((source/'benchmark_results.json').read_text())
    config=yaml.safe_load(Path(scenario).read_text())
    if (manifest['policy']!=policy or manifest['mode']!=mode or data.get('inference_method')!=policy
            or data.get('experiment_mode')!=mode or data.get('run_id')!=manifest['run_id']):
        raise ValueError('Recovery source policy/mode/run ID mismatch')
    digest=hashlib.sha256(Path(scenario).read_bytes()).hexdigest()
    if manifest['scenario']['sha256']!=digest or hashlib.sha256((source/'scenario.yaml').read_bytes()).hexdigest()!=digest:
        raise ValueError('Recovery scenario differs from original capture')
    from protocol_v2 import FROZEN
    if manifest.get('frozen')!=FROZEN:raise ValueError('Original frozen settings mismatch')
    if episode not in config or not episode.startswith('episode_'):raise ValueError('Unknown recovery episode')
    rows=[r for r in data['episodes'] if r.get('episode')==episode]
    if len(rows)>1:raise ValueError('Duplicate source episode')
    if rows:
        row=rows[0]
        if not (row.get('error')=='no_result_file' and any('crash' in str(x).lower() for x in row.get('isaac_sim_errors',[]))):
            raise ValueError('Refusing to repeat a completed navigation outcome or unverified failure')
    return dict(episode_ids=[episode],source_capture=str(source),source_run_id=manifest['run_id'],
        source_result_sha256=hashlib.sha256((source/'benchmark_results.json').read_bytes()).hexdigest(),
        reason='verified_isaac_crash' if rows else 'missing_after_interruption')


def wrap(folder):
    folder=Path(folder);manifest=json.loads((folder/'protocol_manifest.json').read_text())
    ids=manifest.get('selected_episode_ids')
    if not ids or len(ids)!=1:raise ValueError('Not a single-episode recovery capture')
    raw=json.loads((folder/'raw_episode_result.json').read_text())
    if (raw.get('episode')!=ids[0] or raw.get('inference_method')!=manifest['policy'] or raw.get('experiment_mode')!=manifest['mode']):raise ValueError('Wrong episode/policy/mode returned by simulator')
    data=dict(run_id=manifest['run_id'],inference_method=manifest['policy'],experiment_mode=manifest['mode'],
        episodes=[raw],recovery_source=manifest['recovery_source'])
    with (folder/'benchmark_results.json').open('x') as stream:json.dump(data,stream,indent=2)


def create_plan(snapshot,output):
    from protocol_v2 import write_new
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(snapshot,output/'original_remote_snapshot.json')
    jobs=[]
    for item in json.loads(Path(snapshot).read_text()):
        run=item['run_id'];data=item['result'];policy=data['inference_method'];mode=data['experiment_mode']
        level=run.split('_scenario3_crossover_')[1].split('_20trials')[0]
        scenario=f'DynaNav/configs/crossover_scenarios_v2/scenario3_crossover_{level}_20trials.yaml'
        config=yaml.safe_load((ROOT/scenario).read_text());expected=[k for k in config if k.startswith('episode_')]
        batch=run[len(policy)+1:].rsplit('_attempt1_',1)[0]
        original=ROOT.parent/policy/'logs/orchestration'/batch/f'{batch}_{mode}'/'attempt1'
        rows={r['episode']:r for r in data['episodes']}
        if len(rows)!=len(data['episodes']) or set(rows)-set(expected):raise ValueError('Unexpected/duplicate source IDs')
        missing=[ep for ep in expected if ep not in rows or rows[ep].get('error')]
        if not missing:continue
        source=output/'sources'/run;source.mkdir(parents=True)
        for name in ['protocol_manifest.json','scenario.yaml']:shutil.copy2(original/name,source/name)
        write_new(source/'benchmark_results.json',data)
        slug={'OmniVLA':'omnivla','NaVILA':'navila','TIC-VLA':'ticvla'}[policy]
        campaign=batch.split(f'_{slug}_scenario3_crossover_')[0]
        for ep in missing:
            selection=read_selection(ROOT/scenario,policy,mode,ep,source)
            attempt=f'attempt2_{ep}'
            capture=original.parent/attempt
            if capture.exists():raise FileExistsError(f'Recovery attempt already exists: {capture}')
            command=['env',f'BATCH_ID={campaign}_{slug}',f'RUN_ATTEMPT_ID={attempt}',
                f'PROTOCOL_V2_RECOVERY_EPISODE={ep}',f'PROTOCOL_V2_RECOVERY_SOURCE={source}',
                str(ROOT.parent/policy/'run_nova_scenarios.bash'),'--mode',mode,'--no-fix-end',Path(scenario).stem]
            jobs.append(dict(policy=policy,mode=mode,scenario=scenario,episode=ep,source_run_id=run,
                capture=str(capture),command=command,selection=selection))
    write_new(output/'recovery_plan.json',dict(jobs=jobs,episodes=len(jobs)))
    lines=['#!/usr/bin/env bash','set -euo pipefail',
        '[[ -z "${TMUX:-}" ]] || { echo "Run outside tmux to see each experiment session." >&2; exit 2; }',
        'export PROTOCOL_V2_APPROVED=true OBJECT_VELOCITY_SOURCE=estimated CONTINUE_ON_EPISODE_CRASH=false']
    for i,j in enumerate(jobs,1):
        lines += [f"echo 'Recovery {i}/{len(jobs)}: {j['policy']} {j['mode']} {j['episode']}'",shlex.join(j['command'])]
    lines += [shlex.join(['python3',str(Path(__file__).resolve()),'merge','--folder',str(output)])]
    (output/'run_recovery.bash').write_text('\n'.join(lines)+'\n')
    return jobs


def merge(folder):
    """New derived output only; originals and individual attempt results are immutable."""
    from protocol_v2 import write_new
    from save_benchmark_outcomes import save_outcomes
    folder=Path(folder);plan=json.loads((folder/'recovery_plan.json').read_text())
    original=json.loads((folder/'original_remote_snapshot.json').read_text());byrun={x['run_id']:x['result'] for x in original}
    for job in plan['jobs']:
        capture=Path(job['capture']);status=json.loads((capture/'capture_status.json').read_text())
        if not status.get('valid'):raise ValueError(f'Unaccepted recovery: {capture}')
        row=json.loads((capture/'benchmark_results.json').read_text())['episodes'][0]
        if row['episode']!=job['episode']:raise ValueError('Recovery episode mismatch')
        data=byrun[job['source_run_id']];old=[r for r in data['episodes'] if r['episode']==row['episode']]
        if old and not old[0].get('error'):raise ValueError('Refusing to replace navigation outcome')
        data['episodes']=[r for r in data['episodes'] if r['episode']!=row['episode']]+[dict(row,recovery_capture=str(capture))]
    out=folder/'merged';out.mkdir(exist_ok=False)
    for run,data in byrun.items():
        level=run.split('_scenario3_crossover_')[1].split('_20trials')[0]
        scenario=ROOT/f'DynaNav/configs/crossover_scenarios_v2/scenario3_crossover_{level}_20trials.yaml'
        expected={k for k in yaml.safe_load(scenario.read_text()) if k.startswith('episode_')}
        if len(data['episodes'])!=20 or {r['episode'] for r in data['episodes']}!=expected or any(r.get('error') for r in data['episodes']):raise ValueError(f'Incomplete merged condition: {run}')
        dest=out/run;dest.mkdir();write_new(dest/'benchmark_results.json',data)
        summary=save_outcomes(dest/'benchmark_results.json',scenario,data['experiment_mode'],run)
        if not summary['complete']:raise ValueError(f'Merged validation failed: {run}')
    write_new(out/'completion.json',dict(conditions=len(byrun),episodes=sum(len(d['episodes']) for d in byrun.values()),
        note='Navigation results merged with accepted episode-only recoveries. Original telemetry and recovery telemetry remain separate captures.'))
    print(f'Saved merged results: {out}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','wrap','merge'])
    p.add_argument('--snapshot',type=Path);p.add_argument('--folder',type=Path,required=True);args=p.parse_args()
    if args.action=='plan':print(f'Prepared {len(create_plan(args.snapshot,args.folder))} episodes; not launched.')
    elif args.action=='wrap':wrap(args.folder)
    else:merge(args.folder)
