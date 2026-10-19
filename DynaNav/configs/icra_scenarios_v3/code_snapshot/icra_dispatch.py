"""Dispatch validated evaluation groups while serializing access to each policy.

Run inside a held simulator GPU allocation. Results and locks live on the shared
filesystem so additional simulator workers can join without duplicate trials.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def complete(path):
    try:
        d = json.loads(path.read_text())
        return (not d.get('error') and d.get('measurement_version') == 'physx_capsules_fov_v1'
                and d.get('duration_frames', 0) > 0
                and d.get('measurement_frames') == d['duration_frames'])
    except (OSError, ValueError):
        return False


def validated(group):
    try:
        content = Path(group['config']).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        review = json.loads(Path(group['validation_file']).read_text())
        return (review.get('validated') is True and digest == group['config_sha256']
                and review.get('config_sha256') == digest)
    except (OSError, ValueError):
        return False


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default=str(ROOT / 'icra_suite_manifest.json'))
    p.add_argument('--chunk-size', type=int, default=1)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--worker-id', default=f'{socket.gethostname()}_{os.getpid()}')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.chunk_size < 1:
        p.error('chunk size must be positive')
    state_root = ROOT / 'icra_dispatch'
    state_root.mkdir(exist_ok=True)
    state = state_root / f'{a.worker_id}.json'
    log_root = state_root / 'logs'
    log_root.mkdir(exist_ok=True)
    os.environ.update(VEGA_CONTROLLER_TARGET='max_lateral', VEGA_V_MAX='1',
                      VEGA_W_MAX='0.3', VEGA_DISABLE_THIRD_PERSON='1',
                      VEGA_REVIEW_ROOT='', VEGA_OBSTACLE_CONFIG='',
                      VEGA_RENDER_ON_SENSOR_ONLY='0', VEGA_RENDER_TELEMETRY='1')
    while True:
        manifest = json.loads(Path(a.manifest).read_text())
        groups = manifest['groups']
        counts = {g['run_id']: sum(complete(ROOT / 'results' / g['run_id'] / f'{e}.json')
                                  for e in g['episodes']) for g in groups}
        pending = [g for g in groups if counts[g['run_id']] < len(g['episodes'])]
        if a.dry_run:
            print(json.dumps({'completed': sum(counts.values()),
                              'requested': manifest['requested'],
                              'groups': [{'run_id': g['run_id'], 'completed': counts[g['run_id']],
                                          'validated': validated(g)} for g in groups]}, indent=2))
            return
        if not pending:
            atomic_json(state, {'status': 'complete', 'updated_unix': time.time(), 'completed': counts})
            return
        priority = manifest.get('priority_model')
        if priority and any(g['model'] == priority for g in pending):
            pending = [g for g in pending if g['model'] == priority]
        selected = None
        lock = None
        # Balanced progress across models and scenarios; preserve YAML trial order.
        for group in sorted(pending, key=lambda g: (g['scenario'], counts[g['run_id']], g['model']) if priority else (counts[g['run_id']], g['scenario'], g['model'])):
            if not validated(group):
                continue
            candidate = (state_root / f"policy_{group['model']}.lock").open('a+')
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
                continue
            selected, lock = group, candidate
            break
        if selected is None:
            atomic_json(state, {'status': 'waiting_for_validation_or_policy',
                                'updated_unix': time.time(), 'completed': counts})
            time.sleep(20)
            continue
        try:
            url = Path(selected['policy_url_file']).read_text().strip()
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.read() != b'ready':
                    raise RuntimeError(f'Policy is not ready: {url}')
            run_id = selected['run_id']
            env = os.environ.copy()
            env['VEGA_POLICY_URL'] = url
            env['VEGA_TRACE_ROOT'] = str(ROOT / 'results' / run_id / 'predictions')
            output = ROOT / 'results' / run_id
            output.mkdir(parents=True, exist_ok=True)
            code_paths = [ROOT / 'persistent_suite.py', ROOT / 'isaac_python.sh',
                          ROOT.parent / 'TIC-VLA/DynaNav/benchmark.py',
                          ROOT.parent / 'TIC-VLA/DynaNav/behavior/nova_carter_test_ticvla.py',
                          ROOT.parent / 'TIC-VLA/DynaNav/contact_fov_metrics.py',
                          ROOT.parent / 'TIC-VLA/DynaNav/render_cadence.py',
                          ROOT.parent / 'TIC-VLA/DynaNav/vega_client.py',
                          ROOT.parent / 'TIC-VLA/DynaNav/scenario_target.py']
            provenance = {'group': selected, 'policy_url': url,
                          'worker': a.worker_id, 'host': socket.gethostname(),
                          'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                          'runtime_root': os.environ.get('ISAAC_SIM_ROOT'),
                          'started_unix': time.time(),
                          'environment': {k: v for k, v in env.items() if k.startswith('VEGA_')},
                          'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in code_paths}}
            service_provenance = Path(selected['policy_url_file']).parent / 'provenance.json'
            if service_provenance.exists():
                provenance['policy_provenance'] = json.loads(service_provenance.read_text())
            atomic_json(output / f'launch_{int(time.time())}_{a.worker_id}.json', provenance)
            command = [str(ROOT / 'isaac_python.sh'), str(ROOT / 'persistent_suite.py'),
                       '--config', selected['config'], '--run-id', run_id,
                       '--max-episodes', str(a.chunk_size), '--cpu-threads', str(a.cpu_threads)]
            log = log_root / f'{a.worker_id}_{run_id}_{int(time.time())}.log'
            atomic_json(state, {'status': 'running', 'updated_unix': time.time(),
                                'group': selected, 'log': str(log), 'command': command,
                                'completed_before_chunk': counts[run_id]})
            with log.open('w') as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Worker failed with code {result.returncode}; inspect {log}')
            summary = ROOT / 'summarize_icra.py'
            if summary.exists():
                subprocess.run([sys.executable, str(summary)], check=True)
        except BaseException as error:
            atomic_json(state, {'status': 'error', 'updated_unix': time.time(),
                                'group': selected, 'error': str(error)})
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


if __name__ == '__main__':
    main()
