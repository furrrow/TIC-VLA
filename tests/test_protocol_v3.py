"""V3 scenario preservation and launch isolation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import protocol_v2


@pytest.mark.parametrize('level', ['easy', 'medium', 'hard'])
def test_v3_preserves_trials_and_validates(level):
    name = f'scenario3_crossover_{level}_20trials.yaml'
    path = ROOT / 'DynaNav/configs/crossover_scenarios_v3' / name
    old = yaml.safe_load((ROOT / 'DynaNav/configs/crossover_scenarios_v2' / name).read_text())
    new = yaml.safe_load(path.read_text())
    assert {k for k in new if new[k] != old[k]} == {'protocol_version', 'success_threshold'}
    info = protocol_v2.check_scenario(path)
    assert info['protocol_version'] == 3
    assert info['success_radius_m'] == 1.5
    assert len(info['episode_ids']) == 20


def test_v3_suite_dry_run_never_launches(tmp_path):
    for command in ('ssh', 'scp', 'tmux', 'uv', 'ros2'):
        path = tmp_path / command
        path.write_text('#!/bin/sh\necho UNEXPECTED_LAUNCH >&2\nexit 99\n')
        path.chmod(0o755)
    env = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}')
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_crossover_v3.bash'), '--dry-run'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('"no_launch": true') == 9
