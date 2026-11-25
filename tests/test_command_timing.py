import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('command_summary',Path(__file__).resolve().parents[1]/'scripts/summarize_command_timing.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_hold_estimate_uses_forward_intervals_and_isolates_resets():
    def apply(sim, v, episode='a'):
        return dict(event='bridge_apply',pid=1,episode_id=episode,simulation_seconds=sim,
                    selected_linear_x=v,selected_angular_z=0.,command_fresh=True)
    rows = [apply(0.,1.),apply(.05,0.),apply(.1,0.),apply(0.,1.),apply(.02,0.),
            apply(0.,0.,'b'),apply(1.,0.,'b')]
    out = summary.summarize(rows)
    assert out['bridge']['observed_nonzero_hold_sim_seconds'] == pytest.approx(.07)
    assert out['bridge']['observed_zero_hold_sim_seconds'] == pytest.approx(1.05)
    assert out['bridge']['clock_regressions'] == 1
    assert 'controller_publish' in out['missing_evidence']


def test_old_ordinary_log_is_missing_evidence_not_zero_motion(tmp_path):
    p = tmp_path/'controller.log'
    p.write_text('[INFO] Publishing new velocity: 1.3, 0.0\n')
    out = summary.summarize(summary.read_rows(p))
    assert out['missing_evidence'] == ['controller_publish','bridge_apply','cmd_vel']
