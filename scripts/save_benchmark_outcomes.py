#!/usr/bin/env python3
"""Validate complete episode outcomes and save local termination artifacts."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import yaml


def save_outcomes(result_path, scenario_path, mode, run_id):
    result_path, scenario_path = Path(result_path), Path(scenario_path)
    data = json.loads(result_path.read_text())
    config = yaml.safe_load(scenario_path.read_text())
    expected = {k for k in config if k.startswith('episode_')}
    episodes = data.get('episodes', [])
    ids = [r.get('episode') for r in episodes]
    errors = []
    if not expected: errors.append('Scenario contains no episodes')
    if set(ids) != expected or len(ids) != len(expected):
        errors.append('Expected exactly one outcome for every scenario episode')
    if data.get('experiment_mode') != mode:
        errors.append('Result experiment mode does not match requested mode')
    for row in episodes:
        if not row.get('termination_condition') or row.get('termination_condition') == 'unknown':
            errors.append(f"{row.get('episode')}: missing/unknown termination condition")
        if not row.get('termination_reason'):
            errors.append(f"{row.get('episode')}: missing termination reason")
        if row.get('error') or row.get('termination_reason') == 'no_result_file':
            errors.append(f"{row.get('episode')}: infrastructure failure: {row.get('error') or 'no_result_file'}")
    for row in episodes:
        episode = config.get(row.get('episode'), {})
        margin = episode.get('goal_stop_collision_margin_m', config.get('goal_stop_collision_margin_m'))
        if margin is not None:
            collision = episode.get('human_collision_threshold_m', config.get('human_collision_threshold_m', .2))
            score = episode.get('success_threshold', config.get('success_threshold'))
            if (row.get('goal_stop_protocol') != 'collision_margin'
                or not isinstance(row.get('goal_stop_radius_m'), (int,float))
                or not math.isclose(row['goal_stop_radius_m'], collision+margin, abs_tol=1e-9)
                or row.get('success_threshold_m') != score):
                errors.append(f"{row.get('episode')}: reported stop/scoring protocol does not match scenario")
    columns = ['episode', 'success', 'success_threshold_m', 'success_criteria_met', 'goal_stop_protocol', 'goal_stop_radius_m', 'goal_stop_collision_margin_m', 'human_collision_threshold_m', 'termination_condition', 'termination_reason', 'failure_reason',
               'closest_goal_distance_m', 'normalized_progress', 'navigation_error', 'human_collision', 'non_goal_collision', 'goal_human_collision', 'physical_collision', 'min_non_goal_human_distance_m', 'min_goal_human_distance_m', 'collision_identities', 'path_length', 'duration', 'child_log']
    with (result_path.parent/'episode_outcomes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(episodes)
    summary = dict(run_id=run_id, experiment_mode=mode, expected_episodes=len(expected),
        recorded_episodes=len(episodes), complete=not errors,
        termination_conditions=dict(Counter(r.get('termination_condition') or 'missing' for r in episodes)),
        goal_stop_protocols=dict(Counter(r.get('goal_stop_protocol') or 'legacy_unrecorded' for r in episodes)),
        non_goal_hc_trials=sum(bool(r.get('human_collision')) for r in episodes),
        goal_human_contacts=sum(bool(r.get('goal_human_collision')) for r in episodes),
        physical_collision_trials=sum(bool(r.get('physical_collision')) for r in episodes),
        successes=sum(r.get('success') is True for r in episodes),
        infrastructure_or_validation_errors=errors)
    (result_path.parent/'termination_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--scenario', required=True, type=Path)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--run-id', required=True)
    args=parser.parse_args()
    summary=save_outcomes(args.result,args.scenario,args.mode,args.run_id)
    print(json.dumps(summary,indent=2))
    raise SystemExit(0 if summary['complete'] else 1)
