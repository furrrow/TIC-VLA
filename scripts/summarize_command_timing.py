#!/usr/bin/env python3
"""Offline summary of opt-in command traces and passive receipt observations."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        if 'COMMAND_TRACE ' in line:
            line = line.split('COMMAND_TRACE ', 1)[1]
        elif not line.startswith('{'):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize(rows):
    by_event = defaultdict(list)
    for row in rows:
        by_event[row.get('event')].append(row)
    result = dict(events=dict(Counter(row.get('event') for row in rows)))
    commands = by_event['controller_publish']
    result['controller'] = dict(publications=len(commands),
        nonzero=sum(abs(r['linear_x']) > .001 or abs(r['angular_z']) > .001 for r in commands),
        reasons=dict(Counter(r['reason'] for r in commands)),
        watchdog_zeros=sum(r['reason']=='control_tick_wall_timeout' and
                          abs(r['linear_x']) <= .001 and abs(r['angular_z']) <= .001 for r in commands))
    applied = by_event['bridge_apply']
    groups = defaultdict(list)
    for r in applied:
        groups[(r.get('pid'),r.get('episode_id'))].append(r)
    nonzero_sim = zero_sim = 0.
    resets = 0
    for group in groups.values():
        for a,b in zip(group,group[1:]):
            dt = b['simulation_seconds']-a['simulation_seconds']
            if dt < 0:
                resets += 1
                continue
            # Hold-interval estimate between observed API submissions; no extrapolation.
            if abs(a['selected_linear_x']) > .001 or abs(a['selected_angular_z']) > .001:
                nonzero_sim += dt
            else:
                zero_sim += dt
    result['bridge'] = dict(submissions=len(applied),
        stale_command_submissions=sum(not r['command_fresh'] for r in applied),
        observed_nonzero_hold_sim_seconds=nonzero_sim,
        observed_zero_hold_sim_seconds=zero_sim, clock_regressions=resets,
        note='Forward intervals between logged wheel API submissions; excludes final unobserved interval. Not measured robot motion.')
    received = by_event['cmd_vel']
    result['observer'] = dict(cmd_vel_receipts=len(received),
        nonzero_receipts=sum(abs(r['linear_x'])>.001 or abs(r['angular_z'])>.001 for r in received),
        note='BEST_EFFORT receipt observations may miss messages; does not identify originating publisher.')
    result['missing_evidence'] = [name for name, data in [('controller_publish',commands),('bridge_apply',applied),('cmd_vel',received)] if not data]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs',nargs='+',type=Path)
    args = parser.parse_args()
    print(json.dumps({str(p):summarize(read_rows(p)) for p in args.logs},indent=2))


if __name__ == '__main__':
    main()
