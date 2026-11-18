#!/usr/bin/env python3
"""Summarize receiver callback order; does not infer network arrival times."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time


def summarize(records):
    odom = defaultdict(list)
    queries = []
    sources = Counter()
    for row in records:
        if row['event'] == 'odom_callback':
            odom[row.get('episode_id')].append(row)
        elif row['event'] == 'pose_query':
            queries.append(row)
            sources[row['source']] += 1
    index = {}
    for episode, rows in odom.items():
        rows.sort(key=lambda r: r['odom_stamp'])
        index[episode] = ([r['odom_stamp'] for r in rows], rows)
    matches = Counter()
    delays = []
    for query in queries:
        stamps, rows = index.get(query.get('episode_id'), ([], []))
        i = bisect_left(stamps, query['image_stamp'] - 1e-9)
        candidates = []
        while i < len(stamps) and abs(stamps[i]-query['image_stamp']) <= 1e-9:
            candidates.append(rows[i])
            i += 1
        if not candidates:
            matches['no matching callback in capture'] += 1
            continue
        first = min(candidates, key=lambda r: r['callback_monotonic_ns'])
        delay = (first['callback_monotonic_ns'] - query['query_monotonic_ns'])/1e6
        if delay > 0:
            matches['matching callback after query'] += 1
            delays.append(delay)
        else:
            matches['matching callback before query'] += 1
            if query['source'] != 'exact':
                matches['nonexact despite earlier matching callback (inspect reset/eviction/tolerance)'] += 1
    return dict(queries=len(queries), alignment_sources=dict(sources),
                matching_odom_callbacks=dict(matches),
                later_callback_delay_ms=dict(count=len(delays),
                    median=sorted(delays)[len(delays)//2] if delays else None,
                    maximum=max(delays) if delays else None))


def read_summary(path):
    rows = []
    malformed = 0
    with open(path) as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    result = summarize(rows)
    result['malformed_lines'] = malformed
    result['writer_summaries'] = [r for r in rows if r['event'] == 'writer_summary']
    result['capture_finalized'] = bool(result['writer_summaries'])
    result['caution'] = ('Callback entry is not DDS arrival. Missing matching callbacks can reflect '
                         'drops, capture boundaries, stamp differences or reset; inspect raw records. '
                         'A later matching callback does not prove its message was queued at query time.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jsonl')
    parser.add_argument('--watch', action='store_true', help='Refresh every 2 s while capture runs')
    parser.add_argument('--done-file', type=Path, help='Stop refreshing when capture exit-status file exists')
    args = parser.parse_args()
    try:
        while True:
            if args.watch and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            done = args.done_file is not None and args.done_file.exists()
            if Path(args.jsonl).exists():
                result = read_summary(args.jsonl)
                if args.watch:
                    print('FINAL CAPTURE' if done else 'LIVE CAPTURE (counts provisional)')
                print(json.dumps(result, indent=2), flush=True)
            elif args.watch:
                print('Waiting for receiver_timing.jsonl; model initialization may take a few minutes.', flush=True)
            else:
                parser.error(f'Log does not exist: {args.jsonl}')
            if done:
                print('Capture exit status: ' + args.done_file.read_text().strip())
            if not args.watch or done:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
