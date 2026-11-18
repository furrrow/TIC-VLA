#!/usr/bin/env python3
"""Offline shadow comparison using later exact/bracketed odometry, not live alignment.

Simulator finite differences are a reference, not noise-free physical truth.
Matching is geometric and uncertain. Rejected visual estimates remain in metrics.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def reference_pose(rows, stamp, max_bracket=.25):
    times = [r['odom_stamp'] for r in rows]
    i = bisect_left(times, stamp-1e-9)
    if i < len(rows) and abs(times[i]-stamp) <= 1e-9:
        position = rows[i]['position']
        rotation = Rotation.from_quat(rows[i]['quaternion_xyzw']).as_matrix()
    elif 0 < i < len(rows) and 0 < times[i]-times[i-1] <= max_bracket:
        a, b = rows[i-1], rows[i]
        fraction = (stamp-times[i-1])/(times[i]-times[i-1])
        position = (1-fraction)*np.array(a['position'])+fraction*np.array(b['position'])
        rotation = Slerp([times[i-1], times[i]], Rotation.from_quat(
            [a['quaternion_xyzw'], b['quaternion_xyzw']]))([stamp]).as_matrix()[0]
    else:
        return None
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, position
    return pose


def states_at(rows, stamp, max_age):
    if not rows:
        return None
    times = [r['reference_stamp'] for r in rows]
    i = bisect_left(times, stamp)
    candidates = rows[max(0, i-1):i+1]
    best = min(candidates, key=lambda r: abs(r['reference_stamp']-stamp))
    gap = stamp-best['reference_stamp']
    if abs(gap) > max_age:
        return None
    states = {key: (np.asarray(v['position_world'])+gap*np.asarray(v['velocity_world']),
                    np.asarray(v['velocity_world'])) for key, v in best['pedestrians'].items()}
    return states, gap


def metrics(estimate, reference):
    estimate, reference = np.asarray(estimate), np.asarray(reference)
    a, b = np.linalg.norm(estimate[:2]), np.linalg.norm(reference[:2])
    direction = None
    if a > .1 and b > .1:
        direction = float(np.degrees(np.arccos(np.clip(np.dot(estimate[:2], reference[:2])/(a*b), -1, 1))))
    return dict(vector_error_mps=float(np.linalg.norm(estimate-reference)),
                reference_speed_mps=float(b), estimated_speed_mps=float(a),
                speed_error_mps=float(a-b), absolute_speed_error_mps=float(abs(a-b)), direction_error_deg=direction)


def distribution(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return dict(count=0)
    return dict(count=len(values), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(max(values)))


def analyze(rows, steernav_root, max_age=.1, match_gate=1., ambiguity_margin=.2):
    sys.path.insert(0, str(Path(steernav_root)/'steernav'))
    from sim_pedestrians import match_pedestrians_to_tracks, pedestrian_states_in_camera
    settings = next(r for r in rows if r['event'] == 'settings')
    extrinsic = np.asarray(settings['T_base_from_cam'])
    odom, pedestrians, groups = defaultdict(list), defaultdict(list), defaultdict(list)
    records, coverage = [], Counter()
    reasons = Counter()
    for row in rows:
        ep = row.get('episode_id')
        if row['event'] == 'odom_callback': odom[ep].append(row)
        if row['event'] == 'pedestrian_reference': pedestrians[ep].append(row)
        if row['event'] in ('visual_velocity', 'external_velocity'):
            row = dict(row, velocity_source='sim-ground-truth' if row['event']=='external_velocity' else 'estimated')
            reasons[row['reason']] += 1
            groups[ep, row['image_stamp'], row.get('observation_stamp')].append(row)
    for values in odom.values(): values.sort(key=lambda r: r['odom_stamp'])
    for values in pedestrians.values(): values.sort(key=lambda r: r['reference_stamp'])
    for (ep, image_stamp, stamp), group in groups.items():
        if stamp is None:
            coverage['missing_observation_stamp'] += len(group)
            continue
        pose = reference_pose(odom[ep], stamp)
        reference = states_at(pedestrians[ep], stamp, max_age)
        if pose is None or reference is None:
            coverage['missing_reference_pose' if pose is None else 'missing_reference_snapshot'] += len(group)
            continue
        camera = pose @ extrinsic
        states, snapshot_gap = reference
        camera_states = pedestrian_states_in_camera(states, camera)
        positions = {r['track_id']: r['position_camera'] for r in group if r.get('position_camera') is not None}
        matches = match_pedestrians_to_tracks(positions, camera_states, max_ground_distance=match_gate)
        for row in group:
            item = dict(episode_id=ep, image_stamp=image_stamp, observation_stamp=stamp,
                        track_id=row['track_id'], velocity_source=row.get('velocity_source','estimated'), reason=row['reason'],
                        projection_applied=row['projection_applied'], reference_snapshot_gap=snapshot_gap)
            match = matches.get(row['track_id'])
            if match is None:
                item['comparison_status'] = 'unmatched'
            else:
                pid = match['pedestrian_id']
                pos = np.asarray(positions[row['track_id']])
                distances = sorted(float(np.linalg.norm(pos[[0,2]]-p[[0,2]]))
                    for p, v in camera_states.values() if p[2] > 0)
                ambiguous = (len(distances) > 1 and distances[1]-distances[0] < ambiguity_margin) or (bool(distances) and match['distance']-distances[0] > 1e-6)
                item.update(pedestrian_id=pid, supplied_pedestrian_id=row.get('supplied_pedestrian_id'), supplied_snapshot_stamp=row.get('supplied_snapshot_stamp'), match_distance_m=match['distance'],
                            ambiguous_match=bool(ambiguous))
                estimate = row.get('estimated_velocity_camera')
                if estimate is not None: estimate = np.asarray(estimate, dtype=float)
                if row.get('supplied_pedestrian_id') is not None and row['supplied_pedestrian_id'] != pid:
                    item['comparison_status'] = 'supplied_identity_disagrees_with_reference_match'
                elif ambiguous:
                    item['comparison_status'] = 'ambiguous_match'
                elif estimate is None:
                    item['comparison_status'] = 'no_supplied_velocity' if row.get('velocity_source') == 'sim-ground-truth' else 'no_visual_estimate'
                elif not np.isfinite(estimate).all():
                    item['comparison_status'] = 'nonfinite_visual_estimate'
                else:
                    estimated_world = camera[:3,:3] @ np.asarray(estimate)
                    reference_world = states[pid][1]
                    item.update(comparison_status='compared', estimated_velocity_world=estimated_world.tolist(),
                                reference_velocity_world=reference_world.tolist(),
                                **metrics(estimated_world, reference_world))
                    item['displacement_error_at_horizon_m'] = item['vector_error_mps']*row.get('prediction_horizon', 1.5)
                    # Compare against the same robust temporal fit too: an instantaneous
                    # reference can differ legitimately during starts/stops/turns.
                    observations = row.get('observation_stamps', [])
                    reference_positions = []
                    for t in observations:
                        sample = states_at(pedestrians[ep], t, max_age)
                        if sample is None or pid not in sample[0]: break
                        reference_positions.append(camera[:3,:3].T @ sample[0][pid][0])
                    slopes = []
                    if len(reference_positions) == len(observations):
                        for i in range(len(observations)):
                            for j in range(i+1, len(observations)):
                                dt = observations[j]-observations[i]
                                if dt >= row.get('min_velocity_interval', .075):
                                    slopes.append((reference_positions[j]-reference_positions[i])/dt)
                    if slopes:
                        fit_world = camera[:3,:3] @ np.median(slopes, axis=0)
                        item['window_reference_velocity_world'] = fit_world.tolist()
                        item['window_vector_error_mps'] = float(np.linalg.norm(estimated_world-fit_world))
            coverage[item['comparison_status']] += 1
            records.append(item)
    compared = [r for r in records if r['comparison_status'] == 'compared']
    summary = dict(taper_pedestrian_sweep=bool(settings.get('arguments',{}).get('taper_pedestrian_sweep',False)), velocity_source=settings.get('arguments',{}).get('object_velocity_source','estimated'), diagnostic_reasons=dict(reasons), coverage=dict(coverage),
        metrics={key: distribution([r.get(key) for r in compared]) for key in
            ['vector_error_mps', 'speed_error_mps', 'absolute_speed_error_mps', 'direction_error_deg', 'window_vector_error_mps', 'displacement_error_at_horizon_m']},
        stationary_reference_count=sum(r['reference_speed_mps'] <= .1 for r in compared),
        stationary_reference_projected=sum(r['reference_speed_mps'] <= .1 and r['projection_applied'] for r in compared),
        writer_summaries=[r for r in rows if r['event']=='writer_summary'],
        reference_note='Later exact/bracketed odometry, nearest snapshot propagated <= max_age; geometric identity matching can be wrong. No extrapolated reference pose. Metrics include gate-rejected estimates; missing estimates are coverage failures, not zero error.',
        parameters=dict(max_reference_age=max_age, max_match_distance=match_gate, ambiguity_margin=ambiguity_margin))
    return summary, records


def add_bag_reference(rows, bag):
    """Use the independent recorder for denser reference data after capture ends."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String
    from nav_msgs.msg import Odometry
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[
        '/isaacsim/episode', '/isaacsim/pedestrians', '/isaacsim/odom']))
    references = []
    ep = None
    while reader.has_next():
        topic, data, receipt = reader.read_next()
        if topic == '/isaacsim/odom':
            if ep is None: continue
            m = deserialize_message(data, Odometry)
            p, q = m.pose.pose.position, m.pose.pose.orientation
            references.append(dict(event='odom_callback', episode_id=ep,
                odom_stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
                position=[p.x,p.y,p.z], quaternion_xyzw=[q.x,q.y,q.z,q.w]))
        else:
            m = json.loads(deserialize_message(data, String).data)
            if topic == '/isaacsim/episode':
                if m.get('state') == 'started': ep = m['episode_id']
            else:
                references.append(dict(event='pedestrian_reference', episode_id=m['episode_id'],
                    reference_stamp=m['stamp']['sec']+m['stamp']['nanosec']*1e-9,
                    pedestrians={v['id']: dict(position_world=v['position'], velocity_world=v['velocity'])
                                 for v in m['pedestrians']}))
    # Use one reference stream consistently, retaining node records for diagnostics.
    return [r for r in rows if r['event'] not in ('odom_callback','pedestrian_reference')] + references


def episode_aggregation(rows, records, episode_ids=None):
    """Equal episode weighting; bootstrap resamples entire episodes, not frames."""
    episodes = sorted(set(episode_ids or []) | {r.get('episode_id') for r in rows if r.get('episode_id') is not None})
    output = []
    for ep in episodes:
        raw = [r for r in rows if r.get('episode_id') == ep and r.get('event') == 'visual_velocity']
        matched = [r for r in records if r.get('episode_id') == ep]
        compared = [r for r in matched if r.get('comparison_status') == 'compared']
        unambiguous = [r for r in matched if r.get('pedestrian_id') is not None and not r.get('ambiguous_match')]
        n = len(raw)
        def ratio(count): return count/n if n else None
        errors = [r['vector_error_mps'] for r in compared]
        poses = Counter(r.get('source','unknown') for r in rows if r.get('episode_id') == ep and r.get('event') == 'pose_query')
        output.append(dict(episode_id=ep, observations=n, compared=len(compared),
            median_vector_error_mps=float(np.median(errors)) if errors else None,
            p95_vector_error_mps=float(np.percentile(errors,95)) if errors else None,
            median_displacement_error_1_5s_m=float(np.median(errors)*1.5) if errors else None,
            unambiguous_match_coverage=ratio(len(unambiguous)),
            ambiguous_unmatched_fraction=ratio(n-len(unambiguous)),
            estimate_availability=ratio(sum(r.get('estimated_velocity_camera') is not None and np.isfinite(np.asarray(r['estimated_velocity_camera'],dtype=float)).all() for r in raw)),
            projection_coverage=ratio(sum(bool(r.get('projection_applied')) for r in raw)),
            pose_availability_breakdown=dict(poses)))
    metrics = {}
    rng = np.random.default_rng(36)
    for key in ('median_vector_error_mps','p95_vector_error_mps','median_displacement_error_1_5s_m',
                'unambiguous_match_coverage','ambiguous_unmatched_fraction','estimate_availability','projection_coverage'):
        values = np.asarray([r[key] for r in output if r[key] is not None],dtype=float)
        if len(values):
            samples = np.median(rng.choice(values,size=(10000,len(values)),replace=True),axis=1)
            metrics[key] = dict(median=float(np.median(values)), episodes_with_data=len(values),
                ci95=[float(v) for v in np.percentile(samples,[2.5,97.5])])
        else: metrics[key] = dict(median=None,episodes_with_data=0,ci95=None)
    return dict(episodes=output,condition_metrics=metrics,expected_episodes=len(episodes),
        confidence_interval='95% percentile bootstrap, 10000 whole-episode resamples, seed 36; episodes without measurable error excluded from error CI and retained in coverage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jsonl', type=Path)
    parser.add_argument('--bag', type=Path, help='Finalized ROS bag; source ROS setup before analysis')
    parser.add_argument('--steernav-root', default='/home/jim/Projects/steernav')
    args = parser.parse_args()
    rows, malformed = [], 0
    for line in args.jsonl.read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: malformed += 1
    if args.bag:
        rows = add_bag_reference(rows, args.bag)
    summary, records = analyze(rows, args.steernav_root)
    scenario = args.jsonl.parent/'scenario.yaml'
    expected = None
    if scenario.exists():
        import yaml
        expected = [k for k in yaml.safe_load(scenario.read_text()) if k.startswith('episode_')]
    summary['episode_aggregation'] = episode_aggregation(rows, records, expected)
    summary['frame_metrics_use'] = 'diagnostic only; paper metrics use episode_aggregation'
    summary['reference_stream'] = 'independent_rosbag' if args.bag else 'node_callbacks' 
    summary['malformed_lines'] = malformed
    output = args.jsonl.parent/'velocity_comparison.json'
    output.write_text(json.dumps(summary, indent=2)+'\n')
    (args.jsonl.parent/'velocity_comparison_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(json.dumps(summary, indent=2))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
