# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Adapted for INTERACTION-style CSV trajectories.

import argparse
import csv
import os
import pickle
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _normalize_col_name(name):
    return name.strip().lower()


def _find_col(fieldnames, candidates, required=True):
    lowered = {_normalize_col_name(x): x for x in fieldnames}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    if required:
        raise ValueError(f'Cannot find required column among {candidates}. Available columns: {fieldnames}')
    return None


def _to_float(v, default=np.nan):
    if v is None:
        return default
    if isinstance(v, (float, int)):
        return float(v)
    v = str(v).strip()
    if v == '':
        return default
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v, default=-1):
    f = _to_float(v, default=np.nan)
    if np.isnan(f):
        return default
    return int(round(f))


def _sanitize_for_filename(text):
    text = str(text)
    return re.sub(r'[^0-9A-Za-z_\-\.]+', '_', text)


def _map_object_type(raw_type):
    if raw_type is None:
        return 'TYPE_OTHER'

    raw = str(raw_type).strip().lower()
    if raw == '':
        return 'TYPE_OTHER'

    if raw in ['1', 'vehicle']:
        return 'TYPE_VEHICLE'
    if raw in ['2', 'pedestrian', 'ped', 'person']:
        return 'TYPE_PEDESTRIAN'
    if raw in ['3', 'cyclist', 'bicycle', 'bike', 'motorcycle']:
        return 'TYPE_CYCLIST'

    if 'veh' in raw or 'car' in raw or 'truck' in raw or 'bus' in raw:
        return 'TYPE_VEHICLE'
    if 'ped' in raw or 'person' in raw:
        return 'TYPE_PEDESTRIAN'
    if 'cycl' in raw or 'bike' in raw or 'motor' in raw:
        return 'TYPE_CYCLIST'

    return 'TYPE_OTHER'


def _default_box_size(object_type):
    if object_type == 'TYPE_VEHICLE':
        return 4.5, 1.9, 1.6
    if object_type == 'TYPE_PEDESTRIAN':
        return 0.8, 0.8, 1.7
    if object_type == 'TYPE_CYCLIST':
        return 1.8, 0.7, 1.6
    return 2.0, 1.0, 1.6


def _fill_missing_velocity(x, y, valid_mask, timestamps, vx, vy):
    n = len(x)
    for i in range(n):
        if valid_mask[i] <= 0:
            continue
        if not (np.isnan(vx[i]) or np.isnan(vy[i])):
            continue

        prev_idx = i - 1
        while prev_idx >= 0 and valid_mask[prev_idx] <= 0:
            prev_idx -= 1
        next_idx = i + 1
        while next_idx < n and valid_mask[next_idx] <= 0:
            next_idx += 1

        if prev_idx >= 0 and next_idx < n:
            dt = timestamps[next_idx] - timestamps[prev_idx]
            dx = x[next_idx] - x[prev_idx]
            dy = y[next_idx] - y[prev_idx]
        elif next_idx < n:
            dt = timestamps[next_idx] - timestamps[i]
            dx = x[next_idx] - x[i]
            dy = y[next_idx] - y[i]
        elif prev_idx >= 0:
            dt = timestamps[i] - timestamps[prev_idx]
            dx = x[i] - x[prev_idx]
            dy = y[i] - y[prev_idx]
        else:
            dt = 0.0
            dx, dy = 0.0, 0.0

        if abs(dt) < 1e-6:
            vx[i], vy[i] = 0.0, 0.0
        else:
            vx[i], vy[i] = dx / dt, dy / dt

    vx[np.isnan(vx)] = 0.0
    vy[np.isnan(vy)] = 0.0
    return vx, vy


def _fill_missing_heading(valid_mask, vx, vy, heading):
    n = len(heading)
    for i in range(n):
        if valid_mask[i] <= 0:
            continue
        if np.isnan(heading[i]):
            speed = np.hypot(vx[i], vy[i])
            if speed > 1e-4:
                heading[i] = np.arctan2(vy[i], vx[i])
            else:
                prev_idx = i - 1
                while prev_idx >= 0 and (valid_mask[prev_idx] <= 0 or np.isnan(heading[prev_idx])):
                    prev_idx -= 1
                heading[i] = heading[prev_idx] if prev_idx >= 0 else 0.0

    heading[np.isnan(heading)] = 0.0
    return heading


def _collect_csv_files(split_root):
    files = sorted(split_root.rglob('*.csv'))
    return [x for x in files if x.is_file()]


def _resolve_split(raw_root, split_candidates):
    for split_name in split_candidates:
        split_path = raw_root / split_name
        if split_path.is_dir():
            files = _collect_csv_files(split_path)
            if len(files) > 0:
                return split_path, files
    raise FileNotFoundError(f'Cannot locate split folder from {split_candidates} under {raw_root}')


def _build_timestamps_seconds(selected_times, time_col_name, time_delta):
    t = np.asarray(selected_times, dtype=np.float64)
    t = t - t[0]
    time_col_name = _normalize_col_name(time_col_name)
    if 'timestamp' in time_col_name and 'ms' in time_col_name:
        t = t / 1000.0
    elif 'timestamp' in time_col_name and ('sec' in time_col_name or 's' == time_col_name):
        t = t
    else:
        t = t * time_delta
    return t.astype(np.float32)


def _rotate_to_local(dx, dy, heading):
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    local_x = dx * cos_h + dy * sin_h
    local_y = -dx * sin_h + dy * cos_h
    return np.array([local_x, local_y], dtype=np.float32)


def _kmeans(points, num_centers=64, max_iters=100, seed=42):
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return np.zeros((num_centers, 2), dtype=np.float32)

    rng = np.random.RandomState(seed)
    replace = points.shape[0] < num_centers
    init_idx = rng.choice(points.shape[0], size=num_centers, replace=replace)
    centers = points[init_idx].copy()

    for _ in range(max_iters):
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        assign = d2.argmin(axis=1)
        new_centers = centers.copy()
        for k in range(num_centers):
            mask = assign == k
            if mask.any():
                new_centers[k] = points[mask].mean(axis=0)
            else:
                new_centers[k] = points[rng.randint(0, points.shape[0])]

        shift = np.linalg.norm(new_centers - centers, axis=-1).mean()
        centers = new_centers
        if shift < 1e-4:
            break

    return centers.astype(np.float32)


def _load_case_tracks(csv_file):
    with open(csv_file, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f'CSV has no header: {csv_file}')

        fieldnames = reader.fieldnames
        col_case = _find_col(fieldnames, ['case_id', 'scenario_id', 'scene_id'], required=False)
        col_track = _find_col(fieldnames, ['track_id', 'agent_id', 'object_id'], required=True)
        col_time = _find_col(fieldnames, ['timestamp_ms', 'timestamp', 'timestamp_sec', 'frame_id', 'frame'], required=True)
        col_agent_type = _find_col(fieldnames, ['agent_type', 'object_type', 'type'], required=False)

        col_x = _find_col(fieldnames, ['x', 'position_x', 'x_center'], required=True)
        col_y = _find_col(fieldnames, ['y', 'position_y', 'y_center'], required=True)

        col_vx = _find_col(fieldnames, ['vx', 'v_x', 'velocity_x'], required=False)
        col_vy = _find_col(fieldnames, ['vy', 'v_y', 'velocity_y'], required=False)
        col_heading = _find_col(fieldnames, ['psi_rad', 'heading', 'yaw', 'theta'], required=False)
        col_length = _find_col(fieldnames, ['length', 'agent_length'], required=False)
        col_width = _find_col(fieldnames, ['width', 'agent_width'], required=False)

        case_tracks = {}
        case_times = {}
        for row in reader:
            case_id = row[col_case] if col_case is not None else '0'
            case_id = str(case_id).strip()
            if case_id == '':
                case_id = '0'

            track_id_raw = row[col_track]
            track_id = str(track_id_raw).strip()
            if track_id == '':
                continue

            t = _to_float(row[col_time], default=np.nan)
            if np.isnan(t):
                continue
            t = float(round(t, 6))

            x = _to_float(row[col_x], default=np.nan)
            y = _to_float(row[col_y], default=np.nan)
            if np.isnan(x) or np.isnan(y):
                continue

            object_type = _map_object_type(row[col_agent_type] if col_agent_type is not None else None)
            length = _to_float(row[col_length], default=np.nan) if col_length is not None else np.nan
            width = _to_float(row[col_width], default=np.nan) if col_width is not None else np.nan
            vx = _to_float(row[col_vx], default=np.nan) if col_vx is not None else np.nan
            vy = _to_float(row[col_vy], default=np.nan) if col_vy is not None else np.nan
            heading = _to_float(row[col_heading], default=np.nan) if col_heading is not None else np.nan

            case_tracks.setdefault(case_id, {})
            case_times.setdefault(case_id, set())
            case_times[case_id].add(t)

            if track_id not in case_tracks[case_id]:
                case_tracks[case_id][track_id] = {
                    'object_type': object_type,
                    'length': length,
                    'width': width,
                    'states': {}
                }
            else:
                if case_tracks[case_id][track_id]['object_type'] == 'TYPE_OTHER' and object_type != 'TYPE_OTHER':
                    case_tracks[case_id][track_id]['object_type'] = object_type
                if not np.isnan(length):
                    case_tracks[case_id][track_id]['length'] = length
                if not np.isnan(width):
                    case_tracks[case_id][track_id]['width'] = width

            case_tracks[case_id][track_id]['states'][t] = {
                'x': x,
                'y': y,
                'vx': vx,
                'vy': vy,
                'heading': heading,
                'length': length,
                'width': width
            }

    return case_tracks, case_times, col_time


def _build_windowed_scenarios_for_case(
    case_id,
    tracks_dict,
    sorted_times,
    col_time_name,
    split_tag,
    source_stem,
    output_dir,
    past_steps,
    future_steps,
    time_delta,
    window_stride,
    valid_object_types
):
    infos = []
    goal_points = {'TYPE_VEHICLE': [], 'TYPE_PEDESTRIAN': [], 'TYPE_CYCLIST': []}
    total_steps = past_steps + future_steps
    current_idx = past_steps - 1

    if len(sorted_times) < total_steps:
        return infos, goal_points

    max_start = len(sorted_times) - total_steps
    start_indices = [0]
    if max_start > 0:
        start_indices = list(range(0, max_start + 1, max(window_stride, 1)))
        if start_indices[-1] != max_start:
            start_indices.append(max_start)

    for window_idx, start_idx in enumerate(start_indices):
        selected_times = sorted_times[start_idx:start_idx + total_steps]
        timestamps_seconds = _build_timestamps_seconds(selected_times, col_time_name, time_delta)
        time2idx = {t: i for i, t in enumerate(selected_times)}

        track_object_ids = []
        track_object_types = []
        track_trajs = []

        for track_id, track_data in tracks_dict.items():
            object_type = track_data['object_type']
            default_l, default_w, default_h = _default_box_size(object_type)
            length = track_data['length']
            width = track_data['width']
            length = default_l if np.isnan(length) else float(length)
            width = default_w if np.isnan(width) else float(width)
            height = default_h

            x = np.zeros(total_steps, dtype=np.float32)
            y = np.zeros(total_steps, dtype=np.float32)
            vx = np.full(total_steps, np.nan, dtype=np.float32)
            vy = np.full(total_steps, np.nan, dtype=np.float32)
            heading = np.full(total_steps, np.nan, dtype=np.float32)
            valid = np.zeros(total_steps, dtype=np.float32)

            for t, state in track_data['states'].items():
                if t not in time2idx:
                    continue
                idx = time2idx[t]
                x[idx] = state['x']
                y[idx] = state['y']
                vx[idx] = state['vx']
                vy[idx] = state['vy']
                heading[idx] = state['heading']
                valid[idx] = 1.0

            if valid.sum() <= 0:
                continue

            vx, vy = _fill_missing_velocity(x, y, valid, timestamps_seconds, vx, vy)
            heading = _fill_missing_heading(valid, vx, vy, heading)

            traj = np.zeros((total_steps, 10), dtype=np.float32)
            traj[:, 0] = x
            traj[:, 1] = y
            traj[:, 2] = 0.0
            traj[:, 3] = length
            traj[:, 4] = width
            traj[:, 5] = height
            traj[:, 6] = heading
            traj[:, 7] = vx
            traj[:, 8] = vy
            traj[:, 9] = valid
            traj[valid <= 0] = 0.0

            track_object_ids.append(track_id)
            track_object_types.append(object_type)
            track_trajs.append(traj)

        if len(track_trajs) == 0:
            continue

        track_trajs = np.stack(track_trajs, axis=0)
        track_object_types = np.asarray(track_object_types)
        track_object_ids = np.asarray(track_object_ids)

        track_index_to_predict = []
        for idx in range(track_trajs.shape[0]):
            if track_object_types[idx] not in valid_object_types:
                continue
            if track_trajs[idx, current_idx, 9] <= 0:
                continue
            if track_trajs[idx, current_idx + 1:, 9].sum() <= 0:
                continue
            track_index_to_predict.append(idx)

        if len(track_index_to_predict) == 0:
            continue

        track_index_to_predict = np.asarray(track_index_to_predict, dtype=np.int64)
        track_types_to_predict = track_object_types[track_index_to_predict].tolist()
        difficulty = [0 for _ in range(len(track_index_to_predict))]

        vehicle_in_predict = np.where(track_object_types[track_index_to_predict] == 'TYPE_VEHICLE')[0]
        if len(vehicle_in_predict) > 0:
            sdc_track_index = int(track_index_to_predict[vehicle_in_predict[0]])
        else:
            sdc_track_index = int(track_index_to_predict[0])

        scenario_id_raw = f'{split_tag}_{source_stem}_{case_id}_w{window_idx}'
        scenario_id = _sanitize_for_filename(scenario_id_raw)

        save_info = {
            'scenario_id': scenario_id,
            'timestamps_seconds': timestamps_seconds.tolist(),
            'current_time_index': int(current_idx),
            'sdc_track_index': int(sdc_track_index),
            'objects_of_interest': track_object_ids[track_index_to_predict].tolist(),
            'tracks_to_predict': {
                'track_index': track_index_to_predict.tolist(),
                'difficulty': difficulty,
                'object_type': track_types_to_predict
            },
            'track_infos': {
                'object_id': track_object_ids.tolist(),
                'object_type': track_object_types.tolist(),
                'trajs': track_trajs.astype(np.float32)
            },
            'dynamic_map_infos': {
                'lane_id': [],
                'state': [],
                'stop_point': []
            },
            'map_infos': {
                'all_polylines': np.zeros((2, 7), dtype=np.float32)
            }
        }

        with open(output_dir / f'sample_{scenario_id}.pkl', 'wb') as f:
            pickle.dump(save_info, f)

        infos.append({
            'scenario_id': scenario_id,
            'current_time_index': int(current_idx),
            'tracks_to_predict': {
                'track_index': track_index_to_predict.tolist(),
                'difficulty': difficulty,
                'object_type': track_types_to_predict
            }
        })

        for obj_idx in track_index_to_predict.tolist():
            obj_type = track_object_types[obj_idx]
            if obj_type not in goal_points:
                continue
            cur_state = track_trajs[obj_idx, current_idx]
            future_valid = np.where(track_trajs[obj_idx, current_idx + 1:, 9] > 0)[0]
            if len(future_valid) == 0:
                continue
            last_future_idx = current_idx + 1 + future_valid[-1]
            end_state = track_trajs[obj_idx, last_future_idx]
            local_goal = _rotate_to_local(
                dx=end_state[0] - cur_state[0],
                dy=end_state[1] - cur_state[1],
                heading=cur_state[6]
            )
            goal_points[obj_type].append(local_goal)

    return infos, goal_points


def _preprocess_split(
    split_name,
    csv_files,
    output_dir,
    past_steps,
    future_steps,
    time_delta,
    window_stride,
    valid_object_types
):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_infos = []
    all_goal_points = {'TYPE_VEHICLE': [], 'TYPE_PEDESTRIAN': [], 'TYPE_CYCLIST': []}

    for csv_file in tqdm(csv_files, desc=f'Processing {split_name}', dynamic_ncols=True):
        case_tracks, case_times, col_time_name = _load_case_tracks(csv_file)
        for case_id in sorted(case_tracks.keys(), key=lambda x: str(x)):
            sorted_times = sorted(case_times[case_id])
            infos, goal_points = _build_windowed_scenarios_for_case(
                case_id=case_id,
                tracks_dict=case_tracks[case_id],
                sorted_times=sorted_times,
                col_time_name=col_time_name,
                split_tag=split_name,
                source_stem=csv_file.stem,
                output_dir=output_dir,
                past_steps=past_steps,
                future_steps=future_steps,
                time_delta=time_delta,
                window_stride=window_stride,
                valid_object_types=set(valid_object_types)
            )
            all_infos.extend(infos)
            for k in all_goal_points.keys():
                all_goal_points[k].extend(goal_points[k])

    return all_infos, all_goal_points


def _merge_goal_points(dst, src):
    for key in dst.keys():
        dst[key].extend(src.get(key, []))
    return dst


def create_infos_from_interaction_csv(
    raw_data_root,
    output_root,
    train_split_candidates=('train', 'training'),
    val_split_candidates=('val', 'validation'),
    past_steps=10,
    future_steps=80,
    time_delta=0.1,
    window_stride=10,
    valid_object_types=('TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST'),
    num_center_clusters=64,
    seed=42,
    skip_intention_points=False
):
    raw_root = Path(raw_data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_split_root, train_files = _resolve_split(raw_root, train_split_candidates)
    val_split_root, val_files = _resolve_split(raw_root, val_split_candidates)
    print(f'Use train split: {train_split_root} ({len(train_files)} files)')
    print(f'Use val split: {val_split_root} ({len(val_files)} files)')

    train_infos, train_goal_points = _preprocess_split(
        split_name='train',
        csv_files=train_files,
        output_dir=output_root / 'processed_scenarios_training',
        past_steps=past_steps,
        future_steps=future_steps,
        time_delta=time_delta,
        window_stride=window_stride,
        valid_object_types=valid_object_types
    )
    val_infos, _ = _preprocess_split(
        split_name='val',
        csv_files=val_files,
        output_dir=output_root / 'processed_scenarios_validation',
        past_steps=past_steps,
        future_steps=future_steps,
        time_delta=time_delta,
        window_stride=window_stride,
        valid_object_types=valid_object_types
    )

    train_info_file = output_root / 'processed_scenarios_training_infos.pkl'
    val_info_file = output_root / 'processed_scenarios_val_infos.pkl'
    with open(train_info_file, 'wb') as f:
        pickle.dump(train_infos, f)
    with open(val_info_file, 'wb') as f:
        pickle.dump(val_infos, f)

    print(f'Train scenarios: {len(train_infos)} -> {train_info_file}')
    print(f'Val scenarios: {len(val_infos)} -> {val_info_file}')

    if not skip_intention_points:
        rng_seed = int(seed)
        intention_points = {}
        merged_goals = {'TYPE_VEHICLE': [], 'TYPE_PEDESTRIAN': [], 'TYPE_CYCLIST': []}
        _merge_goal_points(merged_goals, train_goal_points)

        for obj_type in ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']:
            points = np.asarray(merged_goals[obj_type], dtype=np.float32)
            intention_points[obj_type] = _kmeans(
                points=points,
                num_centers=num_center_clusters,
                max_iters=100,
                seed=rng_seed
            )
            print(f'Intention points for {obj_type}: {intention_points[obj_type].shape[0]} centers (from {points.shape[0]} samples)')

        intention_file = output_root / 'cluster_64_center_dict.pkl'
        with open(intention_file, 'wb') as f:
            pickle.dump(intention_points, f)
        print(f'Saved intention points to: {intention_file}')


def parse_config():
    parser = argparse.ArgumentParser(description='INTERACTION data preprocess for MTR')
    parser.add_argument('raw_data_path', type=str, help='Raw INTERACTION root folder')
    parser.add_argument('output_path', type=str, help='Output folder, e.g. data/interaction')

    parser.add_argument('--train_split_candidates', nargs='+', default=['train', 'training'], help='Train split directory candidates')
    parser.add_argument('--val_split_candidates', nargs='+', default=['val', 'validation'], help='Validation split directory candidates')
    parser.add_argument('--past_steps', type=int, default=10, help='History steps (10Hz). 10 means 1s.')
    parser.add_argument('--future_steps', type=int, default=80, help='Future steps (10Hz). 80 means 8s.')
    parser.add_argument('--time_delta', type=float, default=0.1, help='Frame time delta in seconds when using frame id as timestamp.')
    parser.add_argument('--window_stride', type=int, default=10, help='Sliding window stride in frames for long cases.')
    parser.add_argument('--num_center_clusters', type=int, default=64, help='Number of intention centers per object type.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for kmeans initialization.')
    parser.add_argument('--skip_intention_points', action='store_true', default=False, help='Skip generating cluster_64_center_dict.pkl')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    create_infos_from_interaction_csv(
        raw_data_root=args.raw_data_path,
        output_root=args.output_path,
        train_split_candidates=tuple(args.train_split_candidates),
        val_split_candidates=tuple(args.val_split_candidates),
        past_steps=args.past_steps,
        future_steps=args.future_steps,
        time_delta=args.time_delta,
        window_stride=args.window_stride,
        num_center_clusters=args.num_center_clusters,
        seed=args.seed,
        skip_intention_points=args.skip_intention_points
    )
