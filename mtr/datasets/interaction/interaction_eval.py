# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved

import numpy as np


def _flatten_pred_dicts(pred_dicts):
    flat_pred_dicts = []
    for item in pred_dicts:
        if isinstance(item, list):
            flat_pred_dicts.extend(item)
        else:
            flat_pred_dicts.append(item)
    return flat_pred_dicts


def _safe_mean(total, count):
    return total / max(count, 1)


def _canonical_object_type(object_type):
    if object_type == 'TYPE_PEDESTRAIN':
        return 'TYPE_PEDESTRIAN'
    return object_type


def interaction_evaluation(
    pred_dicts,
    num_future_frames=80,
    miss_threshold=2.0,
    valid_type_list=('TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST')
):
    pred_dicts = _flatten_pred_dicts(pred_dicts)
    per_type_stats = {
        cur_type: {'minADE': 0.0, 'minFDE': 0.0, 'MissRate': 0.0, 'count': 0}
        for cur_type in valid_type_list
    }
    all_stats = {'minADE': 0.0, 'minFDE': 0.0, 'MissRate': 0.0, 'count': 0}

    for cur_pred in pred_dicts:
        pred_trajs = np.asarray(cur_pred['pred_trajs'])
        gt_trajs = np.asarray(cur_pred['gt_trajs'])

        if pred_trajs.ndim != 3 or pred_trajs.shape[-1] != 2:
            continue
        if gt_trajs.ndim != 2 or gt_trajs.shape[-1] < 3:
            continue

        eval_horizon = min(num_future_frames, pred_trajs.shape[1], gt_trajs.shape[0])
        if eval_horizon <= 0:
            continue

        gt_future = gt_trajs[-eval_horizon:, 0:2]
        gt_valid_mask = gt_trajs[-eval_horizon:, -1] > 0
        if gt_valid_mask.sum() <= 0:
            continue

        pred_future = pred_trajs[:, :eval_horizon, :]
        l2_dist = np.linalg.norm(pred_future - gt_future[None, :, :], axis=-1)  # (num_modes, T)

        valid_mask_float = gt_valid_mask.astype(np.float32)[None, :]
        ade_per_mode = (l2_dist * valid_mask_float).sum(axis=-1) / np.clip(valid_mask_float.sum(axis=-1), a_min=1.0, a_max=None)
        last_valid_idx = np.where(gt_valid_mask)[0][-1]
        fde_per_mode = l2_dist[:, last_valid_idx]

        min_ade = float(ade_per_mode.min())
        min_fde = float(fde_per_mode.min())
        miss_rate = float(min_fde > miss_threshold)

        all_stats['minADE'] += min_ade
        all_stats['minFDE'] += min_fde
        all_stats['MissRate'] += miss_rate
        all_stats['count'] += 1

        object_type = _canonical_object_type(cur_pred.get('object_type', 'TYPE_OTHER'))
        if object_type in per_type_stats:
            per_type_stats[object_type]['minADE'] += min_ade
            per_type_stats[object_type]['minFDE'] += min_fde
            per_type_stats[object_type]['MissRate'] += miss_rate
            per_type_stats[object_type]['count'] += 1

    metric_results = {
        'minADE': _safe_mean(all_stats['minADE'], all_stats['count']),
        'minFDE': _safe_mean(all_stats['minFDE'], all_stats['count']),
        'MissRate': _safe_mean(all_stats['MissRate'], all_stats['count']),
        'EvalCount': int(all_stats['count']),
    }

    for cur_type in valid_type_list:
        type_stats = per_type_stats[cur_type]
        metric_results[f'minADE - {cur_type}'] = _safe_mean(type_stats['minADE'], type_stats['count'])
        metric_results[f'minFDE - {cur_type}'] = _safe_mean(type_stats['minFDE'], type_stats['count'])
        metric_results[f'MissRate - {cur_type}'] = _safe_mean(type_stats['MissRate'], type_stats['count'])
        metric_results[f'Count - {cur_type}'] = int(type_stats['count'])

    result_format_str = '\n'.join([
        'INTERACTION Evaluation Summary',
        f'minADE={metric_results["minADE"]:.4f}, minFDE={metric_results["minFDE"]:.4f}, MissRate={metric_results["MissRate"]:.4f}, Count={metric_results["EvalCount"]}',
    ])
    return metric_results, result_format_str
