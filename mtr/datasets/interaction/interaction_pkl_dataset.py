# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Adapted for direct DIGIR INTERACTION PKL input.

from pathlib import Path
import pickle
import numpy as np

from mtr.config import cfg
from .interaction_dataset import InteractionDataset


class InteractionPKLDataset(InteractionDataset):
    """
    Directly reads DIGIR-style pkl:
        {
            'train': [sample_dict, ...],
            'val': [sample_dict, ...],
            ...
        }
    each sample_dict includes:
        trajectory: (N, hist_len, 4) -> [x, y, vx, vy]
        future_trajectory: (N, future_len, 2) -> [x, y]
    """

    DEFAULT_OBJECT_TYPE_MAP = {
        0: 'TYPE_VEHICLE',
        1: 'TYPE_PEDESTRIAN',
        2: 'TYPE_CYCLIST',
    }

    def __init__(self, dataset_cfg, training=True, logger=None):
        from mtr.datasets.dataset import DatasetTemplate

        DatasetTemplate.__init__(self, dataset_cfg=dataset_cfg, training=training, logger=logger)
        self.time_delta = float(self.dataset_cfg.get('TIME_DELTA', 0.1))
        self.num_future_frames = int(self.dataset_cfg.get('NUM_FUTURE_FRAMES', 30))
        self.num_history_frames = int(self.dataset_cfg.get('NUM_HISTORY_FRAMES', 10))
        self.valid_object_types = set(self.dataset_cfg.get('OBJECT_TYPE', ['TYPE_VEHICLE']))

        pkl_file = self.dataset_cfg.get('PKL_DATA_FILE', None)
        assert pkl_file is not None, 'DATA_CONFIG.PKL_DATA_FILE must be set for InteractionPKLDataset'
        pkl_path = Path(pkl_file)
        if not pkl_path.is_absolute():
            pkl_path = cfg.ROOT_DIR / pkl_file
        self.pkl_path = pkl_path

        split_key_map = self.dataset_cfg.get('SPLIT_KEY', {'train': 'train', 'test': 'val'})
        split_key = split_key_map[self.mode] if isinstance(split_key_map, dict) else ('train' if self.training else 'val')

        with open(self.pkl_path, 'rb') as f:
            data_dict = pickle.load(f)

        assert split_key in data_dict, f'Split key "{split_key}" not found in {self.pkl_path}, available: {list(data_dict.keys())}'
        self.samples = data_dict[split_key]
        self.src_config = data_dict.get('config', {})

        sample_interval = int(self.dataset_cfg.SAMPLE_INTERVAL[self.mode])
        self.samples = self.samples[::max(sample_interval, 1)]

        self.object_type_map = self.DEFAULT_OBJECT_TYPE_MAP.copy()
        cfg_type_map = self.dataset_cfg.get('OBJECT_TYPE_MAP', {})
        if isinstance(cfg_type_map, dict):
            for key, val in cfg_type_map.items():
                try:
                    self.object_type_map[int(key)] = str(val)
                except Exception:
                    continue

        if self.logger is not None:
            self.logger.info(f'Load direct PKL from: {self.pkl_path}')
            self.logger.info(f'Use split: {split_key}, total samples after interval={sample_interval}: {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.create_scene_level_data(index)

    def _map_object_type(self, raw_type):
        try:
            raw_type = int(raw_type)
        except Exception:
            pass

        if raw_type in self.object_type_map:
            return self.object_type_map[raw_type]

        if isinstance(raw_type, str):
            v = raw_type.strip().upper()
            if v in ['TYPE_VEHICLE', 'VEHICLE', 'CAR']:
                return 'TYPE_VEHICLE'
            if v in ['TYPE_PEDESTRIAN', 'PEDESTRIAN', 'PED']:
                return 'TYPE_PEDESTRIAN'
            if v in ['TYPE_CYCLIST', 'CYCLIST', 'BIKE', 'BICYCLE']:
                return 'TYPE_CYCLIST'
        return 'TYPE_OTHER'

    @staticmethod
    def _default_box_size(object_type):
        if object_type == 'TYPE_VEHICLE':
            return 4.5, 1.9, 1.6
        if object_type == 'TYPE_PEDESTRIAN':
            return 0.8, 0.8, 1.7
        if object_type == 'TYPE_CYCLIST':
            return 1.8, 0.7, 1.6
        return 2.0, 1.0, 1.6

    def _build_trajs_and_meta(self, sample):
        traj_hist = np.asarray(sample['trajectory'], dtype=np.float32)  # (N, H, 4): x,y,vx,vy
        fut_xy = np.asarray(sample['future_trajectory'], dtype=np.float32)  # (N, F, 2): x,y

        num_objects = int(sample.get('num_vehicles', traj_hist.shape[0]))
        traj_hist = traj_hist[:num_objects]
        fut_xy = fut_xy[:num_objects]
        assert traj_hist.ndim == 3 and traj_hist.shape[-1] >= 2
        assert fut_xy.ndim == 3 and fut_xy.shape[-1] == 2

        hist_len = min(traj_hist.shape[1], self.num_history_frames)
        fut_len = min(fut_xy.shape[1], self.num_future_frames)
        traj_hist = traj_hist[:, :hist_len]
        fut_xy = fut_xy[:, :fut_len]

        hist_xy = traj_hist[:, :, 0:2]
        hist_vel_fallback = np.zeros((num_objects, hist_len, 2), dtype=np.float32)
        if hist_len > 1:
            hist_vel_fallback[:, 1:] = (hist_xy[:, 1:] - hist_xy[:, :-1]) / max(self.time_delta, 1e-3)
            hist_vel_fallback[:, 0] = hist_vel_fallback[:, 1]
        if traj_hist.shape[-1] >= 4:
            hist_vel = traj_hist[:, :, 2:4].copy()
            finite_mask = np.isfinite(hist_vel).all(axis=-1)
            hist_vel[~finite_mask] = hist_vel_fallback[~finite_mask]
        else:
            hist_vel = hist_vel_fallback

        hist_vel = np.nan_to_num(hist_vel, nan=0.0, posinf=0.0, neginf=0.0)

        fut_vel = np.zeros((num_objects, fut_len, 2), dtype=np.float32)
        if fut_len > 0:
            fut_vel[:, 0] = (fut_xy[:, 0] - hist_xy[:, -1]) / max(self.time_delta, 1e-3)
            if fut_len > 1:
                fut_vel[:, 1:] = (fut_xy[:, 1:] - fut_xy[:, :-1]) / max(self.time_delta, 1e-3)
        fut_vel = np.nan_to_num(fut_vel, nan=0.0, posinf=0.0, neginf=0.0)

        hist_heading = np.arctan2(hist_vel[:, :, 1], hist_vel[:, :, 0])
        fut_heading = np.arctan2(fut_vel[:, :, 1], fut_vel[:, :, 0])

        total_len = hist_len + fut_len
        trajs = np.zeros((num_objects, total_len, 10), dtype=np.float32)
        # [x, y, z, dx, dy, dz, heading, vel_x, vel_y, valid]
        trajs[:, :hist_len, 0:2] = hist_xy
        trajs[:, :hist_len, 6] = hist_heading
        trajs[:, :hist_len, 7:9] = hist_vel
        trajs[:, :hist_len, 9] = 1.0

        trajs[:, hist_len:hist_len + fut_len, 0:2] = fut_xy
        trajs[:, hist_len:hist_len + fut_len, 6] = fut_heading
        trajs[:, hist_len:hist_len + fut_len, 7:9] = fut_vel
        trajs[:, hist_len:hist_len + fut_len, 9] = 1.0

        vehicle_types = np.asarray(sample.get('vehicle_types', np.zeros((num_objects,), dtype=np.int64))).reshape(-1)[:num_objects]
        obj_types = np.array([self._map_object_type(x) for x in vehicle_types])
        for k in range(num_objects):
            dx, dy, dz = self._default_box_size(obj_types[k])
            trajs[k, :, 3] = dx
            trajs[k, :, 4] = dy
            trajs[k, :, 5] = dz

        location_name = str(sample.get('location_name', 'loc'))
        case_id = sample.get('case_id', -1)
        start_frame = sample.get('start_frame', -1)
        obj_ids = np.array([f'{location_name}_{case_id}_{start_frame}_{k}' for k in range(num_objects)])

        return trajs, obj_types, obj_ids, hist_len, fut_len

    def _build_scene_info(self, sample, index):
        trajs, obj_types, obj_ids, hist_len, fut_len = self._build_trajs_and_meta(sample)
        current_time_index = hist_len - 1
        track_index_to_predict = [
            k for k in range(trajs.shape[0])
            if obj_types[k] in self.valid_object_types
            and trajs[k, current_time_index, 9] > 0
            and trajs[k, current_time_index + 1:, 9].sum() > 0
        ]
        if len(track_index_to_predict) == 0:
            # keep one center object to avoid empty training sample
            track_index_to_predict = [0]

        scenario_id = f'pkl_{sample.get("location_name", "loc")}_{sample.get("case_id", -1)}_{sample.get("start_frame", -1)}_{index}'
        timestamps_seconds = (np.arange(hist_len, dtype=np.float32) * self.time_delta).tolist()

        info = {
            'scenario_id': scenario_id,
            'timestamps_seconds': timestamps_seconds,
            'current_time_index': current_time_index,
            'sdc_track_index': int(track_index_to_predict[0]),
            'objects_of_interest': obj_ids[track_index_to_predict].tolist(),
            'tracks_to_predict': {
                'track_index': track_index_to_predict,
                'difficulty': [0] * len(track_index_to_predict),
                'object_type': obj_types[track_index_to_predict].tolist(),
            },
            'track_infos': {
                'object_id': obj_ids.tolist(),
                'object_type': obj_types.tolist(),
                'trajs': trajs,
            },
            'dynamic_map_infos': {'lane_id': [], 'state': [], 'stop_point': []},
            'map_infos': {'all_polylines': np.zeros((2, 7), dtype=np.float32)},
        }
        return info

    def create_scene_level_data(self, index):
        info = self._build_scene_info(self.samples[index], index=index)

        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)

        track_infos = info['track_infos']
        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_ids = np.array(track_infos['object_id'])
        obj_trajs_full = track_infos['trajs']
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=info['scenario_id']
        )

        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs,
            center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids) = self.create_agent_data_for_center_objects(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types, obj_ids=obj_ids
        )

        ret_dict = {
            'scenario_id': np.array([info['scenario_id']] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,
            'obj_types': obj_types,
            'obj_ids': obj_ids,

            'center_objects_world': center_objects,
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
        }

        if not self.dataset_cfg.get('WITHOUT_HDMAP', False):
            map_polylines_data, map_polylines_mask, map_polylines_center = self.create_map_data_for_center_objects(
                center_objects=center_objects, map_infos=info['map_infos'],
                center_offset=self.dataset_cfg.get('CENTER_OFFSET_OF_MAP', (30.0, 0)),
            )
            ret_dict['map_polylines'] = map_polylines_data
            ret_dict['map_polylines_mask'] = (map_polylines_mask > 0)
            ret_dict['map_polylines_center'] = map_polylines_center

        return ret_dict
