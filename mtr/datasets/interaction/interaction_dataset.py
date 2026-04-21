# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved

from mtr.datasets.waymo.waymo_dataset import WaymoDataset
from .interaction_eval import interaction_evaluation


class InteractionDataset(WaymoDataset):
    def evaluation(self, pred_dicts, output_path=None, eval_method='interaction', **kwargs):
        if eval_method == 'waymo':
            return super().evaluation(pred_dicts=pred_dicts, output_path=output_path, eval_method=eval_method, **kwargs)

        metric_results, result_format_str = interaction_evaluation(
            pred_dicts=pred_dicts,
            num_future_frames=self.dataset_cfg.get('NUM_FUTURE_FRAMES', 80),
            miss_threshold=self.dataset_cfg.get('MISS_THRESHOLD', 2.0),
            valid_type_list=tuple(self.dataset_cfg.get('OBJECT_TYPE', ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']))
        )

        metric_result_str = '\n'
        for key in metric_results:
            if isinstance(metric_results[key], float):
                metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
            else:
                metric_result_str += '%s: %s \n' % (key, metric_results[key])
        metric_result_str += '\n'
        metric_result_str += result_format_str
        return metric_result_str, metric_results
