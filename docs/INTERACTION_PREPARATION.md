# INTERACTION Dataset Preparation for MTR

This recipe prepares INTERACTION-style CSV data for the new `InteractionDataset` in this repo.

## 1. Organize Raw Data

Use a root folder with train/val splits:
```
<raw_interaction_root>/
  train/        # or training/
  val/          # or validation/
```

Each split can contain nested folders. The script scans all `*.csv` files recursively.

## 2. Preprocess (history=1s, future=8s)

`past_steps=10` and `future_steps=80` correspond to 10Hz data.

```
cd mtr/datasets/interaction
python data_preprocess.py <raw_interaction_root> ../../../data/interaction --past_steps 10 --future_steps 80
```

## 3. Output Structure

The preprocessing script will generate:
```
data/interaction/
  processed_scenarios_training/
  processed_scenarios_validation/
  processed_scenarios_training_infos.pkl
  processed_scenarios_val_infos.pkl
  cluster_64_center_dict.pkl
```

## Notes

- Object types are mapped to:
  - `TYPE_VEHICLE`
  - `TYPE_PEDESTRIAN`
  - `TYPE_CYCLIST`
- Missing velocity/heading fields are auto-filled from trajectories.
- Empty HD map placeholders are created automatically so MTR can run without map parsing.
