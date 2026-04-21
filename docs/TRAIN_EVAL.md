## Training 
For example, train with 8 GPUs: 
```
cd tools

bash scripts/dist_train.sh 8 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 80 --epochs 30 --extra_tag my_first_exp
```
Actually, during the training process, the evaluation results will be logged to the log file under `output/waymo/mtr+100_percent_data/my_first_exp/log_train_xxxx.txt`

## Testing
For example, test with 8 GPUs: 
```
cd tools
bash scripts/dist_test.sh 8 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --ckpt ../output/waymo/mtr+100_percent_data/my_first_exp/ckpt/checkpoint_epoch_30.pth --extra_tag my_first_exp --batch_size 80 
```

## Training & Testing on INTERACTION (history=1s, future=3s)

Train:
```
cd tools
python train.py --cfg_file cfgs/interaction/mtr_interaction_h1s_f3s.yaml --extra_tag interaction_h1s_f3s
```

Test:
```
cd tools
python test.py --cfg_file cfgs/interaction/mtr_interaction_h1s_f3s.yaml --ckpt ../output/interaction/mtr_interaction_h1s_f3s/interaction_h1s_f3s/ckpt/best_model.pth --extra_tag interaction_h1s_f3s
```
