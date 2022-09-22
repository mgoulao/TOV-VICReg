#!/bin/bash

arch="vit_tiny"
games="Alien Assault BankHeist Breakout ChopperCommand Freeway Frostbite Kangaroo MsPacman Pong"
max_size=100000
checkpoints="1 25 50"
outPath="./out"
exp_name="tov_vicreg_$(date +%s)"
data_path="/media/storage/dqn"
patch_size=8
batch_size=512
num_workers=8
out_dim=512
encoder_out=192
temporal_coeff=0.1
cov_coeff=10.0
inv_coeff=10.0
var_coeff=10.0
warmup_epochs=2
epochs=10
save_only_final_arg="--save_only_final"
image_size=84
dqn_frames=3

python tov_vicreg/main.py  \
    --arch $arch \
    --data $data_path \
    --output_dir $outPath \
    --experiment_name $exp_name \
    --patch_size $patch_size \
    --image_size $image_size \
    --dqn_frames $dqn_frames \
    --batch-size $batch_size \
    --workers $num_workers \
    --temporal-coeff $temporal_coeff \
    --cov-coeff $cov_coeff \
    --sim-coeff $inv_coeff \
    --std-coeff $var_coeff \
    --dqn_games $games \
    --dqn_single_dataset_max_size $max_size \
    --dqn_checkpoints $checkpoints \
    --warmup-epochs $warmup_epochs \
    --epochs $epochs \
    --gpu 2 \
    $save_only_final_arg
