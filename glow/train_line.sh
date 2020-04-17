#!/bin/bash

project='density'
model='glow'
layer_type='fc'
in_channels=2
mid_channels=64
num_levels=1
dataset=$in_channels'Dline'
fdata='./data/synthetic/d'$in_channels'_std_train.npy'

optim='Adam'
lr=1e-4
wd=1e-5
bt=128
use_val=1

wb_name="glow_G1D_lr$lr_wd$wd_bt$bt"

# WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=2 python train.py \
  --mode='train' \
  --dataset=$dataset \
  --fdata=$fdata \
  --layer-type=$layer_type \
  --in-channels=$in_channels \
  --mid-channels=$mid_channels \
  --num-levels=$num_levels \
  --optim=$optim \
  --lr=$lr \
  --wd=$wd \
  --batch-size=$bt \
  --use-val=$use_val \
  --project=$project \
  --model=$model \
  --wb-name=$wb_name

