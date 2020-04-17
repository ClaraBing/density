#!/bin/bash

project='flow'
dataset='cifar10'
layer_type='conv'
in_channels=12
mid_channels=512

optim='Adam'
lr=0.001
wd=1e-5
bt=32
use_val=1

wb_name="glow_cifar_lr$lr_wd$wd_bt$bt"

# WANDB_MODE=dryrun
CUDA_VISIBLE_DEVICES=2 python train.py \
  --mode='train' \
  --dataset=$dataset \
  --fdata=$fdata \
  --layer-type=$layer_type \
  --in-channels=$in_channels \
  --optim=$optim \
  --lr=$lr \
  --wd=$wd \
  --batch-size=$bt \
  --use-val=$use_val \
  --project=$project \
  --wb-name=$wb_name

