#!/bin/bash

dataset='GaussianLine'
# dataset='GaussianMixture'
# dataset='uniform'

CUDA_VISIBLE_DEVICES=3 python -W ignore rbig.py \
  --model='rbig' \
  --dataset=$dataset \
  --use-val=1 \
  --d=32 \
  --dlen=50000 \
  --bt=2000 \
  --rotation-type='PCA'
