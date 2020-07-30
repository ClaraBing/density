#!/bin/bash

# dataset='GaussianLine'
dataset='GaussianMixture'
# dataset='uniform'

CUDA_VISIBLE_DEVICES=0 python -W ignore rbig.py \
  --model='rbig' \
  --dataset=$dataset \
  --use-val=1 \
  --dlen=100000 \
  --bt=10000 \
  --rotation-type='random'
