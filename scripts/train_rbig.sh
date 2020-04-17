#!/bin/bash

# dataset='GaussianLine'
dataset='GaussianMixture'

CUDA_VISIBLE_DEVICES=0 python -W ignore rbig.py \
  --dataset=$dataset \
  --use-val=1 \
  --d=2 \
  --dlen=50000 \
  --bt=100 \
  --n-layer=5 \
  --rotation-type='random'
