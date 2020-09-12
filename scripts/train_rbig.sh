#!/bin/bash

# dataset='GaussianLine'
# dataset='GaussianMixture'
# dataset='uniform'
# dataset='GAS8'
# dataset='miniboone'

dataset='GM'
n_layer=10

save_suffix='_data200k'
save_suffix='_tmp'

CUDA_VISIBLE_DEVICES=1 python -W ignore rbig.py \
  --model='rbig' \
  --dataset=$dataset \
  --use-val=1 \
  --bt=15000 \
  --rotation-type='PCA' \
  --n-layer=$n_layer \
  --save-suffix=$save_suffix

#  --dlen=100000 \

