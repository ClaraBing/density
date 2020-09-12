#!/bin/bash

# dataset='GaussianLine'
# dataset='GaussianMixture'
# dataset='uniform'
# dataset='GAS8'
# dataset='miniboone'
dataset='GaussianMixture'
dlen=50000

dataset='GM'
n_layer=10

# save_suffix='_data200k_run1'
save_suffix='_btVal100_checkKL'

for n_layer in 5 6 8 10 12
do
CUDA_VISIBLE_DEVICES=0 python -W ignore rbig.py \
  --model='rbig' \
  --dataset=$dataset \
  --use-val=1 \
  --bt=50000 \
  --bt-test=100 \
  --rotation-type='ICA' \
  --n-layer=$n_layer \
  --save-suffix=$save_suffix \
  --dlen=$dlen
done

