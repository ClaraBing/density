#!/bin/bash

rot_type='ICA'

# dataset='GaussianLine'
# dataset='GaussianMixture'
# dataset='uniform'
# dataset='GAS8'
# dataset='miniboone'
# dataset='GaussianMixture'
dataset='MNISTtab'
dlen=50000
bt_test=100

# dataset='GM'

# save_suffix='_data200k_run1'
# save_suffix='_btVal100_randProj'
# save_suffix='_btVal'$bt_test'_checkCov_proj10'
save_suffix='_btVal'$bt_test

for run in 1
do
save_suffix='_btVal'$bt_test'_run'$run
for n_layer in 20
do
CUDA_VISIBLE_DEVICES=1 python -W ignore rbig.py \
  --model='rbig' \
  --dataset=$dataset \
  --use-val=1 \
  --bt=50000 \
  --bt-test=$bt_test \
  --rotation-type=$rot_type \
  --n-layer=$n_layer \
  --save-suffix=$save_suffix \
  --dlen=$dlen
done
done

