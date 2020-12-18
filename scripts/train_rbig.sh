#!/bin/bash

rot_type='ICA'
inverse_cdf_by_thresh=1

# dataset='GaussianLine'
# dataset='GaussianMixture'
# dataset='uniform'
# dataset='GAS8'
dataset='gas'
# dataset='miniboone'
# dataset='GaussianMixture'
# dataset='MNISTtab'
pca_dim=100
# dataset='hepmass'
dlen=0
bt=8000
bt_test=200
n_pts=0

# dataset='GM'

for run in 1
do
for rot_type in 'PCA'
do
for inverse_cdf_by_thresh in 1
do
# 'newData' is for MNISTTab
# save_suffix='_btVal'$bt_test'_newData_run'$run
save_suffix='_bt'$bt'_btVal'$bt_test'_run'$run'_tmpDebug_addLogDet'
if [ $inverse_cdf_by_thresh -eq 1 ]; then
  save_suffix=$save_suffix'_myG1D'
else
  save_suffix=$save_suffix'_origG1D'
fi
for n_layer in 50
do
# WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=1 python -W ignore rbig.py \
  --model='rbig' \
  --dataset=$dataset \
  --n-pts=$n_pts \
  --use-val=1 \
  --bt=$bt \
  --bt-test=$bt_test \
  --inverse-cdf-by-thresh=$inverse_cdf_by_thresh \
  --rotation-type=$rot_type \
  --n-layer=$n_layer \
  --pca-dim=$pca_dim \
  --save-suffix=$save_suffix \
  --dlen=$dlen
done
done
done
done
