#!/bin/bash

gpu_id=1

TIME=0

n_steps=400
data='gas'

for mode in 'ICA'
do
for iter in 1
do
for K in 400
do
for n_em in 20 40 80 160
do
for density_type in 'GM'
do
for n_anchors in 200
do
for log_det_version in 'v1'
# 'v2'
do
for myG1D in 1
do
save_token='_'$mode'_'$density_type'_myG1D'$myG1D'_logDet'$log_det_version
save_token=$save_token'_run'$iter

if [ $mode = 'ICA' ] || [ $mode = 'PCA' ]; then
  n_pts=0
fi

# WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=$gpu_id python em_rbig.py \
  --data=$data \
  --mode=$mode \
  --density-type=$density_type \
  --n-anchors=$n_anchors \
  --n-steps=$n_steps \
  --K=$K \
  --n-em=$n_em \
  --n-pts=$n_pts \
  --inverse-cdf-by-thresh=$myG1D \
  --log-det-version=$log_det_version \
  --save-token=$save_token \
  --time=$TIME 
done
done
done
done
done
done
done
done
