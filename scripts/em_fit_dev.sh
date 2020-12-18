#!/bin/bash

gpu_id=0

TIME=0

data='gas'

g1d_first=0

for iter in 1
do
for mode in 'ICA'
do
for g1d_first in 0
do
for log_det_version in 'v1' 'v2'
do
save_token='_'$mode
if [ $g1d_first -eq 1 ]; then
  save_token=$save_token'_RBIG'
else
  save_token=$save_token'_PP'
fi
save_token=$save_token'_myG1D_logDet'$log_det_version
for K in 20
do
for n_em in 30
do

if [ $mode = 'ICA' ] || [ $mode = 'PCA' ]; then
  n_pts=100000
  n_gd=0
fi
save_token=$save_token'_run'$iter

# WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=$gpu_id python em_fit.py \
  --data=$data \
  --g1d-first=$g1d_first \
  --log-det-version=$log_det_version \
  --mode=$mode \
  --K=$K \
  --n-steps=50 \
  --n-em=$n_em \
  --n-gd=$n_gd \
  --n-pts=$n_pts \
  --save-token=$save_token \
  --time=$TIME
done
done
done
done
done
done
