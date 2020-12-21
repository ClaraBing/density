#!/bin/bash

gpu_id=0

TIME=0

data='gas'
n_pts=400000
# data='miniboone'
# data='power'
# n_pts=800000
# data='hepmass'
# n_pts=100000

n_steps=4000
overwrite=1

# sleep 16000


for iter in 1
do
for mode in 'ICA'
do
for g1d_first in 0
do
for log_det_version in 'v2'
do
for K in 40
# 64 256 512
do
for n_em in 200
do
save_token='_'$mode
if [ $g1d_first -eq 1 ]; then
  save_token=$save_token'_RBIG'
else
  save_token=$save_token'_PP'
fi
# save_token=$save_token'_myG1D_logDet'$log_det_version'_scipy'
save_token=$save_token'_myG1D_logDet'$log_det_version'_ndtri'

# if [ $mode = 'ICA' ] || [ $mode = 'PCA' ]; then
#   n_pts=100000
#   n_gd=0
# fi
save_token=$save_token'_run'$iter

# CUDA_VISIBLE_DEVICES=$gpu_id python -m cProfile -s cumtime em_fit.py \

# WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=$gpu_id python em_fit.py \
  --data=$data \
  --g1d-first=$g1d_first \
  --log-det-version=$log_det_version \
  --mode=$mode \
  --K=$K \
  --n-steps=$n_steps \
  --n-em=$n_em \
  --n-pts=$n_pts \
  --save-token=$save_token \
  --time=$TIME \
  --overwrite=$overwrite
done
done
done
done
done
done
