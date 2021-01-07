#!/bin/bash

gpu_id=0

TIME=0

# data='gas'
# n_pts=400000
# data='miniboone'
# data='power'
# n_pts=800000
# data='hepmass'
# n_pts=100000

data='GM'
n_pts=0

n_steps=100
overwrite=1

# sleep 16000

for iter in 1
do
for mode in 'variational'
do
for g1d_first in 0
do
for log_det_version in 'v2'
do
for K in 40
do
for n_em in 200
do
for var_wd in 1e-4 1e-5 1e-3
do
for var_lr in 0.1 0.3 1 0.03
do
save_token='_'$mode
if [ $g1d_first -eq 1 ]; then
  save_token=$save_token'_RBIG'
else
  save_token=$save_token'_PP'
fi
# save_token=$save_token'_myG1D_logDet'$log_det_version'_scipy'
# save_token=$save_token'_myG1D_logDet'$log_det_version'_ndtri'
save_token=$save_token'_varLR'$var_lr'_varWD'$var_wd
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
  --var-lr=$var_lr \
  --var-wd=$var_wd \
  --save-token=$save_token \
  --time=$TIME \
  --overwrite=$overwrite
done
done
done
done
done
done
done
done