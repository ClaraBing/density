#!/bin/bash

gpu_id=0

TIME=0

# data='gas'
# n_pts=10000
# data='miniboone'
# data='power'
# n_pts=800000
# data='hepmass'
# n_pts=100000

data='GM'
n_pts=0

n_steps=200
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
for var_iter in 3000
do
for var_wd in 1e-6
do
for var_lr in 1e-3 6e-4 2e-3 2e-4
do
for det_lambda in 0.1
do
for det_every in 200
do
for var_num_hidden_nodes in 160
do
for var_num_layers in 1
do
for var_LB in 'E2'
do
for pos_type in 'square'
do
for var_A_mode in 'givens'
do
save_token='_'$mode
if [ $g1d_first -eq 1 ]; then
  save_token=$save_token'_RBIG'
else
  save_token=$save_token'_PP'
fi
# save_token=$save_token'_myG1D_logDet'$log_det_version'_scipy'
# save_token=$save_token'_myG1D_logDet'$log_det_version'_ndtri'
save_token=$save_token'_varIter'$var_iter'_varLR'$var_lr'_varWD'$var_wd'_detLambda'$det_lambda'_detEvery'$det_every
save_token=$save_token'_varA'$var_A_mode'_posType'$pos_type'_nNodes'$var_num_hidden_nodes'_nLayers'$var_num_layers
save_token=$save_token'_'$var_LB
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
  --det-lambda=$det_lambda \
  --det-every=$det_every \
  --var-iter=$var_iter \
  --var-LB=$var_LB \
  --var-lr=$var_lr \
  --var-wd=$var_wd \
  --var-num-hidden-nodes=$var_num_hidden_nodes \
  --var-num-layers=$var_num_layers \
  --var-A-mode=$var_A_mode \
  --var-pos-type=$pos_type \
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
done
done
done
done
done
done
done
done
