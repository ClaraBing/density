#!/bin/bash

gpu_id=3

mode='GA'
# mode='torchGA'
# mode='torchAll'
mode='ICA'
gamma=0
gamma_min=0
# gamma=0.001
# gamma_min=1e-3
# gamma=-1
# gamma_min=-1

TIME=0
check_obj=0

data='miniboone'

save_token='_ICA'

for iter in 1 2
do
save_token=$save_token'_run'$iter
for K in 40
do
for n_em in 40
do

if [ $mode = 'ICA' ]; then
  n_pts=0
  n_gd=0
fi

CUDA_VISIBLE_DEVICES=$gpu_id python em_fit.py \
  --data=$data \
  --mode=$mode \
  --gamma=$gamma \
  --gamma-min=$gamma_min \
  --K=$K \
  --n-steps=200 \
  --n-em=$n_em \
  --n-gd=$n_gd \
  --n-pts=$n_pts \
  --save-token=$save_token \
  --time=$TIME \
  --check-obj=$check_obj
done
done
done
