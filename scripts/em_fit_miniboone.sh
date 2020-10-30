#!/bin/bash

gpu_id=0

# mode='GA'
# mode='torchGA'
# mode='torchAll'
mode='None'
# mode='CF'
gamma=0.003
gamma_min=3e-3
# gamma=-1
# gamma_min=-1

TIME=0
check_obj=0

data='miniboone'

save_token='_debug_newICA'

for iter in 1
do
if [ $iter -eq 1 ]; then
  K=20
  n_em=30
  n_gd=30
  n_pts=27000
fi
if [ $iter -eq 2 ]; then
  K=20
  n_em=40
  n_gd=20
  n_pts=27000
fi
if [ $iter -eq 3 ]; then
  K=40
  n_em=30
  n_gd=30
  n_pts=14000
fi
if [ $iter -eq 4 ]; then
  K=20
  n_em=30
  n_gd=20
  n_pts=14000
fi

if [ $mode = 'ICA' ]; then
  n_pts=0
fi

CUDA_VISIBLE_DEVICES=$gpu_id python em_fit.py \
  --data=$data \
  --mode=$mode \
  --lib='torch' \
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
