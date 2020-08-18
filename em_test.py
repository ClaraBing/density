import os
from glob import glob
import numpy as np
import torch
from scipy.stats import ortho_group
import argparse

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--n-steps', type=int, default=50)
parser.add_argument('--data', type=str, default='GM', choices=[
       # connected
       'normal', 'scaledNormal', 'rotatedNormal', 'ring',
       # disconnected
       'GM', 'GM_scale1', 'GM_scale2', 'GMn2', 'concentric'])
parser.add_argument('--save-dir', type=str)
args = parser.parse_args()

from em_utils_np import *

SAVE_ROOT = 'runs_gaussianization'

def test(X, save_dir):
  x = X
  fimg = os.path.join(save_dir, 'figs/test_hist2d_{}_init.png'.format(data_token))
  plot_hist(x, fimg)

  n_steps = args.n_steps
  D = X.shape[1]
  K = args.K
  for i in range(n_steps):
    # load parameters
    A = np.load(os.path.join(save_dir, 'A_i{}.npy'.format(i)))
    pi = np.load(os.path.join(save_dir, 'pi_i{}.npy'.format(i)))
    mu = np.load(os.path.join(save_dir, 'mu_i{}.npy'.format(i)))
    sigma_sqr = np.load(os.path.join(save_dir, 'sigma_sqr_i{}.npy'.format(i)))

    print('iteration', i)
    print('mu: mean={:.3e}/ std={:.3e}'.format(mu.mean(), mu.std()))
    print('sigma_sqr: min={:.3e} / mean={:.3e}/ std={:.3e}'.format(sigma_sqr.min(), sigma_sqr.mean(), sigma_sqr.std()))
    Y = X.dot(A.T)

    fimg = 'figs/test_hist2d_{}_iter{}_Y.png'.format(data_token, i)
    fimg = os.path.join(args.save_dir, fimg)
    plot_hist(Y, fimg)
    print('NLL (Y):', eval_NLL(Y))
    X = gaussianize_1d(Y, pi, mu, sigma_sqr)
    print('NLL:', eval_NLL(X))
    print()
    x = X
    fimg = 'figs/test_hist2d_{}_iter{}.png'.format(data_token, i)
    fimg = os.path.join(args.save_dir, fimg)
    plot_hist(x, fimg)

if __name__ == '__main__':
  args.save_dir = os.path.join(SAVE_ROOT, args.save_dir)

  data_dir = './datasets/EM'
  data_token = args.data
  mu_low, mu_up = -2, 2
  if data_token == 'GM':
    fdata = 'GM_2d_scale4_test.npy'
    mu_low, mu_up = -4, 4
  if data_token == 'GM_scale1':
    fdata = 'GM_2d_scale1.npy'
    mu_low, mu_up = -4, 4
  if data_token == 'GM_scale2':
    fdata = 'GM_2d_scale2.npy'
    mu_low, mu_up = -4, 4
  if data_token == 'GMn2':
    fdata = 'GM_2d_2centers.npy'
    mu_low, mu_up = -4, 4
  elif data_token == 'normal':
    fdata = 'normal.npy'
  elif data_token == 'rotatedNormal':
    fdata = 'rotatedNormal.npy'
  elif data_token == 'scaledNormal':
    fdata = 'scaledNormal.npy'
  elif data_token == 'ring':
    fdata = 'ring.npy'
    mu_low, mu_up = -1, 1
  elif data_token == 'concentric':
    # 2 circles w/ radius 0.5 and 2. Each with 10k points.
    fdata = 'concentric.npy'
  
  X = np.load(os.path.join(data_dir, fdata))
  test(X, args.save_dir)
  
