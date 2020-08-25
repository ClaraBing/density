import os
import numpy as np
import torch
from scipy.stats import ortho_group
import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

TIME = 0

parser = argparse.ArgumentParser()
parser.add_argument('--lib', type=str, default='torch', choices=['np', 'torch'])
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--n-pts', type=int, default=0)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--gamma-min', type=float, default=0.001)
parser.add_argument('--n-steps', type=int, default=50)
parser.add_argument('--n-em', type=int, default=30)
parser.add_argument('--n-gd', type=int, default=20)
parser.add_argument('--mode', type=str, default='GA', choices=['GA', 'CF', 'ICA'])
parser.add_argument('--data', type=str, default='GM', choices=[
       # connected
       'normal', 'scaledNormal', 'rotatedNormal', 'ring',
       # disconnected
       'GM', 'GM_scale1', 'GM_scale2', 'GMn2', 'concentric'])
parser.add_argument('--save-token', type=str, default='')
parser.add_argument('--save-dir', type=str)
args = parser.parse_args()

if args.lib == 'np':
  from em_utils_np import *
elif args.lib == 'torch':
  from em_utils_torch import *

SAVE_ROOT = 'runs_gaussianization'

def fit(X, mu_low, mu_up, data_token=''):
  x = X
  fimg = 'figs/hist2d_{}_init.png'.format(data_token)
  fimg = os.path.join(args.save_dir, fimg)
  plot_hist(x, fimg)

  A_mode = args.mode
  D = X.shape[1]
  K = args.K
  n_steps = args.n_steps
  gamma_low, gamma_up = args.gamma_min, args.gamma
  gammas = get_aranges(gamma_low, gamma_up, n_steps)
  threshs = get_aranges(1e-9, 1e-5, n_steps)
  A, pi, mu, sigma_sqr = init_params(D, K, mu_low, mu_up)
  NLLs = []
  nll = eval_NLL(X)
  NLLs += nll,
  print('Initial NLL:', nll)

  grad_norms_total = []
  if TIME:
    time_iter = []
    time_em = []
    time_g1d = []
    time_save = []
    time_NLL = []
    time_A, time_E, time_GA, time_Y = [], [], [], []
  for i in range(n_steps):
    iter_start = time()
    print('iteration', i)
    if A_mode == 'ICA':
      Y, A, pi, mu, sigma_sqr, avg_time = EM(X, K, gammas[i], A, pi, mu, sigma_sqr, threshs[i],
                A_mode=A_mode, max_em_steps=args.n_em, n_gd_steps=args.n_gd)
    else:
      if A_mode == 'random':
        A = ortho_group.rvs(D)
      else:
        if TIME:
          em_start = time()
        X, A, pi, mu, sigma_sqr, grad_norms, avg_time = EM(X, K, gammas[i], A, pi, mu, sigma_sqr, threshs[i],
                  A_mode=A_mode, max_em_steps=args.n_em, n_gd_steps=args.n_gd)
        if TIME:
          time_em += time() - em_start,
          time_A += avg_time['A'],
          time_E += avg_time['E'],
          time_GA += avg_time['GA'],
          time_Y += avg_time['Y'],
        if args.mode == 'GA':
          grad_norms_total += np.array(grad_norms).mean(),
      if type(X) is torch.Tensor:
        Y = X.matmul(A.T)
      else:
        Y = X.dot(A.T)
    print('mu: mean={:.3e}/ std={:.3e}'.format(mu.mean(), mu.std()))
    print('sigma_sqr: min={:.3e} / mean={:.3e}/ std={:.3e}'.format(sigma_sqr.min(), sigma_sqr.mean(), sigma_sqr.std()))
    fimg = 'figs/hist2d_{}_mode{}_K{}_gamma{}_gammaMin{}_iter{}_Y.png'.format(data_token, A_mode, K, gamma_up, gamma_low, i)
    fimg = os.path.join(args.save_dir, fimg)
    plot_hist(Y, fimg)
    if TIME:
      nll_start = time()
    nll = eval_NLL(Y)
    if TIME:
      time_NLL += time() - nll_start,
    print('NLL (Y):', nll)

    if TIME:
      g1d_start = time()
    X = gaussianize_1d(Y, pi, mu, sigma_sqr)
    if TIME:
      time_g1d += time() - g1d_start,

    if TIME:
      nll_start = time()
    nll = eval_NLL(X)
    if TIME:
      time_NLL += time() - nll_start,
    NLLs += nll,
    print('NLL:', nll)
    print()
    x = X
    fimg = 'figs/hist2d_{}_mode{}_K{}_gamma{}_gammaMin{}_iter{}.png'.format(data_token, A_mode, K, gamma_up, gamma_low, i)
    fimg = os.path.join(args.save_dir, fimg)
    plot_hist(x, fimg)

    if args.save_dir:
      if TIME:
        save_start = time()
      if args.lib == 'torch':
        np.save(os.path.join(args.save_dir, 'A_i{}.npy'.format(i)), A.cpu().numpy())
        np.save(os.path.join(args.save_dir, 'pi_i{}.npy'.format(i)), pi.cpu().numpy())
        np.save(os.path.join(args.save_dir, 'mu_i{}.npy'.format(i)), mu.cpu().numpy())
        np.save(os.path.join(args.save_dir, 'sigma_sqr_i{}.npy'.format(i)), sigma_sqr.cpu().numpy())
      else:
        np.save(os.path.join(args.save_dir, 'A_i{}.npy'.format(i)), A)
        np.save(os.path.join(args.save_dir, 'pi_i{}.npy'.format(i)), pi)
        np.save(os.path.join(args.save_dir, 'mu_i{}.npy'.format(i)), mu)
        np.save(os.path.join(args.save_dir, 'sigma_sqr_i{}.npy'.format(i)), sigma_sqr)
      if TIME:
        time_save += time() - save_start,
    if TIME:
      time_iter += time() - iter_start,
      print("Timing (data={} / K={} / n_pts={} ):".format(args.data, args.K, args.n_pts))
      print('EM: {:.4e}'.format(np.array(time_em).mean()))
      print('--  A: {:.4e}'.format(np.array(time_A).mean()))
      print('--  E: {:.4e}'.format(np.array(time_E).mean()))
      print('-- GA: {:.4e}'.format(np.array(time_GA).mean()))
      print('--  Y: {:.4e}'.format(np.array(time_Y).mean()))
      print('G1D : {:.4e}'.format(np.array(time_g1d).mean()))
      print('NLL : {:.4e}'.format(np.array(time_NLL).mean()))
      print('Save: {:.4e}'.format(np.array(time_save).mean()))
      print()
  if TIME:
    np.save(os.path.join(args.save_dir, 'time_em.npy'), np.array(time_em))
    np.save(os.path.join(args.save_dir, 'time_A.npy'), np.array(time_A))
    np.save(os.path.join(args.save_dir, 'time_E.npy'), np.array(time_E))
    np.save(os.path.join(args.save_dir, 'time_GA.npy'), np.array(time_GA))
    np.save(os.path.join(args.save_dir, 'time_Y.npy'), np.array(time_Y))
    np.save(os.path.join(args.save_dir, 'time_G1D.npy'), np.array(time_g1d))
    np.save(os.path.join(args.save_dir, 'time_NLL.npy'), np.array(time_NLL))
    np.save(os.path.join(args.save_dir, 'time_save.npy'), np.array(time_save))
  np.save(os.path.join(args.save_dir, 'NLLs.npy'), np.array(NLLs))
  plt.plot(NLLs)
  plt.savefig(os.path.join(args.save_dir, 'figs', 'NLL.png'))
  plt.close()
  if args.mode == 'GA':
    np.save(os.path.join(args.save_dir, 'grad_norms.npy'), np.array(grad_norms_total))
    plt.plot(grad_norms_total)
    plt.savefig(os.path.join(args.save_dir, 'figs', 'grad_norms.png'))
    plt.close()


if __name__ == '__main__':
  # test()
  # gen_data()

  data_dir = './datasets/EM'
  args.save_dir = '{}_mode{}_K{}_iter{}_em{}_gd{}{}'.format(
        args.data, args.mode, args.K, args.n_steps, args.n_em, args.n_gd, '_gamma{}_gammaMin{}'.format(args.gamma, args.gamma_min) if args.mode == 'GA' else '')
  if args.n_pts:
    args.save_dir += '_nPts{}'.format(args.n_pts)
  if args.save_token:
    args.save_dir += '_' + args.save_token
  args.save_dir = os.path.join(SAVE_ROOT, args.save_dir)
  os.makedirs(os.path.join(args.save_dir), exist_ok=True)
  os.makedirs(os.path.join(args.save_dir, 'figs'), exist_ok=True)

  data_token = args.data
  mu_low, mu_up = -2, 2
  if data_token == 'GM':
    fdata = 'GM_2d.npy'
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
  
  data_token += args.save_token

  X = np.load(os.path.join(data_dir, fdata))
  if args.n_pts:
    X = X[:args.n_pts]
  fit(X, mu_low, mu_up, data_token)
  
