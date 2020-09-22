import os
import numpy as np
import torch
from scipy.stats import ortho_group
import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.get_loader import *

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--lib', type=str, default='torch', choices=['np', 'torch'])
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--n-pts', type=int, default=0)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--gamma-min', type=float, default=0.001)
parser.add_argument('--n-steps', type=int, default=50)
parser.add_argument('--n-em', type=int, default=30)
parser.add_argument('--n-gd', type=int, default=20)
parser.add_argument('--mode', type=str, default='GA', choices=['GA', 'torchGA', 'torchAll', 'CF', 'ICA'])
parser.add_argument('--A-first', type=int, default=1, choices=[0,1],
                    help="For ICA: whether to update A first (i.e. before estimating other params).")
parser.add_argument('--grad-mode', type=str, default='GA', choices=['GA', 'CF1', 'CF2', 'BTLS', 'perturb'],
                    help="Ways to update A in EM iterates.")
parser.add_argument('--data', type=str, default='GM', choices=[
       # connected
       'normal', 'scaledNormal', 'rotatedNormal', 'ring',
       # disconnected
       'GM', 'GM_scale1', 'GM_scale2', 'GMn2', 'concentric',
       # UCI
       'gas16_co', 'gas16_methane', 'gas8_co', 'gas8_co_normed', 'miniboone',
       # images,
       'MNIST',
       ])
parser.add_argument('--save-token', type=str, default='')
parser.add_argument('--save-dir', type=str)
parser.add_argument('--time', type=int, default=0)
parser.add_argument('--check-obj', type=int, default=0)
args = parser.parse_args()

if args.lib == 'np':
  from em_utils_np import *
elif args.lib == 'torch':
  from em_utils_torch import *

SAVE_ROOT = 'runs_gaussianization'
SAVE_NPY = False
VERBOSE = False

TIME=args.time
CHECK_OBJ=args.check_obj

def fit(X, Xtest, mu_low, mu_up, data_token=''):
  x = X
  fimg = 'figs/hist2d_{}_init.png'.format(data_token)
  fimg = os.path.join(args.save_dir, fimg)
  plot_hist(x, fimg)
  x = Xtest
  fimg = 'figs/hist2d_test_{}_init.png'.format(data_token)
  fimg = os.path.join(args.save_dir, fimg)
  plot_hist(x, fimg)

  if args.lib == 'torch':
    X = to_tensor(X)
    Xtest = to_tensor(Xtest)

  A_mode = args.mode
  D = X.shape[1]
  K = args.K
  n_steps = args.n_steps
  gamma_low, gamma_up = args.gamma_min, args.gamma
  gammas = get_aranges(gamma_low, gamma_up, n_steps)
  threshs = get_aranges(1e-9, 1e-5, n_steps)
  A, pi, mu, sigma_sqr = init_params(D, K, mu_low, mu_up)
  KLs, KLs_test = [], []

  log_det, log_det_test = 0, 0

  # get initial metrics
  # train
  kl = eval_KL(X, log_det)
  KLs += kl,
  print('Inital KL:', kl)
  # test
  kl_test = eval_KL(X, log_det_test)
  KLs_test += kl_test,
  print('Inital KL (test):', kl_test)
  print()

  grad_norms_total = []
  if TIME:
    avg_time = {'EM':[], 'KL':[], 'G1D':[], 'save':[]}
    default_keys = list(avg_time.keys())
  for i in range(n_steps):
    iter_start = time()
    print('iteration {} - data={} - mode={}{}'.format(i, args.data, args.mode,
          '(grad {})'.format(args.grad_mode) if args.mode in ['GA', 'torchGA', 'torchAll'] else ''))
    if TIME:
      em_start = time()
    A, pi, mu, sigma_sqr, grad_norms, objs, ret_time = update_EM(X, K, gammas[i], A, pi, mu, sigma_sqr, threshs[i],
              A_mode=A_mode, grad_mode=args.grad_mode, max_em_steps=args.n_em, n_gd_steps=args.n_gd)
    if TIME:
      for key in ret_time:
        if key not in avg_time: avg_time[key] = []
        avg_time[key] += ret_time[key],
      avg_time['EM'] += time() - em_start,
    if grad_norms:
      grad_norms_total += np.array(grad_norms).mean(),

    if type(X) is torch.Tensor:
      Y = X.matmul(A.T)
      Ytest = Xtest.matmul(A.T)
    else:
      Y = X.dot(A.T)
      Ytest = Xtest.dot(A.T)

    if VERBOSE:
      print('mu: mean={:.3e}/ std={:.3e}'.format(mu.mean(), mu.std()))
      print('sigma_sqr: min={:.3e} / mean={:.3e}/ std={:.3e}'.format(sigma_sqr.min(), sigma_sqr.mean(), sigma_sqr.std()))
    if CHECK_OBJ:
      for emi, obj in enumerate(objs):
        fimg = 'figs/objs/objs_step{}_em{}.png'.format(i, emi)
        fimg = os.path.join(args.save_dir, fimg)
        plt.plot(obj)
        plt.savefig(fimg)
        plt.close()

    fimg = 'figs/hist2d_{}_mode{}_K{}_gamma{}_gammaMin{}_iter{}_Y.png'.format(data_token, A_mode, K, gamma_up, gamma_low, i)
    fimg = os.path.join(args.save_dir, fimg)
    plot_hist(Y, fimg)
    fimg = 'figs/hist2d_test_{}_mode{}_K{}_gamma{}_gammaMin{}_iter{}_Y.png'.format(data_token, A_mode, K, gamma_up, gamma_low, i)
    fimg = os.path.join(args.save_dir, fimg)
    plot_hist(Ytest, fimg)

    if TIME:
      g1d_start = time()
    X, cdf_mask, [log_cdf, cdf_mask_left], [log_sf, cdf_mask_right] = gaussianize_1d(Y, pi, mu, sigma_sqr)
    if TIME:
      avg_time['G1D'] += time() - g1d_start,
      nll_start = time()
    check_cov(X)
    log_det += compute_log_det(Y, X, pi, mu, sigma_sqr, A, cdf_mask, log_cdf, cdf_mask_left, log_sf, cdf_mask_right)
    kl = eval_KL(X, log_det)
    KLs += kl,
    print('KL:', kl)

    x = X
    fimg = 'figs/hist2d_{}_mode{}_K{}_gamma{}_gammaMin{}_iter{}.png'.format(data_token, A_mode, K, gamma_up, gamma_low, i)
    fimg = os.path.join(args.save_dir, fimg)
    if torch.isnan(x.max()) or torch.isnan(x.min()):
      print('X: NaN')
      pdb.set_trace()
    plot_hist(x, fimg)

    # check on test data
    Xtest, cdf_mask_test, [log_cdf_test, cdf_mask_left_test], [log_sf_test, cdf_mask_right_test] = gaussianize_1d(Ytest, pi, mu, sigma_sqr)
    check_cov(Xtest)
    x = Xtest
    fimg = 'figs/hist2d_test_{}_mode{}_K{}_gamma{}_gammaMin{}_iter{}.png'.format(data_token, A_mode, K, gamma_up, gamma_low, i)
    fimg = os.path.join(args.save_dir, fimg)
    if torch.isnan(x.max()) or torch.isnan(x.min()):
      print('Xtest: NaN')
      pdb.set_trace()
    plot_hist(x, fimg)
    log_det_test += compute_log_det(Ytest, Xtest, pi, mu, sigma_sqr, A, cdf_mask_test, log_cdf_test, cdf_mask_left_test, log_sf_test, cdf_mask_right_test)
    kl_test = eval_KL(Xtest, log_det_test)
    KLs_test += kl_test,
    print('KL (test):', kl_test)
    print()

    if args.save_dir and SAVE_NPY:
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
        avg_time['save'] += time() - save_start,
      np.save(os.path.join(args.save_dir, 'KLs.npy'), np.array(KLs))
      plt.plot(np.array(KLs))
      plt.savefig(os.path.join(args.save_dir, 'figs', 'KL_log.png'))
      plt.close()
      np.save(os.path.join(args.save_dir, 'KLs_test.npy'), np.array(KLs_test))
      plt.plot(np.array(KLs_test))
      plt.savefig(os.path.join(args.save_dir, 'figs', 'KLtest_log.png'))
      plt.close()

    if TIME:
      avg_time['iter'] += time() - iter_start,
      print("Timing (data={} / K={} / n_pts={} ):".format(args.data, args.K, args.n_pts))
      print('EM: {:.4e}'.format(np.array(time_em).mean()))
      for key in avg_time:
        if key not in default_keys:
          # time returned from EM call
          print('--  {}: {:.4e}'.format(key, np.array(avg_time[key]).mean()))
      print('G1D : {:.4e}'.format(np.array(avg_time['G1D']).mean()))
      print('KL  : {:.4e}'.format(np.array(avg_time['KL'].mean())))
      print('Save: {:.4e}'.format(np.array(avg_time['save']).mean()))
      print()
  if TIME:
    ftime = os.path.join(args.save_dir, 'time.pkl')
    with open(ftime, 'wb') as handle:
      pickle.dump(avg_time, handle)
  # train
  KLs = np.array(KLs)
  np.save(os.path.join(args.save_dir, 'KLs.npy'), KLs)
  plt.plot(KLs)
  plt.savefig(os.path.join(args.save_dir, 'figs', 'KL_log.png'))
  plt.close()
  # test
  KLs_test = np.array(KLs_test)
  np.save(os.path.join(args.save_dir, 'KLs_test.npy'), KLs_test)
  plt.plot(KLs_test)
  plt.savefig(os.path.join(args.save_dir, 'figs', 'KL_log_test.png'))
  plt.close()

  if args.mode == 'GA':
    np.save(os.path.join(args.save_dir, 'grad_norms.npy'), np.array(grad_norms_total))
    plt.plot(grad_norms_total)
    plt.savefig(os.path.join(args.save_dir, 'figs', 'grad_norms.png'))
    plt.close()


if __name__ == '__main__':
  # test()
  # gen_data()

  data_dir = './datasets/'
  ga_token = ''
  if args.mode in ['GA', 'torchGA']:
    ga_token = '_'+args.grad_mode
    if args.grad_mode == 'GA':
      ga_token = '_gamma{}_gammaMin{}'.format(args.gamma, args.gamma_min)
  elif args.mode == 'CF':
    ga_token = '_gamma{}'.format(args.gamma)
  args.save_dir = '{}/mode{}_K{}_iter{}_em{}_gd{}{}'.format(
        args.data, args.mode, args.K, args.n_steps, args.n_em, args.n_gd, ga_token)
  if args.n_pts:
    args.save_dir += '_nPts{}'.format(args.n_pts)
  if args.save_token:
    args.save_dir += '_' + args.save_token
  args.save_dir = os.path.join(SAVE_ROOT, args.save_dir)
  if os.path.exists(args.save_dir):
    proceed = input('Dir exist: {} \n Do you want to proceed? (y/N)'.format(args.save_dir))
    if 'y' not in proceed:
      print('Exiting. Bye!')
      exit(0)
  os.makedirs(args.save_dir, exist_ok=True)
  os.makedirs(os.path.join(args.save_dir, 'figs'), exist_ok=True)
  if CHECK_OBJ:
    os.makedirs(os.path.join(args.save_dir, 'figs', 'objs'), exist_ok=True)  

  data_token = args.data
  mu_low, mu_up = -2, 2
  if data_token == 'GM':
    fdata = 'EM/GM_2d.npy'
    fdata_val = 'EM/GM_2d_scale4_test.npy'
    mu_low, mu_up = -4, 4
  if data_token == 'GM_scale1':
    fdata = 'EM/GM_2d_scale1.npy'
    mu_low, mu_up = -4, 4
  if data_token == 'GM_scale2':
    fdata = 'EM/GM_2d_scale2.npy'
    mu_low, mu_up = -4, 4
  if data_token == 'GMn2':
    fdata = 'EM/GM_2d_2centers.npy'
    mu_low, mu_up = -4, 4
  elif data_token == 'normal':
    fdata = 'EM/normal.npy'
  elif data_token == 'rotatedNormal':
    fdata = 'EM/rotatedNormal.npy'
  elif data_token == 'scaledNormal':
    fdata = 'EM/scaledNormal.npy'
  elif data_token == 'ring':
    fdata = 'EM/ring.npy'
    mu_low, mu_up = -1, 1
  elif data_token == 'concentric':
    # 2 circles w/ radius 0.5 and 2. Each with 10k points.
    fdata = 'EM/concentric.npy'
  elif data_token == 'gas16_co':
    fdata = 'GAS/gas_d16_CO_normed_train400k.npy'
    fdata_val = 'GAS/gas_d16_CO_normed_test.npy'
  elif data_token == 'gas16_methane':
    fdata = 'GAS/gas_d16_methane.npy'
  elif data_token == 'gas8_co':
    gas_dir = 'flows_data/gas'
    # train: 852174 points
    fdata = os.path.join(gas_dir, 'gas_train.npy')
    # val: 94685 points
    fdata_val = os.path.join(gas_dir, 'gas_val.npy')
  elif data_token == 'gas8_co_normed':
    # gas-8d, with data normzlied per column to [-1, 1].
    gas_dir = 'flows_data/gas'
    # train: 852174 points
    fdata = os.path.join(gas_dir, 'gas_train_normed.npy')
    # val: 94685 points
    fdata_val = os.path.join(gas_dir, 'gas_val_normed.npy')
  elif data_token == 'miniboone':
    fdata = 'miniboone/train_normed.npy'
    fdata_val = 'miniboone/val_normed.npy'
  elif data_token == 'MNIST':
    mnist_dir = 'mnist/MNIST/processed'
    fdata = os.path.join(mnist_dir, 'train_normed.npy')
    fdata_val = os.path.join(mnist_dir, 'test_normed.npy')
  
  data_token += args.save_token

  X = np.load(os.path.join(data_dir, fdata))
  Xtest = np.load(os.path.join(data_dir, fdata_val))
  if X.ndim > 2: # images
    X = X.reshape(len(X), -1)
    Xtest = Xtest.reshape(len(Xtest), -1)
  if args.n_pts:
    X = X[:args.n_pts]
    Xtest = Xtest[:args.n_pts//2]
  fit(X, Xtest, mu_low, mu_up, data_token)
  
