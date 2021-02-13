import os
import numpy as np
import pickle
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
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--n-pts', type=int, default=0,
                    help="Number of samples used in training.")
parser.add_argument('--n-steps', type=int, default=50,
                    help="Number of layers / Gaussianization iterations.")
parser.add_argument('--n-em', type=int, default=30)
parser.add_argument('--n-gd', type=int, default=20)
parser.add_argument('--mode', type=str, default='GA',
                    choices=['ICA', 'PCA', 'random', 'None', 'variational', 'Wasserstein'])
parser.add_argument('--ica-iters', type=int, default=200)
parser.add_argument('--ica-tol', type=float, default=1e-4)
parser.add_argument('--g1d-first', type=int, default=0,
                    help="Whether to run a Gaussianization step before iterates. If 1, then this should be the same as RBIG.")
parser.add_argument('--log-det-version', type=str, default='v1')
parser.add_argument('--grad-mode', type=str, default='GA', choices=['GA', 'CF1', 'CF2', 'BTLS', 'perturb'],
                    help="Ways to update A in EM iterates.")
parser.add_argument('--data', type=str, default='GM', choices=[
       # connected
       'normal', 'scaledNormal', 'rotatedNormal', 'ring',
       # disconnected
       'GM', 'GM_scale1', 'GM_scale2', 'GMn2', 'concentric',
       # UCI
       'gas16_co', 'gas16_methane', 'gas8_co', 'gas8_co_normed',
       'miniboone', 'hepmass', 'gas', 'gasSmall', 'power',
       # images,
       'MNIST',
       ])
parser.add_argument('--pca-dim', type=int, default=0,
                   help="PCA dimension for high-dim data e.g. MNIST.")
# variational tuning
parser.add_argument('--var-iters', type=int, default=1000)
parser.add_argument('--var-optimizer-option', type=str, default='SGD', choices=['SGD', 'Adam'])
parser.add_argument('--var-lr', type=float, default=1e-1)
parser.add_argument('--var-wd', type=float, default=1e-4)
parser.add_argument('--var-patience', type=int, default=200)
parser.add_argument('--det-lambda', type=float, default=0.1)
parser.add_argument('--det-every', type=int, default=100)
parser.add_argument('--var-LB', type=str, help="Type of KL lower bound.")
parser.add_argument('--var-A-mode', type=str, default='GD', choices=['GD', 'fixed', 'givens'])
parser.add_argument('--var-pos-type', type=str, default='smoothL1')
parser.add_argument('--var-num-hidden-nodes', type=int, default=10)
parser.add_argument('--var-num-layers', type=int, default=1)
parser.add_argument('--var-batch-size', type=int, default=10000)
parser.add_argument('--var-n-Zs', type=int, default=5000,
                    help="Number of Zs (std Gaussian) samples used in each layer.")
parser.add_argument('--var-function-option', type=str, default='net', choices=['net', 'basis'])
parser.add_argument('--var-n-cos', type=int, default=50,
                    help="Number of cos basis func; applicable when function_option = 'basis' ")
parser.add_argument('--var-n-sin', type=int, default=50,
                    help="Number of sin basis func; applicable when function_option = 'basis' ")
# saving & logging
parser.add_argument('--overwrite', type=int, default=0,
                   help="Whether to overwrite an existing directory.")
parser.add_argument('--save-token', type=str, default='')
parser.add_argument('--save-dir', type=str)
parser.add_argument('--time', type=int, default=0)
args = parser.parse_args()

from utils.misc import *
from utils.em_utils_torch import *
# from rbig_util import *

SAVE_ROOT = 'runs_gaussianization'
SAVE_FIG = True
SAVE_NPY = False
VERBOSE = False

TIME=args.time
PROFILE = 1
if PROFILE:
  import cProfile
  pr = cProfile.Profile()


data_dir = './datasets/'

d_token = 'em{}_K{}'.format(args.n_em, args.K)
args.save_dir = '{}/{}_mode{}_iter{}_{}'.format(
      args.data, args.data, args.mode, args.n_steps, d_token)
if args.n_pts:
  args.save_dir += '_nPts{}'.format(args.n_pts)
if args.save_token:
  args.save_dir += '_' + args.save_token
args.save_dir = os.path.join(SAVE_ROOT, args.save_dir)
if not args.overwrite and os.path.exists(args.save_dir):
  proceed = input('Dir exist: {} \n Do you want to proceed? (y/N)'.format(args.save_dir))
  if 'y' not in proceed:
    print('Exiting. Bye!')
    exit(0)
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, 'model'), exist_ok=True)
os.makedirs(os.path.join(args.save_dir, 'figs'), exist_ok=True)

try:
  import wandb
  wandb.init(project='density', name=os.path.basename(args.save_dir), config=args)
  USE_WANDB = True
except Exception as e:
  print('Exception:', e)
  print('Not using wandb. \n\n')
  USE_WANDB = False

if args.log_det_version == 'v1':
  compute_log_det = compute_log_det_v1
else:
  compute_log_det = compute_log_det_v2

def fit(X, Xtest, mu_low, mu_up, data_token='', b_a=0):
  x = X
  if SAVE_FIG:
    fimg = 'figs/hist2d_{}_init.png'.format(data_token)
    fimg = os.path.join(args.save_dir, fimg)
    plot_hist(x, fimg)
  x = Xtest
  if SAVE_FIG:
    fimg = 'figs/hist2d_test_{}_init.png'.format(data_token)
    fimg = os.path.join(args.save_dir, fimg)
    plot_hist(x, fimg)

  X = to_tensor(X)
  Xtest = to_tensor(Xtest)

  A_mode = args.mode
  D = X.shape[1]
  K = args.K
  n_steps = args.n_steps
  threshs = get_aranges(1e-9, 1e-5, n_steps)
  A = init_A(D)
  pi, mu, sigma_sqr = init_GM_params(D, K, mu_low, mu_up)
  NLLs, NLLs_test = [], []
  KLs, KLs_test = [], []

  log_det, log_det_test = to_tensor(np.zeros(X.shape[0])), to_tensor(np.zeros(Xtest.shape[0]))

  # get initial metrics
  # train
  nll = eval_NLL(X, log_det)
  NLLs += nll,
  print('Inital NLL:', nll)
  # test
  nll_test = eval_NLL(X, log_det_test)
  NLLs_test += nll_test,
  print('Inital NLL (test):', nll_test)
  print()

  if args.g1d_first:
    X, _, _, _ = gaussianize_1d(X, pi=pi, mu=mu, sigma_sqr=sigma_sqr) 
    Xtest, _, _, _ = gaussianize_1d(Xtest, pi=pi, mu=mu, sigma_sqr=sigma_sqr)

  if TIME:
    avg_time = {'EM':[], 'NLL':[], 'G1D':[], 'save':[], 'iter':[]}
    default_keys = list(avg_time.keys())
  # Iterating over the layers
  if PROFILE:
    pr.enable()
  for i in range(n_steps):
    iter_start = time()
    print('iteration {} - data={} - mode={}{}'.format(i, args.data, args.mode,
          '(grad {})'.format(args.grad_mode) if args.mode in ['GA', 'torchGA', 'torchAll'] else ''))
    if TIME:
      em_start = time()
    A = update_A(X, args, b_a)
    pi, mu, sigma_sqr, ret_time = update_EM(X, A, pi, mu, sigma_sqr,
              threshs[i], max_em_steps=args.n_em)

    if TIME:
      for key in ret_time:
        if key not in avg_time: avg_time[key] = []
        avg_time[key] += ret_time[key],
      avg_time['EM'] += time() - em_start,

    if type(X) is torch.Tensor:
      Y = X.matmul(A.T)
      Ytest = Xtest.matmul(A.T)
    else:
      Y = X.dot(A.T)
      Ytest = Xtest.dot(A.T)

    if VERBOSE:
      print('mu: mean={:.3e}/ std={:.3e}'.format(mu.mean(), mu.std()))
      print('sigma_sqr: min={:.3e} / mean={:.3e}/ std={:.3e}'.format(sigma_sqr.min(), sigma_sqr.mean(), sigma_sqr.std()))

    if SAVE_FIG or i % 50 == 0:
      fimg = 'figs/hist2d_{}_mode{}_K{}_iter{}_Y.png'.format(data_token, A_mode, K, i)
      fimg = os.path.join(args.save_dir, fimg)
      plot_hist(Y, fimg)
      fimg = 'figs/hist2d_test_{}_mode{}_K{}_iter{}_Y.png'.format(data_token, A_mode, K, i)
      fimg = os.path.join(args.save_dir, fimg)
      plot_hist(Ytest, fimg)

    if TIME: g1d_start = time()
    prevX = X.clone()
    X, cdf_mask, [log_cdf, cdf_mask_left], [log_sf, cdf_mask_right] = gaussianize_1d(Y, pi, mu, sigma_sqr)
    if TIME: avg_time['G1D'] += time() - g1d_start,

    if TIME: nll_start = time()
    log_det += compute_log_det(X, Y, cdf_mask, log_cdf, cdf_mask_left, log_sf, cdf_mask_right,
                               pi=pi, mu=mu, sigma_sqr=sigma_sqr)
    log_det += torch.log(torch.abs(torch.det(A)))
    nll = eval_NLL(X, log_det)
    NLLs += nll,
    kl = eval_KL(X)
    KLs += kl,
    print('NLL+KL:', nll)
    if TIME: avg_time['NLL'] += time() - nll_start,

    diff_from_I, sigma_max, sigma_min, sigma_mean = check_cov(X)

    x = X
    if SAVE_FIG or i % 50 == 0:
      fimg = 'figs/hist2d_{}_mode{}_K{}_iter{}.png'.format(data_token, A_mode, K, i)
      fimg = os.path.join(args.save_dir, fimg)
      if torch.isnan(x.max()) or torch.isnan(x.min()):
        print('X: NaN')
        pdb.set_trace()
      plot_hist(x, fimg)

    # check on test data
    Xtest, cdf_mask_test, [log_cdf_test, cdf_mask_left_test], [log_sf_test, cdf_mask_right_test] = gaussianize_1d(Ytest, pi, mu, sigma_sqr)
    diff_from_I_test, sigma_max_test, sigma_min_test, sigma_mean_test = check_cov(Xtest)
    x = Xtest
    if SAVE_FIG or i % 50 == 0:
      fimg = 'figs/hist2d_test_{}_mode{}_K{}_iter{}.png'.format(data_token, A_mode, K, i)
      fimg = os.path.join(args.save_dir, fimg)
      if torch.isnan(x.max()) or torch.isnan(x.min()):
        print('Xtest: NaN')
        pdb.set_trace()
      plot_hist(x, fimg)
    log_det_test += compute_log_det(Xtest, Ytest, cdf_mask_test, log_cdf_test, cdf_mask_left_test, log_sf_test, cdf_mask_right_test,
                                    pi=pi, mu=mu, sigma_sqr=sigma_sqr)
    log_det_test += torch.log(torch.abs(torch.det(A)))
    nll_test = eval_NLL(Xtest, log_det_test)
    NLLs_test += nll_test,
    print('NLL (test):', nll_test)
    print()
    kl_test = eval_KL(Xtest)
    KLs_test += kl_test,

    if USE_WANDB:
      wandb.log({
        'layer': i,
        'NLL': nll,
        'NLLtest': nll_test,
        'KL': kl,
        'KLtest': kl_test,
        'diff_from_I': diff_from_I,
        'diff_from_Itest': diff_from_I_test,
        'sigma_max': sigma_max,
        'sigma_min': sigma_min,
        'sigma_max_test': sigma_max_test,
        'sigma_min_test': sigma_min_test,
        'sigma_mean': sigma_mean,
        'sigma_mean_test': sigma_mean_test,
        'detA': torch.det(A).item(),
        'num_iter': i,
        })

    if args.save_dir:
      if SAVE_NPY:
        if TIME:
          save_start = time()
        np.save(os.path.join(args.save_dir, 'model', 'A_i{}.npy'.format(i)), A.cpu().numpy())
        np.save(os.path.join(args.save_dir, 'model', 'pi_i{}.npy'.format(i)), pi.cpu().numpy())
        np.save(os.path.join(args.save_dir, 'model', 'mu_i{}.npy'.format(i)), mu.cpu().numpy())
        np.save(os.path.join(args.save_dir, 'model', 'sigma_sqr_i{}.npy'.format(i)), sigma_sqr.cpu().numpy())
        if TIME:
          avg_time['save'] += time() - save_start,
        np.save(os.path.join(args.save_dir, 'NLLs.npy'), np.array(NLLs))
        np.save(os.path.join(args.save_dir, 'NLLs_test.npy'), np.array(NLLs_test))
        np.save(os.path.join(args.save_dir, 'KLs.npy'), np.array(KLs))
        np.save(os.path.join(args.save_dir, 'KLs_test.npy'), np.array(KLs_test))
      if SAVE_FIG or i % 50 == 0:
        plt.plot(np.array(NLLs))
        plt.savefig(os.path.join(args.save_dir, 'figs', 'NLL_log.png'))
        plt.close()
        plt.plot(np.array(NLLs_test))
        plt.savefig(os.path.join(args.save_dir, 'figs', 'NLLtest_log.png'))
        plt.close()
        plt.plot(np.array(KLs))
        plt.savefig(os.path.join(args.save_dir, 'figs', 'KL_log.png'))
        plt.close()
        plt.plot(np.array(KLs_test))
        plt.savefig(os.path.join(args.save_dir, 'figs', 'KLtest_log.png'))
        plt.close()

    if TIME:
      avg_time['iter'] += time() - iter_start,
      print("Timing (data={} / K={} / n_pts={} ):".format(args.data, args.K, args.n_pts))
      print('EM: {:.4e}'.format(np.array(avg_time['EM']).mean()))
      for key in avg_time:
        if key not in default_keys:
          # time returned from EM call
          print('--  {}: {:.4e}'.format(key, np.array(avg_time[key]).mean()))
      print('G1D : {:.4e}'.format(np.array(avg_time['G1D']).mean()))
      print('NLL  : {:.4e}'.format(np.array(avg_time['KL']).mean()))
      print('Save: {:.4e}'.format(np.array(avg_time['save']).mean() if avg_time['save'] else 0))
      print()

    if PROFILE:
      # TODO: delete the tmp file at the end; check the end file is correct first.
      print_profile_stats(pr, 'profile_{}_tmp.txt'.format(args.save_token))


  if TIME:
    ftime = os.path.join(args.save_dir, 'time.pkl')
    with open(ftime, 'wb') as handle:
      pickle.dump(avg_time, handle)
  if PROFILE:
    pr.disable()
    print_profile_stats(pr, 'profile_{}.txt'.format(args.save_token))

  # train
  NLLs = np.array(NLLs)
  np.save(os.path.join(args.save_dir, 'NLLs.npy'), KLs)
  plt.plot(NLLs)
  plt.savefig(os.path.join(args.save_dir, 'figs', 'NLL_log.png'))
  plt.close()
  KLs = np.array(KLs)
  np.save(os.path.join(args.save_dir, 'KLs.npy'), KLs)
  plt.plot(KLs)
  plt.savefig(os.path.join(args.save_dir, 'figs', 'KL_log.png'))
  plt.close()
  # test
  NLLs_test = np.array(NLLs_test)
  np.save(os.path.join(args.save_dir, 'NLLs_test.npy'), KLs_test)
  plt.plot(NLLs_test)
  plt.savefig(os.path.join(args.save_dir, 'figs', 'NLL_log_test.png'))
  plt.close()
  KLs_test = np.array(KLs_test)
  np.save(os.path.join(args.save_dir, 'KLs_test.npy'), KLs_test)
  plt.plot(KLs_test)
  plt.savefig(os.path.join(args.save_dir, 'figs', 'KL_log_test.png'))
  plt.close()


if __name__ == '__main__':
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
    fdata = 'GAS/my_data/gas_d16_CO_normed_train400k.npy'
    fdata_val = 'GAS/my_data/gas_d16_CO_normed_test.npy'
  elif data_token == 'gas16_methane':
    fdata = 'GAS/my_data/gas_d16_methane.npy'
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
    fdata = 'miniboone/trainval_normed.npy'
    fdata_val = 'miniboone/test_normed.npy'
  elif data_token == 'hepmass':
    fdata = 'hepmass/train_normed.npy'
    fdata_val = 'hepmass/val_normed.npy'
  elif data_token == 'MNIST':
    mnist_dir = 'mnist/MNIST/processed'
    if args.pca_dim:
      fdata = os.path.join(mnist_dir, 'train_normed_pca{}.npy'.format(args.pca_dim))
      fdata_val = os.path.join(mnist_dir, 'test_normed_pca{}.npy'.format(args.pca_dim))
    else:
      fdata = os.path.join(mnist_dir, 'train_normed.npy')
      fdata_val = os.path.join(mnist_dir, 'test_normed.npy')
  elif data_token == 'power':
    power_dir = 'POWER/'
    fdata = os.path.join(power_dir, 'train_normed.npy')
    fdata_val = os.path.join(power_dir, 'val_normed.npy')
  elif data_token == 'gas':
    gas_dir = 'GAS/'
    fdata = os.path.join(gas_dir, 'ethylene_CO_train_normed.npy')
    fdata_val = os.path.join(gas_dir, 'ethylene_CO_val_normed.npy')
  elif data_token == 'gasSmall':
    gas_dir = 'GAS/'
    fdata = os.path.join(gas_dir, 'ethylene_CO_trainSmall_normed.npy')
    fdata_val = os.path.join(gas_dir, 'ethylene_CO_valSmall_normed.npy')
  
  data_token += args.save_token

  X = np.load(os.path.join(data_dir, fdata))
  Xtest = np.load(os.path.join(data_dir, fdata_val))
  # zero-centered
  X = X - X.mean(0)
  b_a = X.max() - X.min()
  Xtest = Xtest - X.mean(0)

  if X.ndim > 2: # images
    X = X.reshape(len(X), -1)
    Xtest = Xtest.reshape(len(Xtest), -1)
  if args.n_pts:
    X = X[:args.n_pts]
    Xtest = Xtest[:args.n_pts//2]
  fit(X, Xtest, mu_low, mu_up, data_token, b_a=b_a)
  
