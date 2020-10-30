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

def old_test(X, save_dir):
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

def test(X, Xtest):
  x = X
  fimg = 'figs/test_hist2d_{}_init.png'.format(data_token)
  fimg = os.path.join(args.save_dir, fimg)
  plot_hist(x, fimg)
  x = Xtest
  fimg = 'figs/test_hist2d_test_{}_init.png'.format(data_token)
  fimg = os.path.join(args.save_dir, fimg)
  plot_hist(x, fimg)

  if args.lib == 'torch':
    X = to_tensor(X)
    Xtest = to_tensor(Xtest)

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

  for i in range(n_steps):
    # load parameters
    A = np.load(os.path.join(args.save_dir, 'A_i{}.npy'.format(i)))
    pi = np.load(os.path.join(args.save_dir, 'pi_i{}.npy'.format(i)))
    mu = np.load(os.path.join(args.save_dir, 'mu_i{}.npy'.format(i)))
    sigma_sqr = np.load(os.path.join(args.save_dir, 'sigma_sqr_i{}.npy'.format(i)))

    if type(X) is torch.Tensor:
      Y = X.matmul(A.T)
      Ytest = Xtest.matmul(A.T)
    else:
      Y = X.dot(A.T)
      Ytest = Xtest.dot(A.T)

    X, cdf_mask, [log_cdf, cdf_mask_left], [log_sf, cdf_mask_right] = gaussianize_1d(Y, pi, mu, sigma_sqr)
    diff_from_I, sigma_max, sigma_min, sigma_mean = check_cov(X)
    log_det += compute_log_det(Y, X, pi, mu, sigma_sqr, A, cdf_mask, log_cdf, cdf_mask_left, log_sf, cdf_mask_right)
    kl = eval_KL(X, log_det)
    KLs += kl,
    print('KL:', kl)

    x = X
    fimg = 'figs/test_hist2d_{}_mode{}_K{}_gamma{}_gammaMin{}_iter{}.png'.format(data_token, A_mode, K, gamma_up, gamma_low, i)
    fimg = os.path.join(args.save_dir, fimg)
    if torch.isnan(x.max()) or torch.isnan(x.min()):
      print('X: NaN')
      pdb.set_trace()
    plot_hist(x, fimg)

    # check on test data
    Xtest, cdf_mask_test, [log_cdf_test, cdf_mask_left_test], [log_sf_test, cdf_mask_right_test] = gaussianize_1d(Ytest, pi, mu, sigma_sqr)
    diff_from_I_test, sigma_max_test, sigma_min_test, sigma_mean_test = check_cov(Xtest)
    x = Xtest
    fimg = 'figs/test_hist2d_test_{}_mode{}_K{}_gamma{}_gammaMin{}_iter{}.png'.format(data_token, A_mode, K, gamma_up, gamma_low, i)
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

    if USE_WANDB:
      wandb.log({
        'KL': kl,
        'KLtest': kl_test,
        'diff_from_I': diff_from_I,
        'diff_from_Itest': diff_from_I_test,
        'sigma_max': sigma_max,
        'sigma_min': sigma_min,
        'sigma_max_test': sigma_max_test,
        'sigma_min_test': sigma_min_test,
        'sigma_mean': sigma_mean,
        'sigma_mean_test': sigma_mean_test
        })

    if args.save_dir:
      plt.plot(np.array(KLs))
      plt.savefig(os.path.join(args.save_dir, 'figs', 'test_KL_log.png'))
      plt.close()
      plt.plot(np.array(KLs_test))
      plt.savefig(os.path.join(args.save_dir, 'figs', 'test_KLtest_log.png'))
      plt.close()

  if TIME:
    ftime = os.path.join(args.save_dir, 'time.pkl')
    with open(ftime, 'wb') as handle:
      pickle.dump(avg_time, handle)
  # train
  KLs = np.array(KLs)
  np.save(os.path.join(args.save_dir, 'test_KLs.npy'), KLs)
  plt.plot(KLs)
  plt.savefig(os.path.join(args.save_dir, 'figs', 'test_KL_log.png'))
  plt.close()
  # test
  KLs_test = np.array(KLs_test)
  np.save(os.path.join(args.save_dir, 'test_KLs_test.npy'), KLs_test)
  plt.plot(KLs_test)
  plt.savefig(os.path.join(args.save_dir, 'figs', 'test_KL_log_test.png'))
  plt.close()


if __name__ == '__main__':
  args.save_dir = os.path.join(SAVE_ROOT, args.save_dir)

  data_dir = './datasets/EM'
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
    fdata = os.path.join(mnist_dir, 'train_normed_pca{}.npy'.format(args.pca_dim))
    fdata_val = os.path.join(mnist_dir, 'test_normed_pca{}.npy'.format(args.pca_dim))
   
  X = np.load(os.path.join(data_dir, fdata))
  test(X, args.save_dir)
  
