import numpy as np
from scipy.stats import ortho_group
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# from data import get_loader
from data.dataset_mixture import GaussianMixture
from em_utils import *

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--n-steps', type=int, default=50)
parser.add_argument('--mode', type=str, choices=['GA', 'CF'])
parser.add_argument('--data', type=str, choices=['normal', 'scaledNormal', 'rotatedNormal', 'GM', 'ring'])
args = parser.parse_args()


def fit(X, mu_low, mu_up, data_token=''):
  plt.hist2d(X[:,0], X[:,1], bins=[100,100])
  plt.savefig('figs/hist2d_{}_init.png'.format(data_token))
  plt.clf()

  # A_mode = 'GA'
  A_mode = args.mode
  D = X.shape[1]
  K = args.K
  n_steps = args.n_steps
  gamma_low, gamma_up = 1e-4, 0.1
  gammas = get_aranges(gamma_low, gamma_up, n_steps)
  threshs = get_aranges(1e-9, 1e-5, n_steps)
  A, pi, mu, sigma_sqr = init_params(D, K, mu_low, mu_up)
  print('Initial NLL:', eval_NLL(X))
  for i in range(n_steps):
    print('iteration', i)
    if A_mode == 'random':
      A = ortho_group.rvs(D)
    else:
      A, pi, mu, sigma_sqr = EM(X, K, gammas[i], A, pi, mu, sigma_sqr, threshs[i], A_mode=A_mode)
    print('mu: mean={:.3e}/ std={:.3e}'.format(mu.mean(), mu.std()))
    print('sigma_sqr: min={:.3e} / mean={:.3e}/ std={:.3e}'.format(sigma_sqr.min(), sigma_sqr.mean(), sigma_sqr.std()))
    Y = X.dot(A.T)
    print('NLL (Y):', eval_NLL(Y))
    X = gaussianize_1d(Y, pi, mu, sigma_sqr)
    print('NLL:', eval_NLL(X))
    print()
    plt.hist2d(X[:,0], X[:,1], bins=[100,100])
    plt.savefig('figs/hist2d_{}_mode{}_K{}_gamma{}_iter{}.png'.format(data_token, A_mode, K, gamma_up, i))
    plt.clf()


if __name__ == '__main__':
  # test()
  # gen_data()

  data_token = args.data
  mu_low, mu_up = -2, 2
  if data_token == 'GM':
    X = np.load('GM_2d.npy')
    mu_low, mu_up = -4, 4
  elif data_token == 'normal':
    X = np.load('normal.npy')
  elif data_token == 'rotatedNormal':
    X = np.load('rotatedNormal.npy')
  elif data_token == 'scaledNormal':
    X = np.load('scaledNormal.npy')
  elif data_token == 'ring':
    X = np.load('ring.npy')
    mu_low, mu_up = -1, 1

  fit(X, mu_low, mu_up, data_token)
  
