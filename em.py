import numpy as np
from scipy.stats import ortho_group
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# from data import get_loader
from data.dataset_mixture import GaussianMixture
from em_utils import *

import pdb

def test():
  N = 1000
  D = 2
  K = 10
  gamma = 1e-5
  X = np.random.randn(N, D)
  A, pi, mu, sigma_sqr = EM(X, K, gamma)

def gen_data():
  dlen = 100000
  dset = GaussianMixture(dlen)
  data = []
  for _ in range(dlen):
    data += dset.__getitem__(0),
  data = np.array(data)
  pdb.set_trace()

def fit(X):
  plt.hist2d(X[:,0], X[:,1], bins=[100,100])
  plt.savefig('figs/hist2d_init.png')
  plt.clf()

  A_mode = 'GA'
  D = X.shape[1]
  K = 40
  n_steps = 80
  gamma_low, gamma_up = 1e-7, 1e-5
  gammas = get_aranges(gamma_low, gamma_up, n_steps)
  threshs = get_aranges(1e-9, 1e-5, n_steps)
  A, pi, mu, sigma_sqr = init_params(D, K)
  print('Initial NLL:', eval_NLL(X))
  for i in range(n_steps):
    print('iteration', i)
    # A, pi, mu, sigma_sqr = EM(X, K, gammas[i], A, pi, mu, sigma_sqr, threshs[i], A_mode=A_mode)
    print('mu: mean={:.3e}/ std={:.3e}'.format(mu.mean(), mu.std()))
    print('sigma_sqr: min={:.3e} / mean={:.3e}/ std={:.3e}'.format(sigma_sqr.min(), sigma_sqr.mean(), sigma_sqr.std()))
    Y = X.dot(A.T)
    print('NLL (Y):', eval_NLL(Y))
    X = gaussianize_1d(Y, pi, mu, sigma_sqr)
    print('NLL:', eval_NLL(X))
    print()
    plt.hist2d(X[:,0], X[:,1], bins=[100,100])
    plt.savefig('figs/hist2d_mode{}_K{}_gamma{}_iter{}.png'.format(A_mode, K, gamma_up, i))
    plt.clf()


if __name__ == '__main__':
  # test()
  # gen_data()
  X = np.load('GM_2d.npy')
  fit(X)
  
