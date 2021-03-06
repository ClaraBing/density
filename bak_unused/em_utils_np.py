import numpy as np
from scipy.stats import ortho_group, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# local imports
from data.dataset_mixture import GaussianMixture

import pdb

VERBOSE = 0
SMALL = 1e-10
EPS = 5e-7

def init_params(D, K, mu_low, mu_up):
  # rotation matrix
  # TODO: which way to initialize?
  # A = ortho_group.rvs(D)
  A = np.eye(D)
  # prior - uniform
  pi = np.ones([D, K]) / K
  # GM means
  mu = np.array([np.arange(mu_low, mu_up, (mu_up-mu_low)/K) for _ in range(D)])
  # GM variances
  sigma_sqr = np.ones([D, K])
  print('A:', A.reshape(-1))
  print('mu: mean={} / std={}'.format(mu.mean(), mu.std()))

  return A, pi, mu, sigma_sqr


def EM(X, K, gamma, A, pi, mu, sigma_sqr, threshold=5e-5, A_mode='GA'):
  N, D = X.shape
  max_em_steps = 30
  n_gd_steps = 20

  END = lambda dA, dsigma_sqr: (dA + dsigma_sqr) < threshold
  
  niters = 0
  dA, dsigma_sqr = 10, 10
  while (not END(dA, dsigma_sqr)) and niters < max_em_steps:
    niters += 1
    A_prev, sigma_sqr_prev = A.copy(), sigma_sqr.copy()

    def E(pi, mu, sigma_sqr):
      # E-step - update posterior counts
      Y = A.dot(X.T) # D x N

      diff_square = (Y.transpose(1,0).reshape(N, D, 1) - mu)**2
      exponents = diff_square / sigma_sqr
      exp = np.exp(-0.5 * exponents)
      w = exp * pi / (sigma_sqr**0.5)
      Ksum = np.maximum(w.sum(-1, keepdims=1), SMALL)
      w = w / Ksum

      w_sumN = np.maximum(w.sum(0), SMALL)
      w_sumNK = np.maximum(w_sumN.sum(-1), SMALL)
      return Y, w, w_sumN, w_sumNK

    def update_pi_mu_sigma(X, A, w_sumN, w_sumNK):
      Y = A.dot(X.T) # D x N
      pi = w_sumN / w_sumNK.reshape(-1, 1)
      mu = (Y.transpose(1,0).reshape(N, D, 1) * w).sum(0) / w_sumN
      diff_square = (Y.transpose(1,0).reshape(N, D, 1) - mu)**2
      sigma_sqr = (w * diff_square).sum(0) / w_sumN
      mu[np.abs(mu) < SMALL] = SMALL 
      sigma_sqr = np.maximum(sigma_sqr, SMALL)
      return pi, mu, sigma_sqr

    Y, w, w_sumN, w_sumNK = E(pi, mu, sigma_sqr)

    # M-step
    if A_mode == 'GA': # gradient ascent
      for _ in range(n_gd_steps):
        if VERBOSE: print(A.reshape(-1))
        
        if False: # TODO: should I update w per GD step?
          Y, w, w_sumN, w_sumNK = E(pi, mu, sigma_sqr)


        pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w_sumN, w_sumNK)

        Y = A.dot(X.T)
        scaled = (-Y.T.reshape(N, D, 1) + mu) / sigma_sqr
        weighted_X = (w * scaled).reshape(N, D, 1, K) * X.reshape(N, 1, D, 1)
        B = weighted_X.sum(0).sum(-1) / N

        A += gamma * (np.linalg.inv(A).T + B)
        _, ss, _ = np.linalg.svd(A)
        A /= ss[0]


    elif A_mode == 'CF': # closed form
      raise NotImplementedError("mode CF: not implemented yet.")
            
    # difference from the previous iterate
    dA, dsigma_sqr = np.linalg.norm(A - A_prev), np.linalg.norm(sigma_sqr.reshape(-1) - sigma_sqr_prev.reshape(-1))

  print('#{}: dA={:.3e} / dsigma_sqr={:.3e}'.format(niters, dA, dsigma_sqr))
  print('A:', A.reshape(-1))
  return X, A, pi, mu, sigma_sqr

def gaussianize_1d(X, pi, mu, sigma_sqr):
   N, D = X.shape

   scaled = (X.reshape(N, D, 1) - mu) / sigma_sqr**0.5
   cdf = norm.cdf(scaled)
   cdf = np.maximum(cdf, EPS)
   cdf=  np.minimum(cdf, 1 - EPS)
   new_distr = (pi * cdf).sum(-1)
   # new_distr = np.maximum(new_distr, EPS)
   # new_distr = np.minimum(new_distr, 1 - EPS)
   new_X = norm.ppf(new_distr)
   return new_X

def eval_NLL(X):
  # evaluate the negative log likelihood of X coming from a standard normal.
  exponents = 0.5 * (X**2).sum(1)
  if exponents.max() > 10:
    print("NLL: exponents large.")
  return 0.5 * X.shape[1] * np.log(2*np.pi) + exponents.mean()

def get_aranges(low, up, n_steps):
  return np.arange(low, up, (up-low)/n_steps)[::-1]

def plot_hist(data, fimg):
  plt.figure(figsize=[8,8])
  plt.hist2d(data[:,0], data[:,1], bins=[100,100])
  plt.xlim([-2.5, 2.5])
  plt.ylim([-2.5, 2.5])
  plt.savefig(fimg)
  plt.clf()
  plt.close()

def gen_data(dlen=100000, scale=4, token=''):
  dset = GaussianMixture(dlen, scale)
  data = []
  for _ in range(dlen):
    data += dset.__getitem__(0),
  data = np.array(data)
  np.save('GM_2d_scale{}{}.npy'.format(scale, token), data)

if __name__ == '__main__':
  scale = 4
  token = '_test'
  dlen = 20000
  gen_data(dlen, scale, token)

