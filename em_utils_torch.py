import numpy as np
import torch
from scipy.stats import ortho_group, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from time import time

# local imports
from data.dataset_mixture import GaussianMixture

import pdb

VERBOSE = 0
TIME = 0
CHECK_OBJ = 1

SMALL = 1e-10
EPS = 5e-7

DTYPE = torch.DoubleTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_params(D, K, mu_low, mu_up):
  # rotation matrix
  # TODO: which way to initialize?
  # A = ortho_group.rvs(D)
  A = torch.eye(D)
  # prior - uniform
  pi = torch.ones([D, K]) / K
  # GM means
  mu = torch.tensor([np.arange(mu_low, mu_up, (mu_up-mu_low)/K) for _ in range(D)])
  # GM variances
  sigma_sqr = torch.ones([D, K])
  if VERBOSE:
    print('A:', A.view(-1))
    print('mu: mean={} / std={}'.format(mu.mean(), mu.std()))

  A = A.type(DTYPE).to(device)
  pi = pi.type(DTYPE).to(device)
  mu = mu.type(DTYPE).to(device)
  sigma_sqr = sigma_sqr.type(DTYPE).to(device)

  return A, pi, mu, sigma_sqr


def EM(X, K, gamma, A, pi, mu, sigma_sqr, threshold=5e-5, A_mode='GA',
       max_em_steps=30, n_gd_steps=20):
  if type(X) is not torch.Tensor:
    X = torch.tensor(X)
  X = X.type(DTYPE).to(device)

  N, D = X.shape

  END = lambda dA, dsigma_sqr: (dA + dsigma_sqr) < threshold
  
  niters = 0
  dA, dsigma_sqr = 10, 10
  avg_time = {}
  time_A, time_E, time_GA, time_Y = [], [], [], []
  grad_norms = []
  objs = []
  while (not END(dA, dsigma_sqr)) and niters < max_em_steps:
    niters += 1
    A_prev, sigma_sqr_prev = A.clone(), sigma_sqr.clone()
    objs += [],

    def E(pi, mu, sigma_sqr, Y=None):
      # E-step - update posterior counts
      if Y is None:
        Y = A.matmul(X.T) # D x N

      diff_square = (Y.T.view(N, D, 1) - mu)**2
      exponents = diff_square / sigma_sqr
      exp = torch.exp(-0.5 * exponents)
      w = exp * pi / (sigma_sqr**0.5)
      Ksum = w.sum(-1, keepdim=True)
      Ksum[Ksum<SMALL] = SMALL
      w = w / Ksum

      w_sumN = w.sum(0)
      w_sumNK = w_sumN.sum(-1)
      w_sumN[w_sumN < SMALL] = SMALL
      w_sumNK[w_sumNK < SMALL] = SMALL
      return Y, w, w_sumN, w_sumNK

    def update_pi_mu_sigma(X, A, w_sumN, w_sumNK, Y=None):
      if Y is None:
        Y = A.matmul(X.T) # D x N
      pi = w_sumN / w_sumNK.view(-1, 1)
      mu = (Y.T.view(N, D, 1) * w).sum(0) / w_sumN
      diff_square = (Y.T.view(N, D, 1) - mu)**2
      sigma_sqr = (w * diff_square).sum(0) / w_sumN
      mu[torch.abs(mu) < SMALL] = SMALL 
      sigma_sqr[sigma_sqr < SMALL] = SMALL
      return pi, mu, sigma_sqr

    def get_objetive(X, A, pi, mu, sigma_sqr, w, Y=None):
      if Y is None:
        Y = A.matmul(X.T)
      Y = Y.T
      exponents = (Y.view(N, D, 1) - mu)**2 / sigma_sqr
      exp = torch.exp(-0.5 * exponents) / (2*np.pi)**(D/2)
      log = torch.log(exp * pi)
      obj = (w * log).sum() / N + torch.log(torch.abs(torch.det(A)))
      return obj

    if TIME:
      e_start = time()
    Y, w, w_sumN, w_sumNK = E(pi, mu, sigma_sqr)
    if TIME:
      time_E += time() - e_start,

    # M-step
    if A_mode == 'GA': # gradient ascent
      if CHECK_OBJ:
        objs[-1] += get_objetive(X, A, pi, mu, sigma_sqr, w),
      for i in range(n_gd_steps):
        ga_start = time()
        if VERBOSE: print(A.view(-1))
        
        if False: # TODO: should I update w per GD step?
          Y, w, w_sumN, w_sumNK = E(pi, mu, sigma_sqr)

        pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w_sumN, w_sumNK)

        if TIME:
          y_start = time()
        Y = A.matmul(X.T)
        if TIME:
          time_Y += time() - y_start,

        scaled = (-Y.T.view(N, D, 1) + mu) / sigma_sqr
        weighted_X = (w * scaled).view(N, D, 1, K) * X.view(N, 1, D, 1)
        B = weighted_X.sum(0).sum(-1) / N

        if TIME:
          a_start = time()
        grad = gamma * (torch.inverse(A).T + B)
        # A += grad
        A -= grad
        _, ss, _ = torch.svd(A)
        A /= ss[0]
        if TIME:
          time_A += time() - a_start,
          time_GA += time() - ga_start,
        grad_norms += torch.norm(grad).item(),

        if CHECK_OBJ:
          obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
          objs[-1] += obj,
          print('iter {}: obj= {:.5f}'.format(i, obj))

    elif A_mode == 'ICA':
      # pdb.set_trace()
      cov = X.T.matmul(X) / len(X)
      cnt = 0
      n_tries = 20
      while cnt < n_tries:
        # multiple
        try:
          ica = FastICA()
          Y = ica.fit_transform(X.cpu())
          Y = to_tensor(Y).T
          _, ss, _ = np.linalg.svd(ica.mixing_)
          cnt = 2*n_tries
          Y *= ss[0]
        except:
          cnt += 1
      if cnt != 2*n_tries:
        print('ICA failed. Use random.')
        A = to_tensor(ortho_group.rvs(D))
        Y = A.matmul(X)

      # NOTE: passing in Y as an argument since A is not explicitly calculated.
      Y, w, w_sumN, w_sumNK = E(pi, mu, sigma_sqr, Y=Y)
      pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w_sumN, w_sumNK, Y=Y)
      
      # NOTE: returning directly since no EM iteration is required.
      return Y.T, A, pi, mu, sigma_sqr, avg_time

    elif A_mode == 'CF': # closed form
      raise NotImplementedError("mode CF: not implemented yet.")
           
    # difference from the previous iterate
    dA, dsigma_sqr = torch.norm(A - A_prev), torch.norm(sigma_sqr.view(-1) - sigma_sqr_prev.view(-1))

  print('#{}: dA={:.3e} / dsigma_sqr={:.3e}'.format(niters, dA, dsigma_sqr))
  if VERBOSE:
    print('A:', A.view(-1))

  if TIME:
    avg_time = {
      'A': np.array(time_A).mean(),
      'E': np.array(time_E).mean(),
      'GA': np.array(time_GA).mean(),
      'Y': np.array(time_Y).mean(),
    }
  return X, A, pi, mu, sigma_sqr, grad_norms, objs, avg_time 

def gaussianize_1d(X, pi, mu, sigma_sqr):
   N, D = X.shape

   scaled = (X.view(N, D, 1) - mu) / sigma_sqr**0.5
   scaled = scaled.cpu()
   cdf = norm.cdf(scaled)
   # remove outliers 
   cdf[cdf<EPS] = EPS
   cdf[cdf>1-EPS] = 1 - EPS

   new_distr = (pi.cpu().numpy() * cdf).sum(-1)
   new_X = norm.ppf(new_distr)
   new_X = to_tensor(new_X)
   return new_X

def eval_NLL(X):
  # evaluate the negative log likelihood of X coming from a standard normal.
  exponents = 0.5 * (X**2).sum(1)
  if exponents.max() > 10:
    print("NLL: exponents large.")
  return 0.5 * X.shape[1] * np.log(2*np.pi) + exponents.mean()

def get_aranges(low, up, n_steps):
  if low == up:
    return np.array([low] * n_steps)
  return np.arange(low, up, (up-low)/n_steps)[::-1]

def plot_hist(data, fimg):
  if type(data) is torch.Tensor:
    data = data.cpu().numpy()

  plt.figure(figsize=[8,8])
  plt.hist2d(data[:,0], data[:,1], bins=[100,100])
  plt.xlim([-2.5, 2.5])
  plt.ylim([-2.5, 2.5])
  plt.savefig(fimg)
  plt.clf()
  plt.close()

def to_tensor(data):
  return torch.tensor(data).type(DTYPE).to(device)


def gen_data(scale):
  dlen = 100000
  dset = GaussianMixture(dlen, scale)
  data = []
  for _ in range(dlen):
    data += dset.__getitem__(0),
  data = np.array(data)
  np.save('GM_2d_scale{}.npy'.format(scale), data)

if __name__ == '__main__':
  scale = 2
  gen_data()

