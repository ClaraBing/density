import numpy as np
import torch
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
  print('A:', A.view(-1))
  print('mu: mean={} / std={}'.format(mu.mean(), mu.std()))

  A = A.type(DTYPE).to(device)
  pi = pi.type(DTYPE).to(device)
  mu = mu.type(DTYPE).to(device)
  sigma_sqr = sigma_sqr.type(DTYPE).to(device)

  return A, pi, mu, sigma_sqr


def EM(X, K, gamma, A, pi, mu, sigma_sqr, threshold=5e-5, A_mode='GA'):
  if type(X) is not torch.Tensor:
    X = torch.tensor(X)
  X = X.type(DTYPE).to(device)

  N, D = X.shape
  max_em_steps = 30
  n_gd_steps = 20

  END = lambda dA, dsigma_sqr: (dA + dsigma_sqr) < threshold
  
  niters = 0
  dA, dsigma_sqr = 10, 10
  while (not END(dA, dsigma_sqr)) and niters < max_em_steps:
    niters += 1
    A_prev, sigma_sqr_prev = A.clone(), sigma_sqr.clone()

    def E(pi, mu, sigma_sqr):
      # E-step - update posterior counts
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

    def update_pi_mu_sigma(X, A, w_sumN, w_sumNK):
      Y = A.matmul(X.T) # D x N
      pi = w_sumN / w_sumNK.view(-1, 1)
      mu = (Y.T.view(N, D, 1) * w).sum(0) / w_sumN
      diff_square = (Y.T.view(N, D, 1) - mu)**2
      sigma_sqr = (w * diff_square).sum(0) / w_sumN
      mu[torch.abs(mu) < SMALL] = SMALL 
      sigma_sqr[sigma_sqr < SMALL] = SMALL
      return pi, mu, sigma_sqr

    Y, w, w_sumN, w_sumNK = E(pi, mu, sigma_sqr)

    # M-step
    if A_mode == 'GA': # gradient ascent
      for _ in range(n_gd_steps):
        if VERBOSE: print(A.view(-1))
        
        if False: # TODO: should I update w per GD step?
          Y, w, w_sumN, w_sumNK = E(pi, mu, sigma_sqr)

        pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w_sumN, w_sumNK)

        Y = A.matmul(X.T)
        scaled = (-Y.T.view(N, D, 1) + mu) / sigma_sqr
        weighted_X = (w * scaled).view(N, D, 1, K) * X.view(N, 1, D, 1)
        B = weighted_X.sum(0).sum(-1) / N

        A += gamma * (torch.inverse(A).T + B)

    elif A_mode == 'CF': # closed form
      update_pi_mu_sigma()

      if VERBOSE: print(A.view(-1))
      det = np.linalg.det(A)
      if np.abs(det) > 0.05:
        # invertible A: adjugate = det * inv.T
        cofs = det * np.linalg.inv(A).T
      else:
        cofs = np.zeros_like(A)
        for i in range(D):
          for j in range(D):
            cofs[i,j] = cof(A, i, j)

      new_A = np.zeros_like(A)
      weights = w / sigma_sqr
      for i in range(D):
        common_sums = A[i].view(-1, 1) * X.T # D x N
        total_sum = common_sums.sum(0)
        for j in range(D):
          j_common_sum = total_sum - common_sums[j]
          cof_sum = A[i].dot(cofs[i]) - A[i,j]*cofs[i,j]

          c1, c2a, c2b, c3 = 0, 0, 0, 0
          for k in range(K):
            t1 = (weights[:,j,k] * (X[:,j]**2)).sum()
            t2 = (weights[:,i,k] * X[:,j] * (j_common_sum - mu[i,k])).sum()
            c1 += t1
            c2a += t2
            c2b += t1
            c3 += t2

          c1 *= cofs[i, j]
          c2 = cofs[i,j] * c2a + cof_sum * c2b
          c3 = cof_sum * c3 - N*cofs[i,j]
          
          tmp = np.sqrt(c2**2 - 4*c1*c3)
          if np.abs(c1) < SMALL:
            new_A[i,j] = A[i,j]
          else:
            new_A[i,j] = 0.5 * (-c2 + tmp) / c1

      A = new_A
            
    # difference from the previous iterate
    dA, dsigma_sqr = torch.norm(A - A_prev), torch.norm(sigma_sqr.view(-1) - sigma_sqr_prev.view(-1))

  print('#{}: dA={:.3e} / dsigma_sqr={:.3e}'.format(niters, dA, dsigma_sqr))
  print('A:', A.view(-1))
  return X, A, pi, mu, sigma_sqr

def gaussianize_1d(X, pi, mu, sigma_sqr):
   N, D = X.shape

   scaled = (X.view(N, D, 1) - mu) / sigma_sqr**0.5
   scaled = scaled.cpu()
   cdf = norm.cdf(scaled)
   cdf[cdf<EPS] = EPS
   cdf[cdf>1-EPS] = 1 - EPS
   new_distr = (pi.cpu().numpy() * cdf).sum(-1)
   # new_distr = np.maximum(new_distr, EPS)
   # new_distr = np.minimum(new_distr, 1 - EPS)
   new_X = norm.ppf(new_distr)
   new_X = torch.tensor(new_X).type(DTYPE).to(device)
   return new_X

def eval_NLL(X):
  # evaluate the negative log likelihood of X coming from a standard normal.
  exponents = 0.5 * (X**2).sum(1)
  if exponents.max() > 10:
    print("NLL: exponents large.")
  return 0.5 * X.shape[1] * np.log(2*np.pi) + exponents.mean()

def get_aranges(low, up, n_steps):
  return np.arange(low, up, (up-low)/n_steps)[::-1]

def cof(A, i, j):
  Am = np.zeros([A.shape[0]-1, A.shape[1]-1])
  Am[:i,:j] = A[:i, :j]
  Am[i:, :j] = A[i+1:, :j]
  Am[:i, j:] = A[:i, j+1:]
  Am[i:, j:] = A[i+1:, j+1:]
  return (-1)**(i+j) * np.linalg.det(Am)

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


def gen_data():
  dlen = 100000
  scale = 2
  dset = GaussianMixture(dlen, scale)
  data = []
  for _ in range(dlen):
    data += dset.__getitem__(0),
  data = np.array(data)
  np.save('GM_2d_scale{}.npy'.format(scale), data)

if __name__ == '__main__':
  # TODO: larger variance -> almost connected modes? 
  gen_data()

