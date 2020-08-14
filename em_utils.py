import numpy as np
from scipy.stats import ortho_group, norm

# local imports
from data.dataset_mixture import GaussianMixture

import pdb

VERBOSE = 0
SMALL = 1e-10

def init_params(D, K, mu_low, mu_up):
  # rotation matrix
  # TODO: which way to initialize?
  A = ortho_group.rvs(D)
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
  max_em_steps = 50
  n_gd_steps = 10

  END = lambda dA, dsigma_sqr: (dA + dsigma_sqr) < threshold
  
  niters = 0
  dA, dsigma_sqr = 10, 10
  while (not END(dA, dsigma_sqr)) and niters < max_em_steps:
    niters += 1
    A_prev, sigma_sqr_prev = A.copy(), sigma_sqr.copy()

    def E():
      # E-step - update posterior counts
      w = np.zeros([N, D, K])
      Y = A.dot(X.T) # D x N
      diff_square = np.zeros([N, D, K])
      exponents = np.zeros([N, D, K])
      for d in range(D):
        cur_y = Y[d]
        for k in range(K):
          diff_square[:, d, k] = (cur_y - mu[d,k])**2 
          if sigma_sqr[d,k] != 0:
            exponents[:, d, k] = diff_square[:, d, k] / np.maximum(sigma_sqr[d,k], SMALL)
            w[:, d, k] = pi[d,k] * np.exp(-0.5*exponents[:, d, k]) / np.maximum(sigma_sqr[d,k]**0.5, SMALL)
        Ksum = w[:, d].sum(-1)
        if False:
          mask_good = np.abs(Ksum) > 1e-3
          mask_bad = np.abs(Ksum) <= 1e-3
          w[:, d][mask_good] /= np.maximum(Ksum[mask_good].reshape(-1,1), SMALL)
          w[:, d][mask_bad] = 1/K
        if True:
          w[:, d] /= np.maximum(Ksum.reshape(-1,1),  SMALL)
      w_sumN = w.sum(0)
      w_sumNK = w_sumN.sum(-1)
      return Y, w, w_sumN, w_sumNK, diff_square, exponents

    def update_intermediate(X, w, A):
      Y = A.dot(X.T) # D x N
      diff_square = np.zeros([N, D, K])
      exponents = np.zeros([N, D, K])
      for d in range(D):
        cur_y = Y[d]
        for k in range(K):
          diff_square[:, d, k] = (cur_y - mu[d,k])**2 
          if sigma_sqr[d,k] != 0:
            exponents[:, d, k] = diff_square[:, d, k] / np.maximum(sigma_sqr[d,k], SMALL)
      return Y, diff_square, exponents

    def update_pi_mu_sigma():
      for d in range(D):
        cur_y = Y[d]
        for k in range(K):
          pi[d,k] = w_sumN[d,k] / np.maximum(w_sumNK[d], SMALL)
          if w_sumN[d, k] != 0:
            mu[d,k] = w[:, d, k].dot(Y[d]) / np.maximum(w_sumN[d,k], SMALL)
            diff_square[:, d, k] = (cur_y - mu[d,k])**2 
            sigma_sqr[d, k] = w[:, d, k].dot(diff_square[:, d, k]) / np.maximum(w_sumN[d,k], SMALL)

    Y, w, w_sumN, w_sumNK, diff_square, exponents = E()

    # M-step
    if A_mode == 'GA': # gradient ascent
      for _ in range(n_gd_steps):
        if VERBOSE: print(A.reshape(-1))
        
        if False: # TODO: should I update w per GD step?
          Y, w, w_sumN, w_sumNK, diff_square, exponents = E()

        if True:
          Y, diff_square, exponents = update_intermediate(X, w, A)

        update_pi_mu_sigma()
        # pdb.set_trace()
    
        B = np.zeros([D, D])
        weights = w * exponents
        for d in range(D):
          for k in range(K):
            scaled = (- Y[d] + mu[d,k]) / np.maximum(sigma_sqr[d,k], SMALL)
            weighted_X = (w[:, d, k] * scaled).reshape(-1,1) * X
            B[d] += weighted_X.sum(0)
        B /= N

        A += gamma * (np.linalg.inv(A).T + B)

    elif A_mode == 'CF': # closed form
      update_pi_mu_sigma()

      if VERBOSE: print(A.reshape(-1))
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
      weights = w / np.maximum(sigma_sqr, SMALL)
      for i in range(D):
        common_sums = A[i].reshape(-1, 1) * X.T # D x N
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
          if c1 < SMALL:
            new_A[i,j] = A[i,j]
          else:
            new_A[i,j] = 0.5 * (-c2 + tmp) / c1

      A = new_A
            
    # difference from the previous iterate
    dA, dsigma_sqr = np.linalg.norm(A - A_prev), np.linalg.norm(sigma_sqr.reshape(-1) - sigma_sqr_prev.reshape(-1))

  print('#{}: dA={:.3e} / dsigma_sqr={:.3e}'.format(niters, dA, dsigma_sqr))
  print('A:', A.reshape(-1))
  return A, pi, mu, sigma_sqr

def gaussianize_1d(X, pi, mu, sigma_sqr):
   N, D = X.shape
   K = mu.shape[1]
   new_X = np.zeros_like(X)
   for i in range(D):
     cur_sum = np.zeros(N)
     for k in range(K):
       scaled = (X[:, i] - mu[i,k]) / np.maximum(sigma_sqr[i,k]**0.5, SMALL)
       cur_sum += pi[i,k] * norm.cdf(scaled)
     new_X[:, i] += norm.ppf(cur_sum)
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

def gen_data():
  dlen = 100000
  dset = GaussianMixture(dlen)
  data = []
  for _ in range(dlen):
    data += dset.__getitem__(0),
  data = np.array(data)
  pdb.set_trace()



