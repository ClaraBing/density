import numpy as np
from scipy.stats import ortho_group

import pdb

# pseudo-code

def EM(X, K, gamma):
  N, D = X.shape

  # init
  # rotation matrix
  A = ortho_group.rvs(D)
  # prior
  pi = np.ones([D, K]) / K
  # GM means
  mu = np.random.randn(D, K)
  # GM variances
  sigma = np.ones([D, K])
  # posterior counts
  w = np.zeros([N, D, K])
  
  threshold = 5e-5
  END = lambda dA, dsigma: (dA + dsigma) < threshold
  
  niters = 0
  dA, dsigma = 10, 10
  while not END(dA, dsigma):
    niters += 1
    A_prev, sigma_prev = A.copy(), sigma.copy()
  
    # E-step - update posterior counts
    w = np.zeros([N, D, K])
    y = A.dot(X.T) # D x N
    diff_square = np.zeros([N, D, K])
    exponents = np.zeros([N, D, K])
    for d in range(D):
      cur_y = y[d]
      for k in range(K):
        diff_square[:, d, k] = (cur_y - mu[d,k])**2 
        if sigma[d,k] != 0:
          exponents[:, d, k] = diff_square[:, d, k] / sigma[d,k]**2
          w[:, d, k] = pi[d,k] * np.exp(-0.5*exponents[:, d, k])
        else:
          w[:, d, k] = 0
      w[:, d] /= w[:, d].sum(-1).reshape(-1, 1)
  
    # M-step
    w_sumN = w.sum(0)
    total_counts = w_sumN.sum()
    for d in range(D):
      for k in range(K):
        pi[d,k] = w_sumN[d,k] / total_counts
        if w_sumN[d, k] != 0:
          mu[d,k] = w[:, d, k].dot(y[d]) / w_sumN[d,k]
          sigma[d, k] = w[:, d, k].dot(diff_square[:, d, k]) / w_sumN[d,k]
        else:
          mu[d,k] = 0
          sigma[d,k] = 0

    B = np.zeros([D, D])
    weights = w * exponents
    for d in range(D):
      for k in range(K):
        weighted_X = (weights[:, d, k].reshape(-1, 1)) * X
        B[d] = weighted_X.sum(0)

    A += gamma * (np.linalg.inv(A).T + B)
  
    # difference from the previous iterate
    dA, dsigma = np.linalg.norm(A - A_prev), np.linalg.norm(sigma.reshape(-1) - sigma_prev.reshape(-1))
    print('#{}: dA={:.3e} / dsigma={:.3e}'.format(niters, dA, dsigma))

  return A, pi, mu, sigma

def test():
  N = 1000
  D = 2
  K = 10
  gamma = 1e-5
  X = np.random.randn(N, D)
  A, pi, mu, sigma = EM(X, K, gamma)

if __name__ == '__main__':
  test()
  
