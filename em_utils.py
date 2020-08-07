import numpy as np
from scipy.stats import ortho_group, norm

# local imports
# from data import get_loader
from data.dataset_mixture import GaussianMixture

import pdb

def EM(X, K, gamma, A=None, pi=None, mu=None, sigma=None, threshold=5e-5, A_mode='GD'):
  N, D = X.shape
  max_steps = 50

  # init
  if A is None:
    # rotation matrix
    # A = ortho_group.rvs(D)
    A = np.eye(D)
    # prior
    pi = np.ones([D, K]) / K
    # GM means
    mu = np.random.randn(D, K)
    # GM variances
    sigma = np.ones([D, K])
  
  END = lambda dA, dsigma: (dA + dsigma) < threshold
  
  niters = 0
  dA, dsigma = 10, 10
  while (not END(dA, dsigma)) and niters < max_steps:
    niters += 1
    A_prev, sigma_prev = A.copy(), sigma.copy()
  
    # E-step - update posterior counts
    w = np.zeros([N, D, K])
    Y = A.dot(X.T) # D x N
    diff_square = np.zeros([N, D, K])
    exponents = np.zeros([N, D, K])
    for d in range(D):
      cur_y = Y[d]
      for k in range(K):
        diff_square[:, d, k] = (cur_y - mu[d,k])**2 
        if sigma[d,k] != 0:
          exponents[:, d, k] = diff_square[:, d, k] / sigma[d,k]**2
          w[:, d, k] = pi[d,k] * np.exp(-0.5*exponents[:, d, k])
        else:
          print('w_[:, {}, {}] is 0'.format(d, k))
          pdb.set_trace()
          # w[:, d, k] = 0
      row_sum = w[:, d].sum(-1).reshape(-1, 1)
      row_sum[row_sum==0] = 1 # avoid divide-by-zero
      w[:, d] /= row_sum
  
    # M-step
    w_sumN = w.sum(0)
    w_sumNK = w_sumN.sum(-1)
    for d in range(D):
      for k in range(K):
        pi[d,k] = w_sumN[d,k] / w_sumNK[d]
        if w_sumN[d, k] != 0:
          mu[d,k] = w[:, d, k].dot(Y[d]) / w_sumN[d,k]
          sigma[d, k] = w[:, d, k].dot(diff_square[:, d, k]) / w_sumN[d,k]
        else:
          mu[d,k] = 0
          sigma[d,k] = 1

    B = np.zeros([D, D])
    weights = w * exponents
    for d in range(D):
      for k in range(K):
        scaled = (Y[d] - mu[d,k]) / sigma[d,k]**2
        weighted_X = (w[:, d, k] * scaled).reshape(-1,1) * X
        # weighted_X = (weights[:, d, k].reshape(-1, 1)) * X
        B[d] = weighted_X.sum(0)

    if A_mode == 'GD':
      A += gamma * (np.linalg.inv(A).T + B)
    elif A_mode == 'CF': # closed form
      print(A.reshape(-1))
      cofs = np.zeros_like(A)
      for i in range(D):
        for j in range(D):
          cofs[i,j] = cof(A, i, j)

      new_A = np.zeros_like(A)
      for i in range(D):
        common_sums = A[j].reshape(-1, 1) * X.T
        total_sum = common_sums.sum(0)
        for j in range(D):
          j_common_sum = total_sum - common_sums[j]
          cof_common_term = A[i].dot(cofs[i]) - A[i,j]*cofs[i,j]

          c1, c2a, c2b, c3 = 0, 0, 0, 0
          for k in range(K):
            t1 = (w[:,j,k] * (X[:,j]**2) / sigma[i,k]**2).sum()
            t2 = (w[:,i,k] * X[:,j] * (j_common_sum - mu[i,k]) / sigma[i,k]**2).sum()
            c1 += t1
            c2a += t2
            c2b += t1
            c3 += t2

          c1 *= cofs[i, j]
          c2a *= cofs[i, j]
          c2b *= cof_common_term
          c2 = c2a + c2b
          c3 *= cof_common_term
          c3 -= cofs[i,j]
          
          tmp = np.sqrt(c2**2 - 4*c1*c3)
          if c1 == 0:
            new_A[i,j] = A[i,j]
          else:
            new_A[i,j] = 0.5 * (-c2 + tmp) / c1
      A = new_A
          
    # difference from the previous iterate
    dA, dsigma = np.linalg.norm(A - A_prev), np.linalg.norm(sigma.reshape(-1) - sigma_prev.reshape(-1))
    # wandb.log({'niters'})
    # print('#{}: dA={:.3e} / dsigma={:.3e}'.format(niters, dA, dsigma))

  print('#{}: dA={:.3e} / dsigma={:.3e}'.format(niters, dA, dsigma))
  return A, pi, mu, sigma

def gaussianize_1d(X, pi, mu, sigma):
   N, D = X.shape
   K = mu.shape[1]
   new_X = np.zeros_like(X)
   for d in range(D):
     cur_sum = np.zeros(N)
     for k in range(K):
       scaled = (X[:, d] - mu[d,k]) / sigma[d,k]
       # scaled[np.isnan(scaled)] = 0
       cur_sum += pi[d,k] * norm.cdf(scaled)
     new_X[:, d] += norm.ppf(cur_sum)
   return new_X

def eval_NLL(X):
  # evaluate the negative log likelihood of X coming from a standard normal.
  exponents = 0.5 * (X**2).sum(1)
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


