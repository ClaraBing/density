import numpy as np
import torch
import torch.optim as optim
import torch.distributions as dists
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
TIME = 1
CHECK_OBJ = 1

SMALL = 1e-10
EPS = 5e-8
EPS_GRAD = 1e-2
SINGULAR_SMALL = 1e-2

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

def update_EM(X, K, gamma, A, pi, mu, sigma_sqr, threshold=5e-5,
       A_mode='GA', grad_mode='GA',
       max_em_steps=30, n_gd_steps=20):

  if type(X) is not torch.Tensor:
    X = torch.tensor(X)
  X = X.type(DTYPE).to(device)

  N, D = X.shape

  END = lambda dA, dsigma_sqr: (dA + dsigma_sqr) < threshold

  Y = None
  niters = 0
  dA, dsigma_sqr = 10, 10
  ret_time = {'E':[], 'obj':[]}
  grad_norms, objs= [], []

  if A_mode == 'random':
    A = ortho_group.rvs(D)
    A = to_tensor(A)
  elif A_mode == 'ICA':
    cov = X.T.matmul(X) / len(X)
    cnt = 0
    n_tries = 20
    while cnt < n_tries:
      try:
        ica = FastICA()
        _ = ica.fit_transform(X.cpu())
        Aorig = ica.mixing_

        # avoid numerical instability
        U, ss, V = np.linalg.svd(Aorig)
        ss /= ss[0]
        ss[ss < SINGULAR_SMALL] = SINGULAR_SMALL
        Aorig = (U * ss).dot(V)

        A = np.linalg.inv(Aorig)
        _, ss, _ = np.linalg.svd(A)
        A = to_tensor(A / ss[0])
        cnt = 2*n_tries
      except:
        cnt += 1
    if cnt != 2*n_tries:
      print('ICA failed. Use random.')
      A = to_tensor(ortho_group.rvs(D))

  while (not END(dA, dsigma_sqr)) and niters < max_em_steps:
    niters += 1
    A_prev, sigma_sqr_prev = A.clone(), sigma_sqr.clone()
    objs += [],

    if TIME: e_start = time()
    Y, w, w_sumN, w_sumNK = E(X, A, pi, mu, sigma_sqr, Y=Y)
    if TIME: ret_time['E'] += time() - e_start,

    # M-step
    if A_mode == 'ICA' or A_mode == 'None':
      pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w, w_sumN, w_sumNK)
      obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
      objs[-1] += obj,

    if A_mode == 'CF': # gradient ascent
      if CHECK_OBJ:
        objs[-1] += get_objetive(X, A, pi, mu, sigma_sqr, w),

      for i in range(n_gd_steps):
        cf_start = time()
        if VERBOSE: print(A.view(-1))
        
        pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w, w_sumN, w_sumNK)

        if TIME: a_start = time()
        if grad_mode == 'CF1':
          A = set_grad_zero(X, A, w, mu, sigma_sqr)
          A = A.T
        elif grad_mode == 'CF2':
          cofs = get_cofactors(A)
          det = torch.det(A)
          if det < 0: # TODO: ignore neg det for now
            cofs = cofs * -1

          newA = A.clone()
          for i in range(D):
            for j in range(D):
              t1 = (w[:, i] * X[:,j,None]**2 / sigma_sqr[i]).sum() / N
              diff = (Y[i] - A[i,j] * X[:, j])[:, None] - mu[i]
              t2 = (w[:, i] * X[:,j,None] * diff / sigma_sqr[i]).sum() / N
              c1 = t1 * cofs[i,j]
              c2 = t1 * (det - A[i,j]*cofs[i,j]) + t2 * cofs[i,j]
              c3 = t2 * (det - A[i,j]*cofs[i,j]) - cofs[i,j]
              inner = c2**2 - 4*c1*c3
              if inner < 0:
                print('Problme at solving for A[{},{}]: no real sol.'.format(i,j))
                pdb.set_trace()
              if c1 == 0:
                sol = - c3 / c2
              else:
                sol = (inner**0.5 - c2) / (2*c1)
              if False:
                # check whether obj gets improved with each updated entry of A
                curr_A = newA.clone()
                curr_A[i,j] = sol
                curr_obj = get_objetive(X, curr_A, pi, mu, sigma_sqr, w)
              newA[i,j] = sol
          A = newA.double()

        # avoid numerical instability
        U, ss, V = torch.svd(A)
        ss = ss / ss[0]
        ss[ss < SINGULAR_SMALL] = SINGULAR_SMALL
        A = (U * ss).matmul(V)

        if TIME:
          if 'A' not in ret_time: ret_time['A'] = []
          ret_time['A'] += time() - a_start,
          if 'CF' not in ret_time: ret_time['CF'] = []
          ret_time['CF'] += time() - cf_start,

        if CHECK_OBJ:
          if TIME: obj_start = time()
          obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
          if TIME: ret_time['obj'] += time() - obj_start,
          objs[-1] += obj,
          if VERBOSE:
            print('iter {}: obj= {:.5f}'.format(i, obj))
        # pdb.set_trace()
      # pdb.set_trace()

    if A_mode == 'GA': # gradient ascent
      if CHECK_OBJ:
        objs[-1] += get_objetive(X, A, pi, mu, sigma_sqr, w),

      for i in range(n_gd_steps):
        ga_start = time()
        if VERBOSE: print(A.view(-1))
        
        pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w, w_sumN, w_sumNK)

        if TIME: a_start = time()
        # gradient steps
        grad, y_time = get_grad(X, A, w, mu, sigma_sqr)
        if TIME:
          if 'Y' not in ret_time:
            ret_time['Y'] = []
          ret_time['Y'] += y_time,
        if grad_mode == 'BTLS':
          # backtracking line search
          if TIME: obj_start = time()
          obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
          if TIME: ret_time['obj'] += time() - obj_start,

          beta, t, flag = 0.6, 1, True
          gnorm = torch.norm(grad)
          n_iter, ITER_LIM = 0, 10
          while flag and n_iter < ITER_LIM:
            n_iter += 1
            Ap = A + t * grad
            _, ss, _ = torch.svd(Ap)
            Ap /= ss[0]
            if TIME: obj_start = time()
            obj_p = get_objetive(X, Ap, pi, mu, sigma_sqr, w)
            if TIME: ret_time['obj'] += time() - obj_start,
            t *= beta
            base = obj - 0.5 * t * gnorm
            flag = obj_p < base
          gamma = t
          ret_time['btls_nIters'] += n_iter,
        elif grad_mode == 'perturb':
          # perturb
          perturb = A.std() * 0.1 * torch.randn(A.shape).type(DTYPE).to(device)
          perturbed = A + perturb
          perturbed_grad, _ = get_grad(X, perturbed, w, mu, sigma_sqr)

          grad_diff = torch.norm(grad - perturbed_grad)
          gamma = 1 /(EPS_GRAD + grad_diff) * 0.03

        grad_norms += torch.norm(grad).item(),
        A += gamma * grad

        _, ss, _ = torch.svd(A)
        A /= ss[0]

        if TIME:
          if 'A' not in ret_time: ret_time['A'] = []
          ret_time['A'] += time() - a_start,
          if 'GA' not in ret_time: ret_time['GA'] = []
          ret_time['GA'] += time() - ga_start,

        if CHECK_OBJ:
          if TIME: obj_start = time()
          obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
          if TIME: ret_time['obj'] += time() - obj_start,
          objs[-1] += obj,
          if VERBOSE:
            print('iter {}: obj= {:.5f}'.format(i, obj))
        # pdb.set_trace()
      # pdb.set_trace()

  if VERBOSE:
    print('#{}: dA={:.3e} / dsigma_sqr={:.3e}'.format(niters, dA, dsigma_sqr))
    print('A:', A.view(-1))

  if TIME:
    for key in ret_time:
      ret_time[key] = np.array(ret_time[key]) if ret_time[key] else 0

  # pdb.set_trace()
  return A, pi, mu, sigma_sqr, grad_norms, objs, ret_time 

# util funcs in EM steps
def E(X, A, pi, mu, sigma_sqr, Y=None):
  N, D = X.shape
  # E-step - update posterior counts
  if Y is None:
    Y = A.matmul(X.T) # D x N

  diff_square = (Y.T.unsqueeze(-1) - mu)**2
  exponents = diff_square / sigma_sqr
  exp = torch.exp(-0.5 * exponents)
  w = exp * pi / (sigma_sqr**0.5)
  Ksum = w.sum(-1, keepdim=True)
  Ksum[Ksum<SMALL] = SMALL
  w = w / Ksum

  w_sumN = w.sum(0)
  w_sumN[w_sumN < SMALL] = SMALL
  w_sumNK = w_sumN.sum(-1)
  w_sumNK[w_sumNK < SMALL] = SMALL
  return Y, w, w_sumN, w_sumNK

def update_pi_mu_sigma(X, A, w, w_sumN, w_sumNK, Y=None):
  N, D = X.shape
  if Y is None:
    Y = A.matmul(X.T) # D x N
  pi = w_sumN / w_sumNK.view(-1, 1)
  mu = (Y.T.unsqueeze(-1) * w).sum(0) / w_sumN
  diff_square = (Y.T.unsqueeze(-1) - mu)**2
  sigma_sqr = (w * diff_square).sum(0) / w_sumN
  mu[torch.abs(mu) < SMALL] = SMALL 
  sigma_sqr[sigma_sqr < SMALL] = SMALL
  return pi, mu, sigma_sqr

def get_objetive(X, A, pi, mu, sigma_sqr, w, Y=None):
  N, D = X.shape
  if Y is None:
    Y = A.matmul(X.T)
  Y = Y.T
  exponents = (Y.unsqueeze(-1) - mu)**2 / sigma_sqr
  exp = torch.exp(-0.5 * exponents) / (2*np.pi * sigma_sqr)**(1/2)
  # pi = torch.max(pi, to_tensor(np.array([0.])))
  prob = (exp * pi).sum(-1)
  log = torch.log(prob) 
  log[prob == 0] = 0 # mask out NaN
  obj = log.sum() / N + torch.log(torch.abs(torch.det(A)))
  if torch.isnan(obj):
    print('objective is NaN.') 
    pdb.set_trace()
  return obj.item()

def set_grad_zero(X, A, w, mu, sigma_sqr):
  N, D, K = w.shape
  Y = A.matmul(X.T)

  scaled = (-Y.T.unsqueeze(-1) + mu) / sigma_sqr
  weights = (w * scaled).sum(-1) / N
  B = weights.T.matmul(X)
  A_opt = torch.inverse(-B.T)
  return A_opt

def get_grad(X, A, w, mu, sigma_sqr):
  N, D, K = w.shape
  if TIME: y_start = time()
  Y = A.matmul(X.T)
  if TIME:
    y_time = time() - y_start
  else:
    y_time = 0

  scaled = (-Y.T.unsqueeze(-1) + mu) / sigma_sqr
  weights = (w * scaled).sum(-1) / N
  B = weights.T.matmul(X)
  grad = torch.inverse(A).T + B
  return grad, y_time


def gaussianize_1d_old(X, pi, mu, sigma_sqr):
   N, D = X.shape

   scaled = (X.unsqueeze(-1) - mu) / sigma_sqr**0.5
   scaled = scaled.cpu()
   cdf = norm.cdf(scaled)
   # remove outliers 
   cdf[cdf<EPS] = EPS
   cdf[cdf>1-EPS] = 1 - EPS

   new_distr = (pi.cpu().numpy() * cdf).sum(-1)
   new_X = norm.ppf(new_distr)
   new_X = to_tensor(new_X)
   if torch.isnan(new_X.max()):
     print("gaussianize_1d: new_X has nan.")
     pdb.set_trcae()
   return new_X

def eval_NLL(X):
  # evaluate the negative log likelihood of X coming from a standard normal.
  exponents = 0.5 * (X**2).sum(1)
  NLL = 0.5 * X.shape[1] * np.log(2*np.pi) + exponents.mean()
  return NLL.item()

normal_distribution = dists.Normal(0, 1)

# compute inverse normal CDF
def gaussianize_1d(X, pi, mu, sigma_sqr):
  mask_bound = 5e-8

  N, D = X.shape

  # for calculations please see: https://www.overleaf.com/6125358376rgmjjgdsmdmm
  scaled = (X.unsqueeze(-1) - mu) / sigma_sqr**0.5
  scaled = scaled.cpu()
  normal_cdf = to_tensor(norm.cdf(scaled))
  cdf = (pi * normal_cdf).sum(-1)
  log_cdfs = to_tensor(norm.logcdf(scaled))
  log_cdf = torch.logsumexp(torch.log(pi) + log_cdfs, dim=-1)
  log_sfs = to_tensor(norm.logcdf(-1*scaled))
  log_sf = torch.logsumexp(torch.log(pi) + log_sfs, dim=-1)

  # Approximate Gaussian CDF
  # inv(CDF) ~ np.sqrt(-2 * np.log(1-x)) #right, x -> 1
  # inv(CDF) ~ -np.sqrt(-2 * np.log(x)) #left, x -> 0
  # 1) Step1: invert good CDF
  cdf_mask = ((cdf > mask_bound) & (cdf < 1 - (mask_bound))).double()
  # Keep good CDF, mask the bad CDF values to 0.5(inverse(0.5)=0.)
  cdf_good = cdf * cdf_mask + 0.5 * (1. - cdf_mask)
  inverse_cdf = normal_distribution.icdf(cdf_good)

  # 2) Step2: invert BAD large CDF
  cdf_mask_right = (cdf >= 1. - (mask_bound)).double()
  # Keep large bad CDF, mask the good and small bad CDF values to 0.
  cdf_bad_right_log = log_sf * cdf_mask_right
  inverse_cdf += torch.sqrt(-2. * cdf_bad_right_log)

  # 3) Step3: invert BAD small CDF
  cdf_mask_left = (cdf <= mask_bound).double()
  # Keep small bad CDF, mask the good and large bad CDF values to 1.
  cdf_bad_left_log = log_cdf * cdf_mask_left
  inverse_cdf += (-torch.sqrt(-2 * cdf_bad_left_log))
  if torch.isnan(inverse_cdf.max()) or torch.isnan(inverse_cdf.min()):
    print('inverse CDF: NaN.')
    pdb.set_trace()
  if torch.isinf(inverse_cdf.max()) or torch.isinf(inverse_cdf.min()):
    print('inverse CDF: Inf.')
    exit(0)
    pdb.set_trace()

  # old simple (and possibly numerically unstable) way
  cdf2 = norm.cdf(scaled)
  # remove outliers 
  cdf2[cdf2<EPS] = EPS
  cdf2[cdf2>1-EPS] = 1 - EPS
  new_distr = (pi.cpu().numpy() * cdf2).sum(-1)
  new_X = norm.ppf(new_distr)
  new_X = to_tensor(new_X)

  if False and torch.norm(new_X - inverse_cdf) > 10:
    print('Gaussianization 1D mismatch.')
    pdb.set_trace()

  # return inverse_cdf, cdf_mask, [log_cdf, cdf_mask_left], [log_sf, cdf_mask_right]
  return new_X, cdf_mask, [log_cdf, cdf_mask_left], [log_sf, cdf_mask_right]


def compute_log_det(Y, X, pi, mu, sigma_sqr, A,
                    cdf_mask, log_cdf_l, cdf_mask_left, log_sf_l, cdf_mask_right):
  N, D = Y.shape
  scaled = (Y.unsqueeze(-1) - mu) / sigma_sqr**0.5
  log_pdfs = - 0.5 * scaled**2 + torch.log((2*np.pi)**(-0.5) * pi / sigma_sqr)
  log_pdf = torch.logsumexp(log_pdfs, dim=-1).double()

  t2 = (X**2).sum() / N + 0.5*np.log(2*np.pi)

  log_gaussian_derivative_good = dists.Normal(0, 1).log_prob(X) * cdf_mask
  cdf_l_bad_right_log = log_sf_l * cdf_mask_right + (-1.) * (1. - cdf_mask_right)
  cdf_l_bad_left_log = log_cdf_l * cdf_mask_left + (-1.) * (1. - cdf_mask_left)
  log_gaussian_derivative_left = (torch.log(torch.sqrt(-2 * cdf_l_bad_left_log))
                                  - log_cdf_l) * cdf_mask_left
  log_gaussian_derivative_right = (torch.log(torch.sqrt(-2. * cdf_l_bad_right_log))
                                   - log_sf_l) * cdf_mask_right
  log_gaussian_derivative = log_gaussian_derivative_good + log_gaussian_derivative_left + log_gaussian_derivative_right

  lgd_sum = log_gaussian_derivative.sum() / N

  log_det = (log_pdf - log_gaussian_derivative).sum() / N + torch.log(torch.abs(torch.det(A)))
  return log_det

def eval_KL(X, log_det):
  N, D = X.shape
  # term for std normal
  log_probs = - (X**2).sum() - 0.5*D *np.log(2*np.pi)
  KL = - log_probs / N - log_det
  return KL.item()

def eval_KL_old(X, pi, mu, sigma_sqr):
  N, D = X.shape
  exponents_normal = -0.5 * (X**2).sum(1)
  log_prob_normal = -0.5 * D * np.log(2*np.pi) + exponents_normal
  prob_normal = torch.exp(log_prob_normal)
  prob_normal /= prob_normal.sum()
  log_prob_normal = torch.log(prob_normal)

  # uncomment the following two lines for sanity check w/ std normal
  # mu = torch.zeros_like(mu).to(device)
  # sigma_sqr = torch.ones_like(sigma_sqr).to(device)

  exponents = - (X.unsqueeze(-1) - mu)**2 / (2 * sigma_sqr)
  prob = pi * (2*np.pi*sigma_sqr)**(-0.5) * torch.exp(exponents)
  prob = prob.sum(-1) # shape: N x D
  log_prob_curr = torch.log(prob).sum(-1)
  prob = torch.exp(log_prob_curr)
  prob /= prob.sum()
  log_prob_curr = torch.log(prob)

  KL = (prob * (log_prob_curr - log_prob_normal)).sum().item()
  if KL < -SMALL:
    print('ERROR: negative value of KL.')
    pdb.set_trace()
  return KL

  # prob = torch.exp(-0.5 * exponents) * pi / (sigma_sqr**0.5)
  # log_prob_curr = torch.log(prob)

def check_cov(X):
  N, D = X.shape
  cov = X.T.matmul(X) / N
  diag = torch.diag(cov)
  diag = diag.cpu().numpy()
  print('check cov:')
  print('  diag: max={:.4e} / min={:.4e} / mean={:.4e} / std={:.4e}'.format(
    diag.max(), diag.min(), diag.mean(), diag.std()))
  _, ss, _ = torch.svd(cov)
  ss = ss.cpu().numpy()
  print('  ss: max={:.4e} / min={:.4e} / mean={:.4e} / std={:.4e}'.format(
    ss.max(), ss.min(), ss.mean(), ss.std()))
  diff_from_I = torch.norm(torch.eye(D).to(device) - cov).item()
  print('  diff from I: {:.4e}'.format(diff_from_I))
  return diff_from_I

def get_cofactors(A):
  cofs = torch.zeros_like(A)
  D = A.shape[0]
  if D == 2:
    cofs[0,0] = A[1,1]
    cofs[0,1] = -A[1,0]
    cofs[1,0] = -A[0,1]
    cofs[1,1] = A[0,0]
    return cofs

  for i in range(D):
    for j in range(D):
      B = A.clone()
      B[i] = 0
      B[:, j] = 0
      B = B[B.sum(0) != 0]
      B = B.T
      B = B[B.sum(1) != 0]
      B = B.T
      cof = (-1)**(i+j) * torch.det(B)
      cofs[i,j] = cof.item()
  return cofs

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

