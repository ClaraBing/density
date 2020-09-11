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
EPS = 5e-7
EPS_GRAD = 1e-2

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
  BACKTRACK = gamma < 0

  if type(X) is not torch.Tensor:
    X = torch.tensor(X)
  X = X.type(DTYPE).to(device)

  N, D = X.shape

  END = lambda dA, dsigma_sqr: (dA + dsigma_sqr) < threshold

  if A_mode == 'ICA':
    cov = X.T.matmul(X) / len(X)
    cnt = 0
    n_tries = 20
    while cnt < n_tries:
      # multiple
      try:
        ica = FastICA()
        Y = ica.fit_transform(X.cpu())
        # Y = to_tensor(Y).T
        _, ss, _ = np.linalg.svd(ica.mixing_)
        # Y *= ss[0]
        A = ica.mixing_ / ss[0]
        A = np.linalg.inv(A)
        A = to_tensor(A)
        # A = to_tensor(torch.ica.mixing_)
        Y = X.matmul(A.T)
        Y = Y.T
        cnt = 2*n_tries
      except:
        cnt += 1
    if cnt != 2*n_tries:
      print('ICA failed. Use random.')
      A = to_tensor(ortho_group.rvs(D))
      Y = A.matmul(X)

    Y, w, w_sumN, w_sumNK = E(X, pi, mu, sigma_sqr, Y=Y)
    pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w, w_sumN, w_sumNK, Y=Y)
    return Y.T, A, pi, mu, sigma_sqr

  Y = None
  niters = 0
  dA, dsigma_sqr = 10, 10
  avg_time = {}
  time_A, time_E, time_GA, time_Y = [], [], [], []
  time_obj, n_iters_btls = [], []
  grad_norms = []
  objs = []

  while (not END(dA, dsigma_sqr)) and niters < max_em_steps:
    niters += 1
    A_prev, sigma_sqr_prev = A.clone(), sigma_sqr.clone()
    objs += [],


    if TIME:
      e_start = time()

    Y, w, w_sumN, w_sumNK = E(X, pi, mu, sigma_sqr, Y=Y)
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
          Y, w, w_sumN, w_sumNK = E(X, pi, mu, sigma_sqr)

        pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w, w_sumN, w_sumNK)


        if TIME:
          a_start = time()
        # grad = torch.inverse(A).T + B
        grad, y_time = get_grad(X, A, w, mu, sigma_sqr)
        if TIME:
          time_Y += y_time,

        if TIME:
          obj_start = time()
        obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
        if TIME:
          time_obj += time() - obj_start,
        if BACKTRACK:
          # backtracking line search
          beta = 0.6
          t = 1
          flag = True
          gnorm = torch.norm(grad)
          n_iter = 0
          while flag:
            n_iter += 1
            Ap = A + t * grad
            _, ss, _ = torch.svd(Ap)
            Ap /= ss[0]
            if TIME:
              obj_start = time()
            obj_p = get_objetive(X, Ap, pi, mu, sigma_sqr, w)
            if TIME:
             time_obj += time() - obj_start,
            t *= beta
            base = obj - 0.5 * t * gnorm
            flag = obj_p < base
          gamma = t
          n_iters_btls += n_iter,
        elif gamma == 0:
          # perturb
          perturb = A.std() * 0.1 * torch.randn(A.shape).type(DTYPE).to(device)
          perturbed = A + perturb
          perturbed_grad, _ = get_grad(X, perturbed, w, mu, sigma_sqr)

          grad_diff = torch.norm(grad - perturbed_grad)
          gamma = 1 /(EPS_GRAD + grad_diff) * 0.03

        A += gamma * grad
        _, ss, _ = torch.svd(A)
        A /= ss[0]
        if TIME:
          time_A += time() - a_start,
          time_GA += time() - ga_start,
        grad_norms += torch.norm(grad).item(),

        if CHECK_OBJ:
          if TIME:
            obj_start = time()
          obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
          if TIME:
            time_obj += time() - obj_start,
          objs[-1] += obj,
          if VERBOSE:
            print('iter {}: obj= {:.5f}'.format(i, obj))

    elif A_mode == 'torchGA': # gradient ascent with torch
      if CHECK_OBJ:
        obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
        objs[-1] += obj.item(),

      A.requires_grad = True
      # pi.requires_grad = False
      # mu.requires_grad = False
      # sigma_sqr.requires_grad = False
      # w.requires_grad = False

      optimizer = optim.Adam([A], lr=gamma)

      for i in range(n_gd_steps):
        ga_start = time()
        if VERBOSE: print(A.view(-1))
        
        # TODO: should I update them as well?
        pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w, w_sumN, w_sumNK)

        if TIME:
          a_start = time()
          obj_start = time()

        optimizer.zero_grad()
        obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
        objs[-1] += obj.item(),
        obj *= -1
        if TIME:
          time_obj += time() - obj_start,
        if VERBOSE:
          print('iter {}: obj= {:.5f}'.format(i, obj.item()))

        obj.backward()
        A.grad[torch.where(torch.isnan(A.grad))] = 0
        optimizer.step()
        # _, ss, _ = torch.svd(A)
        # A /= ss[0]
        if TIME:
          time_A += time() - a_start,
          time_GA += time() - ga_start,
        grad_norms += torch.norm(A.grad).item(),

        sgd_grad = A.grad
        grad, y_time = get_grad(X, A, w, mu, sigma_sqr)
        # print('A:', A.view(-1))
        
      A = A.detach()
      pi = pi.detach()
      mu = mu.detach()
      sigma_sqr = sigma_sqr.detach()
      w = w.detach()

      # normalize
      _, ss, _ = torch.svd(A)
      A = A/ss[0]

    elif A_mode == 'torchAll': # gradient ascent with torch
      if CHECK_OBJ:
        obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
        objs[-1] += obj.item(),

      A.requires_grad = True
      pi.requires_grad = True
      mu.requires_grad = True
      sigma_sqr.requires_grad = True
      w.requires_grad = True

      optimizer = optim.SGD([A, pi, mu, sigma_sqr, w], lr=gamma)

      for i in range(n_gd_steps):
        ga_start = time()
        if VERBOSE: print(A.view(-1))
        
        # TODO: should I update them as well?
        # pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w_sumN, w_sumNK)

        if TIME:
          a_start = time()
          obj_start = time()

        optimizer.zero_grad()
        obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
        objs[-1] += obj.item(),
        obj *= -1
        if TIME:
          time_obj += time() - obj_start,
        if VERBOSE:
          print('iter {}: obj= {:.5f}'.format(i, obj.item()))

        obj.backward()
        optimizer.step()
        # _, ss, _ = torch.svd(A)
        # A /= ss[0]
        if TIME:
          time_A += time() - a_start,
          time_GA += time() - ga_start,
        grad_norms += torch.norm(A.grad).item(),
        # print('A:', A.view(-1))

        
      A = A.detach()
      pi = pi.detach()
      mu = mu.detach()
      sigma_sqr = sigma_sqr.detach()
      w = w.detach()

      # normalize
      _, ss, _ = torch.svd(A)
      A = A/ss[0]
      pi[pi < 0] = SMALL
      pi /= pi.sum(1, keepdim=True)


    elif A_mode == 'ICA':
      # NOTE: passing in Y as an argument since A is not explicitly calculated.
      Y, w, w_sumN, w_sumNK = E(X, pi, mu, sigma_sqr, Y=Y)
      pi, mu, sigma_sqr = update_pi_mu_sigma(X, A, w, w_sumN, w_sumNK, Y=Y)
      

    elif A_mode == 'CF': # closed form
      raise NotImplementedError("mode CF: not implemented yet.")
           
    # difference from the previous iterate
    dA, dsigma_sqr = torch.norm(A - A_prev), torch.norm(sigma_sqr.view(-1) - sigma_sqr_prev.view(-1))

  if VERBOSE:
    print('#{}: dA={:.3e} / dsigma_sqr={:.3e}'.format(niters, dA, dsigma_sqr))
    print('A:', A.view(-1))

  if TIME:
    avg_time = {
      'A': np.array(time_A).mean() if time_A else 0,
      'E': np.array(time_E).mean() if time_E else 0,
      'GA': np.array(time_GA).mean() if time_GA else 0,
      'Y': np.array(time_Y).mean() if time_Y else 0,
      'obj': np.array(time_obj).mean() if time_obj else 0,
      'btls_nIters': np.array(n_iters_btls).mean() if n_iters_btls else 0,
    }

  if A_mode == 'ICA':
    # NOTE: returning directly since no EM iteration is required.
    return Y.T, A, pi, mu, sigma_sqr, avg_time
  return X, A, pi, mu, sigma_sqr, grad_norms, objs, avg_time 

# util funcs in EM steps
def E(X, pi, mu, sigma_sqr, Y=None):
  N, D = X.shape
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

def update_pi_mu_sigma(X, A, w, w_sumN, w_sumNK, Y=None):
  N, D = X.shape
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
  N, D = X.shape
  if Y is None:
    Y = A.matmul(X.T)
  Y = Y.T
  exponents = (Y.view(N, D, 1) - mu)**2 / sigma_sqr
  exp = torch.exp(-0.5 * exponents) / (2*np.pi * sigma_sqr)**(1/2)
  # pi = torch.max(pi, to_tensor(np.array([0.])))
  prob = (exp * pi).sum(-1)
  log = torch.log(prob) 
  log[prob == 0] = 0 # mask out NaN
  obj = log.sum() / N + torch.log(torch.abs(torch.det(A)))
  if torch.isnan(obj):
    print('objective is NaN.') 
    pdb.set_trace()
  return obj

def get_grad(X, A, w, mu, sigma_sqr):
  N, D = X.shape
  if TIME:
    y_start = time()
  Y = A.matmul(X.T)
  if TIME:
    y_time = time() - y_start
  else:
    y_time = 0

  scaled = (-Y.T.view(N, D, 1) + mu) / sigma_sqr
  weighted_X = (w * scaled).view(N, D, 1, K) * X.view(N, 1, D, 1)
  B = weighted_X.sum(0).sum(-1) / N
  grad = torch.inverse(A).T + B
  return grad, y_time


def gaussianize_1d_old(X, pi, mu, sigma_sqr):
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
  mask_bound = 0.5e-7

  N, D = X.shape

  scaled = (X.view(N, D, 1) - mu) / sigma_sqr**0.5
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
    pdb.set_trace()
  return inverse_cdf, cdf_mask, [log_cdf, cdf_mask_left], [log_sf, cdf_mask_right]


def compute_log_det(Y, pi, mu, sigma_sqr, A,
                    cdf_mask, log_cdf_l, cdf_mask_left, log_sf_l, cdf_mask_right):
  N, D = Y.shape
  scaled = (Y.view(N, D, 1) - mu) / sigma_sqr**0.5
  log_pdfs = - 0.5 * scaled + torch.log((2*np.pi)**(-0.5) * pi / sigma_sqr)
  log_pdf = torch.logsumexp(log_pdfs, dim=-1).double()

  log_gaussian_derivative_good = dists.Normal(0, 1).log_prob(Y) * cdf_mask
  cdf_l_bad_right_log = log_sf_l * cdf_mask_right + (-1.) * (1. - cdf_mask_right)
  cdf_l_bad_left_log = log_cdf_l * cdf_mask_left + (-1.) * (1. - cdf_mask_left)
  log_gaussian_derivative_left = (torch.log(torch.sqrt(-2 * cdf_l_bad_left_log))
                                  - log_cdf_l) * cdf_mask_left
  log_gaussian_derivative_right = (torch.log(torch.sqrt(-2. * cdf_l_bad_right_log))
                                   - log_sf_l) * cdf_mask_right
  log_gaussian_derivative = log_gaussian_derivative_good + log_gaussian_derivative_left + log_gaussian_derivative_right

  log_det = (log_pdf - log_gaussian_derivative).sum() / N + torch.log(torch.abs(torch.det(A)))
  return log_det

def eval_KL(X, log_det):
  N, D = X.shape
  # term for std normal
  log_probs = - (X**2).sum() - 0.5*D *np.log(2*np.pi)
  KL = - log_probs / N - log_det
  return KL

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

  exponents = - (X.view(N, D, 1) - mu)**2 / (2 * sigma_sqr)
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

