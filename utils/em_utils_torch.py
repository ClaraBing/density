import numpy as np
import torch
import torch.optim as optim
import torch.distributions as dists
import torch.nn.functional as F
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
from utils.rbig_util import logistic_kernel_cdf, logistic_kernel_log_cdf, logistic_kernel_log_sf, logistic_inverse_normal_cdf
from utils.variational_util import variational_KL

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

def init_A(D):
  # rotation matrix
  A = torch.eye(D)
  A = A.type(DTYPE).to(device)
  return A

def init_GM_params(D, K, mu_low, mu_up):
  # prior - uniform
  pi = torch.ones([D, K]) / K
  # GM means
  mu = torch.tensor([np.arange(mu_low, mu_up, (mu_up-mu_low)/K) for _ in range(D)])
  # GM variances
  sigma_sqr = torch.ones([D, K])
  if VERBOSE:
    print('mu: mean={} / std={}'.format(mu.mean(), mu.std()))

  pi = pi.type(DTYPE).to(device)
  mu = mu.type(DTYPE).to(device)
  sigma_sqr = sigma_sqr.type(DTYPE).to(device)

  return pi, mu, sigma_sqr

def update_EM(X, A, pi, mu, sigma_sqr,
              threshold=5e-5, max_em_steps=30):

  if type(X) is not torch.Tensor:
    X = torch.tensor(X)
  X = X.type(DTYPE).to(device)

  N, D = X.shape

  END = lambda dsigma_sqr: dsigma_sqr < threshold

  Y = None
  niters = 0
  dsigma_sqr = 10,
  ret_time = {'E':[], 'obj':[]}

  while (not END(dsigma_sqr)) and niters < max_em_steps:
    niters += 1

    if TIME: e_start = time()
    Y, w, w_sumN, w_sumNK = E(X, A, pi, mu, sigma_sqr, Y=Y)
    if TIME: ret_time['E'] += time() - e_start,

    # M-step
    pi, mu, sigma_sqr = M(X, A, w, w_sumN, w_sumNK)
    # obj = get_objetive(X, A, pi, mu, sigma_sqr, w)
    # objs[-1] += obj,

  if VERBOSE:
    print('#{}: dsigma_sqr={:.3e}'.format(niters, dsigma_sqr))

  if TIME:
    for key in ret_time:
      ret_time[key] = np.array(ret_time[key]) if ret_time[key] else 0

  return pi, mu, sigma_sqr, ret_time 

def update_A(A_mode, X):
  N, D = X.shape

  if A_mode == 'random':
    A = ortho_group.rvs(D)
    A = to_tensor(A)
  elif A_mode == 'PCA':
    cov = X.T.matmul(X) / len(X)
    _, _, A = torch.svd(cov)
  elif A_mode == 'ICA':
    cov = X.T.matmul(X) / len(X)
    cnt = 0
    n_tries = 20
    SUCC_FLAG = False
    while cnt < n_tries and not SUCC_FLAG:
      try:
        ica = FastICA()
        _ = ica.fit_transform(X.cpu())
        Aorig = ica.mixing_

        # avoid numerical instability
        U, ss, V = np.linalg.svd(Aorig)
        ss /= ss[0]
        if ss[-1] < SINGULAR_SMALL:
          Aorig += np.eye(Aorig.shape[0]) * SINGULAR_SMALL
        # ss[ss < SINGULAR_SMALL] = SINGULAR_SMALL
        # Aorig = (U * ss).dot(V)

        A = np.linalg.inv(Aorig)
        _, ss, _ = np.linalg.svd(A)
        A = to_tensor(A / ss[0])
        SUCC_FLAG = True
      except:
        cnt += 1
    if not SUCC_FLAG:
      print('ICA failed. Use random orthonormal matrix.')
      A = to_tensor(ortho_group.rvs(D))
  elif A_mode == 'variational':
    A = to_tensor(variational_KL(X, 1000))
    _, ss, _ = np.linalg.svd(A)
    A = to_tensor(A / ss[0])
  elif A_mode == 'Wasserstein':
    raise NotImplementedError("Wasserstein")
  return A
 
# util funcs in EM steps
def E(X, A, pi, mu, sigma_sqr, Y=None):
  N, D = X.shape
  # E-step - update posterior counts
  if Y is None:
    if A is None:
      Y = X.T.clone()
    else:
      Y = A.matmul(X.T) # shape: D x N

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

def M(X, A, w, w_sumN, w_sumNK, Y=None):
  N, D = X.shape
  if Y is None:
    if A is None:
      Y = X.T.clone()
    else:
      Y = A.matmul(X.T) # shape: D x N
  pi = w_sumN / w_sumNK.view(-1, 1)
  mu = (Y.T.unsqueeze(-1) * w).sum(0) / w_sumN
  diff_square = (Y.T.unsqueeze(-1) - mu)**2
  sigma_sqr = (w * diff_square).sum(0) / w_sumN
  mu[torch.abs(mu) < SMALL] = SMALL 
  sigma_sqr[sigma_sqr < SMALL] = SMALL
  return pi, mu, sigma_sqr

# from RBIG code
def generate_bandwidth(datapoints):
    total_datapoints = datapoints.shape[0]
    # scale ~ 0.13
    scale = (4. * np.sqrt(np.pi) / ((np.pi ** 4) * total_datapoints)) ** (0.2)
    bandwidth = torch.std(datapoints, dim=0, keepdim=True) * scale
    return bandwidth


def get_objetive(X, A, pi, mu, sigma_sqr, w, Y=None):
  N, D = X.shape
  if Y is None:
    Y = A.matmul(X.T)
  Y = Y.T
  exponents = (Y.unsqueeze(-1) - mu)**2 / sigma_sqr
  exp = torch.exp(-0.5 * exponents) / (2*np.pi * sigma_sqr)**(1/2)
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

normal_distribution = dists.Normal(0, 1)

# compute inverse normal CDF
def gaussianize_1d(X, pi, mu, sigma_sqr, datapoints=None, bandwidth=None):
  mask_bound = 5e-8

  N, D = X.shape

  # pdb.set_trace()
  if bandwidth is None:
    # for calculations please see: https://www.overleaf.com/6125358376rgmjjgdsmdmm
    ##  pdb.set_trace()
    z = (X.unsqueeze(-1) - mu) / sigma_sqr**0.5
    z = z.cpu()
    normal_cdf = to_tensor(norm.cdf(z))
    cdf = (pi * normal_cdf).sum(-1)
    log_cdfs = to_tensor(norm.logcdf(z))
    log_cdf = torch.logsumexp(0.5*torch.log(2*pi) + 0.5*log_cdfs, dim=-1)
    log_sfs = to_tensor(norm.logcdf(-1*z))
    log_sf = torch.logsumexp(0.5*torch.log(2*pi) + 0.5*log_sfs, dim=-1)
  elif False: # not used: KDE uses the routine in rbig_utils
    mask_bound = 0.5e-7
    cdf = logistic_kernel_cdf(x, datapoints, h=bandwidth)
    log_cdf = logistic_kernel_log_cdf(x, datapoints, h=bandwidth)  # log(CDF)
    log_sf = logistic_kernel_log_sf(x, datapoints, h=bandwidth)  # log(1-CDF)

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

  cdf2 = norm.cdf(z)
  # remove outliers 
  cdf2[cdf2<EPS] = EPS
  cdf2[cdf2>1-EPS] = 1 - EPS
  new_distr = (pi.cpu().numpy() * cdf2).sum(-1)
  new_X = norm.ppf(new_distr)
  new_X = to_tensor(new_X)

  if False and torch.norm(new_X - inverse_cdf) > 10:
    print('Gaussianization 1D mismatch.')
    pdb.set_trace()

  return new_X, cdf_mask, [log_cdf, cdf_mask_left], [log_sf, cdf_mask_right]

# compute inverse normal CDF - RBIG version
def logistic_inverse_normal_cdf_cp(x, bandwidth, datapoints, inverse_cdf_by_thresh=False):
    if x.dtype != bandwidth.dtype:
      bandwidth = bandwidth.type(x.dtype)
    if x.dtype != datapoints.dtype:
      datapoints = datapoints.type(x.dtype)
    mask_bound = 0.5e-7
    cdf_l = logistic_kernel_cdf(x, datapoints, h=bandwidth)
    log_cdf_l = logistic_kernel_log_cdf(x, datapoints, h=bandwidth)  # log(CDF)
    log_sf_l = logistic_kernel_log_sf(x, datapoints, h=bandwidth)  # log(1-CDF)

    # Approximate Gaussian CDF
    # inv(CDF) ~ np.sqrt(-2 * np.log(1-x)) #right, x -> 1
    # inv(CDF) ~ -np.sqrt(-2 * np.log(x)) #left, x -> 0
    # 1) Step1: invert good CDF
    cdf_mask = ((cdf_l > mask_bound) & (cdf_l < 1 - (mask_bound))).double()
    # Keep good CDF, mask the bad CDF values to 0.5(inverse(0.5)=0.)
    cdf_l_good = cdf_l * cdf_mask + 0.5 * (1. - cdf_mask)
    inverse_l = normal_distribution.icdf(cdf_l_good)

    # 2) Step2: invert BAD large CDF
    cdf_mask_right = (cdf_l >= 1. - (mask_bound)).double()
    # Keep large bad CDF, mask the good and small bad CDF values to 0.
    cdf_l_bad_right_log = log_sf_l * cdf_mask_right
    inverse_l += torch.sqrt(-2. * cdf_l_bad_right_log)

    # 3) Step3: invert BAD small CDF
    cdf_mask_left = (cdf_l <= mask_bound).double()
    # Keep small bad CDF, mask the good and large bad CDF values to 1.
    cdf_l_bad_left_log = log_cdf_l * cdf_mask_left
    inverse_l += (-torch.sqrt(-2 * cdf_l_bad_left_log))

    if inverse_cdf_by_thresh:
      # remove outliers 
      cdf_l[cdf_l<mask_bound] = mask_bound
      cdf_l[cdf_l>1-mask_bound] = 1 - mask_bound
      # pdb.set_trace()
      # new_distr = cdf_l.sum(-1)
      new_X = norm.ppf(cdf_l.cpu())
      new_X = to_tensor(new_X)
      ret_x = new_X
    else:
      ret_x = inverse_l 

    return ret_x, cdf_mask, [log_cdf_l, cdf_mask_left], [log_sf_l, cdf_mask_right]


def compute_log_det_v2(X, Y, cdf_mask, log_cdf_l, cdf_mask_left, log_sf_l, cdf_mask_right,
                    pi=None, mu=None, sigma_sqr=None,
                    datapoints=None, h=None):
  # NOTE: currently debugging this function.
  # Y should be rotated & before Gaussianization,
  # X should be Gaussiznied.
  N, D = Y.shape

  # dp / dy
  if h is None:
    scaled = (Y.unsqueeze(-1) - mu)**2 / sigma_sqr
    # log_pdfs = - 0.5 * scaled + torch.log((2*np.pi)**(-0.5) * pi / sigma_sqr**0.75)
    log_pdfs = - 0.5 * scaled + torch.log(pi / (2*np.pi * sigma_sqr)**(-0.5)
    pdb.set_trace()
    log_pdf = torch.logsumexp(log_pdfs, dim=-1).double()
  else:
    Nd = datapoints.shape[0]
    log_pdfs = -(Y[None, ...] - datapoints[:, None, :]) / h[None, ...] \
               - torch.log(h[None, ...]) \
               - 2 * F.softplus(-(Y[None, ...] - datapoints[:, None, :]) / h[None, ...]) - np.log(Nd)
    log_pdf = torch.logsumexp(log_pdfs, dim=0).double()

  # d phi^{-1} / dp
  p1 = 0.5 * np.log(2*np.pi) + 0.5 * X**2

  # log_det = (log_pdf + p1).sum() / N
  log_det = (log_pdf + p1).sum(-1)
  return log_det

def compute_log_det_v1(X, Y, cdf_mask, log_cdf_l, cdf_mask_left, log_sf_l, cdf_mask_right,
                    pi=None, mu=None, sigma_sqr=None,
                    datapoints=None, h=None):
  # NOTE: currently debugging this function.
  # Y should be rotated & before Gaussianization,
  # X should be Gaussiznied.
  N, D = X.shape
  if h is None and pi is not None:
    scaled = (Y.unsqueeze(-1) - mu)**2 / sigma_sqr
    # log_pdfs = - 0.5 * scaled + torch.log((2*np.pi)**(-0.5) * pi / sigma_sqr**0.75)
    log_pdfs = - 0.5 * scaled + torch.log(pi / (2*np.pi * sigma_sqr)**(-0.5)
    log_pdf = torch.logsumexp(log_pdfs, dim=-1).double()
  else:
    Nd = datapoints.shape[0]
    log_pdfs = -(Y[None, ...] - datapoints[:, None, :]) / h[None, ...] \
               - torch.log(h[None, ...]) \
               - 2 * F.softplus(-(Y[None, ...] - datapoints[:, None, :]) / h[None, ...]) - np.log(Nd)
    log_pdf = torch.logsumexp(log_pdfs, dim=0).double()

  # TODO: is this correct?
  log_gaussian_derivative_good = dists.Normal(0, 1).log_prob(X) * cdf_mask
  # log_gaussian_derivative_good = dists.Normal(0, 1).log_prob(Y) * cdf_mask

  cdf_l_bad_right_log = log_sf_l * cdf_mask_right + (-1.) * (1. - cdf_mask_right)
  cdf_l_bad_left_log = log_cdf_l * cdf_mask_left + (-1.) * (1. - cdf_mask_left)
  log_gaussian_derivative_left = (torch.log(torch.sqrt(-2 * cdf_l_bad_left_log))
                                  - log_cdf_l) * cdf_mask_left
  log_gaussian_derivative_right = (torch.log(torch.sqrt(-2. * cdf_l_bad_right_log))
                                   - log_sf_l) * cdf_mask_right
  log_gaussian_derivative = log_gaussian_derivative_good + log_gaussian_derivative_left + log_gaussian_derivative_right

  # lgd_sum = log_gaussian_derivative.sum(-1) / N

  # log_det = (log_pdf - log_gaussian_derivative).sum() / N
  log_det = (log_pdf - log_gaussian_derivative).sum(-1)
  return log_det

def eval_NLL(X, log_det):
  N, D = X.shape
  # term for std normal
  log_probs = - 0.5*(X**2) - 0.5*np.log(2*np.pi)
  # NLL = - log_probs / N - log_det
  NLL = - (log_probs.sum() + log_det.sum()) / N
  return NLL.item()

def eval_KL(X):
  N, D = X.shape
  KL = 0.5*(X**2).sum() / N + 0.5*D *np.log(2*np.pi)
  return KL.item()

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
  return diff_from_I, ss[0], ss[-1], ss.mean()

def get_aranges(low, up, n_steps):
  if low == up:
    return np.array([low] * n_steps)
  return np.arange(low, up, (up-low)/n_steps)[::-1]

def plot_hist(data, fimg):
  if type(data) is torch.Tensor:
    data = data.cpu().numpy()

  if data.shape[1] == 2:
    x, y = data[:, 0], data[:, 1]
    plt.figure(figsize=[8,8])
    plt.hist2d(x, y, bins=[100,100])
    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    plt.savefig(fimg)
    plt.clf()
    plt.close()
  else:
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8), sharex=True, sharey=True)
    # check the projection at 10 random directions
    for pi in range(12):
      v1 = np.random.randn(data.shape[1])
      v2 = np.random.randn(data.shape[1])
      v2 -= v1.dot(v2) * v1
      v1 /= np.linalg.norm(v1)
      v2 /= np.linalg.norm(v2)
      x = data.dot(v1)
      y = data.dot(v2)

      ri = pi // n_cols
      ci = pi % n_cols
      axes[ri, ci].hist2d(x, y, bins=[100,100])
    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    plt.savefig(fimg)
    plt.close()

def to_tensor(data):
  if not isinstance(data, torch.Tensor):
    data = torch.tensor(data)
  return data.type(DTYPE).to(device)

def gen_data(scale):
  """
  Generate toy data with 2D Gaussian mixtures.
  """
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

