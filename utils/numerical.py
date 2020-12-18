"""
Scipy's special -> ndtri (i.e. inverse Gaussian CDF).
Code from: https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtri.c
"""
import torch
import numpy as np
from scipy.stats import norm

import pdb

def polevl(x, coeffs, deg):
  # NOTE: deg is not used;
  # I keep it to make the call signagture same as the C code.

  out = coeffs[0]
  for coeff in coeffs[1:1+deg]:
    out = out * x + coeff
  return out

def p1evl(x, coeffs, deg):
  out = 1
  for coeff in coeffs[:deg]:
    out = out * x + coeff
  return out

P0 = [
    -5.99633501014107895267E1,
    9.80010754185999661536E1,
    -5.66762857469070293439E1,
    1.39312609387279679503E1,
    -1.23916583867381258016E0,
]

Q0 = [
    # 1.00000000000000000000E0
    1.95448858338141759834E0,
    4.67627912898881538453E0,
    8.63602421390890590575E1,
    -2.25462687854119370527E2,
    2.00260212380060660359E2,
    -8.20372256168333339912E1,
    1.59056225126211695515E1,
    -1.18331621121330003142E0,
]

# Approximation for interval z = sqrt(-2 log y ) between 2 and 8
# i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.

P1 = [
    4.05544892305962419923E0,
    3.15251094599893866154E1,
    5.71628192246421288162E1,
    4.40805073893200834700E1,
    1.46849561928858024014E1,
    2.18663306850790267539E0,
    -1.40256079171354495875E-1,
    -3.50424626827848203418E-2,
    -8.57456785154685413611E-4,
]

Q1 = [
    #  1.00000000000000000000E0,
    1.57799883256466749731E1,
    4.53907635128879210584E1,
    4.13172038254672030440E1,
    1.50425385692907503408E1,
    2.50464946208309415979E0,
    -1.42182922854787788574E-1,
    -3.80806407691578277194E-2,
    -9.33259480895457427372E-4,
]

# Approximation for interval z = sqrt(-2 log y ) between 8 and 64
# i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.

P2 = [
    3.23774891776946035970E0,
    6.91522889068984211695E0,
    3.93881025292474443415E0,
    1.33303460815807542389E0,
    2.01485389549179081538E-1,
    1.23716634817820021358E-2,
    3.01581553508235416007E-4,
    2.65806974686737550832E-6,
    6.23974539184983293730E-9,
]

Q2 = [
    # 1.00000000000000000000E0,
    6.02427039364742014255E0,
    3.67983563856160859403E0,
    1.37702099489081330271E0,
    2.16236993594496635890E-1,
    1.34204006088543189037E-2,
    3.28014464682127739104E-4,
    2.89247864745380683936E-6,
    6.79019408009981274425E-9,
]

# Global constants
ndtri_thresh = 0.13533528323661269189
# sqrt(2*pi)
s2pi = 2.50662827463100050242E0


def ndtri(y0):
  negInf_mask = y0 == 0
  Inf_mask = y0 == 1
  y0[negInf_mask] = 0.5
  y0[Inf_mask] = 0.5
  if torch.any(y0 < 0.0) or torch.any(y0 > 1.0):
    raise ValueError("CDF values should be between 0 and 1.")

  code = torch.ones_like(y0)
  y = y0

  # case 1
  mask_right = y > (1.0 - ndtri_thresh)
  code[mask_right] = 0
  y[mask_right] = 1 - y[mask_right]

  # case 2
  mask_checked = y > ndtri_thresh
  y[mask_checked] -= 0.5
  y2 = y[mask_checked]**2
  x = y[mask_checked] + y[mask_checked] * (y2 * polevl(y2, P0, 4) / p1evl(y2, Q0, 8))
  x = x * s2pi
  y[mask_checked] = x.clone()
  # NOTE: no need to flip x here

  # case 3
  mask_todo = ~mask_checked
  x = torch.sqrt(-2.0 * torch.log(y[mask_todo]))
  x0 = x - torch.log(x) / x
  z = 1.0 / x
  x1 = torch.ones_like(x)
  mask_x = x < 8.0
  x1[mask_x] = z[mask_x] * polevl(z[mask_x], P1, 8) / p1evl(z[mask_x], Q1, 8)
  x1[~mask_x] = z[~mask_x] * polevl(z[~mask_x], P2, 8) / p1evl(z[~mask_x], Q2, 8)
  x = x0 - x1
  y[mask_todo] = x
  
  mask_flip = code.bool() * mask_todo
  y[mask_flip] = -y[mask_flip]

  y[negInf_mask] = -np.inf
  y[Inf_mask] = np.inf

  return y

if __name__ == '__main__':
  # out = polevl(2, [1,2,3], 0)
  # print(out)
  y = np.array([1e-10, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1-1e-10])
  out = ndtri(torch.tensor(y))
  print(out)
  checks = norm.ppf(y)
  pdb.set_trace()
