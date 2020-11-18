import os
import numpy as np
import pickle
import torch
from scipy.stats import ortho_group, gaussian_kde
import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.get_loader import *

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--n-pts', type=int, default=0)
parser.add_argument('--data', type=str, default='GM', choices=[
       # connected
       'normal', 'scaledNormal', 'rotatedNormal', 'ring',
       # disconnected
       'GM', 'GM_scale1', 'GM_scale2', 'GMn2', 'concentric',
       # UCI
       'gas16_co', 'gas16_methane', 'gas8_co', 'gas8_co_normed',
       'miniboone', 'hepmass', 'gas', 'power',
       # images,
       'MNIST',
       ])
parser.add_argument('--pca-dim', type=int, default=0,
                   help="PCA dimension for high-dim data e.g. MNIST.")

args = parser.parse_args()

from utils.em_utils_torch import *

def fit(X, Xtest, mu_low, mu_up, data_token=''):
  kernel = gaussian_kde(X.T)
  nll_train = np.log(kernel.evaluate(X.T))
  print('NLL (train): {:.3f}'.format(nll_train.mean()))
  nll_test = np.log(kernel.evaluate(Xtest.T))
  print('NLL (test): {:.3f}'.format(nll_test.mean()))

if __name__ == '__main__':
  data_dir = './datasets/'
  data_token = args.data
  mu_low, mu_up = -2, 2
  if data_token == 'GM':
    fdata = 'EM/GM_2d.npy'
    fdata_val = 'EM/GM_2d_scale4_test.npy'
    mu_low, mu_up = -4, 4
  if data_token == 'GM_scale1':
    fdata = 'EM/GM_2d_scale1.npy'
    mu_low, mu_up = -4, 4
  if data_token == 'GM_scale2':
    fdata = 'EM/GM_2d_scale2.npy'
    mu_low, mu_up = -4, 4
  if data_token == 'GMn2':
    fdata = 'EM/GM_2d_2centers.npy'
    mu_low, mu_up = -4, 4
  elif data_token == 'normal':
    fdata = 'EM/normal.npy'
  elif data_token == 'rotatedNormal':
    fdata = 'EM/rotatedNormal.npy'
  elif data_token == 'scaledNormal':
    fdata = 'EM/scaledNormal.npy'
  elif data_token == 'ring':
    fdata = 'EM/ring.npy'
    mu_low, mu_up = -1, 1
  elif data_token == 'concentric':
    # 2 circles w/ radius 0.5 and 2. Each with 10k points.
    fdata = 'EM/concentric.npy'
  elif data_token == 'gas16_co':
    fdata = 'GAS/my_data/gas_d16_CO_normed_train400k.npy'
    fdata_val = 'GAS/my_data/gas_d16_CO_normed_test.npy'
  elif data_token == 'gas16_methane':
    fdata = 'GAS/my_data/gas_d16_methane.npy'
  elif data_token == 'gas8_co':
    gas_dir = 'flows_data/gas'
    # train: 852174 points
    fdata = os.path.join(gas_dir, 'gas_train.npy')
    # val: 94685 points
    fdata_val = os.path.join(gas_dir, 'gas_val.npy')
  elif data_token == 'gas8_co_normed':
    # gas-8d, with data normzlied per column to [-1, 1].
    gas_dir = 'flows_data/gas'
    # train: 852174 points
    fdata = os.path.join(gas_dir, 'gas_train_normed.npy')
    # val: 94685 points
    fdata_val = os.path.join(gas_dir, 'gas_val_normed.npy')
  elif data_token == 'miniboone':
    fdata = 'miniboone/train_normed.npy'
    fdata_val = 'miniboone/val_normed.npy'
  elif data_token == 'hepmass':
    fdata = 'hepmass/train_normed.npy'
    fdata_val = 'hepmass/val_normed.npy'
  elif data_token == 'MNIST':
    mnist_dir = 'mnist/MNIST/processed'
    if args.pca_dim:
      fdata = os.path.join(mnist_dir, 'train_normed_pca{}.npy'.format(args.pca_dim))
      fdata_val = os.path.join(mnist_dir, 'test_normed_pca{}.npy'.format(args.pca_dim))
    else:
      fdata = os.path.join(mnist_dir, 'train_normed.npy')
      fdata_val = os.path.join(mnist_dir, 'test_normed.npy')
  elif data_token == 'power':
    power_dir = 'POWER/'
    fdata = os.path.join(power_dir, 'train_normed.npy')
    fdata_val = os.path.join(power_dir, 'val_normed.npy')
  elif data_token == 'gas':
    gas_dir = 'GAS/'
    fdata = os.path.join(gas_dir, 'ethylene_CO_trainSmall_normed.npy')
    fdata_val = os.path.join(gas_dir, 'ethylene_CO_valSmall_normed.npy')
  
  X = np.load(os.path.join(data_dir, fdata))
  Xtest = np.load(os.path.join(data_dir, fdata_val))
  # zero-centered
  X = X - X.mean(0)
  Xtest = Xtest - X.mean(0)

  if X.ndim > 2: # images
    X = X.reshape(len(X), -1)
    Xtest = Xtest.reshape(len(Xtest), -1)
  if args.n_pts:
    X = X[:args.n_pts]
    Xtest = Xtest[:args.n_pts//2]
  fit(X, Xtest, mu_low, mu_up, data_token)
  
