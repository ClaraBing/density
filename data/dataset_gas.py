import numpy as np
import torch
import torch.utils.data as data
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

import pdb

class gas(data.Dataset):

  def __init__(self, n_pts=0):
    fdata = './datasets/GAS/ethylene_CO_trainval_normed.npy'
    # TODO: for dev only.
    # fdata = './datasets/GAS/ethylene_CO_trainSmall_normed.npy'
    self.X = np.load(fdata)
    if n_pts > 0:
      idx = np.random.choice(len(self.X), n_pts)
      self.X = self.X[idx]
    self.len = len(self.X)

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    return self.X[idx]


