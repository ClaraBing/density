import numpy as np
import torch
import torch.utils.data as data
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

import pdb

class GAS8(data.Dataset):

  def __init__(self, norm_by_col=0):
    fdata = './datasets/flows_data/gas/gas_trainval.npy'
    self.X = np.load(fdata)
    self.len = len(self.X)

    if norm_by_col:
      # normalize each column w/ its std
      self.X /= self.X.std(0)

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    return self.X[idx]


