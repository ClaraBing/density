import numpy as np
import torch
import torch.utils.data as data

import pdb

class HEPMASS(data.Dataset):

  def __init__(self):
    # the data is normalized.
    # For preprocess please see Gaussianization Flow's code:
    # `datasets/hepmass.py`
    fdata = './datasets/hepmass/trainval_normed.npy'
    self.X = np.load(fdata)
    self.len = len(self.X)

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    return self.X[idx]


