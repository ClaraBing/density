import numpy as np
import torch
import torch.utils.data as data
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

import pdb

class GaussianMixture(data.Dataset):
  def __init__(self, dlen, scale=4):
    self.len = dlen
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
               (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                     1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    self.centers = [(scale * x, scale * y) for x, y in centers]

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    # rng = np.random.RandomState()

    point = np.random.randn(2) * 0.5
    idx = np.random.randint(8)
    center = self.centers[idx]
    point[0] += center[0]
    point[1] += center[1]
    point /= 1.414213
    return point


