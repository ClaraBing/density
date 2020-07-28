import numpy as np
import torch
import torch.utils.data as data
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

import pdb

class Uniform(data.Dataset):
  def __init__(self, dlen, cx=-2, cy=-2):
    self.len = dlen
    self.cx = cx
    self.cy = cy

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    point = 4*np.random.rand(2)
    point[0] += self.cx
    point[1] += self.cy
    return point


