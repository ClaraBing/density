import numpy as np
import torch
import torch.utils.data as data

class GaussianLine(data.Dataset):
  def __init__(self, d, bt, dlen, xdir=None):
    self.d = d # data dimension
    self.bt = bt # batch_size
    self.len = dlen # dataset length
  
    # direction of the line
    self.xdir = xdir
    if self.xdir is None:
      # sample a random direction
      self.xdir = np.random.rand(self.d)
      self.xdir /= np.linalg.norm(self.xdir)

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    # scales = np.random.randn(self.bt, 1)
    # batch = scales.dot(self.xdir)
    # return batch
    scale = np.random.randn()
    return scale * self.xdir

