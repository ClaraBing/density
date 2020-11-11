import numpy as np
import torch
import torch.utils.data as data

import pdb

class MNISTtab(data.Dataset):

  def __init__(self, pca_dim=300):
    # the data is normalized by column.
    # please see `./data/preproces/dataset_miniboone.py` for details.
    fdata = './datasets/mnist/MNIST/processed/train_normed_pca{}.npy'.format(pca_dim)
    self.X = np.load(fdata)
    self.len = len(self.X)

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    return self.X[idx]


