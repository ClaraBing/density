import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import datasets

from . import util

class Hyperbolic(data.Dataset):
    def __init__(self, s):
        self.sum = s

    def show_histograms(self, split):
        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        util.plot_hist_marginals(data_split.x)
        plt.show()

    def __getitem__(self, idx):
      v = np.random.rand(self.d)
      v /= np.linalg.norm(v)
      return v

