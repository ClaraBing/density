import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import datasets

from . import util

class LINE:
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, d):

        fname = datasets.root + 'line/d{}.h5'.format(d)
        f = h5py.File(fname)
        trn = f['train'][:]
        val = f['valid'][:]
        tst = f['test'][:]

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        util.plot_hist_marginals(data_split.x)
        plt.show()

