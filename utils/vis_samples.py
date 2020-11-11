import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

# LOW = -4
# HIGH = 4

def plot_samples(x, title, n_bins=100):
  # plt.hist2d(x[:, 0], x[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=n_bins)
  plt.hist2d(x[:, 0], x[:, 1], bins=n_bins)
  plt.savefig(title)
