import os
from glob import glob
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pdb

# LOW = -4
# HIGH = 4

def plot_samples(x, title, n_bins=100):
  # plt.hist2d(x[:, 0], x[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=n_bins)
  plt.hist2d(x[:, 0], x[:, 1], bins=n_bins)
  plt.savefig(title)

FIG_ROOT = '/home/bingbin/density/runs_gaussianization'

def merge_and_compare(data, fdirs, fout):
  """
  put figs from two runs side-by-side for easier comparison.
  """
  ffig_dirs = []
  for fdir in fdirs:
    ffig_dirs += os.path.join(FIG_ROOT, data, fdir, 'figs'),

  mfigs = []
  padding = None
  for i in [1, 10, 20, 40, 80, 100]:
    i -= 1
    cur_figs = []
    for ffig_dir in ffig_dirs:
      ffig = glob(os.path.join(ffig_dir, '*test*iter{}.png'.format(i)))[0]
      fig = cv2.imread(ffig)
      cur_figs += fig,
    mfig = np.concatenate(cur_figs, 0)
    if padding is None:
      padding = np.ones([mfig.shape[0], int(mfig.shape[1]*0.05), mfig.shape[2]]) * 255
    mfigs += mfig,
    mfigs += padding,
  merged = np.concatenate(mfigs, 1)
  cv2.imwrite(fout, merged)


if __name__ == '__main__':
  data = 'GM'
  fdir1 = 'GM_modevariational_iter100_em200_K40__variational_PP_varLR1_varWD1e-5_run1'
  fdir2 = 'GM_modevariational_iter100_em200_K40__variational_PP_varLR0.03_varWD1e-5_run1'
  # fdir3 = 'GM_modeICA_iter100_em200_K40__ICA_iter400_tol1e-4_PP_myG1D_logDetv1_ndtri_run1'
  fdir4 = 'GM_modeICA_iter100_em200_K40__ICA_iter400_tol1e-4_PP_myG1D_logDetv2_ndtri_run1'
  fdir5 = 'GM_modeICA_iter100_em100_K80__ICA_iter400_tol1e-4_PP_myG1D_logDetv2_ndtri_run1'
  fdir6 = 'GM_modeICA_iter100_em100_K40__ICA_iter200_tol1e-4_PP_myG1D_logDetv2_ndtri_run1'

  fdirs = [fdir1, fdir2, fdir4, fdir5, fdir6]

  fout = os.path.join(FIG_ROOT, data, 'var_compare.png')
  merge_and_compare(data, fdirs, fout)

