import numpy as np
from sklearn.decomposition import PCA

import pdb

# assuming the current file is run under subfoler './utils/'
DATA_ROOT = '../datasets/mnist/MNIST/processed'

def run_PCA(data, n_components=100):
  pca = PCA()
  pca.n_components = n_components
  pca_data = pca.fit_transform(data)
  return pca_data

def PCA_wrapper(fdata, fdata_out, n_components=100):
  data = np.load(os.path.join(DATA_ROOT, fdata))
  data = data.reshape(len(data), -1)
  pca_data = run_PCA(data, n_components)

  stds = np.array([pca_data[:, i].std() for i in range(n_components)])
  np.save(os.path.join(DATA_ROOT, fdata_out), pca_data)

def PCA_train_only(ftrain, ftest, fout_format, n_components=100):
  train = np.load(os.path.join(DATA_ROOT, ftrain))
  train = train.reshape(len(train), -1)
  test = np.load(os.path.join(DATA_ROOT, ftest))
  test = test.reshape(len(test), -1)

  centered = train - train.mean(0)
  cov = centered.T.dot(centered) / len(centered)
  U, ss, _ = np.linalg.svd(cov)

  train_pca = centered.dot(U[:, :n_components])
  np.save(os.path.join(DATA_ROOT, fout_format.format('train')), train_pca)
  test_pca = (test - train.mean(0)).dot(U[:, :n_components])
  np.save(os.path.join(DATA_ROOT, fout_format.format('test')), test_pca)


if __name__ == '__main__':
  n_components = 70
  
  ftrain = 'train_normed.npy'
  ftest = 'test_normed.npy'

  if False:
    ftrain_out = ftrain.replace('.npy', '_pca{}.npy'.format(n_components))
    PCA_wrapper(ftrain, ftrain_out, n_components)

    ftest_out = ftest.replace('.npy', '_pca{}.npy'.format(n_components))
    PCA_wrapper(ftest, ftest_out, n_components)

  for n_components in [70, 100, 200, 300, 400]:
    fout_format = '{}_normed' + '_pca{}'.format(n_components) + '.npy'
    PCA_train_only(ftrain, ftest, fout_format, n_components)

