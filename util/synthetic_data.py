import numpy as np
from scipy.stats import ortho_group

d = 2
n_train = 50000
n_test = 10000

def gen(n):
  z = np.random.normal(size=n)
  X = np.zeros([d, n])
  X[0] = z
  rot = ortho_group.rvs(d)
  X = rot.dot(X).T
  return X

Xtrain = gen(n_train)
np.save('d{}_std_train.npy'.format(d), Xtrain)
Xtest = gen(n_test)
np.save('d{}_std_test.npy'.format(d), Xtest)

