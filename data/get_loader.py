import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from .dataset_gas8 import GAS8
from .dataset_gas16 import GAS16
from .dataset_gas128 import GAS128
from .dataset_hepmass import HEPMASS
from .dataset_line import GaussianLine
from .dataset_mixture import GaussianMixture
from .dataset_miniboone import MINIBooNE
from .dataset_mnist import MNISTtab
from .dataset_uniform import Uniform

import pdb

def get_loader(args, is_train):
  dataset = args.dataset

  if dataset.lower() == 'cifar10':
    dset = get_cifar(is_train)
  elif dataset == 'GaussianLine':
    # dset = GaussianLine(args.fdata)
    xdir = np.load(args.fxdir) if args.fxdir else None
    dset = GaussianLine(args.d, bt=args.bt, dlen=args.dlen, xdir=xdir)
  elif dataset == 'GaussianMixture':
    dset = GaussianMixture(dlen=args.dlen)
  elif dataset == 'uniform':
    dset = Uniform(dlen=args.dlen)
  elif dataset == 'GAS8':
    dset = GAS8(args.norm_by_col)
  elif dataset == 'GAS16':
    dset = GAS16(args.norm_by_col)
  elif dataset == 'GAS128':
    dset = GAS128(args.norm_by_col)
  elif dataset == 'hepmass':
    dset = HEPMASS()
  elif dataset == 'miniboone':
    dset = MINIBooNE()
  elif dataset == 'MNISTtab':
   # treat mnist as tabular data
   dset = MNISTtab()
  elif dataset == 'MNIST':
    channel = 1
    image_size = 28
    lambd = 1e-5
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    dset = MNIST(os.path.join('datasets', 'mnist'), train=is_train, download=True,
                          transform=transform)
  elif dataset == 'FMNIST':
    channel = 1
    image_size = 28
    lambd = 1e-6
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    dset = FashionMNIST(os.path.join('datasets', 'fmnist'), train=is_train, download=True,
                          transform=transform)
  else:
    print('args.dataset:', dataset)
    raise NotImplementedError('Sorry but we are not supporting dataset {}. \n Ciao~! :)'.format(dataset))

  if is_train and args.use_val:
    n_train = int(0.8 * len(dset))
    n_val = len(dset) - n_train
    dset_train, dset_val = torch.utils.data.random_split(dset, [n_train, n_val])
    train_loader = data.DataLoader(dset_train, args.bt, shuffle=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(dset_val, args.bt_test, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader

  return data.DataLoader(dset, args.bt, shuffle=is_train)

def get_cifar(is_train):
    if is_train:
      # No normalization applied, since Glow expects inputs in (0, 1)
      transform_train = transforms.Compose([
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor()
      ])
      trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
      return trainset
    else:
      transform_test = transforms.Compose([
          transforms.ToTensor()
      ])
      testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
      return testset

