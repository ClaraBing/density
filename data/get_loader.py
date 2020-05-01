import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from .dataset_line import GaussianLine
from .dataset_mixture import GaussianMixture
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
    return data.DataLoader(dset_train, args.bt, shuffle=True, num_workers=args.num_workers), \
           data.DataLoader(dset_val, args.bt, shuffle=False, num_workers=args.num_workers)

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

