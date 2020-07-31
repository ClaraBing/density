"""Train Glow on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import numpy as np
import os
import random

def str2bool(s):
    return s.lower().startswith('t')

class BaseArgs:
  def __init__(self):
    self.is_train = None
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('--mode', type=str, choices=['train', 'test'])
    # Model
    self.parser.add_argument('--model', type=str, default='glow', choices=['glow', 'rbig'])
    # -- Glow
    self.parser.add_argument('--in-channels', type=int, help='Number of channels in input layer')
    self.parser.add_argument('--mid-channels', default=512, type=int, help='Number of channels in hidden layers')
    self.parser.add_argument('--num-levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    self.parser.add_argument('--num-steps', '-K', default=32, type=int, help='Number of steps of flow in each level')
    self.parser.add_argument('--layer-type', type=str, choices=['conv', 'fc'])
    # -- RBIG
    self.parser.add_argument('--n-layer', type=int, default=10,
                             help='Number of layers (iterations) used for RBIG.')
    self.parser.add_argument('--rotation-type', type=str, choices=['random', 'PCA', 'ICA'],
                             help="Rotation type for RBIG.")
    # Data
    self.parser.add_argument('--dataset', type=str, choices=['cifar10', 'MNIST', 'FashionMNIST',
                             'GAS16', 'GAS128',
                             'GaussianLine', 'GaussianMixture', 'uniform'])
    self.parser.add_argument('--num-workers', type=int, default=4)
    self.parser.add_argument('--bt', default=64, type=int, help='Batch size')
    self.parser.add_argument('--dlen', default=1, type=int, 
                             help='Dataset length. Used for synthetic dataset e.g. GaussianLine.')
    self.parser.add_argument('--d', type=int, default=16,
                             help="Data dimension. Used by GaussianLine")
    self.parser.add_argument('--fxdir', type=str, default='',
                             help="File path to a pre-stored direction; used for GaussianLine.")
    self.parser.add_argument('--fdata', type=str, help="Path to data file.")
    self.parser.add_argument('--norm-by-col', type=int, default=0,
                             help="Whether to normalize each data column by its std.")
    # Misc
    self.parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    self.parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    self.parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    self.parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    self.parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')
    # wandb
    self.parser.add_argument('--project', type=str)
    self.parser.add_argument('--wb-name', type=str)
    

  def parse(self):
    args = self.parser.parse_args()
    return args

