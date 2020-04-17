"""Train Glow on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util
from data_loader import get_loader

from models import Glow
from tqdm import tqdm

import pdb

parser = argparse.ArgumentParser(description='Glow on CIFAR-10')

def str2bool(s):
    return s.lower().startswith('t')

parser.add_argument('--mode', type=str, choices=['train', 'test'])
# Model
parser.add_argument('--model', type=str, default='glow', choices=['glow'])
parser.add_argument('--in-channels', type=int, required=True, help='Number of channels in input layer')
parser.add_argument('--mid-channels', default=512, type=int, help='Number of channels in hidden layers')
parser.add_argument('--num-levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
parser.add_argument('--num-steps', '-K', default=32, type=int, help='Number of steps of flow in each level')
parser.add_argument('--layer-type', type=str, choices=['conv', 'fc'])
# Training
parser.add_argument('--batch-size', default=64, type=int, help='Batch size per GPU')
parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
parser.add_argument('--wd', default=1e-5, type=float, help="Weight decay.")
parser.add_argument('--use-val', type=int, help="Whether to use a val set during training.")
parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
# Data
parser.add_argument('--dataset', type=str, choices=['cifar10', '2Dline', '16Dline'])
parser.add_argument('--fdata', type=str, help="Path to data file.")
# Misc
parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')
# wandb
parser.add_argument('--project', type=str)
parser.add_argument('--wb-name', type=str)

args = parser.parse_args()

import wandb
wandb.init(project=args.project, name=args.wb_name, config=args)

def main():
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Model
    print('Building model..')
    net = Glow(in_channels=args.in_channels,
               mid_channels=args.mid_channels,
               num_levels=args.num_levels,
               num_steps=args.num_steps,
               layer_type=args.layer_type)
    net = net.to(device)
    print('Model built.')
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    # Train / Test loop
    if args.mode == 'train':
      trainloader, testloader = get_loader(args, is_train=True)
      start_epoch = 0
      if args.resume:
          # Load checkpoint.
          print('Resuming from checkpoint at ckpts/best.pth.tar...')
          assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
          checkpoint = torch.load('ckpts/best.pth.tar')
          net.load_state_dict(checkpoint['net'])
          global best_loss
          global global_step
          best_loss = checkpoint['test_loss']
          start_epoch = checkpoint['epoch']
          global_step = start_epoch * len(trainset)
  
      loss_fn = util.NLLLoss().to(device)
      if args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
      elif args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)
      scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))
  
      for epoch in range(start_epoch, start_epoch + args.num_epochs):
          train(epoch, net, trainloader, device, optimizer, scheduler,
                loss_fn, args.max_grad_norm)
          test(epoch, net, testloader, device, loss_fn, args.num_samples, args.layer_type, args.in_channels)

    elif args.mode == 'test':
      testloader = get_loader(args, is_train=False)


@torch.enable_grad()
def train(epoch, net, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm):
  global global_step
  print('\nEpoch: %d' % epoch)
  net.train()
  loss_meter = util.AverageMeter()
  with tqdm(total=len(trainloader.dataset)) as progress_bar:
    for bi, x in enumerate(trainloader):
      if type(x) is tuple or type(x) is list:
        x = x[0]
      x = x.type(torch.FloatTensor).to(device)
      optimizer.zero_grad()
      # pdb.set_trace()
      z, sldj = net(x, reverse=False)
      loss = loss_fn(z, sldj)
      loss_meter.update(loss.item(), x.size(0))
      loss.backward()
      if max_grad_norm > 0:
          util.clip_grad_norm(optimizer, max_grad_norm)
      optimizer.step()
      scheduler.step(global_step)

      progress_bar.set_postfix(nll=loss_meter.avg,
                               bpd=util.bits_per_dim(x, loss_meter.avg),
                               lr=optimizer.param_groups[0]['lr'])
      progress_bar.update(x.size(0))
      global_step += x.size(0)

      if bi % 500 == 0:
        wandb.log({
          'loss': loss,
        })


@torch.no_grad()
def sample(net, layer_type, batch_size, device, in_channels=16):
  """Sample from RealNVP model.

  Args:
      net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
      batch_size (int): Number of samples to generate.
      device (torch.device): Device to use.
  """
  if layer_type == 'conv':
    z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
  elif layer_type == 'fc':
    z = torch.randn((batch_size, in_channels), dtype=torch.float32, device=device)
  x, _ = net(z, reverse=True)
  x = torch.sigmoid(x)

  return x


@torch.no_grad()
def test(epoch, net, testloader, device, loss_fn, num_samples, layer_type, in_channels=16):
  global best_loss
  net.eval()
  loss_meter = util.AverageMeter()
  with tqdm(total=len(testloader.dataset)) as progress_bar:
    for x in testloader:
      if type(x) is tuple or type(x) is list:
        x = x[0]
      x = x.type(torch.FloatTensor).to(device)
      z, sldj = net(x, reverse=False)
      loss = loss_fn(z, sldj)
      loss_meter.update(loss.item(), x.size(0))
      progress_bar.set_postfix(nll=loss_meter.avg,
                               bpd=util.bits_per_dim(x, loss_meter.avg))
      progress_bar.update(x.size(0))

  # Save checkpoint
  if loss_meter.avg < best_loss:
      print('Saving...')
      state = {
          'net': net.state_dict(),
          'test_loss': loss_meter.avg,
          'epoch': epoch,
      }
      os.makedirs('ckpts', exist_ok=True)
      torch.save(state, 'ckpts/best.pth.tar')
      best_loss = loss_meter.avg

  # Save samples and data
  images = sample(net, layer_type, num_samples, device, in_channels)
  os.makedirs(os.path.join('samples', args.dataset), exist_ok=True)
  images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
  torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    best_loss = 0
    global_step = 0
    main()
