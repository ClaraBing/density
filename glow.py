"""Train Glow on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import pdb

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

import args
import utils
from models import Glow
from data.get_loader import get_loader

from tqdm import tqdm

def str2bool(s):
    return s.lower().startswith('t')

args = args.TrainArgs().parse()

if args.dataset == 'MNIST':
  image_size = (1, 28, 28)
elif args.dataset == 'CIFAR10':
  image_size = (3, 32, 32)

import wandb
wandb.init(project=args.project, name=args.wb_name, config=args)

def main():
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    args.bt *= max(1, len(args.gpu_ids))

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
  
      loss_fn = utils.NLLLoss().to(device)
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
  loss_meter = utils.AverageMeter()
  with tqdm(total=len(trainloader.dataset)) as progress_bar:
    for bi, x in enumerate(trainloader):
      if type(x) is tuple or type(x) is list:
        x = x[0]
      x = x.type(torch.FloatTensor).to(device)
      optimizer.zero_grad()
      z, sldj = net(x, reverse=False)
      loss = loss_fn(z, sldj)
      loss_meter.update(loss.item(), x.size(0))
      loss.backward()
      if max_grad_norm > 0:
          utils.clip_grad_norm(optimizer, max_grad_norm)
      optimizer.step()
      scheduler.step(global_step)

      progress_bar.set_postfix(nll=loss_meter.avg,
                               bpd=utils.bits_per_dim(x, loss_meter.avg),
                               lr=optimizer.param_groups[0]['lr'])
      progress_bar.update(x.size(0))
      global_step += x.size(0)

      if bi % 100 == 0:
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
    z = torch.randn((batch_size, image_size[0], image_size[1], image_size[2]), dtype=torch.float32, device=device)
  elif layer_type == 'fc':
    z = torch.randn((batch_size, in_channels), dtype=torch.float32, device=device)
  x, _ = net(z, reverse=True)
  x = torch.sigmoid(x)

  return x


@torch.no_grad()
def test(epoch, net, testloader, device, loss_fn, num_samples, layer_type, in_channels=16):
  global best_loss
  net.eval()
  loss_meter = utils.AverageMeter()
  with tqdm(total=len(testloader.dataset)) as progress_bar:
    for x in testloader:
      if type(x) is tuple or type(x) is list:
        x = x[0]
      x = x.type(torch.FloatTensor).to(device)
      z, sldj = net(x, reverse=False)
      loss = loss_fn(z, sldj)
      loss_meter.update(loss.item(), x.size(0))
      progress_bar.set_postfix(nll=loss_meter.avg,
                               bpd=utils.bits_per_dim(x, loss_meter.avg))
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
  out_dir = os.path.join('samples', args.dataset+'_glow')
  os.makedirs(out_dir, exist_ok=True)
  images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
  torchvision.utils.save_image(images_concat, os.path.join(out_dir, 'epoch_{}.png'.format(epoch)))

if __name__ == '__main__':
    best_loss = 0
    global_step = 0
    main()
