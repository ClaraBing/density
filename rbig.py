import os
import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.distributions as tdist

# local imports
from util.rbig_util import *
import args
from data.get_loader import *

import random
import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = args.TrainArgs().parse()

os.makedirs('images', exist_ok = True)

# fix random seed
def set_seed(seed=None):
  if seed is None:
    seed = random.randint(0, 1000)
  torch.manual_seed(seed)
  np.random.seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

def main(DATA, lambd, train_loader, val_loader, log_batch=False):
  test_bpd = 0
  val_loss = 0
  total = 0
  rotation_matrices = []
  FAIL_FLAG = False
  with torch.no_grad():
      data_anchors = [DATA]
      bandwidths = []
      vectors = []
      for batch_idx, data in enumerate(val_loader):
          if type(data) is tuple:
            data = data[0]
          total += data.shape[0]
          data = data.to(device)

          if data.ndim == 4: # images
            data = dequantization(data, lambd)
            data = data.view(data.shape[0], -1)

          log_det = torch.zeros(data.shape[0]).to(device)
          log_det_logit = F.softplus(-data).sum() + F.softplus(data).sum() + np.prod(
                data.shape) * np.log(1 - 2 * lambd)

          # Pass the data through the first l-1 gaussian layer
          for prev_l in range(n_layer):
              # initialize rotation matrix
              if batch_idx == 0:
                  bandwidth = generate_bandwidth(data_anchors[prev_l])
                  bandwidths.append(bandwidth)
                  rotation_matrix = generate_orthogonal_matrix(data_anchors[prev_l], rot_type=args.rotation_type)
                  if rotation_matrix is None:
                    FAIL_FLAG = True
                    break
                  rotation_matrices.append(rotation_matrix.to(device))
                  vector = 2 * (torch.rand((1, data.shape[-1])) - 0.5).to(device)
                  vectors.append(vector)

              if data.dtype != bandwidth.dtype:
                bandwidth = bandwidth.type(data.dtype)

              inverse_l, cdf_mask, [log_cdf_l, cdf_mask_left], [log_sf_l, cdf_mask_right] \
                  = logistic_inverse_normal_cdf(data, bandwidth=bandwidth, datapoints=data_anchors[prev_l])
              log_det += compute_log_det(data, inverse_l, data_anchors[prev_l], cdf_mask,
                                               log_cdf_l, cdf_mask_left, log_sf_l, cdf_mask_right, h=bandwidth)
              data = torch.mm(inverse_l, rotation_matrices[prev_l])

              # Update cur_data
              if batch_idx == 0:
                  cur_data = data_anchors[prev_l]
                  update_data_arrays = []
                  assert (total_datapoints % process_size == 0), "Process_size does not divide total_datapoints!"
                  for b in range(total_datapoints // process_size):
                      if b % 20 == 0:
                          print("Layer {0} generating new datapoints: {1}/{2}".format(prev_l, b,
                                                                                      total_datapoints // process_size))
                      cur_data_batch = cur_data[process_size * b: process_size * (b + 1), :]
                      inverse_data, _, _, _ = logistic_inverse_normal_cdf(cur_data_batch, bandwidth=bandwidth,
                                                                       datapoints=data_anchors[prev_l])
                      cur_data_batch = torch.mm(inverse_data, rotation_matrices[prev_l])

                      update_data_arrays.append(cur_data_batch)
                  cur_data = torch.cat(update_data_arrays, dim=0)
                  data_anchors.append(cur_data[:cur_data.shape[0]])

          if not FAIL_FLAG:
            val_loss_r = flow_loss(data, log_det)
            test_bpd_r = (val_loss_r.item() * data.shape[0] - log_det_logit) * (
                    1 / (np.log(2) * np.prod(data.shape))) + 8

            test_bpd += test_bpd_r * data.shape[0]
            val_loss += val_loss_r * data.shape[0]
            if log_batch and batch_idx % 100 == 0:
                print("Batch {} loss {} bpd {}".format(batch_idx, val_loss_r, test_bpd_r))
          else:
            break

      print("Total loss {} bpd {}".format(val_loss / total, test_bpd / total))

      # sampling(rotation_matrices, data_anchors, bandwidths, image_name='images/RBIG_samples_{}.png'.format(args.dataset),
      #                channel=channel, image_size=image_size, process_size=10)
      if FAIL_FLAG:
        return None, None
      else:
        return val_loss / total, test_bpd / total

if __name__ == '__main__':
  total_datapoints = args.bt # 10000
  process_size = 100
  n_layer = args.n_layer
  print("Total layer {}".format(n_layer))

  if args.dataset == 'MNIST':
    lambd = 1e-5
  elif args.dataset == 'FashionMNIST':
    lambd = 1e-6
  else:
    lambd = 1e-5

  print("Loading dataset {}".format(args.dataset))
  normal_distribution = tdist.Normal(0, 1)

  train_loader, val_loader = get_loader(args, is_train=True)
  print('Val_loader: #', len(val_loader))

  train_iter = iter(train_loader)
  if type(train_iter) is tuple:
    train_iter = train_iter[0]
  DATA = next(train_iter)
  DATA = DATA.to(device)
  if DATA.ndim == 4: # images
    DATA = dequantization(DATA, lambd)
    DATA = DATA.view(DATA.shape[0], -1)

  n_runs = 100
  losses = []
  for i in range(n_runs):
    print(i)
    set_seed()
    loss, _ = main(DATA, lambd, train_loader, val_loader)
    if loss is not None:
      losses += loss.item(),
  best_60 = sorted(losses)[:60]
  losses = np.array(best_60)
  print('\nType:', args.rotation_type)
  print('d={} / n_layer={}'.format(args.d, args.n_layer))
  print('Loss: mean={} / std={}'.format(losses.mean(), losses.std()))


