import os
import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.distributions as tdist

# local imports
from utils.rbig_util import *
import args
from data.get_loader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import random
import pdb

SILENT = False
VERBOSE = False
DTYPE = torch.DoubleTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device='cpu'

TIME = 0

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

def main(DATA, lambd, train_loader, val_loader, log_batch=False, n_run=0, out_dir='outputs/'):
  test_bpd = 0
  val_loss = 0
  val_log_prob = 0
  total = 0
  rotation_matrices = []
  FAIL_FLAG = False
  time_keys = ['gen_bandwidth', 'gen_rotation', 'inverse_cdf', 'log_det', 'sampling', 'cur_data_proc', 'flow_loss']
  times = {key:[] for key in time_keys}

  with torch.no_grad():
      DATA = DATA.type(DTYPE).to(device)
      data_anchors = [DATA]
      bandwidths = []
      vectors = []
      kl_layer = [[] for _ in range(n_layer)]
      bpd_layer = [[] for _ in range(n_layer)]
      log_prob_layer = [[] for _ in range(n_layer)]
      log_det_train = torch.zeros(process_size).double().to(device)
      kl_layer_train = []
      for batch_idx, data in enumerate(val_loader):
          if type(data) is list or type(data) is tuple:
            data = data[0]
          total += data.shape[0]
          data = data.type(DTYPE).to(device)

          if batch_idx == 0 and False: # TODO: remove this 
            # show original data
            if args.dataset in ['FashionMNIST', 'MNIST']:
              sampling(rotation_matrices, [], [],
                       image_name='images/RBIG_samples_{}_init.png'.format(args.dataset),
                       channel=channel, image_size=image_size, process_size=10)
            else:
              # args.dataset in ['GaussianLine', 'GaussianMixture', 'uniform']:
              sampling(rotation_matrices, [], [],
                       image_name='images/RBIG_samples_{}_init.png'.format(args.dataset),
                       d=data.shape[1], sample_num=10000, process_size=100)

          if data.ndim == 4: # images
            data = dequantization(data, lambd)
            data = data.view(data.shape[0], -1)

          log_det = torch.zeros(data.shape[0]).double().to(device)
          log_det_logit = F.softplus(-data).sum() + F.softplus(data).sum() + np.prod(
                data.shape) * np.log(1 - 2 * lambd)

          # Pass the data through the first l-1 gaussian layer
          for prev_l in range(n_layer):
              # initialize rotation matrix
              if batch_idx == 0:
                  if TIME:
                    start = time()
                  bandwidth = generate_bandwidth(data_anchors[prev_l])
                  bandwidths.append(bandwidth)
                  if TIME:
                    times['gen_bandwidth'] += time() - start,
                    start = time()
                  rotation_matrix, Y = generate_orthogonal_matrix(data_anchors[prev_l], rot_type=args.rotation_type)
                  if rotation_matrix is None:
                    print('Failed')
                    FAIL_FLAG = True
                    break
                  rotation_matrices.append(rotation_matrix.to(device))
                  if TIME:
                    times['gen_rotation'] += time() - start,
                  vector = 2 * (torch.rand((1, data.shape[-1])) - 0.5).to(device)
                  vectors.append(vector)

              if data.dtype != bandwidth.dtype:
                bandwidth = bandwidth.type(data.dtype)

              # inverse CDF
              if TIME:
                start = time()
              inverse_l, cdf_mask, [log_cdf_l, cdf_mask_left], [log_sf_l, cdf_mask_right] \
                  = logistic_inverse_normal_cdf(data, bandwidth=bandwidth, datapoints=data_anchors[prev_l], inverse_cdf_by_thresh=args.inverse_cdf_by_thresh)
              if TIME:
                times['inverse_cdf'] += time() - start,
                start = time()
              log_det += compute_log_det(data, inverse_l, data_anchors[prev_l], cdf_mask,
                                               log_cdf_l, cdf_mask_left, log_sf_l, cdf_mask_right, h=bandwidth)
              if TIME:
                times['log_det'] += time() - start,
              # rotation
              data = torch.mm(inverse_l, rotation_matrices[prev_l])

              # Update cur_data
              if batch_idx == 0:
                  cur_data = data_anchors[prev_l]
                  update_data_arrays = []
                  assert (total_datapoints % process_size == 0), "Process_size does not divide total_datapoints!"
                  if TIME:
                    proc_start = time()
                  for b in range(total_datapoints // process_size):
                      if VERBOSE and b % 20 == 0:
                          print("Layer {0} generating new datapoints: {1}/{2}".format(prev_l, b,
                                                                                      total_datapoints // process_size))
                      cur_data_batch = cur_data[process_size * b: process_size * (b + 1), :]
                      # inverse CDF
                      if TIME:
                        start = time()
                      inverse_data, cdf_mask_train, [log_cdf_l_train, cdf_mask_left_train], [log_sf_l_train, cdf_mask_right_train] = logistic_inverse_normal_cdf(cur_data_batch, bandwidth=bandwidth,
                                                                       datapoints=data_anchors[prev_l], inverse_cdf_by_thresh=args.inverse_cdf_by_thresh)
                      if b == 0:
                        log_det_train += compute_log_det(cur_data_batch, inverse_data, data_anchors[prev_l], cdf_mask_train,
                                                         log_cdf_l_train, cdf_mask_left_train, log_sf_l_train, cdf_mask_right_train, h=bandwidth)
                        train_loss_curr_layer, train_log_prob_curr_layer = flow_loss(cur_data_batch, log_det)
                        kl_layer_train += train_loss_curr_layer.item(),

                      if TIME:
                        times['inverse_cdf'] += time() - start,
                      # rotation
                      cur_data_batch = torch.mm(inverse_data, rotation_matrices[prev_l])

                      update_data_arrays.append(cur_data_batch)
                  if TIME:
                    times['cur_data_proc'] += time() - proc_start,
                  cur_data = torch.cat(update_data_arrays, dim=0)
                  data_anchors.append(cur_data[:cur_data.shape[0]])
              val_loss_curr_layer, val_log_prob_curr_layer = flow_loss(data, log_det)
              check_cov(data)
              kl_layer[prev_l] += val_loss_curr_layer.item(),
              test_bpd_curr_layer = (val_loss_curr_layer.item() * data.shape[0] - log_det_logit) * (
                      1 / (np.log(2) * np.prod(data.shape))) + 8
              bpd_layer[prev_l] += test_bpd_curr_layer.item(),
              log_prob_layer[prev_l] += val_log_prob_curr_layer.item(),

          if not FAIL_FLAG:
            if TIME:
              start = time()
            val_loss_r, val_log_prob_r = flow_loss(data, log_det)
            check_cov(data)
            if TIME:
              times['flow_loss'] += time() - start,
            test_bpd_r = (val_loss_r.item() * data.shape[0] - log_det_logit) * (
                    1 / (np.log(2) * np.prod(data.shape))) + 8
            test_bpd += test_bpd_r * data.shape[0]
            val_loss += val_loss_r * data.shape[0]
            val_log_prob += val_log_prob_r * data.shape[0]

            if not SILENT and log_batch and batch_idx % 100 == 0:
                print("Batch {} loss {} (log prob: {}) bpd {}".format(batch_idx, val_loss_r, val_log_prob, test_bpd_r))

            if batch_idx == 0:
              if VERBOSE:
                print('Train KL: {}\n'.format(' / '.join(['layer{}: {:.4e}'.format(i+1, kl_train) for (i, kl_train) in enumerate(kl_layer_train)])))
              plt.plot(np.array(kl_layer_train))
              plt.savefig('{}/images/RBIG_trainKLbyLayer_{}_{}_run{}_tmp.png'.format(out_dir, args.dataset, args.rotation_type, n_run))
              plt.close()

            kl_means, bpd_means, log_prob_means = [], [], []
            for li in range(n_layer):
              curr_kl = np.array(kl_layer[li])
              curr_bpd = np.array(bpd_layer[li])
              curr_log_prob = np.array(log_prob_layer[li])
              if VERBOSE:
                print('Layer {}: mean={:.3e} / std={:.3e} / max={:.3e} / min={:.3e}'.format(
                  li, curr.mean(), curr.std(), curr.max(), curr.min()))
              kl_means += curr_kl.mean(),
              bpd_means += curr_bpd.mean(),
              log_prob_means += curr_log_prob.mean(),
            plt.plot(kl_means)
            plt.savefig('{}/images/RBIG_KLbyLayer_{}_{}_run{}_tmp.png'.format(out_dir, args.dataset, args.rotation_type, n_run))
            plt.close()
            plt.plot(bpd_means)
            plt.savefig('{}/images/RBIG_BPDbyLayer_{}_{}_run{}_tmp.png'.format(out_dir, args.dataset, args.rotation_type, n_run))
            plt.close()
            plt.plot(log_prob_means)
            plt.savefig('{}/images/RBIG_LogProbbyLayer_{}_{}_run{}_tmp.png'.format(out_dir, args.dataset, args.rotation_type, n_run))
            plt.close()

          else:
            break

      print('KL by layer.')
      for li in range(n_layer):
        kl_layer[li] = np.array(kl_layer[li])
        print('Layer {}: mean={:.3e} / std={:.3e} / max={:.3e} / min={:.3e}'.format(
           li, kl_layer[li].mean(), kl_layer[li].std(), kl_layer[li].max(), kl_layer[li].min()))
      print()
      plt.plot([each.mean() for each in kl_layer])
      plt.savefig('{}/images/RBIG_KLbyLayer_{}_{}_run{}.png'.format(out_dir, args.dataset, args.rotation_type, n_run))
      plt.close()

      print('BPD by layer.')
      for li in range(n_layer):
        bpd_layer[li] = np.array(bpd_layer[li])
        print('Layer {}: mean={:.3e} / std={:.3e} / max={:.3e} / min={:.3e}'.format(
           li, bpd_layer[li].mean(), bpd_layer[li].std(), bpd_layer[li].max(), bpd_layer[li].min()))
      print()
      plt.plot([each.mean() for each in bpd_layer])
      plt.savefig('{}/images/RBIG_BPDbyLayer_{}_{}_run{}.png'.format(out_dir, args.dataset, args.rotation_type, n_run))
      plt.close()

      print('Log-prob by layer.')
      for li in range(n_layer):
        log_prob_layer[li] = np.array(log_prob_layer[li])
        print('Layer {}: mean={:.3e} / std={:.3e} / max={:.3e} / min={:.3e}'.format(
           li, log_prob_layer[li].mean(), log_prob_layer[li].std(), log_prob_layer[li].max(), log_prob_layer[li].min()))
      print()
      plt.plot([each.mean() for each in log_prob_layer])
      plt.savefig('{}/images/RBIG_LogProbbyLayer_{}_{}_run{}.png'.format(out_dir, args.dataset, args.rotation_type, n_run))
      plt.close()

      if FAIL_FLAG:
        return None, None

      if not SILENT:
        print("Total loss {} (log prob: {}) bpd {}".format(val_loss / total, val_log_prob / total, test_bpd / total))

      if TIME:
        start = time()
      if args.dataset in ['FashionMNIST', 'MNIST']:
        sampling(rotation_matrices, data_anchors, bandwidths,
                 image_name='{}/images/RBIG_samples_{}_{}_layer{}_run{}.png'.format(out_dir, args.dataset, args.rotation_type, args.n_layer, n_run),
                 channel=channel, image_size=image_size, process_size=10)
      else:
        if data_anchors[0].shape[1] > 2:
          proj_mtrxs = []
          for _ in range(10):
            v1 = np.random.randn(data_anchors[0].shape[1])
            v2 = np.random.randn(data_anchors[0].shape[1])
            v2 -= v1.dot(v2) * v1
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            proj_mtrx = np.stack([v1, v2]).T
            proj_mtrxs += proj_mtrx,
        # args.dataset in ['GaussianLine', 'GaussianMixture', 'uniform']:
        for li, pts in enumerate(data_anchors):
          pts = pts.cpu().numpy()
          plt.figure(figsize=[8,8])

          if pts.shape[1] == 2:
            plt.hist2d(img2d[:,0], img2d[:,1], bins=[100,100])
            plt.xlim([-2.5, 2.5])
            plt.ylim([-2.5, 2.5])
            plt.savefig('{}/images/RBIG_transformed_{}_{}_layer{}_run{}.png'.format(out_dir, args.dataset, args.rotation_type, li, n_run))
            plt.close()
          else:
            for pi,proj_mtrx in enumerate(proj_mtrxs):
              img2d = pts.dot(proj_mtrx)
              plt.hist2d(img2d[:,0], img2d[:,1], bins=[100,100])
              plt.xlim([-2.5, 2.5])
              plt.ylim([-2.5, 2.5])
              plt.savefig('{}/images/RBIG_transformed_{}_{}_layer{}_run{}_proj{}.png'.format(out_dir, args.dataset, args.rotation_type, li, n_run, pi))
              plt.close()
        if False: # TODO: remove this
          sampling(rotation_matrices, data_anchors.copy(), bandwidths.copy(),
                   image_name='{}/images/RBIG_samples_{}_{}_layer{}_run{}.png'.format(out_dir, args.dataset, args.rotation_type, args.n_layer, n_run),
                   d=data.shape[1], sample_num=10000, process_size=100)
      if TIME:
        times['sampling'] += time() - start,

      if len(rotation_matrices) > 0:
        rotation_matrices = torch.stack(rotation_matrices, 0)
        rotation_matrices = rotation_matrices.detach().cpu().numpy()
        fname = os.path.join(out_dir, 'rotMtrx_{}_layer{}_run{}.npy'.format(args.rotation_type, args.n_layer, n_run))
        np.save(fname, rotation_matrices)

      if TIME:
        for key in time_keys:
          times[key] = np.array(times[key])
          print('{}: mean={:.3e} / std={:.3e}'.format(key, times[key].mean(), times[key].std()))
      return val_loss / total, test_bpd / total

if __name__ == '__main__':
  total_datapoints = args.bt # 10000
  process_size = 100
  n_layer = args.n_layer
  if not SILENT:
    print("Total layer {}".format(n_layer))

  if args.dataset == 'MNIST':
    lambd = 1e-5
  elif args.dataset == 'FashionMNIST':
    lambd = 1e-6
  else:
    lambd = 1e-5

  if not SILENT:
    print("Loading dataset {}".format(args.dataset))
  normal_distribution = tdist.Normal(0, 1)

  means, stds = [], []
  # n_layers = list(range(20))
  # n_layers = range(10)

  args.bt = total_datapoints
  train_loader, val_loader = get_loader(args, is_train=True)
  # args.bt = process_size
  # val_loader = get_loader(args, is_train=False)
  if not SILENT:
    print('Val_loader: #', len(val_loader))

  train_iter = iter(train_loader)
  if type(train_iter) is tuple:
    train_iter = train_iter[0]
  DATA = next(train_iter)
  if type(DATA) is list or type(DATA) is tuple:
    DATA = DATA[0]
  DATA = DATA.to(device)
  if DATA.ndim == 4: # images
    channel = DATA.shape[1]
    image_size = DATA.shape[-1]
    DATA = dequantization(DATA, lambd)
    DATA = DATA.view(DATA.shape[0], -1)

  n_runs = 1
  losses = []
  for i in range(n_runs):
    if not SILENT:
      print(i)
    set_seed()
    out_dir = os.path.join('outputs/', 'RBIG', args.dataset)
    out_dir = os.path.join(out_dir, '{}_L{}_bt{}{}'.format(args.rotation_type, args.n_layer, args.bt, args.save_suffix))
    if os.path.exists(out_dir):
      print('Dir exist:', out_dir)
      proceed = input("Do you want to proceed? (y/N)")
      if 'y' not in proceed:
        print('Exiting. Bye!')
        exit(0)
    print("out_dir:", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    loss, _ = main(DATA, lambd, train_loader, val_loader, n_run=i, out_dir=out_dir)
    if loss is not None:
      losses += loss.item(),
  best_60 = sorted(losses)[:60]
  losses = np.array(best_60)
  print('\nType:', args.rotation_type)
  print('n_layer={}'.format(args.n_layer))
  print('Loss: mean={} / std={}'.format(losses.mean(), losses.std()))

  means += losses.mean(),
  stds += losses.std(),

  # means = np.array(means)
  # stds = np.array(stds)
  # means = np.minimum(means, means.min()*3)
  # stds = np.minimum(stds, stds.min()*10)
  # plt.errorbar(n_layers, means, stds, marker='^')
  # plt.savefig('plt_{}_layer{}_runs{}.png'.format(args.rotation_type, max(n_layers), n_runs))


