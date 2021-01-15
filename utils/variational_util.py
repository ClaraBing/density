import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from scipy.stats import ortho_group

import pdb

import wandb

USE_WANDB = True
DTYPE = torch.FloatTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEBUG = True

def to_tensor(data):
    if not isinstance(data, torch.Tensor):
      data = torch.tensor(data)
    return data.type(DTYPE).to(device)

class variationalNet(torch.nn.Module):

    def __init__(self, num_hidden_nodes, n_layers=1, embed_size=1, pos_type='smoothL1'):
        super(variationalNet, self).__init__()
        
        layers = [nn.Linear(embed_size, num_hidden_nodes)]
        for _ in range(n_layers):
          layers += nn.ReLU(),
          layers += nn.Linear(num_hidden_nodes, num_hidden_nodes),
        layers += nn.ReLU(),
        layers += nn.Linear(num_hidden_nodes, embed_size),
        self.net = nn.Sequential(*layers)
        if pos_type == 'smoothL1':
          self.make_positive = lambda x: nn.SmoothL1Loss(reduction='none')(x, torch.zeros(x.shape).to(device))
        elif pos_type == 'sigmoid':
          self.make_positive = nn.Sigmoid()
        elif pos_type == 'square':
          self.make_positive = lambda x: torch.square(x)
        elif pos_type == 'exp':
          self.make_positive = lambda x: torch.exp(x)

    def forward(self, x):
        # out = x
        # out = self.input(out)
        # out = self.relu(out)
        # out = self.output(out)
        out = self.net(x)
        # out = self.make_positive(out, torch.zeros(out.shape).to(device))
        out = self.make_positive(out)
        return out

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def variational_KL(X, n_iters, n=1000, num_hidden_nodes=10, det_lambda=0.1, det_every=100,
    lr=1e-2, wd=1e-4, patience=200, A_mode='GD', pos_type='smoothL1', n_layers=1):
    """
    Input:
    X: torch tensor N * D
    n: number of standand gaussian normal dim=D
    Output: 
    A: torch tensor D * D
    """
    X = to_tensor(X)
    N, D = X.shape
    mean = np.ones(D)
    cov = np.eye(D,D)
    Z = to_tensor(np.random.multivariate_normal(mean, cov, n))
    g_function = variationalNet(num_hidden_nodes, n_layers=n_layers, pos_type=pos_type).to(device)
    params = [{'params': g_function.parameters()}]
    if A_mode == 'fixed':
      A = to_tensor(np.eye(D))
    elif A_mode == 'givens':
      A = ortho_group.rvs(D)
      A = to_tensor(A)
    elif A_mode == 'GD':
      A = torch.randn((D,D), requires_grad=True, device=device)
      torch.nn.init.orthogonal_(A)
      params += {'params': A},
    else:
      raise NotImplementedError("A_mode should be in ['GD', 'fixed', 'givens'].\n Got {}".format(A_mode))
    optimizer = optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, verbose=True)
    g_function.train()

    def helper_loss(M):
      sum_loss = to_tensor(torch.tensor(0.0)).to(device)
      # Compute first term for X
      y_X = to_tensor(torch.flatten(M.T.matmul(X.T))).view(-1,1) 
      g_y_X = g_function(y_X)
      log_g_y_X = torch.log(g_y_X)
      sum_loss -= torch.mean(log_g_y_X)
      # Compute second term for Z
      y_Z = torch.flatten(M.T.matmul(Z.T)).view(-1,1)
      g_y_Z = g_function(y_Z)
      sum_loss += torch.mean(g_y_Z)
      return sum_loss, y_X, g_y_X, y_Z, g_y_Z

    for i in range(n_iters):
        optimizer.zero_grad()
        # sum_loss = to_tensor(torch.tensor(0.0)).to(device)
        # # Compute first term for X
        # y_X = to_tensor(torch.flatten(A.T.matmul(X.T))).view(-1,1) 
        # g_y_X = g_function(y_X)
        # log_g_y_X = torch.log(g_y_X)
        # sum_loss -= torch.mean(log_g_y_X)
        # # Compute second term for Z
        # y_Z = torch.flatten(A.T.matmul(Z.T)).view(-1,1)
        # g_y_Z = g_function(y_Z)
        # sum_loss += torch.mean(g_y_Z)
        sum_loss, y_X, g_y_X, y_Z, g_y_Z = helper_loss(A)
        if i % det_every == 0:
            det_lambda *= 0.5
        raw_loss = sum_loss
        reg_loss = det_lambda * torch.log(torch.abs(torch.det(A)))
        sum_loss -= reg_loss
        sum_loss.backward()
        if torch.isnan(sum_loss):
          print("Nan.")
          break
        optimizer.step()
        scheduler.step(raw_loss)
        if i % 20 == 0:
          print(i, sum_loss.item())
          if USE_WANDB:
            wandb.log({
              'var_loss': sum_loss.item(),
              'var_loss_raw': raw_loss.item(),
              'var_reg': reg_loss.item()})
            if DEBUG:
              wandb.log({
                'var_input_mean': y_X.mean().item(),
                'var_input_Z_mean': y_Z.mean().item(),
                'var_out_mean': g_y_X.mean().item(),
                'var_out_Z_mean': g_y_Z.mean().item()})
        if A_mode == 'GD':
          with torch.no_grad():
              _, ss, _ = np.linalg.svd(A.detach().cpu())
              if ss[0] > 1e-5:
                  A /= ss[0]
        elif A_mode == 'givens':
          # alg 1 from https://arxiv.org/pdf/1312.0624.pdf
          n_A_iters = D**2
          for _ in range(n_A_iters):
            i, j = np.random.choice(range(D), 2, replace=False)
            if j < i: i,j = j, i
            # TODO: find step size
            best_loss, best_G = None, None
            for eta in np.arange(-2*np.pi, 2*np.pi, 20):
              # Givens matrix
              G = to_tensor(np.eye(D))
              G[i,i], G[j,j] = np.cos(eta), np.cos(eta)
              G[i,j], G[j,i] = -np.sin(eta), np.sin(eta)
              tmp = A.mm(G)
              cur_loss, _, _, _, _ = helper_loss(tmp)
              if best_loss is None or cur_loss < best_loss:
                best_loss = cur_loss
                best_G = G.clone()
            A = A.mm(best_G)
    return A.detach()


if __name__ == '__main__':
  D = 2
  mu = 4
  N = 10000
  X = np.random.normal(loc=mu, scale=1, size=(N, D))
  n_iters=1000
  A_mode = 'givens'
  
  # for wd in [1e-6, 1e-5, 1e-4]:
  for wd in [1e-4, 1e-5, 1e-6]:
    for lr in [0.01, 0.003, 0.001]:
      for num_hidden_nodes in [160, 320, 640]:
        for n_layers in [1, 2, 3]:
          for pos_type in ['square']:
            cfgs={'var_lr':lr, 'var_wd':wd, 'var_num_hidden_nodes':num_hidden_nodes}
            cfgs['pos_type'] = pos_type
            cfgs['var_n_layers'] = n_layers
            cfgs['data'] = '1dGaussian'
            cfgs['D'] = D
            cfgs['data_mu'] = mu
            cfgs['A_mode'] = A_mode
            wandb.init(project='density', config=cfgs)
            variational_KL(X, n_iters, n=10000, num_hidden_nodes=num_hidden_nodes,
                           det_lambda=0.1, det_every=100, lr=lr, wd=wd, patience=200, A_mode=A_mode,
                           n_layers=n_layers, pos_type=pos_type)
            wandb.finish()

