import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

import pdb

import wandb

USE_WANDB = True
DTYPE = torch.FloatTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEBUG = True

def to_tensor(data):
    return torch.tensor(data).type(DTYPE).to(device)

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
    lr=1e-2, wd=1e-4, patience=200, fix_A=False, pos_type='smoothL1', n_layers=1):
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
    if fix_A:
      A = torch.eye(D).to(device)
    else:
      A = torch.randn((D,D), requires_grad=True, device=device)
      torch.nn.init.orthogonal_(A)
      params += {'params': A},
    optimizer = optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, verbose=True)
    g_function.train()
    for i in range(n_iters):
        optimizer.zero_grad()
        sum_loss = to_tensor(torch.tensor(0.0)).to(device)
        # Compute first term for X
        y_X = to_tensor(torch.flatten(A.T.matmul(X.T))).view(-1,1) 
        g_y_X = g_function(y_X)
        log_g_y_X = torch.log(g_y_X)
        sum_loss -= torch.mean(log_g_y_X)
        # Compute second term for Z
        y_Z = torch.flatten(A.T.matmul(Z.T)).view(-1,1)
        g_y_Z = g_function(y_Z)
        sum_loss += torch.mean(g_y_Z)
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
                'var_out_mean': g_y_X.mean().item(),
                'var_out_Z_mean': g_y_Z.mean().item()})
        if not fix_A:
          with torch.no_grad():
              _, ss, _ = np.linalg.svd(A.detach().cpu())
              if ss[0] > 1e-5:
                  A /= ss[0]
    return A.detach()


if __name__ == '__main__':
  D = 1
  mu = 2
  N = 10000
  X = np.random.normal(loc=mu, scale=1, size=(N, 1))
  n_iters=600
  
  # for wd in [1e-6, 1e-5, 1e-4]:
  for wd in [0]:
    for lr in [0.01, 0.03, 0.003]:
      for num_hidden_nodes in [80, 160]:
        for n_layers in [1, 2, 3]:
          for pos_type in ['smoothL1']:
            cfgs={'var_lr':lr, 'var_wd':wd, 'var_num_hidden_nodes':num_hidden_nodes}
            cfgs['pos_type'] = pos_type
            cfgs['var_n_layers'] = n_layers
            cfgs['data'] = '1dGaussian'
            cfgs['data_mu'] = mu
            wandb.init(project='density', config=cfgs)
            variational_KL(X, n_iters, n=10000, num_hidden_nodes=num_hidden_nodes,
                           det_lambda=0.1, det_every=100, lr=lr, wd=wd, patience=200, fix_A=True,
                           n_layers=n_layers, pos_type=pos_type)
            wandb.finish()

