import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from scipy.stats import ortho_group
from torch.utils.data import Dataset, DataLoader
import math

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

# Data loader for the training data
class trainset(Dataset):

    def __init__(self, X):
        """
        Args:
            X: training data numpy array
        """
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        entry = self.X[idx]
        return entry

class variationalNet(torch.nn.Module):

    def __init__(self, num_hidden_nodes, n_layers=1, embed_size=1, pos_type='smoothL1'):
        super(variationalNet, self).__init__()
        
        if n_layers == 0:
          layers = [nn.Linear(embed_size, 1)]
        else:
          layers = [nn.Linear(embed_size, num_hidden_nodes)]
          for _ in range(n_layers):
            layers += nn.ReLU(),
            layers += nn.Linear(num_hidden_nodes, num_hidden_nodes),
          layers += nn.ReLU(),
          layers += nn.Linear(num_hidden_nodes, 1),
        self.net = nn.Sequential(*layers)
        if pos_type == 'smoothL1':
          self.make_positive = lambda x: nn.SmoothL1Loss(reduction='none')(x, torch.zeros(x.shape).to(device))
        elif pos_type == 'sigmoid':
          self.make_positive = nn.Sigmoid()
        elif pos_type == 'square':
          self.make_positive = lambda x: torch.square(x)
        elif pos_type == 'exp':
          self.make_positive = lambda x: torch.exp(x)
        elif pos_type == 'relu':
          self.make_positive = nn.ReLU()
        elif pos_type == 'relu6':
          self.make_positive = nn.ReLU6()
        elif pos_type == 'none':
          self.make_positive = lambda x: x

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

class basis_function_Net(torch.nn.Module):

    def __init__(self, n_cos, n_sin, b_a, pos_type='smoothL1'):
        super(basis_function_Net, self).__init__()
        
        self.coef = torch.randn(1+n_cos+n_sin, requires_grad=True).to(device)
        self.n_cos = n_cos
        self.n_sin = n_sin
        self.b_a = b_a
        self.pi = torch.Tensor([math.pi]).to(device)
        if pos_type == 'smoothL1':
          self.make_positive = lambda x: nn.SmoothL1Loss(reduction='none')(x, torch.zeros(x.shape).to(device))
        elif pos_type == 'sigmoid':
          self.make_positive = nn.Sigmoid()
        elif pos_type == 'square':
          self.make_positive = lambda x: torch.square(x)
        elif pos_type == 'exp':
          self.make_positive = lambda x: torch.exp(x)
        elif pos_type == 'relu':
          self.make_positive = nn.ReLU()
        elif pos_type == 'relu6':
          self.make_positive = nn.ReLU6()
        elif pos_type == 'none':
          self.make_positive = lambda x: x

    def forward(self, x):
        out = torch.zeros(x.shape).to(device) + self.coef[0]
        for m in range(self.n_cos):
            out += self.coef[1+m] * torch.cos(2*self.pi*m*x/self.b_a)
        for n in range(self.n_sin):
            out += self.coef[1+self.n_cos+n] * torch.sin(2*self.pi*n*x/self.b_a)
        out = self.make_positive(out)
        return out

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def variational_KL(X, n_iters, n_Zs=1000, num_hidden_nodes=10, det_lambda=0.1, det_every=100,
    lr=1e-2, wd=1e-4, patience=200, A_mode='GD', pos_type='smoothL1', n_layers=1,
    var_LB='E1', batch_size=10000, optimizer_option="Adam", function_option="basis"):
    """
    Input:
    X: torch tensor N * D
    n: number of standand gaussian normal dim=D
    Output: 
    A: torch tensor D * D
    """
    N, D = X.shape
    mean = np.ones(D)
    cov = np.eye(D,D)
    Z = to_tensor(np.random.multivariate_normal(mean, cov, n_Zs))
    if var_LB == 'E1':
      embed_size = 1
    elif var_LB == 'E2' or var_LB == 'E3' or var_LB == 'E4':
      embed_size = D
    params = []
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
    if function_option == "net":
        g_function = variationalNet(num_hidden_nodes, n_layers=n_layers, pos_type=pos_type, embed_size=embed_size).to(device)
    elif function_option == "basis":
        AX = to_tensor(torch.flatten(A.T.matmul(to_tensor(X).T))).view(-1,1)
        minimum = torch.min(AX)
        maximum = torch.max(AX)
        b_a = to_tensor(maximum-minimum)
        g_function = basis_function_Net(n_cos=50, n_sin=50, b_a=b_a, pos_type=pos_type).to(device)
        print("Number of parameters: {}".format(g_function.parameter_count()))
    else:
        raise NotImplementedError("function_option should be in ['net', 'basis'].\n Got {}".format(g_function))
    params += [{'params': g_function.parameters()}]
    if optimizer_option == "SGD":
        optimizer = optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    elif optimizer_option == "Adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError("optimizer_option should be in ['SGD', 'Adam'].\n Got {}".format(optimizer_option))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, verbose=True)
    g_function.train()

    def helper_loss_E1(M, X):
      sum_loss = to_tensor(torch.tensor(0.0)).to(device)
      # Compute first term for X
      y_X = to_tensor(torch.flatten(M.T.matmul(X.T))).view(-1,1) 
      g_y_X = g_function(y_X)
      log_g_y_X = torch.log(g_y_X)
      loss1 = torch.mean(log_g_y_X)
      sum_loss -= loss1
      # Compute second term for Z
      y_Z = torch.flatten(M.T.matmul(Z.T)).view(-1,1)
      g_y_Z = g_function(y_Z)
      loss2 = torch.mean(g_y_Z)
      sum_loss += loss2
      return sum_loss, y_X, g_y_X, y_Z, g_y_Z, loss1, loss2 
    
    def helper_loss_E2(M, X):
      sum_loss = to_tensor(torch.tensor(0.0)).to(device)
      # Compute first term for X
      y_X = to_tensor(M.T.matmul(X.T)).T
      g_y_X = g_function(y_X)
      g_y_X = torch.clamp(g_y_X, -100, 60)
      loss1 = torch.mean(g_y_X)
      sum_loss -= loss1
      # Compute second term for Z
      y_Z = to_tensor(M.T.matmul(Z.T)).T
      g_y_Z = g_function(y_Z)
      # print('mean={:.5f} / max={:.5f}'.format(g_y_Z.mean(), g_y_Z.max()))
      g_y_Z = torch.clamp(g_y_Z, -100, 60)
      loss2 = torch.log(torch.exp(g_y_Z).mean())
      sum_loss += loss2
      return sum_loss, y_X, g_y_X, y_Z, g_y_Z, loss1, loss2

    def helper_loss_E3(M, X):
      # same as E2, except using Taylor series to approximate exp
      sum_loss = to_tensor(torch.tensor(0.0)).to(device)
      # Compute first term for X
      y_X = to_tensor(M.T.matmul(X.T)).T
      g_y_X = g_function(y_X)
      loss1 = torch.mean(g_y_X)
      sum_loss -= loss1
      # Compute second term for Z
      y_Z = to_tensor(M.T.matmul(Z.T)).T
      g_y_Z = g_function(y_Z)
      approx_exp = 1 + g_y_Z + g_y_Z**2 / 2
      # + g_y_Z**3 / 6
      loss2 = torch.log(approx_exp.mean())
      sum_loss += loss2
      return sum_loss, y_X, g_y_X, y_Z, g_y_Z, loss1, loss2

    def helper_loss_E4(M, X):
      """
      Variational form
      """
      sum_loss = to_tensor(torch.tensor(0.0)).to(device)
      # Compute first term for X
      y_X = to_tensor(M.T.matmul(X.T)).T 
      g_y_X = g_function(y_X)
      loss1 = torch.mean(g_y_X)
      sum_loss -= loss1
      # Compute second term for Z
      y_Z = to_tensor(M.T.matmul(Z.T)).T
      g_y_Z = g_function(y_Z)
      loss2 = torch.mean(g_y_Z)
      sum_loss += loss2
      return sum_loss, y_X, g_y_X, y_Z, g_y_Z, loss1, loss2 
 
    if var_LB == 'E1':
      helper_loss = helper_loss_E1
    elif var_LB == 'E2':
      helper_loss = helper_loss_E2
    elif var_LB == 'E3':
      helper_loss = helper_loss_E3
    elif var_LB == 'E4':
      helper_loss = helper_loss_E4
    train_dataset = trainset(X)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0)
    train_iter = iter(train_loader)
    for i in range(n_iters):
        optimizer.zero_grad()
        try:
            X_batch = train_iter.next()
        except:
            train_iter = iter(train_loader)
            X_batch = train_iter.next()
        X_batch = to_tensor(X_batch)
        sum_loss, y_X, g_y_X, y_Z, g_y_Z, loss1, loss2 = helper_loss(A, X_batch)
        raw_loss = sum_loss
        reg_loss = None
        if A_mode != "givens":
          if i % det_every == 0:
              det_lambda *= 0.5
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
              'var_loss_raw': raw_loss.item()})
            if reg_loss:
              wandb.log({
              'var_reg': reg_loss.item()})
            if DEBUG:
              wandb.log({
                'var_input_mean': y_X.mean().item(),
                'var_input_Z_mean': y_Z.mean().item(),
                'var_out_mean': g_y_X.mean().item(),
                'var_out_Z_mean': g_y_Z.mean().item(),
                'var_out_term1': loss1.item(),
                'var_out_term2': loss2.item()})
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
            for eta in np.arange(-np.pi, np.pi, 20):
              # Givens matrix
              G = to_tensor(np.eye(D))
              G[i,i], G[j,j] = np.cos(eta), np.cos(eta)
              G[i,j], G[j,i] = -np.sin(eta), np.sin(eta)
              tmp = A.mm(G)
              cur_loss, _, _, _, _, _, _ = helper_loss(tmp, X_batch)
              if best_loss is None or cur_loss < best_loss:
                best_loss = cur_loss
                best_G = G.clone()
            A = A.mm(best_G)
    return A.detach()


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  # model
  parser.add_argument('--pos-type', type=str)
  parser.add_argument('--A-mode', type=str)
  parser.add_argument('--var-LB', type=str, default='E1')
  # opt
  parser.add_argument('--var-n-iters', type=int)
  parser.add_argument('--var-lr', type=float)
  parser.add_argument('--var-wd', type=float)
  parser.add_argument('--var-num-hidden-nodes', type=int)
  parser.add_argument('--var-n-layers', type=int)
  # data
  parser.add_argument('--data', type=str, default='')
  parser.add_argument('--data-dim', type=int)
  parser.add_argument('--data-mu', type=int)
  parser.add_argument('--wb-name', type=str)
 
  args = parser.parse_args()
  args.data = '{}dGaussian'.format(args.data_dim)

  args.wb_name = '{}d_mu{}_lr{}_wd{}_nHidden{}_nLayers{}_posType{}_A{}_lb{}'.format(
      args.data_dim, args.data_mu, args.var_lr, args.var_wd,
      args.var_num_hidden_nodes, args.var_n_layers, args.pos_type, args.A_mode,
      args.var_LB
    )
      
  N = 10000
  X = np.random.normal(loc=args.data_mu, scale=1, size=(N, args.data_dim))

  wandb.init(project='density', config=args, name=args.wb_name)
  variational_KL(X, args.var_n_iters, n_Zs=10000, num_hidden_nodes=args.var_num_hidden_nodes,
                 det_lambda=0.1, det_every=100, lr=args.var_lr, wd=args.var_wd, patience=200,
                 A_mode=args.A_mode, n_layers=args.var_n_layers, pos_type=args.pos_type,
                 var_LB=args.var_LB)
  wandb.finish()

