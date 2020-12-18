import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

DTYPE = torch.FloatTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_tensor(data):
    return torch.tensor(data).type(DTYPE).to(device)

class variationalNet(torch.nn.Module):

    def __init__(self, num_hidden_nodes, embed_size=1):
        super(variationalNet, self).__init__()
        
        self.input = nn.Linear(embed_size, num_hidden_nodes)
        self.relu = nn.ReLU()
        self.output = nn.Linear(num_hidden_nodes, embed_size)
        self.make_positive = nn.SmoothL1Loss()

    def forward(self, x):
        out = x
        out = self.input(out)
        out = self.relu(out)
        out = self.output(out)
        out = self.make_positive(out, torch.zeros(out.shape).to(device))
        return out

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def variational_KL(X, n_epochs, n=1000, num_hidden_nodes=10, det_lambda=0.1):
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
    g_function = variationalNet(num_hidden_nodes).to(device)
    A = torch.randn((D,D), requires_grad=True, device=device)
    torch.nn.init.orthogonal_(A)
    optimizer = optim.SGD([
                {'params': g_function.parameters()},
                {'params': A}
            ], lr=1e-2, momentum=0.9)
    g_function.train()
    for i in range(n_epochs):
        optimizer.zero_grad()
        sum_loss = to_tensor(torch.tensor(0.0)).to(device)
        # Compute first term for X
        y_X = to_tensor(torch.flatten(A.T.matmul(X.T))).view(N*D,1) 
        g_y_X = g_function(y_X)
        log_g_y_X = torch.log(g_y_X)
        sum_loss -= torch.sum(log_g_y_X)/N
        # Compute second term for Z
        y_Z = torch.flatten(A.T.matmul(Z.T)).view(n*D,1)
        g_y_Z = g_function(y_Z)
        sum_loss += torch.sum(g_y_Z)/n
        if i % 100 == 0:
            det_lambda *= 0.5
        sum_loss -= det_lambda * torch.log(torch.abs(torch.det(A)))
        sum_loss.backward()
        optimizer.step()
        print(i, sum_loss.item())
        with torch.no_grad():
            _, ss, _ = np.linalg.svd(A.detach().cpu())
            if ss[0] > 1:
                A /= ss[0]
    return A.detach()
