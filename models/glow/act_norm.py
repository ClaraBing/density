import torch
import torch.nn as nn

from util import mean_dim

import pdb

class ActNorm(nn.Module):
    """Activation normalization for 2D inputs.

    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.

    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, num_features, layer_type, scale=1., return_ldj=False):
        super(ActNorm, self).__init__()
        self.register_buffer('is_initialized', torch.zeros(1))
        if layer_type == 'conv':
          self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
          self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        elif layer_type == 'fc':
          self.bias = nn.Parameter(torch.zeros(1, num_features))
          self.logs = nn.Parameter(torch.zeros(1, num_features))

        self.num_features = num_features
        self.scale = float(scale)
        self.eps = 1e-6
        self.return_ldj = return_ldj

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            dim = list(range(x.ndim))
            dim.pop(1)
            bias = -mean_dim(x.clone(), dim=dim, keepdims=True)
            v = mean_dim((x.clone() + bias) ** 2, dim=dim, keepdims=True)
            logs = (self.scale / (v.sqrt() + self.eps)).log()
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x, sldj, reverse=False):
        logs = self.logs
        if reverse:
            x = x * logs.mul(-1).exp()
        else:
            x = x * logs.exp()

        if sldj is not None:
            if x.ndim == 4:
              ldj = logs.sum() * x.size(2) * x.size(3)
            else:
              ldj = logs.sum()
            if reverse:
                sldj = sldj - ldj
            else:
                sldj = sldj + ldj

        return x, sldj

    def forward(self, x, ldj=None, reverse=False):
        if not self.is_initialized:
          self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)

        if self.return_ldj:
            return x, ldj

        return x
