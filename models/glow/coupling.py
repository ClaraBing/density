import torch
import torch.nn as nn
import torch.nn.functional as F

from models.glow.act_norm import ActNorm

import pdb

class Coupling(nn.Module):
    """Affine coupling layer originally used in Real NVP and described by Glow.

    Note: The official Glow implementation (https://github.com/openai/glow)
    uses a different affine coupling formulation than described in the paper.
    This implementation follows the paper and Real NVP.

    Args:
        in_dim (int): Number of channels in the input.
        mid_dim (int): Number of channels in the intermediate activation
            in NN.
    """
    def __init__(self, in_dim, mid_dim, layer_type):
        super(Coupling, self).__init__()
        self.nn = NN(in_dim, mid_dim, 2 * in_dim, layer_type=layer_type)
        if layer_type == 'conv':
          self.scale = nn.Parameter(torch.ones(in_dim, 1, 1))
        else:
          self.scale = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if reverse:
            x_change = x_change * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = (x_change + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, ldj


class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.

    Args:
        in_dim (int): Number of channels in the input.
        mid_dim (int): Number of channels in the hidden activations.
        out_dim (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_dim, mid_dim, out_dim,
                 use_act_norm=False, layer_type='conv'):
        super(NN, self).__init__()
        if use_act_norm:
          norm_fn = ActNorm
        else:
          norm_fn = nn.BatchNorm2d if layer_type=='conv' else nn.BatchNorm1d

        self.in_norm = norm_fn(in_dim)
        if layer_type == 'conv':
          self.in_layer = nn.Conv2d(in_dim, mid_dim,
                                   kernel_size=3, padding=1, bias=False)
          self.mid_layer = nn.Conv2d(mid_dim, mid_dim,
                                    kernel_size=1, padding=0, bias=False)
          self.out_layer = nn.Conv2d(mid_dim, out_dim,
                                    kernel_size=3, padding=1, bias=True)
        else:
          self.in_layer = nn.Linear(in_dim, mid_dim, bias=False)
          self.mid_layer = nn.Linear(mid_dim, mid_dim, bias=False)
          self.out_layer = nn.Linear(mid_dim, out_dim, bias=True)

        nn.init.normal_(self.in_layer.weight, 0., 0.05)
        nn.init.normal_(self.mid_layer.weight, 0., 0.05)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

        self.mid_norm = norm_fn(mid_dim)
        self.out_norm = norm_fn(mid_dim)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_layer(x)

        x = self.mid_norm(x)
        x = F.relu(x)
        x = self.mid_layer(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_layer(x)

        return x
