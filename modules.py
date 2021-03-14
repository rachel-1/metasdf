import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.parallel
from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear)
from torch.nn.init import _calculate_correct_fan
import numpy as np
from collections import OrderedDict
import dataio
import math


def init_weights_normal(m):
    if type(m) == BatchLinear or nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)


def init_weights_uniform(m):
    if hasattr(m, 'weight'):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0.)


def sal_init(m):
    if type(m) == BatchLinear or nn.Linear:
        if hasattr(m, 'weight'):
            std = np.sqrt(2) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_out'))

            with torch.no_grad():
                m.weight.normal_(0., std)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.0)


def sal_init_last_layer(m):
    if hasattr(m, 'weight'):
        val = np.sqrt(np.pi) / np.sqrt(_calculate_correct_fan(m.weight, 'fan_in'))
        with torch.no_grad():
            m.weight.fill_(val)
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0.0)


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape)-2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output

class ResidualBlock(MetaModule):
    def __init__(self, in_features, num_hidden_layers, hidden_features):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(BatchLinear(in_features, hidden_features))
        for i in range(num_hidden_layers):
            self.layers.append(BatchLinear(hidden_features, hidden_features))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, params=None):
        out = x
        for name, layer in self.layers._modules.items():
            out = self.relu(layer(out, self.get_subdict(params, f"layers.{name}")))
            
        return torch.cat([x, out], -1)
    
class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        # self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(2*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        funcs = [torch.sin, torch.cos]
        for freq in self.freq_bands:
            for func in funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)
        
class MetaFCNet(MetaModule):
    def __init__(self, in_features, out_features, num_hidden_layers,
                 hidden_features, positional_encoding, skip_connect):
        super().__init__()
        self.net = []

        # If using positional encoding, prepend the embedding layer.
        if positional_encoding:
            embedding_dim=22 if in_features==2 else 33
            self.net.append(Embedding(in_features, 5))
            in_features = embedding_dim

        # If we are using a skip connection, we use a ResidualBlock to cover the
        # first half of the hidden layers.    
        if skip_connect:
            self.net.append(ResidualBlock(in_features, num_hidden_layers // 2, hidden_features))
            in_features += hidden_features
            num_hidden_layers -= num_hidden_layers // 2 + 1
            
        self.net.append(BatchLinear(in_features, hidden_features))
        self.net.append(nn.ReLU(inplace=True))

        for i in range(num_hidden_layers):
            self.net.append(BatchLinear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(inplace=True))

        self.net.append(BatchLinear(hidden_features, out_features))
        self.net = MetaSequential(*self.net)
        
    def forward(self, coords, params=None, **kwargs):
        output = self.net(coords, params=self.get_subdict(params, 'net'))
        return output
