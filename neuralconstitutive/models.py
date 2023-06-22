from typing import Callable

import scipy
import torch
from torch import nn
from more_itertools import pairwise


class FullyConnectedNetwork(nn.Module):
    def __init__(self, num_nodes: list[int], activation: Callable):
        super().__init__()
        self.nodes = list(num_nodes)
        layers = self._build_layers()
        self.hidden_layers = nn.ModuleList(layers[:-1])
        self.output_layer = layers[-1]
        self.activation = activation
        self.final_activation = nn.functional.softplus

    def _build_layers(self):
        return [nn.Linear(n_in, n_out) for n_in, n_out in pairwise(self.nodes)]

    def forward(self, x):
        x = torch.log10(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)  # output activation is identity
        return self.final_activation(x)


class BernsteinNN(nn.Module):
    def __init__(self, func: Callable, n: int):
        super().__init__()
        x, w = scipy.special.roots_laguerre(n)
        self.register_buffer(
            "nodes", torch.tensor(x, dtype=torch.float32).view(-1, 1)
        )  # shape: (n, 1)
        self.register_buffer(
            "weights", torch.tensor(w, dtype=torch.float32)
        )  # shape: (n,)
        self._func = func  # should map (n, 1) -> (n, 1)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))
        self._initalize_scale()

    def _initalize_scale(self, t0: float = 1e-3):
        y0 = self(torch.tensor([t0]))
        self.scale.data /= y0[0]

    def func(self, t):
        return torch.abs(self._func(t))

    def forward(self, t):  # t has shape (d,); b: batch size
        t = t.view(-1)
        x = self.nodes / t  # (n, d)
        f = self.func(x.view(-1, 1)).view(x.shape)  # (n, d) -> (nd, 1) -> (n, d)
        y = torch.mv(f.T, self.weights)  # (d, n)*(n,) -> (d,)
        return self.scale * (y / t) + self.offset
