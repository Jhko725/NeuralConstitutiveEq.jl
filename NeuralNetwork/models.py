from typing import Callable

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

    def _build_layers(self):
        return [nn.Linear(n_in, n_out) for n_in, n_out in pairwise(self.nodes)]

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)  # output activation is identity
        return x
