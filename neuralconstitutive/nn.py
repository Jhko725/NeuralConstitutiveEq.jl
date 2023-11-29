from typing import Sequence, Literal, Callable

import jax
from jax import Array
import equinox as eqx
from more_itertools import pairwise


class FullyConnectedNetwork(eqx.Module):
    layers: list[eqx.Module]
    # activation: Callable[[Array], Array]
    # final_activation: Callable[[Array], Array] | None
    random_seed: int

    def __init__(
        self,
        nodes: Sequence[int | Literal["scalar"]],
        # activation: Callable[[Array], Array] = jax.nn.tanh,
        # final_activation: Callable[[Array], Array] | None = None,
        random_seed: int = 0,
    ):
        super().__init__()
        self.random_seed = random_seed
        keys = jax.random.split(jax.random.PRNGKey(self.random_seed), len(nodes) - 1)
        self.layers = [
            eqx.nn.Linear(*feats, key=k) for (feats, k) in zip(pairwise(nodes), keys)
        ]
        # self.activation = activation
        # self.final_activation = (
        #    eqx.nn.Identity() if final_activation is None else final_activation
        # )

    def __call__(self, t: Array) -> Array:
        for layer in self.layers:
            # t = self.activation(layer(t))
            t = jax.nn.tanh(layer(t))
        # return self.final_activation(t)
        return jax.nn.softplus(t)
