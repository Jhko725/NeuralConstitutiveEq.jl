# %%
from typing import Sequence, Literal, Callable

import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
from more_itertools import pairwise
import scipy
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class FullyConnectedNetwork(eqx.Module):
    layers: list
    # activation: Callable
    # final_activation: Callable

    def __init__(
        self,
        nodes: Sequence[int | Literal["scalar"]],
        # activation: Callable,
        # final_activation: Callable | None,
        seed: int = 0,
    ):
        super().__init__()
        keys = jax.random.split(jax.random.PRNGKey(seed), len(nodes) - 1)
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
        return t
        # return jax.nn.softplus(t)


class BernsteinNN(eqx.Module):
    net: eqx.Module
    scale: Array
    bias: Array
    nodes: Array
    weights: Array

    def __init__(self, net: eqx.Module, num_quadrature: int = 500):
        super().__init__()
        self.net = net
        self.scale = jnp.asarray(1.0)
        self.bias = jnp.asarray(5.0)
        self.nodes, self.weights = scipy.special.roots_legendre(num_quadrature)

    def __call__(self, t: Array) -> Array:
        nodes = jax.lax.stop_gradient(self.nodes)
        weights = jax.lax.stop_gradient(self.weights)
        h = jax.nn.softplus(jax.vmap(self.net)(nodes))
        expmx = 0.5 * (nodes + 1)
        return jax.nn.elu(self.scale) * 0.5 * jnp.dot(
            h * expmx ** (t - 1), weights
        ) + jax.nn.elu(self.bias)

class BernsteinNN2(eqx.Module):
    net: eqx.Module
    scale: Array
    bias: Array
    nodes: Array
    weights: Array

    def __init__(self, net: Callable, num_quadrature: int = 100):
        super().__init__()
        self.net = net
        self.scale = jnp.asarray(1.0)
        self.bias = jnp.asarray(0.5)
        nodes, weights = scipy.special.roots_laguerre(num_quadrature)
        self.nodes = jnp.asarray(nodes)
        self.weights = jnp.asarray(weights)

    def __call__(self, t: Array) -> Array:
        nodes = jax.lax.stop_gradient(self.nodes)
        weights = jax.lax.stop_gradient(self.weights)
        t_ = t + 1e-2
        h = jax.vmap(self.net)(nodes / t_)
        return jax.nn.elu(self.scale) * jnp.dot(weights, h) / t_ + jax.nn.relu(
            self.bias
        )


# %%
t = jnp.logspace(-2, 2, 50)
nn = FullyConnectedNetwork(["scalar", 10, 10, "scalar"])
nn_bern = BernsteinNN(nn, num_quadrature=100)

y_nn = jax.vmap(nn)(t)
y_bern = jax.vmap(nn_bern)(t)
# %%
fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
axes[0].plot(t, y_nn, label="Normal NN")
axes[1].plot(t, y_bern, label="Bernstein NN")
for ax in axes:
    ax.legend()
#%%