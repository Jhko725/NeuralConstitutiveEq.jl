import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
import scipy


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
        self.bias = jnp.asarray(1.0)
        self.nodes, self.weights = scipy.special.roots_legendre(num_quadrature)

    def __call__(self, t: Array) -> Array:
        nodes = jax.lax.stop_gradient(self.nodes)
        weights = jax.lax.stop_gradient(self.weights)
        h = eqx.filter_vmap(self.net)(nodes)
        expmx = 0.5 * (nodes + 1)
        return jax.nn.relu(self.scale) * 0.5 * jnp.dot(
            h * expmx ** (t - 1), weights
        ) + jax.nn.relu(self.bias)
