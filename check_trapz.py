# %%
from typing import Callable, Sequence, Literal
from functools import partial

import jax
from jax import Array, jit
import jax.numpy as jnp

import jaxopt
import equinox as eqx
import optax
from more_itertools import pairwise
import matplotlib.pyplot as plt

from neuralconstitutive.jax.constitutive import SimpleLinearSolid
from neuralconstitutive.jax.ting import (
    force_approach,
    force_retract,
    find_t1,
    t1_constraint,
    _find_t1,
)

jax.config.update("jax_enable_x64", True)


sls = SimpleLinearSolid(E0=8.0, E_inf=2.0, tau=0.01)
t_app = jnp.linspace(0, 0.2, 100)
t_ret = jnp.linspace(0.2, 0.4, 100)
sls(t_app)
d_app = 10.0 * t_app
d_ret = -10.0 * t_ret
v_app = 10.0 * jnp.ones_like(t_app)
v_ret = -10.0 * jnp.ones_like(t_ret)


# %%
def simulate_data(
    model: Callable,
    t_app: Array,
    t_ret: Array,
    d_app: Array,
    d_ret: Array,
    v_app: Array,
    v_ret: Array,
    noise_strength: float,
    random_seed: int,
) -> tuple[Array, Array]:
    key = jax.random.PRNGKey(random_seed)
    f_app = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
    t1 = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
    f_ret = force_retract(t_ret, t1, model, t_app, d_app, v_app, 1.0, 1.5)
    noise_scale = jnp.max(f_app)
    noise_app = jax.random.normal(key, f_app.shape) * noise_strength * noise_scale
    noise_ret = (
        jax.random.normal(jax.random.split(key, num=1), f_ret.shape)
        * noise_strength
        * noise_scale
    )
    return f_app + noise_app, f_ret + noise_ret


# %%
f_app, f_ret = simulate_data(sls, t_app, t_ret, d_app, d_ret, v_app, v_ret, 5e-3, 0)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t_app, f_app, ".", label="approach")
ax.plot(t_ret, f_ret, ".", label="retract")
ax.legend()
fig
# %%
grads = jax.grad(lambda t, model: model(t), argnums=1)(0.1, sls)
grads.E0

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
F_app = force_approach(t_app, sls, t_app, d_app, v_app, 1.0, 2.0)
t1 = find_t1(t_ret, sls, t_app, t_ret, v_app, v_ret)
F_ret = force_retract(t_ret, t1, sls, t_app, d_app, v_app, 1.0, 2.0)
ax.plot(t_app, F_app, label="approach")
ax.plot(t_ret, F_ret, label="retract")
ax.legend()
fig


# %%
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
            t = jax.nn.elu(layer(t))
        # return self.final_activation(t)
        return jax.nn.softplus(t)


# phi_nn = FullyConnectedNetwork(["scalar", 100, "scalar"], jax.nn.elu, jax.nn.softplus)
phi_nn = FullyConnectedNetwork(["scalar", 100, "scalar"])

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t_app, jax.vmap(phi_nn)(t_app))
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
F_app = force_approach(t_app, phi_nn, t_app, d_app, v_app, 1.0, 2.0)
t1 = find_t1(t_ret, phi_nn, t_app, t_ret, v_app, v_ret)
F_ret = force_retract(t_ret, t1, sls, t_app, d_app, v_app, 1.0, 2.0)
ax.plot(t_app, F_app, label="approach")
ax.plot(t_ret, F_ret, label="retract")
ax.legend()
fig
# %%
t1

# %%
phi_nn


# %%
@eqx.filter_value_and_grad
def compute_loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    # Trains with respect to binary cross-entropy
    return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))


# %%
phi_nn(0.3)
# %%
sls
# %%
