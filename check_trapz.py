# %%
from typing import Callable, Sequence, Literal

import jax
from jax import Array
import jax.numpy as jnp


import equinox as eqx
import optax
from more_itertools import pairwise
import matplotlib.pyplot as plt
import scipy

from neuralconstitutive.jax.constitutive import SimpleLinearSolid
from neuralconstitutive.jax.ting import (
    force_approach,
    force_retract,
    find_t1,
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
sls = SimpleLinearSolid(E0=8, E_inf=2, tau=0.2)
t_app = jnp.linspace(0, 0.5, 100)
t_ret = jnp.linspace(0.5, 1.0, 100)
sls(t_app)
d_app = 1.0 * t_app
d_ret = -1.0 * t_ret
v_app = 1.0 * jnp.ones_like(t_app)
v_ret = -1.0 * jnp.ones_like(t_ret)


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
plt.plot(t_app, jax.vmap(sls)(t_app))


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
            t = jnp.sin(layer(t))
        # return self.final_activation(t)
        return jax.nn.softplus(t)


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
        self.bias = jnp.asarray(0.0)
        self.nodes, self.weights = scipy.special.roots_legendre(num_quadrature)

    def __call__(self, t: Array) -> Array:
        nodes = jax.lax.stop_gradient(self.nodes)
        weights = jax.lax.stop_gradient(self.weights)
        h = jax.vmap(self.net)(nodes)
        expmx = 0.5 * (nodes + 1)
        return (
            jax.nn.relu(self.scale) * 0.5 * jnp.dot(h * expmx ** (t - 1), weights)
            + self.bias
        )


class BernsteinNN2(eqx.Module):
    net: eqx.Module
    scale: Array
    bias: Array
    nodes: Array
    weights: Array

    def __init__(self, net: Callable, num_quadrature: int = 100):
        super().__init__()
        self.net = net
        self.scale = jnp.asarray(0.1)
        self.bias = jnp.asarray(0.5)
        nodes, weights = scipy.special.roots_laguerre(num_quadrature)
        self.nodes = jnp.asarray(nodes)
        self.weights = jnp.asarray(weights)

    def __call__(self, t: Array) -> Array:
        nodes = jax.lax.stop_gradient(self.nodes)
        weights = jax.lax.stop_gradient(self.weights)
        t_ = t + 5e-2
        h = jax.vmap(self.net)(nodes / t_)
        return jax.nn.relu(self.scale) * jnp.dot(weights, h) / t_ + jax.nn.relu(
            self.bias
        )


class PronyNN(eqx.Module):
    weights: Array
    scales: Array
    bias: Array
    # N: int
    # random_seed: int

    def __init__(self, tau_min: float, tau_max: float, N: int = 50, seed: int = 10):
        super().__init__()
        # self.N = N
        # self.random_seed = seed
        key = jax.random.PRNGKey(seed)
        self.weights = jnp.ones(N) / N
        self.scales = 10 ** -jax.random.uniform(
            key, shape=(N,), minval=jnp.log10(tau_min), maxval=jnp.log10(tau_max)
        )
        self.bias = jnp.asarray(0.0)

    def __call__(self, t: Array) -> Array:
        scales = jax.lax.stop_gradient(self.scales)
        return jnp.dot(
            jax.nn.relu(self.weights), jnp.exp(-jax.nn.relu(scales) * t)
        ) + jax.nn.relu(self.bias)


# phi_nn = FullyConnectedNetwork(["scalar", 100, "scalar"], jax.nn.elu, jax.nn.softplus)
phi_nn = FullyConnectedNetwork(["scalar", 20, 20, 20, 20, "scalar"])
phi_bern = BernsteinNN(phi_nn, 100)
# phi_prony = PronyNN(1e-4, 1e3, 50)
# phi_prony.scales
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t_app, jax.vmap(phi_bern)(t_app))
# ax.plot(t_app, jax.vmap(phi_prony)(t_app))
# %%
# plt.plot(t_app, jax.vmap(phi_nn)(-jnp.log1p(t_app) + jnp.log(2)))
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
F_app_pred = force_approach(t_app, phi_bern, t_app, d_app, v_app, 1.0, 1.5)
t1 = find_t1(t_ret, phi_bern, t_app, t_ret, v_app, v_ret)
F_ret_pred = force_retract(t_ret, t1, phi_bern, t_app, d_app, v_app, 1.0, 1.5)
ax.plot(t_app, f_app, label="approach")
ax.plot(t_ret, f_ret, label="retract")
ax.plot(t_app, F_app_pred, label="approach (nn)")
ax.plot(t_ret, F_ret_pred, label="retract (nn)")
ax.legend()
fig


# %%
@eqx.filter_value_and_grad
def compute_loss(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret):
    F_app_pred = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
    t1_pred = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
    F_ret_pred = force_retract(t_ret, t1_pred, model, t_app, d_app, v_app, 1.0, 1.5)
    # Mean squared error loss
    # l1 = jnp.sum(jnp.abs(model.weights)) + jnp.abs(model.bias)
    return (
        jnp.mean((F_app - F_app_pred) ** 2)
        + jnp.mean((F_ret - F_ret_pred) ** 2)
        # + 1e-3 * l1
    )


@eqx.filter_value_and_grad
def compute_loss_approach(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret):
    F_app_pred = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
    # t1_pred = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
    # F_ret_pred = force_retract(t_ret, t1_pred, model, t_app, d_app, v_app, 1.0, 1.5)
    # Mean squared error loss
    # l1 = jnp.sum(jnp.abs(model.weights)) + jnp.abs(model.bias)
    return jnp.mean((F_app - F_app_pred) ** 2)  # + 1e-3 * l1


@eqx.filter_value_and_grad
def compute_loss_retract(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret):
    # F_app_pred = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
    t1_pred = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
    F_ret_pred = force_retract(t_ret, t1_pred, model, t_app, d_app, v_app, 1.0, 1.5)
    # Mean squared error loss
    return jnp.mean((F_ret - F_ret_pred) ** 2)


# %%
import numpy as np

model = phi_bern
# %%
optim = optax.rmsprop(1e-3)
opt_state = optim.init(model)


@eqx.filter_jit
def make_step(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret, opt_state):
    loss, grads = compute_loss(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret)
    updates, opt_state = optim.update(grads, opt_state, params=model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_jit
def make_step_app(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret, opt_state):
    loss, grads = compute_loss_approach(
        model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret
    )
    updates, opt_state = optim.update(grads, opt_state, params=model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_jit
def make_step_ret(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret, opt_state):
    loss, grads = compute_loss_retract(
        model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret
    )
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


# %%
max_epochs = 4000
loss_history = np.empty(max_epochs)
for step in range(max_epochs):
    loss, model, opt_state = make_step(
        model, t_app, t_ret, d_app, v_app, v_ret, f_app, f_ret, opt_state
    )
    # print(model.taus)
    # print(model.weights)
    loss = loss.item()
    loss_history[step] = loss
    print(f"step={step}, loss={loss}")

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
F_app = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
t1 = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
F_ret = force_retract(t_ret, t1, model, t_app, d_app, v_app, 1.0, 1.5)

plot_kwargs = {"markersize": 3.0, "alpha": 0.8}
fig, axes = plt.subplots(2, 1, figsize=(5, 5))

axes[0].plot(t_app, f_app, ".", color="k", label="Simulated data", **plot_kwargs)
axes[0].plot(t_ret, f_ret, ".", color="k", **plot_kwargs)
# axes[0].plot(t_app, F_app, "-", color="royalblue", label="Prediction", **plot_kwargs)
# axes[0].plot(t_ret, F_ret, "-", color="royalblue", **plot_kwargs)
axes[0].plot(t_app, F_app, "-", color="red", label="Initial prediction", **plot_kwargs)
axes[0].plot(t_ret, F_ret, "-", color="red", **plot_kwargs)
axes[0].set_ylabel("Force $F(t)$ (a.u.)")

axes[1].plot(
    t_app, jax.vmap(sls)(t_app), ".", color="k", label="Ground truth", **plot_kwargs
)
axes[1].set_ylabel("Relaxation function $G(t)$ (a.u.)")
# axes[1].plot(
#     t_app,
#     jax.vmap(model)(t_app),
#     "-",
#     color="royalblue",
#     label="Extracted",
#     **plot_kwargs,
# )
axes[1].plot(
    t_app,
    jax.vmap(model)(t_app),
    "-",
    color="red",
    label="Initial prediction",
    **plot_kwargs,
)


for ax in axes:
    ax.grid(color="lightgray", linestyle="--")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.legend()
axes[-1].set_xlabel("Time $t$ (a.u.)")
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(loss_history, color="orangered", linewidth=1.0)
ax.set_xlabel("Training epochs")
ax.set_ylabel("Mean squared loss")
ax.set_yscale("log")
ax.grid(color="lightgray", linestyle="--")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
# %%
