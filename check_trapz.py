# %%
from typing import Callable

import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import scipy

from neuralconstitutive.constitutive import StandardLinearSolid
from neuralconstitutive.jax.ting import (
    force_approach,
    force_retract,
)
from neuralconstitutive.jax.tipgeometry import Spherical, AbstractTipGeometry
from neuralconstitutive.trajectory import make_triangular, Trajectory
from neuralconstitutive.nn import FullyConnectedNetwork
from neuralconstitutive.models import BernsteinNN
from neuralconstitutive.training import loss_total, train_model

jax.config.update("jax_enable_x64", True)

# %%
sls = StandardLinearSolid(E0=8, E_inf=2, tau=0.2)
tip = Spherical(1.0)
app, ret = make_triangular(0.5, 5e-3, 1.0)


# %%
def simulate_data(
    app: Trajectory,
    ret: Trajectory,
    relaxation: Callable[[Array], Array],
    tip: AbstractTipGeometry,
    noise_strength: float,
    random_seed: int,
) -> tuple[Array, Array]:
    key = jax.random.PRNGKey(random_seed)
    f_app = force_approach(app, relaxation, tip)
    f_ret = force_retract(app, ret, relaxation, tip)
    noise_scale = jnp.max(f_app)
    noise_app = jax.random.normal(key, f_app.shape) * noise_strength * noise_scale
    noise_ret = (
        jax.random.normal(jax.random.split(key, num=1), f_ret.shape)
        * noise_strength
        * noise_scale
    )
    return f_app + noise_app, f_ret + noise_ret


# %%
f_app, f_ret = simulate_data(
    app, ret, sls.relaxation_function, tip, noise_strength=5e-3, random_seed=0
)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.t, f_app, ".", label="approach")
ax.plot(ret.t, f_ret, ".", label="retract")
ax.legend()
fig
# %%
plt.plot(app.t, sls.relaxation_function(app.t))


# %%
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


# %%
phi_nn = FullyConnectedNetwork(
    ["scalar", 20, 20, 20, 20, "scalar"],
    activation=jax.nn.tanh,
    final_activation=jax.nn.softplus,
)
phi_bern = BernsteinNN(phi_nn, 100)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.t, phi_bern(app.t))

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
F_app_pred = force_approach(app, phi_bern, tip)
F_ret_pred = force_retract(app, ret, phi_bern, tip)
ax.plot(app.t, f_app, label="approach")
ax.plot(ret.t, f_ret, label="retract")
ax.plot(app.t, F_app_pred, label="approach (nn)")
ax.plot(ret.t, F_ret_pred, label="retract (nn)")
ax.legend()
fig


# %%
trained_model, loss_history = train_model(
    phi_bern,
    (app, ret),
    (f_app, f_ret),
    tip,
    loss_total,
    optimizer=optax.rmsprop(1e-3),
    max_epochs=3000,
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
F_app = force_approach(app, trained_model, tip)
F_ret = force_retract(app, ret, trained_model, tip)

plot_kwargs = {"markersize": 3.0, "alpha": 0.8}
fig, axes = plt.subplots(2, 1, figsize=(5, 5))

axes[0].plot(app.t, f_app, ".", color="k", label="Simulated data", **plot_kwargs)
axes[0].plot(ret.t, f_ret, ".", color="k", **plot_kwargs)
# axes[0].plot(t_app, F_app, "-", color="royalblue", label="Prediction", **plot_kwargs)
# axes[0].plot(t_ret, F_ret, "-", color="royalblue", **plot_kwargs)
axes[0].plot(app.t, F_app, "-", color="red", label="Initial prediction", **plot_kwargs)
axes[0].plot(ret.t, F_ret, "-", color="red", **plot_kwargs)
axes[0].set_ylabel("Force $F(t)$ (a.u.)")

axes[1].plot(
    app.t,
    sls.relaxation_function(app.t),
    ".",
    color="k",
    label="Ground truth",
    **plot_kwargs,
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
    app.t,
    trained_model(app.t),
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
