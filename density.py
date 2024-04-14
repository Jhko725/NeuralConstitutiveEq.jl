# %%
from typing import Callable
from functools import partial

import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
import optax
import scipy
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import StandardLinearSolid, FromLogDiscreteSpectrum
from neuralconstitutive.jax.ting import (
    force_approach,
    force_retract,
)
from neuralconstitutive.jax.tipgeometry import Spherical, AbstractTipGeometry
from neuralconstitutive.trajectory import make_triangular, Trajectory
from neuralconstitutive.nn import FullyConnectedNetwork
from neuralconstitutive.models import BernsteinNN
from neuralconstitutive.training import loss_total, train_model
from neuralconstitutive.relaxation_spectrum import HonerkampWeeseBimodal

jax.config.update("jax_enable_x64", True)


@partial(eqx.filter_vmap, in_axes=(None, 0, 0))
def lognormal(x: Array, mu: float, sigma: float):
    coeff = 1 / (jnp.sqrt(2 * jnp.pi) * sigma)
    return coeff * jnp.exp(-((jnp.log(x) - mu) ** 2) / (2 * sigma**2)) / x


class LogNormalMixture(eqx.Module):
    _weights: Array
    mus: Array
    sigmas: Array

    def __init__(self, n_mixture: int = 50):
        self._weights = jnp.ones(n_mixture)
        means = jnp.geomspace(1e-3, 100, n_mixture)
        stdevs = jnp.ones_like(means) * 5.0
        self.mus = jnp.log(means**2 / jnp.sqrt(means**2 + stdevs**2))
        self.sigmas = jnp.sqrt(jnp.log(1 + stdevs**2 / means**2))

    @property
    def weights(self) -> Array:
        return jax.nn.softmax(self._weights)

    def __call__(self, x: Array) -> Array:
        lognormals = lognormal(x, self.mus, self.sigmas)
        return jnp.matmul(lognormals.T, self.weights)


class BernsteinNN2(eqx.Module):
    net: eqx.Module
    scale: Array
    nodes: Array
    weights: Array

    def __init__(self, net: Callable, scale: float = 1.0, num_quadrature: int = 200):
        super().__init__()
        self.net = net
        self.scale = jnp.asarray(scale)
        nodes, weights = scipy.special.roots_laguerre(num_quadrature)
        self.nodes = jnp.asarray(nodes)
        self.weights = jnp.asarray(weights)

    def __call__(self, t: Array) -> Array:
        # Trick from https://github.com/google/jax/discussions/17243
        # This makes it so that t can be either float or Array
        # and the resulting output is either Scalar or Array
        t = jnp.asarray(t)
        out_shape = t.shape
        t = jnp.atleast_1d(t)
        t_ = jnp.where(t > 0, t, 1.0)

        @eqx.filter_vmap
        def _call(t: float) -> Array:
            nodes = jax.lax.stop_gradient(self.nodes)
            weights = jax.lax.stop_gradient(self.weights)

            h = self.net(nodes / t)
            return jnp.dot(weights, h) / t

        y = jnp.where(t > 0, _call(t_), 1.0)

        return jax.nn.relu(self.scale) * y.reshape(out_shape)


# %%
x = jnp.linspace(0.0, 100.0, 1000)
mean, std = 100, 1.0
mu = jnp.log(mean**2 / jnp.sqrt(mean**2 + std**2))
sigma = jnp.sqrt(jnp.log(1 + std**2 / mean**2))
y = lognormal(x, jnp.atleast_1d(mu), jnp.atleast_1d(sigma))[0]
plt.plot(x, y)
# %%
model = LogNormalMixture()
y2 = model(x)
plt.plot(x, y2)
# %%
nn = BernsteinNN2(model, scale=2.0)
y_nn = nn(x)
plt.plot(x, y_nn, ".")
# %%
sls = StandardLinearSolid(E0=8, E_inf=2, tau=0.2)
bimodal = FromLogDiscreteSpectrum(HonerkampWeeseBimodal())
tip = Spherical(1.0)
app, ret = make_triangular(100, 1.0, 1.0)


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
    app, ret, bimodal.relaxation_function, tip, noise_strength=5e-3, random_seed=0
)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
F_app_pred = force_approach(app, nn, tip)
F_ret_pred = force_retract(app, ret, nn, tip)

ax.plot(app.t, f_app, label="approach")
ax.plot(ret.t, f_ret, label="retract")
ax.plot(app.t, F_app_pred, label="approach (nn)")
ax.plot(ret.t, F_ret_pred, label="retract (nn)")
ax.legend()
fig
# %%
trained_model, loss_history = train_model(
    nn,
    (app, ret),
    (f_app, f_ret),
    tip,
    loss_total,
    optimizer=optax.rmsprop(5e-3),
    max_epochs=1000,
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
    bimodal.relaxation_function(app.t),
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
plt.plot(app.t, trained_model.net(app.t))
# %%
