# %%
from typing import Literal, Sequence, Callable

from more_itertools import pairwise
import numpy as np
import scipy
import matplotlib.pyplot as plt
import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
import optax

from jhelabtoolkit.io.nanosurf import nanosurf
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults
from neuralconstitutive.preprocessing import process_approach_data, estimate_derivative
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.preprocessing import (
    calc_tip_distance,
    estimate_derivative,
    get_sampling_rate,
    get_z_and_defl,
    ratio_of_variances,
    fit_baseline_polynomial,
)
from neuralconstitutive.jax.ting import force_approach, force_retract, find_t1


configure_matplotlib_defaults()

filepath = "Hydrogel AFM data/SD-Sphere-CONT-M/Highly Entangled Hydrogel(10nN, 10s, liquid).nid"
config, data = nanosurf.read_nid(filepath)

forward, backward = data["spec forward"], data["spec backward"]


# %%
z_fwd, defl_fwd = get_z_and_defl(forward)
z_bwd, defl_bwd = get_z_and_defl(backward)
dist_fwd = calc_tip_distance(z_fwd, defl_fwd)
dist_bwd = calc_tip_distance(z_bwd, defl_bwd)
# cp = find_contact_point(dist_fwd, defl_fwd)
# %%
# ROV method
N = 10
rov_fwd, idx_fwd = ratio_of_variances(defl_fwd, N)
rov_bwd, idx_bwd = ratio_of_variances(defl_bwd, N)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(dist_fwd, rov_fwd)
ax.set_xlabel("Distance(forward)")
ax.set_ylabel("ROV")
plt.axvline(
    dist_fwd[N],
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="maximum point",
)
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd * 1e6, defl_fwd * 1e9, label="forward")
ax.plot(dist_bwd * 1e6, defl_bwd * 1e9, label="backward")
ax.set_xlabel("Distance(μm)")
ax.set_ylabel("Force(nN)")
ax.legend()
# %%
# Find contact point
cp_fwd = dist_fwd[idx_fwd]
cp_bwd = dist_bwd[idx_bwd]
print(cp_fwd, cp_bwd)
# %%
cp_fwd = cp_fwd
# %%
# Translation
dist_fwd = dist_fwd - cp_fwd
dist_bwd = dist_bwd - cp_fwd

# %%
# Polynomial fitting
baseline_poly_fwd = fit_baseline_polynomial(dist_fwd, defl_fwd)
defl_processed_fwd = defl_fwd - baseline_poly_fwd(dist_fwd)
baseline_poly_bwd = fit_baseline_polynomial(dist_bwd, defl_bwd)
defl_processed_bwd = defl_bwd - baseline_poly_bwd(dist_bwd)
baseline_poly_bwd
# baseline_poly_bwd
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd * 1e6, defl_fwd * 1e9, label="forward")
ax.plot(dist_bwd * 1e6, defl_bwd * 1e9, label="backward")
ax.set_xlabel("Distance(μm)")
ax.set_ylabel("Force(nN)")
plt.axvline(cp_fwd, color="grey", linestyle="--", linewidth=1)
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd * 1e6, defl_processed_fwd * 1e9, label="forward")
ax.plot(dist_bwd * 1e6, defl_processed_bwd * 1e9, label="backward")
ax.set_xlabel("Distance(μm)")
ax.set_ylabel("Force(nN)")
plt.axvline(cp_fwd, color="grey", linestyle="--", linewidth=1)
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd, z_fwd, label="forward")
ax.plot(dist_bwd, z_bwd, label="backward")
ax.legend()
# %%
dist_total = np.concatenate((dist_fwd, dist_bwd[::-1]), axis=-1)
defl_total = np.concatenate((defl_fwd, defl_bwd[::-1]), axis=-1)
is_contact = dist_total >= 0
indentation = dist_total[is_contact]
# k = 0.2  # N/m
force = defl_total[is_contact]
force -= force[0]
sampling_rate = get_sampling_rate(config)
time = np.arange(len(indentation)) / sampling_rate
print(len(time))
fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
axes[0].plot(time, indentation * 1e6)
axes[0].set_xlabel("Time(s)")
axes[1].set_xlabel("Time(s)")
axes[0].set_ylabel("Indentation(μm)")
axes[1].set_ylabel("Force(nN)")
axes[1].plot(time, force * 1e9)
# %%
max_ind = np.argmax(indentation)
t_max = time[max_ind]
indent_max = indentation[max_ind]
# %%
F_app = force[: max_ind + 1]
F_ret = force[max_ind:]
# %%
# t_max 부분을 겹치게 해야 문제가 안생김
d_app = indentation[: max_ind + 1]
d_ret = indentation[max_ind:]

t_app = time[: max_ind + 1]
t_ret = time[max_ind:]

# %%
# Truncation negrative force region
negative_idx = np.where(F_ret < 0)[0]
negative_idx = negative_idx[0]
# %%
F_ret[negative_idx:] = 0
# F_ret = F_ret[:negative_idx]
# d_ret = d_ret[:negative_idx]
# t_ret = t_ret[:negative_idx]
# %%
t_app, t_ret = t_app * 10, t_ret * 10
d_app, d_ret = d_app * 1e6, d_ret * 1e6  # m -> um
F_app, F_ret = F_app * 1e8, F_ret * 1e8  # N -> nN
# %%
v_app = estimate_derivative(t_app, d_app)
v_ret = estimate_derivative(t_ret, d_ret)

fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
axes[0].plot(t_app, F_app, label="Approach")
axes[0].plot(t_ret, F_ret, label="Retract")
axes[1].plot(t_app, d_app, label="Approach")
axes[1].plot(t_ret, d_ret, label="Retract")
axes[2].plot(t_app, v_app, label="Approach")
axes[2].plot(t_ret, v_ret, label="Retract")
axes[-1].legend()
axes[-1].set_xlabel("Time (s)")
for ax, y_label in zip(
    axes, ("Force (nN)", "Indentation ($\mu$m)", "Velocity ($\mu$m/s)")
):
    ax.set_ylabel(y_label)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(d_app, F_app)
ax.plot(d_ret, F_ret)


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
            t = jax.nn.tanh(layer(t))
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
        self.bias = jnp.asarray(1.0)
        self.nodes, self.weights = scipy.special.roots_legendre(num_quadrature)

    def __call__(self, t: Array) -> Array:
        nodes = jax.lax.stop_gradient(self.nodes)
        weights = jax.lax.stop_gradient(self.weights)
        h = jax.vmap(self.net)(nodes)
        expmx = 0.5 * (nodes + 1)
        return jax.nn.relu(self.scale) * 0.5 * jnp.dot(
            h * expmx ** (t - 1), weights
        ) + jax.nn.relu(self.bias)


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
        self.bias = jnp.asarray(15.0)
        nodes, weights = scipy.special.roots_laguerre(num_quadrature)
        self.nodes = jnp.asarray(nodes)
        self.weights = jnp.asarray(weights)

    def __call__(self, t: Array) -> Array:
        nodes = jax.lax.stop_gradient(self.nodes)
        weights = jax.lax.stop_gradient(self.weights)
        t_ = t + 1e-2
        h = jax.vmap(self.net)(nodes / t_)
        return jax.nn.relu(self.scale) * jnp.dot(weights, h) / t_ + jax.nn.relu(
            self.bias
        )


class BernsteinNN3(eqx.Module):
    net: eqx.Module
    scale: Array
    bias: Array
    nodes: Array
    weights: Array
    N: float

    def __init__(self, net: eqx.Module, num_quadrature: int = 500):
        super().__init__()
        self.net = net
        self.scale = jnp.asarray(0.1)
        self.bias = jnp.asarray(0.5)
        self.nodes, self.weights = scipy.special.roots_legendre(num_quadrature)
        self.N = 1e4

    def __call__(self, t: Array) -> Array:
        nodes = jax.lax.stop_gradient(self.nodes)
        weights = jax.lax.stop_gradient(self.weights)
        y1 = nodes / 2
        y2 = y1 + 0.5
        h1 = jax.vmap(self.net)(-jnp.log(y1)) * (y1 ** (t - 1))
        h2 = jax.vmap(self.net)(-jnp.log(y2)) * (y2 ** (t - 1))
        return jax.nn.relu(self.scale) * 0.5 * jnp.dot(h1 + h2, weights) + jax.nn.relu(
            self.bias
        )


phi_nn = FullyConnectedNetwork(["scalar", 200, "scalar"])
phi_bern = BernsteinNN(phi_nn, 100)
# %%
plt.plot(t_app, jax.vmap(phi_bern)(t_app))
# %%
# plt.plot(t_app, jax.vmap(phi_nn)(-jnp.log1p(t_app) + jnp.log(2)))
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
tip = Spherical(0.4)  # R=0.4um

F_app_pred = force_approach(t_app, phi_bern, t_app, d_app, v_app, tip.alpha, tip.beta)
t1 = find_t1(t_ret, phi_bern, t_app, t_ret, v_app, v_ret)
F_ret_pred = force_retract(
    t_ret, t1, phi_bern, t_app, d_app, v_app, tip.alpha, tip.beta
)
ax.plot(t_app, F_app, label="Approach (data)")
ax.plot(t_ret, F_ret, label="Retract (data)")
ax.plot(t_app, F_app_pred, label="Approach (NN)")
ax.plot(t_ret, F_ret_pred, label="Retract (NN)")
ax.legend()
fig


# %%
@eqx.filter_value_and_grad
def compute_loss(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret):
    F_app_pred = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
    t1_pred = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
    F_ret_pred = force_retract(t_ret, t1_pred, model, t_app, d_app, v_app, 1.0, 1.5)
    # Mean squared error loss
    return jnp.mean(jnp.abs((F_app - F_app_pred))) + jnp.mean(
        jnp.abs((F_ret - F_ret_pred))
    )


@eqx.filter_value_and_grad
def compute_loss_approach(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret):
    F_app_pred = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
    # t1_pred = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
    # F_ret_pred = force_retract(t_ret, t1_pred, model, t_app, d_app, v_app, 1.0, 1.5)
    # Mean squared error loss
    return jnp.mean((F_app - F_app_pred) ** 2)


@eqx.filter_value_and_grad
def compute_loss_retract(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret):
    # F_app_pred = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
    t1_pred = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
    F_ret_pred = force_retract(t_ret, t1_pred, model, t_app, d_app, v_app, 1.0, 1.5)
    # Mean squared error loss
    return jnp.mean((F_ret - F_ret_pred) ** 2)


optim = optax.adam(5e-3)
opt_state = optim.init(phi_bern)


@eqx.filter_jit
def make_step(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret, opt_state):
    loss, grads = compute_loss(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_jit
def make_step_app(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret, opt_state):
    loss, grads = compute_loss_approach(
        model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret
    )
    updates, opt_state = optim.update(grads, opt_state)
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
import numpy as np

max_epochs = 1000
loss_history = np.empty(max_epochs)
for step in range(max_epochs):
    loss, phi_bern, opt_state = make_step(
        phi_bern, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret, opt_state
    )
    loss = loss.item()
    loss_history[step] = loss
    print(f"step={step}, loss={loss}")
# %%
compute_loss(phi_bern, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret)
# %%
phi_bern.nodes
# %%
scipy.special.roots_legendre(100)[0]
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
F_app_pred = force_approach(t_app, phi_bern, t_app, d_app, v_app, 1.0, 1.5)
t1 = find_t1(t_ret, phi_bern, t_app, t_ret, v_app, v_ret)
F_ret_pred = force_retract(t_ret, t1, phi_bern, t_app, d_app, v_app, 1.0, 1.5)
ax.plot(t_app, F_app_pred, label="Approach (pred)")
ax.plot(t_ret, F_ret_pred, label="Retract (pred)")
ax.plot(t_app, F_app, ".", label="Approach (true)")
ax.plot(t_ret, F_ret, ".", label="Retract (true)")
ax.legend()
fig
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(loss_history)
ax.set_yscale("log", base=10)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t_app, jax.vmap(phi_bern)(t_app), label="learned")
# ax.plot(t_app, jax.vmap(sls)(t_app), label="ground truth")
ax.legend()
# %%
