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
    t1_constraint,
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
plt.plot(t_app, jax.vmap(sls)(t_app))


# %%
class PronyNN(eqx.Module):
    weights: Array
    scales: Array
    bias: Array
    t1s: Array
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
        self.t1s = jnp.linspace(
            0.2,
            0.0,
            100,
        )

    def __call__(self, t: Array) -> Array:
        return jnp.dot(
            jax.nn.relu(self.weights), jnp.exp(-jax.nn.relu(self.scales) * t)
        ) + jax.nn.relu(self.bias)


# %%
phi_prony = PronyNN(1e-3, 1e2, 20)
phi_prony.scales
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# ax.plot(t_app, jax.vmap(phi_bern)(t_app))
ax.plot(t_app, jax.vmap(phi_prony)(t_app))
# %%
# plt.plot(t_app, jax.vmap(phi_nn)(-jnp.log1p(t_app) + jnp.log(2)))
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
F_app_pred = force_approach(t_app, phi_prony, t_app, d_app, v_app, 1.0, 1.5)
F_ret_pred = force_retract(
    t_ret, phi_prony.t1s, phi_prony, t_app, d_app, v_app, 1.0, 1.5
)
ax.plot(t_app, f_app, label="approach")
ax.plot(t_ret, f_ret, label="retract")
ax.plot(t_app, F_app_pred, label="approach (nn)")
ax.plot(t_ret, F_ret_pred, label="retract (nn)")
ax.legend()
fig
# %%
jax.vmap(t1_constraint, in_axes=(0, 0, None, None, None, None, None))(
    t_ret, phi_prony.t1s, phi_prony, t_app, d_app, v_app, v_ret
)


# %%
@eqx.filter_value_and_grad
def compute_loss(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret):
    F_app_pred = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
    t1_pred = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
    F_ret_pred = force_retract(t_ret, t1_pred, model, t_app, d_app, v_app, 1.0, 1.5)
    # Mean squared error loss
    # l1 = jnp.sum(jnp.abs(model.weights)) + jnp.abs(model.bias)
    return jnp.mean((F_app - F_app_pred) ** 2) + jnp.mean((F_ret - F_ret_pred) ** 2)


@eqx.filter_value_and_grad
def compute_loss_approach(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret):
    F_app_pred = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
    # t1_pred = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
    # F_ret_pred = force_retract(t_ret, t1_pred, model, t_app, d_app, v_app, 1.0, 1.5)
    # Mean squared error loss
    # l1 = jnp.sum(jnp.abs(model.weights)) + jnp.abs(model.bias)
    return jnp.mean((F_app - F_app_pred) ** 2)  # + 1e-3 * l1


@eqx.filter_value_and_grad
def compute_loss_t1(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret):
    # F_app_pred = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
    t1_vals = jax.vmap(t1_constraint, in_axes=(0, 0, None, None, None, None, None))(
        t_ret, model.t1s, model, t_app, d_app, v_app, v_ret
    )
    # F_ret_pred = force_retract(
    #    t_ret, jnp.clip(model.t1s, 0.0, 0.2), model, t_app, d_app, v_app, 1.0, 1.5
    # )
    return (
        #    np.mean((F_app - F_app_pred) ** 2)
        #    + jnp.mean((F_ret - F_ret_pred) ** 2)
        # +
        jnp.mean(jnp.abs(t1_vals))
    )
    # t1_pred = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
    # F_ret_pred = force_retract(t_ret, t1_pred, model, t_app, d_app, v_app, 1.0, 1.5)
    # Mean squared error loss
    # l1 = jnp.sum(jnp.abs(model.weights)) + jnp.abs(model.bias)
    # return jnp.mean((F_app - F_app_pred) ** 2)  # + 1e-3 * l1


@eqx.filter_value_and_grad
def compute_loss_retract(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret):
    # F_app_pred = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
    t1_pred = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
    F_ret_pred = force_retract(t_ret, t1_pred, model, t_app, d_app, v_app, 1.0, 1.5)
    # Mean squared error loss
    return jnp.mean((F_ret - F_ret_pred) ** 2)


optim = optax.adam(
    1e-3
)  # optax.chain(optax.adam(1e-3), optax.keep_params_nonnegative())
opt_state = optim.init(phi_prony)


@eqx.filter_jit
def make_step(model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret, opt_state):
    loss, grads = compute_loss_t1(
        model, t_app, t_ret, d_app, v_app, v_ret, F_app, F_ret
    )
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
import numpy as np

model = phi_prony
# %%

# %%
max_epochs = 2000
loss_history = np.empty(max_epochs)
for step in range(max_epochs):
    loss, model, opt_state = make_step_app(
        model, t_app, t_ret, d_app, v_app, v_ret, f_app, f_ret, opt_state
    )
    # print(model.taus)
    # print(model.weights)
    loss = loss.item()
    loss_history[step] = loss
    print(f"step={step}, loss={loss}")
# %%
model.t1s
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
F_app = force_approach(t_app, model, t_app, d_app, v_app, 1.0, 1.5)
t1 = find_t1(t_ret, model, t_app, t_ret, v_app, v_ret)
F_ret = force_retract(t_ret, t1, model, t_app, d_app, v_app, 1.0, 1.5)
ax.plot(t_app, F_app, label="Approach (pred)")
ax.plot(t_ret, F_ret, label="Retract (pred)")
ax.plot(t_app, f_app, ".", label="Approach (true)")
ax.plot(t_ret, f_ret, ".", label="Retract (true)")
ax.legend()
fig
# %%
compute_loss_t1(model, t_app, t_ret, d_app, v_app, v_ret, f_app, f_ret)[1].scales
# %%
jax.vmap(t1_constraint, in_axes=(0, 0, None, None, None, None, None))(
    t_ret, model.t1s, model, t_app, d_app, v_app, v_ret
)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(loss_history)
ax.set_yscale("log", base=10)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t_app, jax.vmap(model)(t_app), label="learned")
ax.plot(t_app, jax.vmap(sls)(t_app), label="ground truth")
ax.legend()
# %%
