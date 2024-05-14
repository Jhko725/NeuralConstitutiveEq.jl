# %%
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array

from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
    FromLogDiscreteSpectrum,
)
from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.indentation import Indentation, interpolate_indentation
from neuralconstitutive.io import import_data

# from neuralconstitutive.integrate import integrate
from neuralconstitutive.plotting import plot_relaxation_fn
from neuralconstitutive.relaxation_spectrum import HonerkampWeeseBimodal
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.utils import (
    normalize_forces,
    normalize_indentations,
    smooth_data,
)

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)


class Prony(AbstractConstitutiveEqn):
    coeffs: Array
    # log10_taus: Array
    bias: Array

    def __init__(self, num_components: int = 10):
        # self.log10_taus = jnp.linspace(-5, 5, num_components)
        log10_taus = jnp.linspace(-5, 2, num_components)
        self.coeffs = jnp.ones_like(log10_taus) / num_components
        # self.coeffs = jnp.ones_like(self.log10_taus) / num_components
        self.bias = jnp.zeros(1)[0]

    def relaxation_function(self, t: FloatScalar) -> FloatScalar:
        c = jnp.abs(self.coeffs)
        log10_taus = jnp.linspace(-5, 2, len(self.coeffs))
        return jnp.dot(c, jnp.exp(-t / 10**log10_taus)) + jnp.abs(self.bias)


def L_mspline(s, k0, dK):
    c = jnp.expm1(dK * s) / dK
    c = jnp.where(s == 0, 1.0, c / jnp.where(s == 0, 1.0, s))
    return jnp.exp(-s * (3 * dK + k0)) * (c**3)


class Mspline(AbstractConstitutiveEqn):
    coeffs: Array
    # log10_taus: Array
    bias: Array

    def __init__(self, num_components: int = 100):
        # self.log10_taus = jnp.linspace(-5, 5, num_components)
        knots = jnp.linspace(0.0, 100.0, num_components)
        self.coeffs = jnp.ones_like(knots) / num_components
        # self.coeffs = jnp.ones_like(self.log10_taus) / num_components
        self.bias = jnp.asarray(0.2)

    def relaxation_function(self, t: FloatScalar) -> FloatScalar:
        c = jnp.abs(self.coeffs)
        b = jnp.abs(self.bias)
        knots = jnp.linspace(0.0, 100.0, len(self.coeffs))
        basis_funcs = L_mspline(t, knots, knots[1] - knots[0])
        return jnp.dot(c, basis_funcs) + b


# %%
# %%
constit = Mspline()
# %%
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
G = eqx.filter_vmap(constit.relaxation_function)(t)
# %%

# %%
bimodal = FromLogDiscreteSpectrum(
    HonerkampWeeseBimodal(t_x=5e-3, t_y=0.5, t_a=1e-4, t_b=10)
)
t_app = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
d_app = 1.0 * t_app
t_ret = jnp.arange(1.0, 2.0 + 1e-3, 1e-3)
d_ret = 2.0 - t_ret

app, ret = Indentation(t_app, d_app), Indentation(t_ret, d_ret)
del t_app, d_app, t_ret, d_ret

# app_interp = create_subsampled_interpolants(app)
# ret_interp = create_subsampled_interpolants(ret)
app_interp = interpolate_indentation(app)
ret_interp = interpolate_indentation(ret)
tip = Spherical(1.0)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
G = eqx.filter_vmap(constit.relaxation_function)(t)
ax.plot(t, G, ".", label="pred")
ax = plot_relaxation_fn(ax, bimodal, app.time, label="data")
ax.legend()


# %%
def force_approach_conv(
    constit: AbstractConstitutiveEqn, t_app, dIb_app, a: float = 1.0
):
    G = constit.relaxation_function(t_app)
    dt = t_app[1] - t_app[0]
    return a * jnp.convolve(G, dIb_app)[0 : len(G)] * dt


@partial(eqx.filter_vmap, in_axes=(None, 0, 0, None, None, None))
def _force_retract_conv(
    constit: AbstractConstitutiveEqn, t, t1, t_app, dIb_app, a: float = 1.0
):
    G = constit.relaxation_function(t - t_app)
    dt = t_app[1] - t_app[0]
    dIb_masked = jnp.where(t_app <= t1, dIb_app, 0.0)
    return a * jnp.dot(G, dIb_masked) * dt


@partial(jax.vmap, in_axes=(None, None, 0))
def mask_by_time_lower(t, x, t1):
    return jnp.where(t > t1, x, 0.0)


@eqx.filter_jit
def find_t1(constit: AbstractConstitutiveEqn, t_ret, v_ret, t_app, v_app):
    G_ret = constit.relaxation_function(t_ret)
    dt = t_ret[1] - t_ret[0]
    t1_obj_const = jnp.convolve(G_ret, v_ret)[0 : len(G_ret)]

    G_matrix = constit.relaxation_function(jnp.expand_dims(t_ret, -1) - t_app)

    def t1_objective(t1):
        v_app_ = mask_by_time_lower(t_app, v_app, t1)
        return (jnp.sum(G_matrix * v_app_, axis=-1) + t1_obj_const) * dt

    def Dt1_objective(t1):
        ind_t1 = jnp.rint(t1 / dt).astype(jnp.int_)
        return -constit.relaxation_function(t_ret - t1) * v_app.at[ind_t1].get()

    t1 = jnp.linspace(t_app[-1], 0.0, len(t_ret))

    for _ in range(3):
        t1 = jnp.clip(t1 - t1_objective(t1) / Dt1_objective(t1), 0.0)

    return jnp.rint(t1 / dt) * dt


@eqx.filter_jit
def force_retract_conv(constit, t_ret, t_app, v_ret, v_app, dIb_app, a: float = 1.0):
    t1 = find_t1(constit, t_ret, v_ret, t_app, v_app)
    return _force_retract_conv(constit, t_ret, t1, t_app, dIb_app, a)


# %%
dIb = (
    tip.b()
    * app_interp.derivative(app.time)
    * app_interp.evaluate(app.time) ** (tip.b() - 1)
)
v_app = app_interp.derivative(app.time)
v_ret = ret_interp.derivative(ret.time)
# %%
dIb
# %%timeit
f_app = force_approach_conv(constit, app.time, dIb, tip.a())
f_ret = force_retract_conv(constit, ret.time, app.time, v_ret, v_app, dIb, tip.a())
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app)
ax.plot(ret.time, f_ret, ".")
# %%
v_ret
# %%
f_app_data = force_approach_conv(bimodal, app.time, dIb, tip.a())
f_ret_data = force_retract_conv(bimodal, ret.time, app.time, v_ret, v_app, dIb, tip.a())
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, v_app)
# ax.plot(ret.time, v_ret)
# %%
t1 = find_t1(constit, ret.time, v_ret, app.time, v_app)
plt.plot(ret.time, t1)
# %%
import optax


def l1_norm(prony):
    coeffs = jnp.abs(prony.coeffs)
    bias = jnp.abs(prony.bias)
    return (jnp.sum(jnp.abs(coeffs)) + bias) / len(coeffs)


@eqx.filter_value_and_grad
def compute_loss(constit, l1_penalty=0.0):
    # f_app = force_approach_conv(constit, app.time, dIb, tip.a())
    f_ret = force_retract_conv(constit, ret.time, app.time, v_ret, v_app, dIb, tip.a())
    l1_term = l1_norm(constit)
    return (
        # jnp.sum((f_app - f_app_data) ** 2)
        jnp.sum((f_ret - f_ret_data) ** 2)
        # + l1_penalty * l1_term
    )


@eqx.filter_jit
def make_step(constit, opt_state):
    loss, grads = compute_loss(constit)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(constit, updates)
    return loss, model, opt_state


# %%
constit = Prony(num_components=40)
constit = Mspline()
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_data, label="data")
ax.plot(ret.time, f_ret_data, label="data")
f_app = force_approach_conv(constit, app.time, dIb, tip.a())
f_ret = force_retract_conv(constit, ret.time, app.time, v_ret, v_app, dIb, tip.a())
ax.plot(app.time, f_app, label="prediction")
ax.plot(ret.time, f_ret, label="prediction")
ax.legend()
# %%

# %%
l1_norm(constit)
# %%

optim = optax.adam(5e-3)
opt_state = optim.init(constit)

max_epochs = 10000
loss_history = np.empty(max_epochs)
for step in range(max_epochs):
    loss, constit, opt_state = make_step(constit, opt_state)
    loss = loss.item()
    loss_history[step] = loss
    print(f"step={step}, loss={loss}")
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_data, label="data")
ax.plot(ret.time, f_ret_data, label="data")
f_app = force_approach_conv(constit, app.time, dIb, tip.a())
f_ret = force_retract_conv(constit, ret.time, app.time, v_ret, v_app, dIb, tip.a())
ax.plot(app.time, f_app, label="prediction")
ax.plot(ret.time, f_ret, label="prediction")
ax.legend()

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(np.arange(max_epochs), loss_history)
ax.set_yscale("log", base=10)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
t_h = jnp.linspace(-5, 5, 50)
# H = eqx.filter_vmap(constit.nn)(t_h)
# ax.plot(t_h, H / jnp.sum)
ax.plot(10**bimodal.log10_t_grid, bimodal.h_grid, label="data")
ax.plot(
    10 ** jnp.linspace(-5, 5, len(constit.coeffs)), jax.nn.relu(constit.coeffs), "."
)
ax.set_xlim(0, 10)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
G = eqx.filter_vmap(constit.relaxation_function)(t)
ax.plot(t, G, ".", label="pred", markersize=1.0)
ax = plot_relaxation_fn(ax, bimodal, app.time, label="data")
ax.legend()
# %%
(10**constit.log10_taus)

# %%
jax.nn.relu(constit.coeffs)
# %%
jnp.linspace(-5, 5, 10)
# %%
constit.bias
# %%
plt.plot(10**bimodal.log10_t_grid, bimodal.h_grid, label="data")


# %%
@partial(eqx.filter_vmap, in_axes=(0, None))
def H_approx(t, constit):
    def _inner(logt):
        return constit.relaxation_function(jnp.exp(logt))

    return -eqx.filter_grad(_inner)(jnp.log(t))


# %%
t_test = 10**bimodal.log10_t_grid
H_ = H_approx(t_test, constit)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))

ax.plot(t_test, bimodal.h_grid, label="data")
ax.plot(t_test, H_, ".")
ax.set_xscale("log")

# %%
ret
# %%
datadir = Path("data/abuhattum_iscience_2022/Interphase rep 2")
name = "interphase_speed 2_2nN"
(app, ret), (f_app_data, f_ret_data) = import_data(
    datadir / f"{name}.tab", datadir / f"{name}.tsv"
)
app, ret = smooth_data(app), smooth_data(ret)
f_ret_data = jnp.clip(f_ret_data, 0.0)
# f_ret_data = jnp.trim_zeros(jnp.clip(f_ret_data, 0.0), "b")
# ret = Indentation(ret.time[: len(f_ret_data)], ret.depth[: len(f_ret_data)])
(f_app_data, f_ret_data), _ = normalize_forces(f_app_data, f_ret_data)
(app, ret), (_, h_m) = normalize_indentations(app, ret)

tip = Spherical(2.5e-6 / h_m)

app_interp = interpolate_indentation(smooth_data(app))
ret_interp = interpolate_indentation(smooth_data(ret))

dIb = (
    tip.b()
    * app_interp.derivative(app.time)
    * app_interp.evaluate(app.time) ** (tip.b() - 1)
)
v_app = app_interp.derivative(app.time)
v_ret = ret_interp.derivative(ret.time)

# %%
constit = Mspline(num_components=50)
# constit = Prony(num_components=40)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_data, label="data")
ax.plot(ret.time, f_ret_data, label="data")
f_app = force_approach_conv(constit, app.time, dIb, tip.a())
f_ret = force_retract_conv(constit, ret.time, app.time, v_ret, v_app, dIb, tip.a())
ax.plot(app.time, f_app, label="prediction")
ax.plot(ret.time, f_ret, label="prediction")
ax.legend()
# %%
f_ret_data
# %%
optim = optax.adam(5e-3)
opt_state = optim.init(constit)

max_epochs = 10000
loss_history = np.empty(max_epochs)
for step in range(max_epochs):
    loss, constit, opt_state = make_step(constit, opt_state)
    loss = loss.item()
    loss_history[step] = loss
    print(f"step={step}, loss={loss}")

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_data, label="data")
ax.plot(ret.time, f_ret_data, label="data")
f_app = force_approach_conv(constit, app.time, dIb, tip.a())
f_ret = force_retract_conv(constit, ret.time, app.time, v_ret, v_app, dIb, tip.a())
ax.plot(app.time, f_app, label="prediction")
ax.plot(ret.time, f_ret, label="prediction")
ax.legend()
# %%
constit.bias
# %%
