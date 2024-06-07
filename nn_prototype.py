# %%
from functools import partial
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.interpolate as scinterp
from jaxtyping import Array, Float

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
from neuralconstitutive.ting import _force_approach, _force_retract
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.utils import (
    normalize_forces,
    normalize_indentations,
    smooth_data,
)

# jax.config.update("jax_debug_nans", True)

jax.config.update("jax_enable_x64", True)


def softplus_inverse(x: Array) -> Array:
    return jnp.log(jnp.expm1(x))


class PiecewiseCubic(diffrax.CubicInterpolation):

    def _interpret_t(self, t, left: bool) -> tuple:
        maxlen = self.ts_size - 2
        index = jnp.searchsorted(
            self.ts, t, side="left" if left else "right", method="compare_all"
        )
        index = jnp.clip(index - 1, a_min=0, a_max=maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part


def make_smoothed_cubic_spline(indentation, s=1.5e-4):
    tck = scinterp.splrep(indentation.time, indentation.depth, s=s)
    ppoly = scinterp.PPoly.from_spline(tck)
    cubic_interp = PiecewiseCubic(ppoly.x[3:-3], tuple(ppoly.c[:, 3:-3]))
    return cubic_interp


class Prony(AbstractConstitutiveEqn):
    coeffs: Array
    # log10_taus: Array
    bias: Array

    def __init__(self, num_components: int = 10):
        # self.log10_taus = jnp.linspace(-5, 5, num_components)
        log10_taus = jnp.linspace(-5, 2, num_components)
        self.coeffs = softplus_inverse(jnp.ones_like(log10_taus) / num_components)
        # self.coeffs = jnp.ones_like(self.log10_taus) / num_components
        self.bias = softplus_inverse(jnp.asarray(1.0 / num_components))

    def _relaxation_function_1D(self, t: Float[Array, " N"]) -> Float[Array, " N"]:
        c = jax.nn.softplus(self.coeffs)
        log10_taus = jnp.linspace(-4, 2, len(self.coeffs))
        return jnp.matmul(
            jnp.exp(-jnp.expand_dims(t, -1) / 10**log10_taus), c
        ) + jax.nn.softplus(self.bias)


@partial(jax.vmap, in_axes=(0, None, None))
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
        knots = jnp.linspace(0.0, 10.0, num_components)
        self.coeffs = softplus_inverse(jnp.ones_like(knots) / num_components)
        # self.coeffs = jnp.ones_like(self.log10_taus) / num_components
        self.bias = softplus_inverse(jnp.asarray(1.0 / num_components))

    def _relaxation_function_1D(self, t: Float[Array, " N"]) -> Float[Array, " N"]:
        c = jax.nn.softplus(self.coeffs)
        b = jax.nn.softplus(self.bias)
        knots = jnp.linspace(0.0, 1.0, len(self.coeffs))
        basis_funcs = L_mspline(t, knots, knots[1] - knots[0])
        return jnp.matmul(basis_funcs, c) + b


# %%
# %%
constit = Mspline()
# %%
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
G = constit.relaxation_function(t)
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

# %%
app_interp = make_smoothed_cubic_spline(app)
ret_interp = make_smoothed_cubic_spline(ret)
f_app_data = _force_approach(app.time, bimodal, app_interp, tip)
f_ret_data = _force_retract(ret.time, bimodal, (app_interp, ret_interp), tip)
f_ret_data = jnp.clip(f_ret_data, 0.0)
#(f_app, f_ret), _ = normalize_forces(f_app, f_ret)
#(app, ret), (_, h_m) = normalize_indentations(app, ret)
f_ret = jnp.trim_zeros(jnp.clip(f_ret_data, 0.0), "b")
ret = jtu.tree_map(lambda leaf: leaf[: len(f_ret)], ret)
app_interp = make_smoothed_cubic_spline(app)
ret_interp = make_smoothed_cubic_spline(ret)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_data)
ax.plot(ret.time, f_ret_data)
# %%

# %%


def l1_norm(prony):
    coeffs = jnp.abs(prony.coeffs)
    bias = jnp.abs(prony.bias)
    return (jnp.sum(jnp.abs(coeffs)) + bias) / len(coeffs)


@eqx.filter_value_and_grad
def compute_loss(constit, l1_penalty=0.0):
    # f_app = force_approach_conv(constit, app.time, dIb, tip.a())
    f_app = _force_approach(app.time, constit, app_interp, tip)
    f_ret = _force_retract(ret.time, constit, (app_interp, ret_interp), tip)
    # f_ret = force_retract_conv(constit, ret.time, app.time, v_ret, v_app, dIb, tip.a())
    # l1_term = l1_norm(constit)
    return (
        jnp.sum((f_app - f_app_data) ** 2)
        + jnp.sum((f_ret - f_ret_data) ** 2)
        # + l1_penalty * l1_term
    )


@eqx.filter_jit
def make_step(constit, opt_state):
    loss, grads = compute_loss(constit)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(constit, updates)
    return loss, model, opt_state


# %%
#constit = Prony(num_components=20)
constit = Mspline(num_components=100)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_data, label="data")
ax.plot(ret.time, f_ret_data, label="data")
f_app = _force_approach(app.time, constit, app_interp, tip)
f_ret = _force_retract(ret.time, constit, (app_interp, ret_interp), tip)

ax.plot(app.time, f_app, label="prediction")
ax.plot(ret.time, f_ret, label="prediction")
ax.legend()
# %%

# %%

# %%
l1_norm(constit)
# %%

optim = optax.adam(5e-2)
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
f_app = _force_approach(app.time, constit, app_interp, tip)
f_ret = _force_retract(ret.time, constit, (app_interp, ret_interp), tip)
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
t_test = 10 ** jnp.linspace(-5, 3, 100)
H_ = H_approx(t_test, constit)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))

ax.plot(10**bimodal.log10_t_grid, bimodal.h_grid, label="data")
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
# f_ret_data = jnp.clip(f_ret_data, 0.0)
f_ret_data = jnp.trim_zeros(jnp.clip(f_ret_data, 0.0), "b")
ret = Indentation(ret.time[: len(f_ret_data)], ret.depth[: len(f_ret_data)])
(f_app_data, f_ret_data), _ = normalize_forces(f_app_data, f_ret_data)
(app, ret), (_, h_m) = normalize_indentations(app, ret)

tip = Spherical(2.5e-6 / h_m)

app_interp = make_smoothed_cubic_spline(app)
ret_interp = make_smoothed_cubic_spline(ret)

v_app = app_interp.derivative(app.time)
v_ret = ret_interp.derivative(ret.time)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, v_app)
ax.plot(ret.time, v_ret)
# %%


# %%
constit = Mspline(num_components=100)
# constit = Prony(num_components=40)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_data, label="data")
ax.plot(ret.time, f_ret_data, label="data")
f_app = _force_approach(app.time, constit, app_interp, tip)
f_ret = _force_retract(ret.time, constit, (app_interp, ret_interp), tip)
ax.plot(app.time, f_app, label="prediction")
ax.plot(ret.time, f_ret, label="prediction")
ax.legend()

# %%
optim = optax.adam(1e-2)
opt_state = optim.init(constit)
# %%
import time

t_start = time.time()
max_epochs = 2000
loss_history = np.empty(max_epochs)
for step in range(max_epochs):
    loss, constit, opt_state = make_step(constit, opt_state)
    loss = loss.item()
    loss_history[step] = loss
    print(f"step={step}, loss={loss}")
t_end = time.time()
print(f"Time elapsed: {t_end-t_start}")
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_data, label="data")
ax.plot(ret.time, f_ret_data, label="data")
f_app = _force_approach(app.time, constit, app_interp, tip)
f_ret = _force_retract(ret.time, constit, (app_interp, ret_interp), tip)
ax.plot(app.time, f_app, label="prediction")
ax.plot(ret.time, f_ret, label="prediction")
ax.legend()
# %%

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(np.arange(max_epochs), loss_history)
ax.set_yscale("log", base=10)
# %%
constit.coeffs
# %%
constit.bias
# %%
