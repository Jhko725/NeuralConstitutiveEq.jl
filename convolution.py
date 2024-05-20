# %%
# ruff: noqa: F722
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Sequence
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from jaxtyping import Bool
from lmfit.minimizer import MinimizerResult
from tqdm import tqdm
import scipy.interpolate as scinterp

from neuralconstitutive.constitutive import (
    Hertzian,
    KohlrauschWilliamsWatts,
    ModifiedPowerLaw,
    StandardLinearSolid,
    AbstractConstitutiveEqn
)
from neuralconstitutive.fitting import (
    LatinHypercubeSampler,
    fit_all_lmfit,
    fit_approach_lmfit,
    fit_indentation_data,
)
from neuralconstitutive.io import import_data
from neuralconstitutive.plotting import plot_indentation, plot_relaxation_fn
#from neuralconstitutive.ting import force_approach, force_retract
#from neuralconstitutive.tingx import force_approach as force_approachx
#from neuralconstitutive.tingx import force_retract as force_retractx

from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.utils import (
    normalize_forces,
    normalize_indentations,
    smooth_data,
)

# %%
jax.config.update("jax_enable_x64", True)

# datadir = Path("open_data/PAAM hydrogel/speed 5")
# name = "PAA_speed 5_4nN"

datadir = Path("data/abuhattum_iscience_2022/Interphase rep 2")
name = "interphase_speed 2_2nN"
(app, ret), (f_app, f_ret) = import_data(
    datadir / f"{name}.tab", datadir / f"{name}.tsv"
)
# app, ret = smooth_data(app), smooth_data(ret)
# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0] = plot_indentation(axes[0], app, marker=".")
axes[0] = plot_indentation(axes[0], ret, marker=".")

axes[1].plot(app.time, f_app, ".")
axes[1].plot(ret.time, f_ret, ".")

axes[2].plot(app.depth, f_app, ".")
axes[2].plot(ret.depth, f_ret, ".")
# %%
f_ret = jnp.clip(f_ret, 0.0)
(f_app, f_ret), _ = normalize_forces(f_app, f_ret)
(app, ret), (_, h_m) = normalize_indentations(app, ret)
# %%
tip = Spherical(2.5e-6 / h_m)

app_fit, ret_fit = smooth_data(app), smooth_data(ret)
constit = StandardLinearSolid(
    3.8624659026709196e-10, 0.30914401077841713, 0.0006863434675687636
)
# %%
from neuralconstitutive.indentation import Indentation, interpolate_indentation
from neuralconstitutive.tingx import _force_approach, _force_retract


def create_subsampled_interpolants(indent, num_knots: int = 10):
    interp = scinterp.make_smoothing_spline(indent.time, indent.depth)
    t_knots = jnp.linspace(indent.time[0], indent.time[-1], num_knots)
    d_knots = interp(t_knots)
    return interpolate_indentation(Indentation(t_knots, d_knots))


app_interp = interpolate_indentation(app_fit)
app_interp2 = create_subsampled_interpolants(app)
ret_interp = interpolate_indentation(ret_fit)
ret_interp2 = create_subsampled_interpolants(ret)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, app_interp.derivative(app.time), label="original")
ax.plot(app.time, app_interp2.derivative(app.time), label="sparse")
ax.legend()
# %%
dIb = (
    tip.b()
    * app_interp2.derivative(app.time)
    * app_interp2.evaluate(app.time) ** (tip.b() - 1)
)
#%%
v_app = app_interp2.derivative(app.time)
v_ret = ret_interp2.derivative(ret.time)
v_total = jnp.concatenate(
    [v_app[:-1], jnp.atleast_1d(0.5 * (v_app[-1] + v_ret[0])), v_ret[1:]]
)


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
        t1 = jnp.clip(t1 - t1_objective(t1) / Dt1_objective(t1), 0.0, t_app[-1])
    return t1


@eqx.filter_jit
def force_retract_conv(constit, t_ret, t_app, v_ret, v_app, dIb_app, a: float = 1.0):
    t1 = find_t1(constit, t_ret, v_ret, t_app, v_app)
    return _force_retract_conv(constit, t_ret, t1, t_app, dIb_app, a)

# %%
# %%timeit
ind = 20
out = eqx.filter_grad(lambda c_: find_t1(c_, ret.time, v_ret, app.time, v_app)[ind])(constit)
print(out.E1, out.E_inf, out.tau)
# %%
%%timeit
f_app_conv = eqx.filter_jit(force_approach_conv)(constit, app.time, dIb, tip.a()).block_until_ready()
# %%
%%timeit
f_app_sparse = eqx.filter_jit(_force_approach)(app.time, constit, app_interp2, tip).block_until_ready()
# %%
%%timeit
f_app = eqx.filter_jit(_force_approach)(app.time, constit, app_interp, tip).block_until_ready()
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, label="original")
ax.plot(app.time, f_app_sparse, label="sparse")
ax.plot(app.time, f_app_conv, label="convolution")
ax.legend()
# %%
from neuralconstitutive.ting import t1_scalar

t1 = eqx.filter_jit(eqx.filter_vmap(t1_scalar, in_axes=(0, None, None, None)))(
    ret.time, constit, (app_interp2, ret_interp2), 5
)

t_total = jnp.concatenate((app.time, ret.time[1:]))
dIb_total = jnp.concatenate((dIb, jnp.zeros_like(ret.time[1:])))

# %%
%%timeit
f_ret_conv = eqx.filter_jit(force_retract_conv)(constit, ret.time, app.time, v_ret, v_app, dIb, tip.a()).block_until_ready()
# %%
%%timeit
f_ret_sparse = eqx.filter_jit(_force_retract)(
    ret.time, constit, (app_interp2, ret_interp2), tip
).block_until_ready()
# %%
%%timeit
f_ret = eqx.filter_jit(_force_retract)(ret.time, constit, (app_interp, ret_interp), tip).block_until_ready()
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(ret.time, f_ret, label="original")
ax.plot(ret.time, f_ret_sparse, label="sparse")
ax.plot(ret.time, f_ret_conv, label="convolution")
ax.legend()
# %%

# %%
# @partial(jax.vmap, in_axes = (None, None, 0))
def mask_by_time_up(t, x, t1):
    return jnp.where(t > t1, x, 0.0)


def t1_objective(t1, t_tot, G_tot, v_tot):
    v_masked = mask_by_time_up(t_tot, v_tot, t1)
    return jax.vmap(partial(jnp.convolve, mode="valid"), in_axes=(None, 0))(
        G_tot, v_masked
    ).squeeze(-1)


# %%
obj = t1_objective(t1, t_total, G_total, v_total)
# %%
plt.plot(obj)
# %%
obj
# %%
v_masked = mask_by_time_up(t_total, v_total, t1)
v_masked.shape
# %%
plt.plot(v_masked[0])
# %%
obj
# %%
obj = jax.vmap(jnp.convolve, in_axes=(None, 0))(G_total, v_masked)[
    :, len(t_total) - len(ret.time)
]
# %%
obj
# %%
plt.plot(obj)
# %%
t1
# %%

# %%
@partial(eqx.filter_vmap, in_axes=(0, 0, None, None, None))
def t1_obj1(t, t1, constit, t_app, v_app):
    G = eqx.filter_vmap(constit.relaxation_function)(t - t_app)
    v_ = mask_by_time_up(t_app, v_app, jnp.atleast_1d(t1))
    return jnp.dot(G, v_) * (ret.time[1] - ret.time[0])
#%%
%%timeit
G_ret = eqx.filter_vmap(constit.relaxation_function)(ret.time)
t1_obj2 = jnp.convolve(G_ret, v_ret)[0 : len(G_ret)] * (ret.time[1] - ret.time[0])
t1_obj = t1_obj1(ret.time, t1, constit, app.time, v_app) + t1_obj2
# %%
mask_by_time_up(app.time, v_app, t1[1:2])
# %%
%%timeit
t1_test = find_t1(constit, ret.time, v_ret, app.time, v_app).block_until_ready()
# %%
t1_test
# %%
ret.time
# %%
