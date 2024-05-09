# %%
# ruff: noqa: F722
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Sequence

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
)
from neuralconstitutive.fitting import (
    LatinHypercubeSampler,
    fit_all_lmfit,
    fit_approach_lmfit,
    fit_indentation_data,
)
from neuralconstitutive.io import import_data
from neuralconstitutive.plotting import plot_indentation, plot_relaxation_fn
from neuralconstitutive.ting import force_approach, force_retract
from neuralconstitutive.tingx import force_approach as force_approachx
from neuralconstitutive.tingx import force_retract as force_retractx

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

datadir = Path("open_data/Interphase rep 2")
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
#%%
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

#%%
fig, ax = plt.subplots(1, 1, figsize = (5, 3))
ax.plot(app.time, app_interp.derivative(app.time), label = "original")
ax.plot(app.time, app_interp2.derivative(app.time), label = "sparse")
ax.legend()
#%%


#%%
#%%
%%timeit
f_app_sparse = eqx.filter_jit(_force_approach)(app.time, constit, app_interp2, tip)
#%%
%%timeit
f_app = eqx.filter_jit(_force_approach)(app.time, constit, app_interp, tip)
#%%
%%timeit
f_ret_sparse = eqx.filter_jit(_force_retract)(ret.time, constit, (app_interp2, ret_interp2), tip)
#%%
%%timeit
f_ret = eqx.filter_jit(_force_retract)(ret.time, constit, (app_interp, ret_interp), tip)
#%%
fig, ax = plt.subplots(1, 1, figsize = (5, 3))
ax.plot(app.time, f_app, color = "black", label = "original")
ax.plot(app.time, f_app_sparse, color = "orangered", label = "sparse")
ax.plot(ret.time, f_ret, color = "black")
ax.plot(ret.time, f_ret_sparse, color = "orangered")
ax.legend()
#%%
f_app_fit = force_approach(constit, app_fit, tip)
#%%
#%%timeit
f_app_fitx = force_approachx(constit, app_fit, tip)
#%%
fig, ax = plt.subplots(1, 1, figsize = (5, 3))
ax.plot(app.time, f_app_fit, label = "quadax")
ax.plot(app.time, f_app_fitx, label = "integrax")
ax.legend()
#%%


smooth_spl = scinterp.make_smoothing_spline(app.time, app.depth)
smooth_ppoly = scinterp.PPoly.from_spline(smooth_spl)
#%%
smooth_ppoly.c.shape
#%%
len(app.time)
#%%
%%timeit
f_ret_fit = force_retract(constit, (app_fit, ret_fit), tip)

# %%
%%timeit
f_ret_fitx = force_retractx(constit, (app_fit, ret_fit), tip)

# %%
fig, ax = plt.subplots(1, 1, figsize = (5, 3))
ax.plot(ret.time, f_ret_fit, label = "quadax")
ax.plot(ret.time, f_ret_fitx, label = "integrax")
ax.legend()
# %%
from neuralconstitutive.indentation import interpolate_indentation
fig, ax = plt.subplots(1, 1, figsize = (5, 3))
interp = interpolate_indentation(app_fit)
t_interp = jnp.linspace(interp.t0, interp.t1, len(app_fit.time)*100)
ax.plot(t_interp, eqx.filter_vmap(interp.derivative)(t_interp))
ax.scatter(app_fit.time, eqx.filter_vmap(interp.derivative)(app_fit.time), s = 1, color = "red")
# %%
jnp.repeat(jnp.asarray(False), 256)
# %%
