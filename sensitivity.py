# %%
# ruff: noqa: F722
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.flatten_util
import lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from jaxtyping import Bool
from lmfit.minimizer import MinimizerResult
from tqdm import tqdm

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
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.utils import (
    normalize_forces,
    normalize_indentations,
    smooth_data,
)

# %%
jax.config.update("jax_enable_x64", True)

datadir = Path("open_data/PAAM hydrogel/speed 5")
(app, ret), (f_app, f_ret) = import_data(
    datadir / "PAA_speed 5_4nN.tab", datadir / "PAA_speed 5_4nN.tsv"
)
app, ret = smooth_data(app), smooth_data(ret)
# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0] = plot_indentation(axes[0], app, marker=".")
axes[0] = plot_indentation(axes[0], ret, marker=".")

axes[1].plot(app.time, f_app, ".")
axes[1].plot(ret.time, f_ret, ".")

axes[2].plot(app.depth, f_app, ".")
axes[2].plot(ret.depth, f_ret, ".")
# %%
(f_app, f_ret), _ = normalize_forces(f_app, f_ret)
(app, ret), (_, h_m) = normalize_indentations(app, ret)
# %%
tip = Spherical(2.5e-6 / h_m)  # Scale tip radius by the length scale we are using

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0] = plot_indentation(axes[0], app, marker=".")
axes[0] = plot_indentation(axes[0], ret, marker=".")

axes[1].plot(app.time, f_app, ".")
axes[1].plot(ret.time, f_ret, ".")

axes[2].plot(app.depth, f_app, ".")
axes[2].plot(ret.depth, f_ret, ".")


# constit_sls = StandardLinearSolid(0.00911, 0.30103, 9544133.23332)
# constit_sls = ModifiedPowerLaw(0.31013634, 2.7684e-09, 19806.8520)
constit_sls = KohlrauschWilliamsWatts(0.04057453, 0.26956466, 9997384.77, 0.98299196)
# %%
import jax.numpy as jnp
import equinox as eqx
import jax


def residual(constit, app, f_app, tip):
    f_app_pred = force_approach(constit, app, tip)
    return jnp.sum((f_app - f_app_pred))


grad_func = eqx.filter_grad(residual, has_aux=False)
# %%
g = grad_func(constit_sls, app, f_app, tip)
# %%
g_array, _ = jax.flatten_util.ravel_pytree(g)
g_array
# %%
L_mat = g_array.reshape(-1, 1) * g_array.reshape(1, -1)
L_mat
# %%
eigvals, eigvecs = jnp.linalg.eigh(L_mat)
# %%
eigvals
# %%
eigvecs[:, 3]
# %%
residual(constit_sls, app, f_app, tip)
# %%
f_app_pred = force_approach(constit_sls, app, tip)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, ".")
ax.plot(app.time, f_app_pred)


# %%
def residual_all(constit, indentations, forces, tip):
    app, _ = indentations
    f_app, f_ret = forces
    f_app_pred = force_approach(constit, app, tip)
    f_ret_pred = force_retract(constit, indentations, tip)
    return jnp.sum((f_app - f_app_pred)) + jnp.sum((f_ret - f_ret_pred))


constit_sls = StandardLinearSolid(0.08490180, 0.26819740, 0.35419074)
# constit_sls = ModifiedPowerLaw(0.38759320, 0.08431627, 	0.01657088)
# constit_sls = KohlrauschWilliamsWatts(1.19798431, 0.01304129, 4.8266e-04, 0.05537683)
# %%
grad_func = eqx.filter_grad(residual_all, has_aux=False)
# %%
g = grad_func(constit_sls, (app, ret), (f_app, f_ret), tip)
# %%
g_array, _ = jax.flatten_util.ravel_pytree(g)
g_array
# %%
L_mat = g_array.reshape(-1, 1) * g_array.reshape(1, -1)
L_mat
# %%
eigvals, eigvecs = jnp.linalg.eigh(L_mat)
# %%
eigvals
# %%
eigvecs[:, 3]
