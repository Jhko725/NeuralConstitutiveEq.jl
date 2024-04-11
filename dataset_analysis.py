# %%
# ruff: noqa: F722
from pathlib import Path

from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from neuralconstitutive.plotting import plot_indentation, plot_relaxation_fn
from neuralconstitutive.constitutive import (
    StandardLinearSolid,
)
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.fitting import force_approach, force_retract
from neuralconstitutive.io import import_data
from neuralconstitutive.fitting import fit_approach_lmfit, fit_all_lmfit
from neuralconstitutive.utils import (
    smooth_data,
    normalize_indentations,
    normalize_forces,
)

jax.config.update("jax_enable_x64", True)

datadir = Path("open_data/PAAM hydrogel")
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

# %%
## Fit only the approach portion
constit = StandardLinearSolid(10.0, 10.0, 10.0)
bounds = [(0.0, jnp.inf)] * 3
constit_fit, result = fit_approach_lmfit(constit, bounds, tip, app, f_app)
display(result)

# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(constit_fit, app, tip)
f_fit_ret = force_retract(constit_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time)
# %%
## Fit the entire approach-retract curve
constit = StandardLinearSolid(10.0, 10.0, 10.0)
bounds = [(0.0, jnp.inf)] * 3
constit_fit, result = fit_all_lmfit(constit, bounds, tip, (app, ret), (f_app, f_ret))
display(result)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(constit_fit, app, tip)
f_fit_ret = force_retract(constit_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time)

# %%
