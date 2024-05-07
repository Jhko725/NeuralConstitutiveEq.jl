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
tip = Spherical(2.5e-6 / h_m)  # Scale tip radius by the length scale we are using

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0] = plot_indentation(axes[0], app, marker=".")
axes[0] = plot_indentation(axes[0], ret, marker=".")

axes[1].plot(app.time, f_app, ".")
axes[1].plot(ret.time, f_ret, ".")

axes[2].plot(app.depth, f_app, ".")
axes[2].plot(ret.depth, f_ret, ".")


# %%
## Fit using Latin hypercube sampling
N_SAMPLES = 10
fit_type = "approach"
### Hertzian model

constit_htz = Hertzian(10.0)
bounds_htz = [(0, 1e3)]

sampler = LatinHypercubeSampler(
    sample_range=[(1e-2, 1e2)],
    sample_scale=["log"],
)

htz_fits, htz_results, htz_initvals, htz_minimizers = fit_indentation_data(
    constit_htz,
    bounds_htz,
    (app, ret),
    (f_app, f_ret),
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)
# %%
# %%
### SLS model
constit_sls = StandardLinearSolid(10.0, 10.0, 10.0)
bounds_sls = [(0, 1e3), (0, 1e3), (1e-6, 1e3)]

sampler = LatinHypercubeSampler(
    sample_range=[(1e-2, 1e2), (1e-2, 1e2), (1e-5, 1e2)],
    sample_scale=["log", "log", "log"],
)
sls_fits, sls_results, sls_initvals, sls_minimizers = fit_indentation_data(
    constit_sls,
    bounds_sls,
    (app, ret),
    (f_app, f_ret),
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)

# %%
### Modified PLR model

constit_mplr = ModifiedPowerLaw(10.0, 10.0, 10.0)
bounds_mplr = [(0, 1e3), (0.0, 1.0), (1e-6, 1e3)]

sampler = LatinHypercubeSampler(
    sample_range=[(1e-2, 1e2), (0.0, 1.0), (1e-5, 1e2)],
    sample_scale=["log", "linear", "log"],
)
mplr_fits, mplr_results, mplr_initvals, mplr_minimizers = fit_indentation_data(
    constit_mplr,
    bounds_mplr,
    (app, ret),
    (f_app, f_ret),
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)

# %%
### KWW model

constit_kww = KohlrauschWilliamsWatts(10.0, 10.0, 10.0, 10.0)
bounds_kww = [(0, 1e3), (0, 1e3), (1e-6, 1e3), (0.0, 1.0)]

sampler = LatinHypercubeSampler(
    sample_range=[
        (1e-2, 1e2),
        (1e-2, 1e2),
        (1e-5, 1e2),
        (0.0, 1.0),
    ],
    sample_scale=["log", "log", "log", "linear"],
)

kww_fits, kww_results, kww_initvals, kww_minimizers = fit_indentation_data(
    constit_kww,
    bounds_kww,
    (app, ret),
    (f_app, f_ret),
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)
# %%
for r in mplr_results:
    display(r)

# %%


def get_best_model(results: list[lmfit.minimizer.MinimizerResult]):
    bic = np.array([res.bic if res is not None else np.inf for res in results])
    ind_best = np.argmin(bic)
    return results[ind_best], ind_best


# %%
## Assess results

results_best = {}
fits_best = {}
for results, fits, name in zip(
    [sls_results, mplr_results, kww_results, htz_results],
    [sls_fits, mplr_fits, kww_fits, htz_fits],
    ["SLS", "MPLR", "KWW", "Hertzian"],
):
    res_best, ind_best = get_best_model(results)
    results_best[name] = res_best
    fits_best[name] = fits[ind_best]

# %%
for r in results_best.values():
    display(r)
    print(r.uvars)

# %%
import equinox as eqx
from neuralconstitutive.ting import _force_approach, _force_retract
from neuralconstitutive.fitting import create_subsampled_interpolants

f_app_fits = {}
f_ret_fits = {}
app_interp = create_subsampled_interpolants(app)
ret_interp = create_subsampled_interpolants(ret)
for name, constit in fits_best.items():
    f_app_fits[name] = _force_approach(app.time, constit, app_interp, tip)
    f_ret_fits[name] = _force_retract(ret.time, constit, (app_interp, ret_interp), tip)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
axes[0].plot(app.time, f_app, ".", color="k", markersize=1.0, label="Data", alpha=0.7)
axes[0].plot(ret.time, f_ret, ".", color="k", markersize=1.0, alpha=0.7)

color_palette = np.array(
    [
        [0.86666667, 0.23137255, 0.20784314],
        [1.0, 0.3882, 0.2784],
        [0.92941176, 0.61568627, 0.24705882],
        [0.63921569, 0.85490196, 0.52156863],
        [0.19764706, 0.46431373, 0.34196078],
        [0.17254902, 0.45098039, 0.69803922],
        [0.372549, 0.596078, 1.0],
        [0.7172549, 0.48117647, 0.92313726],
        [0.64313725, 0.14117647, 0.48627451],
    ]
)
names = ["Hertzian", "SLS", "MPLR", "KWW"]
color_inds = [0, 3, 6, 8]
for n, c_ind in zip(names, color_inds):
    constit = fits_best[n]

    color = color_palette[c_ind]
    axes[0].plot(
        app.time, f_app_fits[n], color=color, linewidth=1.0, label=n, alpha=0.9
    )
    axes[0].plot(ret.time, f_ret_fits[n], color=color, linewidth=1.0, alpha=0.9)
    axes[0].set_xlabel("Time (norm.)")
    axes[0].set_ylabel("Force (norm.)")

    axes[1] = plot_relaxation_fn(axes[1], constit, app.time, color=color, linewidth=1.0)
    axes[1].set_ylim((0, 0.8))
    axes[1].set_xlabel("Time (norm.)")
    axes[1].set_ylabel("$G(t)$ (norm.)")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, ncols=2, loc="upper right")
for ax in axes:
    ax.grid(ls="--", color="lightgray")

fig.suptitle("HeLa cell (interphase) curve fit results: Approach only")
# fig.suptitle("pAAm hydrogel curve fit results: Approach only")
# %%
bics = jnp.asarray([results_best[n].bic for n in names])
colors = color_palette[color_inds]
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.grid(ls="--", color="darkgray")
ax.bar(names, bics, color=colors)
ax.set_yscale("symlog")
ax.set_ylabel("BIC")
ax.set_title("HeLa cell (interphase), Approach only")


# %%
def process_uvars(uvars: dict):
    if "E0" in uvars:
        return uvars
    elif ("E1" in uvars) and ("E_inf" in uvars):
        uvars_ = {}
        for k, v in uvars.items():
            if k == "E1":
                uvars_["E0"] = uvars["E1"] + uvars["E_inf"]
            else:
                uvars_[k] = v
        return uvars_
    else:
        raise ValueError("Unexpected combination of parameters!")


def rel_error(uvar):
    return uvar.s / uvar.n


def get_param_rel_errors(uvars: dict):
    uvars = deepcopy(uvars)
    rel_errors = []
    # First element of the list is always that of E0
    E0 = uvars.pop("E0")
    rel_errors.append(rel_error(E0))
    for v in uvars.values():
        rel_errors.append(rel_error(v))
    return np.asarray(rel_errors)


relative_errors = {}
for name, res in results_best.items():
    display(res)
    uvars_proc = process_uvars(res.uvars)
    relative_errors[name] = get_param_rel_errors(uvars_proc)


fig, ax = plt.subplots(1, 1, figsize=(4, 3))
names = ("Hertzian", "MPLR", "SLS", "KWW")
for name in names:
    ax.plot(relative_errors[name], ".-", label=name, linewidth=1.0, markersize=8.0)
ax.set_yscale("log", base=10)
ax.set_xticks([0, 1, 2, 3])
ax.legend()
ax.set_ylabel("Relative error")
ax.set_xlabel("Model free parameters")
xticklabels = ax.get_xticks().tolist()
xticklabels[0] = "0 ($E_0$)"
ax.set_xticklabels(xticklabels)
ax.grid(ls="--", color="lightgray")

# %%
for n, r in results_best.items():
    print(n, r.params.valuesdict())
# %%
