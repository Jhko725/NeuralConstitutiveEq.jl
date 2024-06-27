# %%
# ruff: noqa: F722
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
from IPython.display import display
from jaxtyping import Bool
from lmfit.minimizer import MinimizerResult
from tqdm import tqdm

from neuralconstitutive.constitutive import (
    FractionalKelvinVoigt,
    GeneralizedMaxwellmodel,
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
from neuralconstitutive.smoothing import make_smoothed_cubic_spline
from neuralconstitutive.ting import (
    _force_approach,
    _force_retract,
    force_approach,
    force_retract,
)
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.utils import (
    normalize_forces,
    normalize_indentations,
    smooth_data,
)

# %%
jax.config.update("jax_enable_x64", True)

datadir = Path("open_data/PAAM hydrogel/speed 5")
name = "PAA_speed 5_4nN"

# datadir = Path("data/abuhattum_iscience_2022/Agarose/speed 5")
# name = "Agarose_speed 5_2nN"
(app, ret), (f_app, f_ret) = import_data(
    datadir / f"{name}.tab", datadir / f"{name}.tsv"
)
# app, ret = smooth_data(app), smooth_data(ret)
# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0] = plot_indentation(axes[0], app, marker=".")
axes[0] = plot_indentation(axes[0], ret, marker=".")
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Indentation [m]")
axes[0].grid(ls="--", color="darkgray")

axes[1].plot(app.time, f_app, ".")
axes[1].plot(ret.time, f_ret, ".")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Force [N]")
axes[1].grid(ls="--", color="darkgray")

axes[2].plot(app.depth, f_app, ".")
axes[2].plot(ret.depth, f_ret, ".")
axes[2].set_xlabel("Indentation [m]")
axes[2].set_ylabel("Force [N]")
axes[2].grid(ls="--", color="darkgray")
# %%

## Abstract fitting into three parts: init / step / postprocess
f_ret = jnp.clip(f_ret, 0.0)
(f_app, f_ret), _ = normalize_forces(f_app, f_ret)
(app, ret), (_, h_m) = normalize_indentations(app, ret)
f_ret = jnp.trim_zeros(jnp.clip(f_ret, 0.0), "b")
ret = jtu.tree_map(lambda leaf: leaf[: len(f_ret)], ret)
# %%
tip = Spherical(2.5e-6 / h_m)  # Scale tip radius by the length scale we are using

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0] = plot_indentation(axes[0], app, marker=".")
axes[0] = plot_indentation(axes[0], ret, marker=".")
axes[0].set_xlabel("Time [norm.]")
axes[0].set_ylabel("Indentation [norm.]")
axes[0].grid(ls="--", color="darkgray")

axes[1].plot(app.time, f_app, ".")
axes[1].plot(ret.time, f_ret, ".")
axes[1].set_xlabel("Time [norm.]")
axes[1].set_ylabel("Force [norm.]")
axes[1].grid(ls="--", color="darkgray")

axes[2].plot(app.depth, f_app, ".")
axes[2].plot(ret.depth, f_ret, ".")
axes[2].set_xlabel("Indentation [norm.]")
axes[2].set_ylabel("Force [norm.]")
axes[2].grid(ls="--", color="darkgray")

app_interp = make_smoothed_cubic_spline(app)
ret_interp = make_smoothed_cubic_spline(ret)
# %%
def residual_approach(constit, args):
    t_data, f_data, interp, tip = args
    constit = jtu.tree_map(lambda x: x**10, constit)
    f_app_pred = _force_approach(t_data, constit, interp, tip)
    return f_app_pred - f_data


def residual_all(constit, args):
    t_data, f_data, interps, tip = args
    t_app, t_ret = t_data
    f_app_pred = _force_approach(t_app, constit, interps[0], tip)
    f_ret_pred = _force_retract(t_ret, constit, interps, tip)
    return jnp.concatenate((f_app_pred, f_ret_pred)) - jnp.concatenate(f_data)


def fit_approach_optx(constit, t_data, f_data, interp, tip, bounds):
    solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-6)
    args = (t_data, f_data, interp, tip)
    sol = optx.least_squares(residual_approach, solver, constit, args, max_steps=1000)
    return sol.value


def fit_all_optx(constit, t_data, f_data, interp, tip, bounds):
    solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-6)
    args = (t_data, f_data, interp, tip)
    sol = optx.least_squares(residual_all, solver, constit, args, max_steps=1000)
    return sol.value


# %%
# %%
## Fit using Latin hypercube sampling
N_SAMPLES = 5
fit_type = "both"
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
N_SAMPLES = 5**3
constit_sls = StandardLinearSolid(10.0, 10.0, 10.0)
bounds_sls = [(0, 1e3), (0, 1e3), (1e-6, 1e3)]
# %%
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
N_SAMPLES = 5**3
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
N_SAMPLES = 5**4
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
### Fractional Kelvin Voigt model
N_SAMPLES = 5**3
constit_fkv = FractionalKelvinVoigt(10.0, 10.0, 10.0)
bounds_fkv = [(0, 1e3), (0, 1e3), (0.0, 1.0)]

sampler = LatinHypercubeSampler(
    sample_range=[(1e-2, 1e2), (1e-2, 1e2), (0.0, 1.0)],
    sample_scale=["log", "log", "linear"],
)

fkv_fits, fkv_results, fkv_initvals, fkv_minimizers = fit_indentation_data(
    constit_fkv,
    bounds_fkv,
    (app, ret),
    (f_app, f_ret),
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)
# %%
### Generalized Maxwell model
N_SAMPLES = 5**5
constit_gm = GeneralizedMaxwellmodel(10.0, 10.0, 10.0, 10.0, 10.0)
bounds_gm = [(0, 1e3), (0, 1e3), (0, 1e3), (0, 1e3), (0, 1e3)]

sampler = LatinHypercubeSampler(
    sample_range=[(1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2)],
    sample_scale=["log", "log", "log", "log", "log"],
)

gm_fits, gm_results, gm_initvals, gm_minimizers = fit_indentation_data(
    constit_gm,
    bounds_gm,
    (app, ret),
    (f_app, f_ret),
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)
# %%
# for r in mplr_results:
#     display(r)
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
    [sls_results, mplr_results, kww_results, htz_results, fkv_results, gm_results],
    [sls_fits, mplr_fits, kww_fits, htz_fits, fkv_fits, gm_fits],
    ["SLS", "MPLR", "KWW", "Hertzian", "FKV", "GM"],
):
    res_best, ind_best = get_best_model(results)
    results_best[name] = res_best
    fits_best[name] = fits[ind_best]

# %%
# for r in results_best.values():
#     display(r)
#     print(r.uvars)

# %%
import equinox as eqx

# from neuralconstitutive.fitting import create_subsampled_interpolants
from neuralconstitutive.ting import _force_approach, _force_retract

f_app_fits = {}
f_ret_fits = {}

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
names = ["Hertzian", "SLS", "MPLR", "KWW", "FKV", "GM"]
color_inds = [0, 3, 6, 8, 5, 1]
for n, c_ind in zip(names, color_inds):
    constit = fits_best[n]

    color = color_palette[c_ind]
    axes[0].plot(
        app.time, f_app_fits[n], color=color, linewidth=1.0, label=n, alpha=0.9
    )
    axes[0].plot(ret.time, f_ret_fits[n], color=color, linewidth=1.0, alpha=0.9)
    axes[0].set_xlabel("Time [norm.]")
    axes[0].set_ylabel("Force [norm.]")

    axes[1] = plot_relaxation_fn(axes[1], constit, app.time, color=color, linewidth=1.0)
    axes[1].set_ylim((0, 0.8))
    axes[1].set_xlabel("Time [norm.]")
    axes[1].set_ylabel("$G(t)$ [norm.]")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, ncols=2, loc="upper right")
for ax in axes:
    ax.grid(ls="--", color="lightgray")

# fig.suptitle("PAAM hydrogel curve fit results: Entire curve")
fig.suptitle("Hela cell(mitotic) curve fit results: Entire")

# %%
bics = jnp.asarray([results_best[n].bic for n in names])
colors = color_palette[color_inds]
fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
ax.grid(ls="--", color="darkgray")
ax.bar(names, bics, color=colors)
ax.set_yscale("symlog")
ax.set_ylabel("BIC")
ax.set_title("Hela cell(mitotic), Entire")


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
names = ("Hertzian", "MPLR", "SLS", "KWW")  # , "FKV", "GM")
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

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
names = ("Hertzian", "MPLR", "SLS", "KWW", "FKV", "GM")
for name in names:
    ax.plot(relative_errors[name], ".-", label=name, linewidth=1.0, markersize=8.0)
ax.set_yscale("log", base=10)
ax.set_xticks([0, 1, 2, 3])
ax.legend()
ax.set_ylabel("Relative error")
ax.set_xlabel("Model free parameters")
xticklabels = ax.get_xticks().tolist()
# xticklabels[0] = "0 ($E_0$)"
ax.set_xticklabels(xticklabels)
ax.grid(ls="--", color="lightgray")
# %%
for n, r in results_best.items():
    print(n, r.params.valuesdict())


# %%
def residual(constit):
    f_app_pred = _force_approach(app.time, constit, app_interp, tip)
    if fit_type == "approach":
        return f_app - f_app_pred
    else:
        f_ret_pred = _force_retract(ret.time, constit, (app_interp, ret_interp), tip)
        f_pred = jnp.concatenate((f_app_pred, f_ret_pred), axis=0)
        f = jnp.concatenate((f_app, f_ret), axis=0)
        return f - f_pred


def sensitivity_matrix(constit):
    log_constit = jtu.tree_map(jnp.log, constit)

    def residual_from_logconstit(log_constit):
        constit = jtu.tree_map(jnp.exp, log_constit)
        return residual(constit)

    D_res = jax.jacfwd(residual_from_logconstit)(log_constit)
    D_res_array, _ = jtu.tree_flatten(D_res)
    D_res_array = jnp.stack(D_res_array, axis=-1)

    S = jax.vmap(jnp.outer)(D_res_array, D_res_array)
    return jnp.sum(S, axis=0)

#%%
S_matrix = sensitivity_matrix(fits_best["KWW"])
eigval, eigvec = jnp.linalg.eigh(S_matrix)
# %%
print(eigval)
print(eigvec)


# %%
def plot_eigval_spectrum(ax, eigvals, bar_offset=0.0, bar_length=1.0, **hlines_kwargs):
    eigvals = jnp.clip(jnp.asarray(eigvals), 1e-50)
    ax.hlines(jnp.log10(eigvals), bar_offset, bar_offset + bar_length, **hlines_kwargs)
    return ax


fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
names = ["Hertzian", "SLS", "MPLR", "KWW", "FKV", "GM"]
color_inds = [0, 3, 6, 8, 5, 1]
bar_offset = 0.0
bar_length = 1.0
x_tick_positions = []
for n, c_ind in zip(names, color_inds):
    constit = fits_best[n]
    color = color_palette[c_ind]

    S_matrix = sensitivity_matrix(constit)
    eigval, eigvec = jnp.linalg.eigh(S_matrix)
    eigval = eigval / jnp.max(eigval)
    print(eigval)
    x_tick_positions.append(bar_offset + 0.5 * bar_length)
    plot_eigval_spectrum(ax, eigval, bar_offset, bar_length, color=color)

    bar_offset += 2 * bar_length

ax.set_xticks(x_tick_positions, names)
ax.set_ylabel("log (eigenvalues)")
ax.grid(ls="--", color="lightgray")
#%%
## LatinHyperCube Sampling demonstration
#%%
## Hertz model
params_htz= ['E_0']
fig, ax = plt.subplots(1, len(htz_initvals[1]), figsize=(10, 5))
for i in jnp.arange(len(htz_initvals[1])):
    ax.set_ylabel("Frequency")
    ax.set_xlabel(f"${params_htz[i]}$")
    if sampler.sample_scale[i] is True:
        ax.hist(np.log10(htz_initvals[:, i]), color=color_palette[5])
        ax.grid(ls="--", color="darkgray")
    else:
        ax.hist(htz_initvals[:, i], color=color_palette[5])
        ax.set_xscale("log")
        ax.grid(ls="--", color="darkgray")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].plot(app.time, f_app, ".", color=color_palette[5], label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color=color_palette[5], alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(htz_fits, tqdm(htz_results))):
    f_fit_app = _force_approach(app.time, constit_fit, app_interp, tip)
    f_fit_ret = _force_retract(ret.time, constit_fit, (app_interp, ret_interp), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")

    axes[0].grid(ls="--", color="darkgray")
    axes[0].set_xlabel("Time[norm.]")
    axes[0].set_ylabel("Force[norm.]")
    
    axes[1].grid(ls="--", color="darkgray")
    axes[1].set_xlabel("Time[norm.]")
    axes[1].set_ylabel("$G(t)$[norm.]")

fig.suptitle(f"Hertz model curve fit results(LHS=$5^{len(params_htz)}$) : Entire", position=(0.5,1.0+0.01)) 
#%%
## SLS model
params_sls= ['E_1', 'E_{\infty}', '\\tau']
fig, axes = plt.subplots(1, len(sls_initvals[1]), figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.set_xlabel(f"${params_sls[i]}$")
    if sampler.sample_scale[i] is True:
        ax.hist(np.log10(sls_initvals[:, i]), color=color_palette[5])
        ax.grid(ls="--", color="darkgray")
    else:
        ax.hist(sls_initvals[:, i], color=color_palette[5])
        ax.set_xscale("log")
        ax.grid(ls="--", color="darkgray")
axes[0].set_ylabel("Frequency")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].plot(app.time, f_app, ".", color=color_palette[5], label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color=color_palette[5], alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(sls_fits, tqdm(sls_results))):
    f_fit_app = _force_approach(app.time, constit_fit, app_interp, tip)
    f_fit_ret = _force_retract(ret.time, constit_fit, (app_interp, ret_interp), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray") 
    
    axes[0].grid(ls="--", color="darkgray")
    axes[0].set_xlabel("Time[norm.]")
    axes[0].set_ylabel("Force[norm.]")
    
    axes[1].grid(ls="--", color="darkgray")
    axes[1].set_xlabel("Time[norm.]")
    axes[1].set_ylabel("$G(t)$[norm.]") 

fig.suptitle(f"SLS model curve fit results(LHS=$5^{len(params_sls)}$) : Entire", position=(0.5,1.0+0.01)) 
#%%
## MPLR model
params_mplr= ['E_0', '\\alpha', 't_0']
fig, axes = plt.subplots(1, len(mplr_initvals[1]), figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.set_xlabel(f"${params_mplr[i]}$")
    if sampler.sample_scale[i] is True:
        ax.hist(np.log10(mplr_initvals[:, i]), color=color_palette[5])
        ax.grid(ls="--", color="darkgray")
    else:
        ax.hist(mplr_initvals[:, i], color=color_palette[5])
        ax.grid(ls="--", color="darkgray")

axes[0].set_ylabel("Frequency")
axes[0].set_xscale("log")
axes[2].set_xscale("log")


fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].plot(app.time, f_app, ".", color=color_palette[5], label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color=color_palette[5], alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(mplr_fits, tqdm(mplr_results))):
    f_fit_app = _force_approach(app.time, constit_fit, app_interp, tip)
    f_fit_ret = _force_retract(ret.time, constit_fit, (app_interp, ret_interp), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")    

    axes[0].grid(ls="--", color="darkgray")
    axes[0].set_xlabel("Time[norm.]")
    axes[0].set_ylabel("Force[norm.]")
    
    axes[1].grid(ls="--", color="darkgray")
    axes[1].set_xlabel("Time[norm.]")
    axes[1].set_ylabel("$G(t)$[norm.]")

fig.suptitle(f"MPLR model curve fit results(LHS=$5^{len(params_mplr)}$) : Entire", position=(0.5,1.0+0.01)) 
#%%
## KWW model
params_kww = ['E_1','E_{\infty}','\\tau', '\\beta']
fig, axes = plt.subplots(1, len(kww_initvals[1]), figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.set_xlabel(f"${params_kww[i]}$")
    if sampler.sample_scale[i] is True:
        ax.hist(np.log10(kww_initvals[:, i], ), color=color_palette[5])
        ax.grid(ls="--", color="darkgray")
    else:
        ax.hist(kww_initvals[:, i], color=color_palette[5])
        ax.grid(ls="--", color="darkgray")

axes[0].set_ylabel("Frequency")
axes[0].set_xscale("log")
axes[1].set_xscale("log")
axes[2].set_xscale("log")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].plot(app.time, f_app, ".", color=color_palette[5], label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color=color_palette[5], alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(kww_fits, tqdm(kww_results))):
    f_fit_app = _force_approach(app.time, constit_fit, app_interp, tip)
    f_fit_ret = _force_retract(ret.time, constit_fit, (app_interp, ret_interp), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")

    axes[0].grid(ls="--", color="darkgray")
    axes[0].set_xlabel("Time[norm.]")
    axes[0].set_ylabel("Force[norm.]")
    
    axes[1].grid(ls="--", color="darkgray")
    axes[1].set_xlabel("Time[norm.]")
    axes[1].set_ylabel("$G(t)$[norm.]")
    
fig.suptitle(f"KWW model curve fit results(LHS=$5^{len(params_kww)}$) : Entire", position=(0.5,1.0+0.01)) 

#%%
## FKV model
params_fkv= ['E_1','E_{\infty}', '\\alpha']
fig, axes = plt.subplots(1, len(fkv_initvals[1]), figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.set_xlabel(f"${params_fkv[i]}$")
    if sampler.sample_scale[i] is True:
        ax.hist(np.log10(fkv_initvals[:, i]), color=color_palette[5])
        ax.grid(ls="--", color="darkgray")
    else:
        ax.hist(fkv_initvals[:, i], color=color_palette[5])
        ax.grid(ls="--", color="darkgray")

axes[0].set_ylabel("Frequency")
axes[0].set_xscale("log")
axes[1].set_xscale("log")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].plot(app.time, f_app, ".", color=color_palette[5], label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color=color_palette[5], alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(fkv_fits, tqdm(fkv_results))):
    f_fit_app = _force_approach(app.time, constit_fit, app_interp, tip)
    f_fit_ret = _force_retract(ret.time, constit_fit, (app_interp, ret_interp), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")

    axes[0].grid(ls="--", color="darkgray")
    axes[0].set_xlabel("Time[norm.]")
    axes[0].set_ylabel("Force[norm.]")
    
    axes[1].grid(ls="--", color="darkgray")
    axes[1].set_xlabel("Time[norm.]")
    axes[1].set_ylabel("$G(t)$[norm.]")

fig.suptitle(f"FKV model curve fit results(LHS=$5^{len(params_fkv)}$) : Entire", position=(0.5,1.0+0.01)) 
#%%
## GM model
params_gm= ['E_1','E_2','E_{\infty}','\\tau_1','\\tau_2']
fig, axes = plt.subplots(1, len(gm_initvals[1]), figsize=(12, 6))
for i, ax in enumerate(axes):
    ax.set_xlabel(f"${params_gm[i]}$")
    if sampler.sample_scale[i] is True:
        ax.hist(np.log10(gm_initvals[:, i]), color=color_palette[5])
        ax.grid(ls="--", color="darkgray")
    else:
        ax.hist(gm_initvals[:, i], color=color_palette[5])
        ax.grid(ls="--", color="darkgray")
        ax.set_xscale("log")
        
axes[0].set_ylabel("Frequency")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].plot(app.time, f_app, ".", color=color_palette[5], label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color=color_palette[5], alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(gm_fits, tqdm(gm_results))):
    if constit_fit is None:
        continue
    f_fit_app = _force_approach(app.time, constit_fit, app_interp, tip)
    f_fit_ret = _force_retract(ret.time, constit_fit, (app_interp, ret_interp), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.5)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.5)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")
    
    axes[0].grid(ls="--", color="darkgray")
    axes[0].set_xlabel("Time[norm.]")
    axes[0].set_ylabel("Force[norm.]")
    
    axes[1].grid(ls="--", color="darkgray")
    axes[1].set_xlabel("Time[norm.]")
    axes[1].set_ylabel("$G(t)$[norm.]")

fig.suptitle(f"GM model curve fit results(LHS=$5^{len(params_gm)}$) : Entire", position=(0.5,1.0+0.01))     
#%%

# %%
# check BIC values, Eigen-params and values
import os
import pandas as pd

f_path = './'
f_name = "Slopiness_Entire"


#%%
df = pd.DataFrame()
for model in names:
    S_matrix = sensitivity_matrix(fits_best[f"{model}"])
    eigval, eigvec = jnp.linalg.eigh(S_matrix)

    new = pd.DataFrame({f'{model} eigen value': eigval,
                        f'{model} eigen vector': eigvec[np.argmax(eigval)]})
    df = pd.concat([df, new], axis=1)

df.to_csv(os.path.join(f_path, f_name), index=False)
# %%
data = pd.read_csv("./Slopiness_Entire")
data
# %%
bics
# %%
S_matrix = sensitivity_matrix(fits_best["GM"])
eigval, eigvec = jnp.linalg.eigh(S_matrix)
# %%
print(eigval)
print(eigvec)
# %%
