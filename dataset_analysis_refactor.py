# %%
# ruff: noqa: F722
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Sequence

import equinox as eqx
import jax
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


# %%
## Fit using Latin hypercube sampling
N_SAMPLES = 5
fit_type = "both"  # "approach"
### SLS model
constit_sls = StandardLinearSolid(10.0, 10.0, 10.0)
bounds_sls = [(1e-7, 1e7)] * 3

sampler = LatinHypercubeSampler(
    sample_range=[(1e-5, 1e5), (1e-5, 1e5), (1e-5, 1e5)],
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
bounds_mplr = [(1e-7, 1e7), (0.0, 1.0), (1e-7, 1e7)]

sampler = LatinHypercubeSampler(
    sample_range=[(1e-5, 1e5), (1e-10, 1.0), (1e-5, 1e5)],
    sample_scale=["log", "log", "log"],
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
bounds_kww = [(1e-7, 1e7)] * 3 + [(0.0, 1.0)]

sampler = LatinHypercubeSampler(
    sample_range=[
        (1e-5, 1e5),
        (1e-5, 1e5),
        (1e-7, 1e7),
        (1e-10, 1e1),
    ],
    sample_scale=["log", "log", "log", "log"],
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
### Hertzian model

constit_htz = Hertzian(10.0)
bounds_htz = [(1e-7, 1e7)]

sampler = LatinHypercubeSampler(
    sample_range=[(1e-5, 1e5)],
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


def get_best_model(results: list[lmfit.minimizer.MinimizerResult]):
    bic = np.array([res.bic for res in results])
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


fig, ax = plt.subplots(1, 1, figsize=(5, 3))
names = ("Hertzian", "MPLR", "SLS", "KWW")
for name in names:
    ax.plot(relative_errors[name], ".-", label=name)
ax.set_yscale("log", base=10)
ax.set_xticks([0, 1, 2, 3])
ax.legend()
ax.set_ylabel("Relative error")
ax.set_xlabel("Fit parameter")
ax.grid(ls="--", color="lightgray")
# %%
# BIC > 0 truncation

MinimizerResultArray = Sequence[MinimizerResult]


def is_valid_fit(
    results: MinimizerResultArray,
    criteria: Callable[[MinimizerResult, MinimizerResultArray], bool],
) -> Bool[np.ndarray, " num_valid"]:
    return np.array([criteria(res) for res in results])


def close_to_min_bic(result: MinimizerResult, results: MinimizerResultArray) -> bool:
    bic_min = np.min()


sls_bic = np.array([sls_results[i].bic for i in np.arange(num)])
mplr_bic = np.array([mplr_results[i].bic for i in np.arange(num)])
kww_bic = np.array([kww_results[i].bic for i in np.arange(num)])
htz_bic = np.array([htz_results[i].bic for i in np.arange(num)])

threshold = 2000


def valid_inds(model_bic, threshold):
    return model_bic < np.min(model_bic) + threshold


sls_fits = np.array(sls_fits)[valid_inds(sls_bic, threshold)]
mplr_fits = np.array(mplr_fits)[valid_inds(mplr_bic, threshold)]
kww_fits = np.array(kww_fits)[valid_inds(kww_bic, threshold)]
htz_fits = np.array(htz_fits)[valid_inds(htz_bic, threshold)]

sls_results = np.array(sls_results)[valid_inds(sls_bic, threshold)]
mplr_results = np.array(mplr_results)[valid_inds(mplr_bic, threshold)]
kww_results = np.array(kww_results)[valid_inds(kww_bic, threshold)]
htz_results = np.array(htz_results)[valid_inds(htz_bic, threshold)]

print(f"SLS = {np.sum(valid_inds(sls_bic, threshold)==False)} truncation")
print(f"MPLR = {np.sum(valid_inds(mplr_bic, threshold)==False)} truncation")
print(f"KWW = {np.sum(valid_inds(kww_bic, threshold)==False)} truncation")
print(f"HTZ = {np.sum(valid_inds(htz_bic, threshold)==False)} truncation")
# %%
## SLS model
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, f_app, ".", color="royalblue", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="royalblue", alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(sls_fits, tqdm(sls_results))):
    f_fit_app = force_approach(constit_fit, app, tip)
    f_fit_ret = force_retract(constit_fit, (app, ret), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")
# %%
## Modified PLR model
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, f_app, ".", color="royalblue", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="royalblue", alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(mplr_fits, tqdm(mplr_results))):
    f_fit_app = force_approach(constit_fit, app, tip)
    f_fit_ret = force_retract(constit_fit, (app, ret), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")
# %%
## KWW model
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, f_app, ".", color="royalblue", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="royalblue", alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(kww_fits, tqdm(kww_results))):
    f_fit_app = force_approach(constit_fit, app, tip)
    f_fit_ret = force_retract(constit_fit, (app, ret), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")
# %%
## Hertzian model
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, f_app, ".", color="royalblue", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="royalblue", alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(htz_fits, tqdm(htz_results))):
    f_fit_app = force_approach(constit_fit, app, tip)
    f_fit_ret = force_retract(constit_fit, (app, ret), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")
# %%
## For SLS
sls_samples_fit = []
for r in sls_results:
    sls_samples_fit.append(list(r.params.valuesdict().values()))
sls_samples_fit = np.asarray(sls_samples_fit)
# %%
fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):
    if sls_is_logscale[i] is True:
        s_ = sls_samples_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(sls_samples_fit[:, i])
# %%
sls_results[0].params.valuesdict()
# %%
## For MPLR
mplr_samples_fit = []
for r in mplr_results:
    mplr_samples_fit.append(list(r.params.valuesdict().values()))
mplr_samples_fit = np.asarray(mplr_samples_fit)
# %%
fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):
    if mplr_is_logscale[i] is True:
        s_ = mplr_samples_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(mplr_samples_fit[:, i])
# %%
mplr_results[0].params.valuesdict()
# %%
# For KWW
kww_samples_fit = []
for r in kww_results:
    kww_samples_fit.append(list(r.params.valuesdict().values()))
kww_samples_fit = np.asarray(kww_samples_fit)
# %%
fig, axes = plt.subplots(1, 4, figsize=(7, 3))
for i, ax in enumerate(axes):
    if kww_is_logscale[i] is True:
        s_ = kww_samples_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(kww_samples_fit[:, i])
# %%
kww_results[0]
# %%
## For Hertz
htz_samples_fit = []
for r in htz_results:
    htz_samples_fit.append(list(r.params.valuesdict().values()))
htz_samples_fit = np.asarray(htz_samples_fit)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 3))

i = 0
if htz_is_logscale[i] is True:
    s_ = htz_samples_fit[:, i]
    s_[s_ > 0] = np.log10(s_[s_ > 0])
    ax.hist(s_)
else:
    ax.hist(htz_samples_fit[:, i])
# %%
htz_results[0]
# %%
## Test fitting results
sls_fit = sls_fits[0]
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(sls_fit, app, tip)
f_fit_ret = force_retract(sls_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], sls_fit, app.time)

mplr_fit = mplr_fits[0]
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(mplr_fit, app, tip)
f_fit_ret = force_retract(mplr_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], mplr_fit, app.time)

kww_fit = kww_fits[0]
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(kww_fit, app, tip)
f_fit_ret = force_retract(kww_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], kww_fit, app.time)

htz_fit = htz_fits[0]
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(htz_fit, app, tip)
f_fit_ret = force_retract(htz_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], htz_fit, app.time)
# %%
## check parameter
for i in range(len(kww_results)):
    print(kww_results[i].params.valuesdict())
# %%
## Parameter space
sls_params = ["E1", "E_inf", "tau"]
mplr_params = ["E0", "t0", "alpha"]
kww_params = ["E1", "E_inf", "tau"]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

for i in np.arange(len(sls_results)):
    params = sls_params
    xs = sls_results[i].params.valuesdict()[params[0]]
    ys = sls_results[i].params.valuesdict()[params[1]]
    zs = sls_results[i].params.valuesdict()[params[2]]

    ax.scatter(xs, ys, zs)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_zscale('log')
ax.set_xlabel(params[0])
ax.set_ylabel(params[1])
ax.set_zlabel(params[2])

plt.show()
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# %%
## Fit the entire approach-retract curve for SLS
sls_tot_fit, tot_result_sls = fit_all_lmfit(
    constit_sls, bounds_sls, tip, (app, ret), (f_app, f_ret)
)
# %%
display(tot_result_sls)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(sls_fit, app, tip)
f_fit_ret = force_retract(sls_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], sls_fit, app.time)

# %%
## Fit the entire approach-retract curve for MPLR
mplr_tot_fit, tot_result_mplr = fit_all_lmfit(
    constit_mplr, bounds_mplr, tip, (app, ret), (f_app, f_ret)
)
display(result)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(mplr_fit, app, tip)
f_fit_ret = force_retract(mplr_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], mplr_fit, app.time)
# %%
## Fit the entire approach-retract curve for KWW
# range(1e-5, 1e5)
kww_tot_fit, tot_result_kww = fit_all_lmfit(
    constit_kww, bounds_kww, tip, (app, ret), (f_app, f_ret)
)
display(result)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(kww_fit, app, tip)
f_fit_ret = force_retract(kww_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], kww_fit, app.time)
# %%
## Fit the entire approach-retract curve for Hertz
htz_tot_fit, tot_result_htz = fit_all_lmfit(
    constit_htz, bounds_htz, tip, (app, ret), (f_app, f_ret)
)
display(result)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(htz_fit, app, tip)
f_fit_ret = force_retract(htz_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], htz_fit, app.time)
# %%
f_sls_fit_app = force_approach(sls_tot_fit, app, tip)
f_sls_fit_ret = force_retract(sls_tot_fit, (app, ret), tip)

f_mplr_fit_app = force_approach(mplr_tot_fit, app, tip)
f_mplr_fit_ret = force_retract(mplr_tot_fit, (app, ret), tip)

f_kww_fit_app = force_approach(kww_tot_fit, app, tip)
f_kww_fit_ret = force_retract(kww_tot_fit, (app, ret), tip)

f_htz_fit_app = force_approach(htz_tot_fit, app, tip)
f_htz_fit_ret = force_retract(htz_tot_fit, (app, ret), tip)
# %%
## Graph of 4 different viscoelastic model for approach params
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="black", alpha=0.5)

# Graph for parameters of SLS model fitting (approach portion)
axes[0].plot(app.time, f_sls_fit_app, color="red", alpha=0.7, label="SLS")
axes[0].plot(ret.time, f_sls_fit_ret, color="red", alpha=0.7)
# Graph for parameters of Modified PLR model fitting (approach portion)
axes[0].plot(app.time, f_mplr_fit_app, color="green", alpha=0.7, label="Modified PLR")
axes[0].plot(ret.time, f_mplr_fit_ret, color="green", alpha=0.7)
# Graph for parameters of KWW model fitting (approach portion)
axes[0].plot(app.time, f_kww_fit_app, color="blue", alpha=0.7, label="KWW")
axes[0].plot(ret.time, f_kww_fit_ret, color="blue", alpha=0.7)

axes[0].plot(app.time, f_htz_fit_app, color="yellow", alpha=0.7, label="Hertzian")
axes[0].plot(ret.time, f_htz_fit_ret, color="yellow", alpha=0.7)

axes[0].legend(loc="upper right")

axes[1] = plot_relaxation_fn(axes[1], sls_fit, app.time, color="red", label="SLS")
axes[1] = plot_relaxation_fn(
    axes[1], mplr_fit, app.time, color="blue", label="Modified PLR"
)
axes[1] = plot_relaxation_fn(axes[1], kww_fit, app.time, color="green", label="KWW")
axes[1] = plot_relaxation_fn(
    axes[1], htz_fit, app.time, color="yellow", label="Hertzian"
)

axes[1].legend(loc="upper right")
# %%
## Latinhypercube sampling

sls_tot_fits, sls_tot_results = [], []
mplr_tot_fits, mplr_tot_results = [], []
kww_tot_fits, kww_tot_results = [], []
htz_tot_fits, htz_tot_results = [], []

for i in tqdm(range(samples_sls.shape[0])):
    sample = samples_sls[i]
    constit_i = type(constit_sls)(*sample)
    constit_fit, result = fit_all_lmfit(
        constit_i, bounds_sls, tip, (app, ret), (f_app, f_ret)
    )
    sls_tot_fits.append(constit_fit)
    sls_tot_results.append(result)

for i in tqdm(range(samples_mplr.shape[0])):
    sample = samples_mplr[i]
    constit_i = type(constit_mplr)(*sample)
    constit_fit, result = fit_all_lmfit(
        constit_i, bounds_mplr, tip, (app, ret), (f_app, f_ret)
    )
    mplr_tot_fits.append(constit_fit)
    mplr_tot_results.append(result)

for i in tqdm(range(samples_kww.shape[0])):
    sample = samples_kww[i]
    constit_i = type(constit_kww)(*sample)
    constit_fit, result = fit_all_lmfit(
        constit_i, bounds_kww, tip, (app, ret), (f_app, f_ret)
    )
    kww_tot_fits.append(constit_fit)
    kww_tot_results.append(result)

for i in tqdm(range(samples_htz.shape[0])):
    sample = samples_htz[i]
    constit_i = type(constit_htz)(*sample)
    constit_fit, result = fit_all_lmfit(
        constit_i, bounds_htz, tip, (app, ret), (f_app, f_ret)
    )
    htz_tot_fits.append(constit_fit)
    htz_tot_results.append(result)
# %%
## BIC > 0 truncation

sls_tot_bic = np.array([sls_tot_results[i].bic for i in np.arange(num)])
mplr_tot_bic = np.array([mplr_tot_results[i].bic for i in np.arange(num)])
kww_tot_bic = np.array([kww_tot_results[i].bic for i in np.arange(num)])
htz_tot_bic = np.array([htz_tot_results[i].bic for i in np.arange(num)])

sls_tot_fits = np.array(sls_tot_fits)[valid_inds(sls_tot_bic, threshold)]
mplr_tot_fits = np.array(mplr_tot_fits)[valid_inds(mplr_tot_bic, threshold)]
kww_tot_fits = np.array(kww_tot_fits)[valid_inds(kww_tot_bic, threshold)]
htz_tot_fits = np.array(htz_tot_fits)[valid_inds(htz_tot_bic, threshold)]

sls_tot_results = np.array(sls_tot_results)[valid_inds(sls_tot_bic, threshold)]
mplr_tot_results = np.array(mplr_tot_results)[valid_inds(mplr_tot_bic, threshold)]
kww_tot_results = np.array(kww_tot_results)[valid_inds(kww_tot_bic, threshold)]
htz_tot_results = np.array(htz_tot_results)[valid_inds(htz_tot_bic, threshold)]

print(f"SLS = {np.sum(valid_inds(sls_tot_bic, threshold)==False)} truncation")
print(f"MPLR = {np.sum(valid_inds(mplr_tot_bic, threshold)==False)} truncation")
print(f"KWW = {np.sum(valid_inds(kww_tot_bic, threshold)==False)} truncation")
print(f"HTZ = {np.sum(valid_inds(htz_tot_bic, threshold)==False)} truncation")

# %%
## SLS model
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, f_app, ".", color="royalblue", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="royalblue", alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(sls_tot_fits, tqdm(sls_results))):
    f_fit_app = force_approach(constit_fit, app, tip)
    f_fit_ret = force_retract(constit_fit, (app, ret), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")
# %%
## Modified PLR model
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, f_app, ".", color="royalblue", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="royalblue", alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(mplr_tot_fits, tqdm(mplr_tot_results))):
    f_fit_app = force_approach(constit_fit, app, tip)
    f_fit_ret = force_retract(constit_fit, (app, ret), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")
# %%
## KWW model
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, f_app, ".", color="royalblue", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="royalblue", alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(kww_tot_fits, tqdm(kww_tot_results))):
    f_fit_app = force_approach(constit_fit, app, tip)
    f_fit_ret = force_retract(constit_fit, (app, ret), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")
# %%
## Hertz model
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, f_app, ".", color="royalblue", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="royalblue", alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(htz_tot_fits, tqdm(htz_results))):
    f_fit_app = force_approach(constit_fit, app, tip)
    f_fit_ret = force_retract(constit_fit, (app, ret), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")
# %%
sls_samples_tot_fit = []
for r in sls_tot_results:
    sls_samples_tot_fit.append(list(r.params.valuesdict().values()))
sls_samples_tot_fit = np.asarray(sls_samples_tot_fit)
# %%
fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):
    if sls_is_logscale[i] is True:
        s_ = sls_samples_tot_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(sls_samples_tot_fit[:, i])
# %%
sls_tot_results[0].params.valuesdict()
# %%
mplr_samples_tot_fit = []
for r in mplr_tot_results:
    mplr_samples_tot_fit.append(list(r.params.valuesdict().values()))
mplr_samples_tot_fit = np.asarray(mplr_samples_tot_fit)
# %%
fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):
    if mplr_is_logscale[i] is True:
        s_ = mplr_samples_tot_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(mplr_samples_tot_fit[:, i])
# %%
mplr_tot_results[0].params.valuesdict()
# %%
kww_samples_tot_fit = []
for r in kww_tot_results:
    kww_samples_tot_fit.append(list(r.params.valuesdict().values()))
kww_samples_tot_fit = np.asarray(kww_samples_tot_fit)
# %%
fig, axes = plt.subplots(1, 4, figsize=(7, 3))
for i, ax in enumerate(axes):
    if kww_is_logscale[i] is True:
        s_ = kww_samples_tot_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(kww_samples_tot_fit[:, i])
# %%
kww_tot_results[0].params.valuesdict()
# %%
htz_samples_tot_fit = []
for r in htz_tot_results:
    htz_samples_tot_fit.append(list(r.params.valuesdict().values()))
htz_samples_tot_fit = np.asarray(htz_samples_tot_fit)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 3))
i = 0
if htz_is_logscale[i] is True:
    s_ = htz_samples_tot_fit[:, i]
    s_[s_ > 0] = np.log10(s_[s_ > 0])
    ax.hist(s_)
else:
    ax.hist(htz_samples_tot_fit[:, i])
# %%
htz_tot_results[0].params.valuesdict()
# %%
## Test fitting results
sls_tot_fit = sls_tot_fits[0]
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(sls_tot_fit, app, tip)
f_fit_ret = force_retract(sls_tot_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], sls_tot_fit, app.time)

mplr_tot_fit = mplr_tot_fits[0]
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(mplr_tot_fit, app, tip)
f_fit_ret = force_retract(mplr_tot_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], mplr_tot_fit, app.time)

kww_tot_fit = kww_tot_fits[0]
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(kww_tot_fit, app, tip)
f_fit_ret = force_retract(kww_tot_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], kww_tot_fit, app.time)

htz_tot_fit = htz_tot_fits[0]
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
f_fit_app = force_approach(htz_tot_fit, app, tip)
f_fit_ret = force_retract(htz_tot_fit, (app, ret), tip)
with mpl.rc_context({"lines.markersize": 1.0, "lines.linewidth": 2.0}):
    axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(ret.time, f_ret, ".", color="black", label="Data", alpha=0.5)
    axes[0].plot(app.time, f_fit_app, label="Curve fit", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, label="Curve fit", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], htz_tot_fit, app.time)
# %%
sls_tot_results[0].params.valuesdict()
# %%
## Parameter space
sls_params = ["E1", "E_inf", "tau"]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(projection="3d")

for i in np.arange(len(sls_results)):
    params = sls_params
    xs = sls_results[i].params.valuesdict()[params[0]]
    ys = sls_results[i].params.valuesdict()[params[1]]
    zs = sls_results[i].params.valuesdict()[params[2]]

    ax.scatter(xs, ys, zs, marker="^", alpha=0.7)

for i in np.arange(len(sls_tot_results)):
    xs_tot = sls_tot_results[i].params.valuesdict()[params[0]]
    ys_tot = sls_tot_results[i].params.valuesdict()[params[1]]
    zs_tot = sls_tot_results[i].params.valuesdict()[params[2]]

    ax.scatter(xs_tot, ys_tot, zs_tot, marker="*", alpha=0.7)

ax.set_xlabel(params[0])
ax.set_ylabel(params[1])
ax.set_zlabel(params[2])

plt.show()
# %%
## Bar graph for BIC
sls_bic = np.min(sls_bic)
mplr_bic = np.min(mplr_bic)
kww_bic = np.min(kww_bic)
htz_bic = np.min(htz_bic)

sls_tot_bic = np.min(sls_tot_bic)
mplr_tot_bic = np.min(mplr_tot_bic)
kww_tot_bic = np.min(kww_tot_bic)
htz_tot_bic = np.min(htz_tot_bic)
# %%
# fig, ax = plt.subplots(1, 1, figsize=(10,5))

# x=np.arange(2)

# ax.bar(x[0], height=sls_bic, width=0.25, label = 'SLS')
# ax.bar(x[0]+0.25, height=mplr_bic, width=0.25, label='MPLR')
# ax.bar(x[0]+0.5, height=kww_bic, width=0.25, label='KWW')
# ax.bar(x[0]+0.75, height=htz_bic, width=0.25, label='Hertz')

# ax.bar(x[1]+0.25, height=sls_tot_bic, width=0.25, label = 'SLS')
# ax.bar(x[1]+0.5, height=mplr_tot_bic, width=0.25, label='MPLR')
# ax.bar(x[1]+0.75, height=kww_tot_bic, width=0.25, label='KWW')
# ax.bar(x[1]+1.0, height=htz_tot_bic, width=0.25, label='Hertz')
# ax.set_ylabel("BIC")

# ax.set_xticks(x[0] + 0.25, 'Approach')
# ax.set_xticks(x[1] + 0.25, 'Total')
# ax.legend()
# %%
display(sls_tot_results[0])
# %%
sls_tot_results[0].aborted
# %%
print(f"operation time : {time.time()-t_start}")
# %%
section = ("Approach", "Total")
model_bic = {
    "SLS": (sls_bic, sls_tot_bic),
    "MPLR": (mplr_bic, mplr_tot_bic),
    "KWW": (kww_bic, kww_tot_bic),
    "Hertz": (htz_bic, htz_tot_bic),
}

x = np.arange(len(section))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout="constrained")

for attribute, measurement in model_bic.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("BIC")
ax.set_title("4 different model by section")
ax.set_xticks(x + width * 1.5, section)
ax.legend(loc="lower left", ncols=2)

plt.show()
# %%
