# %%
# ruff: noqa: F722
from tqdm import tqdm
from pathlib import Path
from IPython.display import display
from scipy.stats import qmc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import equinox as eqx

from neuralconstitutive.plotting import plot_indentation, plot_relaxation_fn
from neuralconstitutive.constitutive import (
    StandardLinearSolid,
    ModifiedPowerLaw,
    KohlrauschWilliamsWatts,
    Hertzian,
    FractionalKelvinVoigt,
    GeneralizedMaxwellmodel
)
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.ting import force_approach, force_retract
from neuralconstitutive.io import import_data
from neuralconstitutive.fitting import fit_approach_lmfit, fit_all_lmfit
from neuralconstitutive.utils import (
    smooth_data,
    normalize_indentations,
    normalize_forces,
)
import time

t_start = time.time()
#%%
import os
print(os.getcwd())

# %%
jax.config.update("jax_enable_x64", True)

datadir = Path("open_data/PAAM hydrogel/speed 5")
name = "PAA_speed 5_4nN"
(app, ret), (f_app, f_ret) = import_data(
    datadir / f"{name}.tab", datadir / f"{name}.tsv"
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(app.depth, f_app, ".", color="k", markersize=1)
ax.plot(ret.depth, f_ret, ".", color="k", markersize=1)
ax.set_xlabel("Indentation (m)")
ax.set_ylabel("Force (N)")
ax.grid(ls="--", color="lightgray")
ax.set_title("HeLa, interphase")

# %%
datadir = Path("open_data/PAAM hydrogel/speed 5")
name = "PAA_speed 5_4nN"
(app, ret), (f_app, f_ret) = import_data(
    datadir / f"{name}.tab", datadir / f"{name}.tsv"
)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(app.depth, f_app, ".", color="k", markersize=1)
ax.plot(ret.depth, f_ret, ".", color="k", markersize=1)
ax.set_xlabel("Indentation (m)")
ax.set_ylabel("Force (N)")
ax.grid(ls="--", color="lightgray")
ax.set_title("pAAm hydrogel")
# %%
f_ret = jnp.clip(f_ret, 0.0)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(app.depth, f_app, ".", color="k", markersize=1)
ax.plot(ret.depth, f_ret, ".", color="k", markersize=1)
ax.set_xlabel("Indentation (m)")
ax.set_ylabel("Force (N)")
ax.grid(ls="--", color="lightgray")
ax.set_title("Adhesive region clipped")
# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(app.time, f_app, ".", color="k", markersize=1)
ax.plot(ret.time, f_ret, ".", color="k", markersize=1)
ax.set_xlabel("Time")
ax.set_ylabel("Force (N)")

ax2 = ax.twinx()
ax2.plot(app.time, app.depth, "--", color="b", linewidth=1)
ax2.plot(ret.time, ret.depth, "--", color="b", linewidth=1)
ax2.set_ylabel("Indentation (m)", color="blue", alpha=0.7)
ax2.tick_params(axis="y", colors="blue")
ax2.spines["right"].set_color("blue")
ax.grid(ls="--", color="lightgray")
ax.set_title("Time domain representation")
# %%
(f_app, f_ret), _ = normalize_forces(f_app, f_ret)
(app, ret), (_, h_m) = normalize_indentations(app, ret)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(app.time, f_app, ".", color="k", markersize=1)
ax.plot(ret.time, f_ret, ".", color="k", markersize=1)
ax.set_xlabel("Time")
ax.set_ylabel("Force (N)")

ax2 = ax.twinx()
ax2.plot(app.time, app.depth, "--", color="b", linewidth=1)
ax2.plot(ret.time, ret.depth, "--", color="b", linewidth=1)
ax2.set_ylabel("Indentation (m)", color="blue", alpha=0.7)
ax2.tick_params(axis="y", colors="blue")
ax2.spines["right"].set_color("blue")
ax.grid(ls="--", color="lightgray")
ax.set_title("Time domain, normalized")
# %%
datadir = Path("open_data/Interphase rep 2")
filenames = [
    "interphase_speed 2_2nN",
    "interphase_speed 2_2nN-1",
    "interphase_speed 2_2nN-2",
    "interphase_speed 2_2nN-3",
]

data_list = [
    import_data(datadir / f"{name}.tab", datadir / f"{name}.tsv") for name in filenames
]

fig, axes = plt.subplots(1, 2, figsize=(5, 3))
for (app, ret), (f_app, f_ret) in data_list:
    axes[0].plot(app.time, app.depth)
    axes[0].plot(ret.time, ret.depth)
    axes[1].plot(app.time, f_app, ".")
    axes[1].plot(ret.time, f_ret, ".")
# datadir = Path("open_data/Mitotic Rep 1/Adhesion(X)")
# (app, ret), (f_app, f_ret) = import_data(
#     datadir / "mitotic_speed 2_2nN-3.tab", datadir / "mitotic_speed 2_2nN-3.tsv"
# )
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
tip.a()
# %%
## Fit only the approach portion

dt = app.time[1] - app.time[0]

constit_sls = StandardLinearSolid(10.0, 10.0, 10.0)
bounds_sls = [(1e-7, 1e7)] * 3

constit_mplr = ModifiedPowerLaw(10.0, 10.0, 10.0)
bounds_mplr = [(1e-7, 1e7), (0.0, 1.0), (1e-7, 1e7)]

constit_kww = KohlrauschWilliamsWatts(10.0, 10.0, 10.0, 0.0)
bounds_kww = [(1e-7, 1e7)] * 3 + [(0.0, 1.0)]

constit_htz = Hertzian(10.0)
bounds_htz = [(1e-7, 1e7)]

constit_fkv = FractionalKelvinVoigt(1.0, 1.0, 1.0)
bounds_fkv = [(1e-7, 1e7)] * 2 + [(0.0, 0.7)] * 1

constit_gm = GeneralizedMaxwellmodel(1.0, 1.0, 1.0, 1.0, 1.0)
bounds_gm = [(1e-7, 1e7)] * 5
# %%
# %%timeit
sls_fit, result_sls, min_sls = fit_approach_lmfit(constit_sls, bounds_sls, tip, app, f_app)
mplr_fit, result_mplr, min_mplr = fit_approach_lmfit(constit_mplr, bounds_mplr, tip, app, f_app)
kww_fit, result_kww, min_mplr = fit_approach_lmfit(constit_kww, bounds_kww, tip, app, f_app)
htz_fit, result_htz, min_htz = fit_approach_lmfit(constit_htz, bounds_htz, tip, app, f_app)
fkv_fit, result_fkv, min_fkv = fit_approach_lmfit(constit_fkv, bounds_fkv, tip, app, f_app)
gm_fit, result_gm, min_gm = fit_approach_lmfit(constit_gm, bounds_gm, tip, app, f_app)
# %%
display(result_sls)
display(result_mplr)
display(result_kww)
display(result_htz)
display(result_fkv)
display(result_gm)
# %%
## Calculation of approach & retraction curve for approach params
f_sls_fit_app = force_approach(sls_fit, app, tip)
f_sls_fit_ret = force_retract(sls_fit, (app, ret), tip)

f_mplr_fit_app = force_approach(mplr_fit, app, tip)
f_mplr_fit_ret = force_retract(mplr_fit, (app, ret), tip)

f_kww_fit_app = force_approach(kww_fit, app, tip)
f_kww_fit_ret = force_retract(kww_fit, (app, ret), tip)

f_htz_fit_app = force_approach(htz_fit, app, tip)
f_htz_fit_ret = force_retract(htz_fit, (app, ret), tip)

f_fkv_fit_app = force_approach(fkv_fit, app, tip)
f_fkv_fit_ret = force_retract(fkv_fit, (app, ret), tip)
#%%
import equinox as eqx
from neuralconstitutive.ting import _force_approach, _force_retract
from neuralconstitutive.fitting import create_subsampled_interpolants

app_interp = create_subsampled_interpolants(app)
ret_interp = create_subsampled_interpolants(ret)

f_sls_fit_app = _force_approach(app.time, sls_fit, app_interp, tip)
f_sls_fit_ret = _force_retract(ret.time, sls_fit, (app_interp, ret_interp), tip)

f_mplr_fit_app = _force_approach(app.time, mplr_fit, app_interp, tip)
f_mplr_fit_ret = _force_retract(ret.time, mplr_fit, (app_interp, ret_interp), tip)

f_kww_fit_app = _force_approach(app.time, kww_fit, app_interp, tip)
f_kww_fit_ret = _force_retract(ret.time, kww_fit, (app_interp, ret_interp), tip)

f_htz_fit_app = _force_approach(app.time, htz_fit, app_interp, tip)
f_htz_fit_ret = _force_retract(ret.time, htz_fit, (app_interp, ret_interp), tip)

f_fkv_fit_app = _force_approach(app.time, fkv_fit, app_interp, tip)
f_fkv_fit_ret = _force_retract(ret.time, fkv_fit, (app_interp, ret_interp), tip)

f_gm_fit_app = _force_approach(app.time, gm_fit, app_interp, tip)
f_gm_fit_ret = _force_retract(ret.time, gm_fit, (app_interp, ret_interp), tip)
# %%
## Graph of 4 different viscoelastic model for approach params
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="black", alpha=0.5)

# Graph for parameters of SLS model fitting (approach portion)
axes[0].plot(app.time, f_sls_fit_app, color="#49adad", alpha=0.7, label="SLS")
axes[0].plot(ret.time, f_sls_fit_ret, color="#49adad", alpha=0.7)
# Graph for parameters of Modified PLR model fitting (approach portion)
axes[0].plot(app.time, f_mplr_fit_app, color="#cb5b42", alpha=0.7, label="Modified PLR")
axes[0].plot(ret.time, f_mplr_fit_ret, color="#cb5b42", alpha=0.7)
# Graph for parameters of KWW model fitting (approach portion)
axes[0].plot(app.time, f_kww_fit_app, color="#67a64e", alpha=0.7, label="KWW")
axes[0].plot(ret.time, f_kww_fit_ret, color="#67a64e", alpha=0.7)
#Graph for parameters of Hertz model fitting (approach portion)
axes[0].plot(app.time, f_htz_fit_app, color="#8d70c9", alpha=0.7, label="Hertzian")
axes[0].plot(ret.time, f_htz_fit_ret, color="#8d70c9", alpha=0.7)
# Graph for parameters of Fractional Kelvin-Voigt fitting (approach portion)
axes[0].plot(app.time, f_fkv_fit_app, color="#b69340", alpha=0.7, label="FKV")
axes[0].plot(ret.time, f_fkv_fit_ret, color="#b69340", alpha=0.7)
# Graph for parameters of Fractional Generalized Maxwell model fitting (approach portion)
axes[0].plot(app.time, f_gm_fit_app, color="#c8588c", alpha=0.7, label="GM")
axes[0].plot(ret.time, f_gm_fit_ret, color="#c8588c", alpha=0.7)

axes[0].legend(loc="upper right")

axes[1] = plot_relaxation_fn(axes[1], sls_fit, app.time, color="#49adad", label="SLS")
axes[1] = plot_relaxation_fn(axes[1], mplr_fit, app.time, color="#cb5b42", label="Modified PLR")
axes[1] = plot_relaxation_fn(axes[1], kww_fit, app.time, color="#67a64e", label="KWW")
axes[1] = plot_relaxation_fn(axes[1], htz_fit, app.time, color="#8d70c9", label="Hertzian")
axes[1] = plot_relaxation_fn(axes[1], fkv_fit, app.time, color="#b69340", label="FKV")
axes[1] = plot_relaxation_fn(axes[1], gm_fit, app.time, color="#c8588c", label="GM")

axes[1].legend(loc="upper right")
# %%timeit
f_sls_fit_app = eqx.filter_jit(force_approach)(sls_fit, app, tip)
f_mplr_fit_app = eqx.filter_jit(force_approach)(mplr_fit, app, tip)
f_kww_fit_app = eqx.filter_jit(force_approach)(kww_fit, app, tip)
f_htz_fit_app = eqx.filter_jit(force_approach)(htz_fit, app, tip)
f_fkv_fit_app = eqx.filter_jit(force_approach)(fkv_fit, app, tip)
f_gm_fit_app = eqx.filter_jit(force_approach)(gm_fit, app, tip)
# %%
## Latinhypercube sampling & draw histogram for parameter space(example)

num = 10
seed = 10

sampler = qmc.LatinHypercube(d=3, seed=seed)
sls_range = [(1e-5, 1e5), (1e-5, 1e5), (1e-5, 1e5)]
sls_range = np.asarray(sls_range)

sample_scale = ["log", "log", "log"]
is_logscale = [s == "log" for s in sample_scale]
sls_range[is_logscale, :] = np.log10(sls_range[is_logscale, :])

samples_sls = sampler.random(num)
samples_sls = qmc.scale(samples_sls, sls_range[:, 0], sls_range[:, 1])
samples_sls[:, is_logscale] = 10 ** samples_sls[:, is_logscale]

fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):
    if is_logscale[i] is True:
        ax.hist(np.log10(samples_sls[:, i]))
    else:
        ax.hist(samples_sls[:, i])

sls_is_logscale = is_logscale
# %%
sampler_mplr = qmc.LatinHypercube(d=3, seed=seed)
mplr_range = [(1e-5, 1e5), (1e-10, 1.0), (1e-5, 1e5)]
mplr_range = np.asarray(mplr_range)

sample_scale = ["log", "log", "log"]
is_logscale = [s == "log" for s in sample_scale]

mplr_range[is_logscale, :] = np.log10(mplr_range[is_logscale, :])

samples_mplr = sampler_mplr.random(num)
samples_mplr = qmc.scale(samples_mplr, mplr_range[:, 0], mplr_range[:, 1])
samples_mplr[:, is_logscale] = 10 ** samples_mplr[:, is_logscale]

mplr_is_logscale = is_logscale
# %%
sampler_kww = qmc.LatinHypercube(d=4, seed=seed)
kww_range = [
    (1e-5, 1e5),
    (1e-5, 1e5),
    (1e-7, 1e7),
    (1e-10, 1e1),
]
kww_range = np.asarray(kww_range)

sample_scale = ["log", "log", "log", "log"]
is_logscale = [s == "log" for s in sample_scale]

kww_range[is_logscale, :] = np.log10(kww_range[is_logscale, :])

samples_kww = sampler_kww.random(num)
samples_kww = qmc.scale(samples_kww, kww_range[:, 0], kww_range[:, 1])
samples_kww[:, is_logscale] = 10 ** samples_kww[:, is_logscale]

kww_is_logscale = is_logscale
# %%
sampler_htz = qmc.LatinHypercube(d=1, seed=seed)
htz_range = [(1e-5, 1e5)]
htz_range = np.asarray(htz_range)

sample_scale = ["log"]
is_logscale = [s == "log" for s in sample_scale]

htz_range[is_logscale, :] = np.log10(htz_range[is_logscale, :])

samples_htz = sampler_htz.random(num)
samples_htz = qmc.scale(samples_htz, htz_range[:, 0], htz_range[:, 1])
samples_htz[:, is_logscale] = 10 ** samples_htz[:, is_logscale]

htz_is_logscale = is_logscale
#%%
sampler_fkv = qmc.LatinHypercube(d=3, seed=seed)
fkv_range = [
    (1e-5, 1e5),
    (1e-5, 1e5),
    (1e-10, 1e1)
    ]
fkv_range = np.asarray(fkv_range)

sample_scale = ["log"]
is_logscale = [s == "log" for s in sample_scale]

fkv_range[is_logscale, :] = np.log10(fkv_range[is_logscale, :])

samples_fkv = sampler_fkv.random(num)
samples_fkv = qmc.scale(samples_fkv, fkv_range[:, 0], fkv_range[:, 1])
samples_fkv[:, is_logscale] = 10 ** samples_fkv[:, is_logscale]

fkv_is_logscale = is_logscale
#%%
sampler_gm = qmc.LatinHypercube(d=5, seed=seed)
gm_range = [
    (1e-5, 1e5),
    (1e-5, 1e5),
    (1e-5, 1e5),
    (1e-5, 1e5),
    (1e-5, 1e5)
    ]
gm_range = np.asarray(gm_range)

sample_scale = ["log"]
is_logscale = [s == "log" for s in sample_scale]

gm_range[is_logscale, :] = np.log10(gm_range[is_logscale, :])

samples_gm = sampler_htz.random(num)
samples_gm = qmc.scale(samples_gm, gm_range[:, 0], gm_range[:, 1])
samples_gm[:, is_logscale] = 10 ** samples_gm[:, is_logscale]

gm_is_logscale = is_logscale
#%%
##
sls_fits, sls_results = [], []
mplr_fits, mplr_results = [], []
kww_fits, kww_results = [], []
htz_fits, htz_results = [], []

for i in tqdm(range(samples_sls.shape[0])):
    sample = samples_sls[i]
    constit_i = type(constit_sls)(*sample)
    constit_fit, result = fit_approach_lmfit(constit_i, bounds_sls, tip, app, f_app)
    sls_fits.append(constit_fit)
    sls_results.append(result)

for i in tqdm(range(samples_mplr.shape[0])):
    sample = samples_mplr[i]
    constit_i = type(constit_mplr)(*sample)
    constit_fit, result = fit_approach_lmfit(constit_i, bounds_mplr, tip, app, f_app)
    mplr_fits.append(constit_fit)
    mplr_results.append(result)

for i in tqdm(range(samples_kww.shape[0])):
    sample = samples_kww[i]
    constit_i = type(constit_kww)(*sample)
    constit_fit, result = fit_approach_lmfit(constit_i, bounds_kww, tip, app, f_app)
    kww_fits.append(constit_fit)
    kww_results.append(result)

for i in tqdm(range(samples_htz.shape[0])):
    sample = samples_htz[i]
    constit_i = type(constit_htz)(*sample)
    constit_fit, result = fit_approach_lmfit(constit_i, bounds_htz, tip, app, f_app)
    htz_fits.append(constit_fit)
    htz_results.append(result)
# %%
# BIC > 0 truncation

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
display(result)
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
#%%
f_kww_fit_app = force_approach(kww_tot_fit, app, tip)
f_kww_fit_ret = force_retract(kww_tot_fit, (app, ret), tip)
#%%
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
#%%
for i in tqdm(range(samples_kww.shape[0])):
    sample = samples_kww[i]
    constit_i = type(constit_kww)(*sample)
    constit_fit, result = fit_all_lmfit(
        constit_i, bounds_kww, tip, (app, ret), (f_app, f_ret)
    )
    kww_tot_fits.append(constit_fit)
    kww_tot_results.append(result)
#%%
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

results = sls_results
params = sls_params

    ax.scatter(xs, ys, zs, marker="^", alpha=0.7)

    ax.scatter(xs, ys, zs, marker='^', alpha=0.7, color="red")

    ax.scatter(xs_tot, ys_tot, zs_tot, marker="*", alpha=0.7)

for i in np.arange(len(tot_results)):
    xs_tot = tot_results[i].params.valuesdict()[params[0]]
    ys_tot = tot_results[i].params.valuesdict()[params[1]]
    zs_tot = tot_results[i].params.valuesdict()[params[2]]

    ax.scatter(xs_tot, ys_tot, zs_tot, marker='*', alpha=0.7, color="blue")

ax.scatter(xs, ys, zs, marker='^', alpha=0.7, color="red", label="approach params")
ax.scatter(xs_tot, ys_tot, zs_tot, marker='*', alpha=0.7, color="blue", label="entire params")
ax.set_xlabel(params[0])
ax.set_ylabel(params[1])
ax.set_zlabel(params[2])
ax.legend()

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
#%%
print("SLS parameter")
for i in range(len(sls_tot_results)):
    print(sls_tot_results[i].params.valuesdict())

print("MPLR parameter")
for i in range(len(mplr_tot_results)):
    print(mplr_tot_results[i].params.valuesdict())

print("KWW parameter")
for i in range(len(kww_tot_results)):
    print(kww_tot_results[i].params.valuesdict())
#%%
