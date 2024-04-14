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
    KohlrauschWilliamsWatts
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
#%%
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
## Fit only the approach portion

dt = app.time[1]-app.time[0]

constit_sls = StandardLinearSolid(10.0, 10.0, 10.0)
bounds_sls = [(0.0, jnp.inf)] * 3

constit_mplr = ModifiedPowerLaw(10.0, 10.0, 10.0)
bounds_mplr = [(0.0, jnp.inf), (0.0, 1.0), (dt*1e-3, dt*1e+10)]

constit_kww = KohlrauschWilliamsWatts(10.0, 10.0, 10.0, 10.0)
bounds_kww = [(0.0, jnp.inf)] * 3 + [(0.0, 1.0)]
#%%
# %%timeit
sls_fit, result_sls = fit_approach_lmfit(constit_sls, bounds_sls, tip, app, f_app)
mplr_fit, result_mplr = fit_approach_lmfit(constit_mplr, bounds_mplr, tip, app, f_app)
kww_fit, result_kww = fit_approach_lmfit(constit_kww, bounds_kww, tip, app, f_app)
#%%
display(result_sls)
display(result_mplr)
display(result_kww)
#%%
## Calculation of approach & retraction curve for approach params
f_sls_fit_app = force_approach(sls_fit, app, tip)
f_sls_fit_ret = force_retract(sls_fit, (app, ret), tip)

f_mplr_fit_app = force_approach(mplr_fit, app, tip)
f_mplr_fit_ret = force_retract(mplr_fit, (app, ret), tip)

f_kww_fit_app = force_approach(kww_fit, app, tip)
f_kww_fit_ret = force_retract(kww_fit, (app, ret), tip)
#%%
## Graph of 3 different viscoelastic model for approach params
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(app.time, f_app, ".", color="royalblue", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="royalblue", alpha=0.5)

# Graph for parameters of SLS model fitting (approach portion)
axes[0].plot(app.time, f_sls_fit_app, color="gray", alpha=0.7, label="SLS")
axes[0].plot(ret.time, f_sls_fit_ret, color="gray", alpha=0.7)
# # Graph for parameters of Modified PLR model fitting (approach portion)
axes[0].plot(app.time, f_mplr_fit_app, color="black", alpha=0.7, label="Modified PLR")
axes[0].plot(ret.time, f_mplr_fit_ret, color="black", alpha=0.7)
# Graph for parameters of KWW model fitting (approach portion)
axes[0].plot(app.time, f_kww_fit_app, color="black", alpha=0.7, label="KWW")
axes[0].plot(ret.time, f_kww_fit_ret, color="black", alpha=0.7)

axes[0].legend(loc="upper right")

axes[1] = plot_relaxation_fn(axes[1], sls_fit, app.time, color="gray", label="SLS")
axes[1] = plot_relaxation_fn(axes[1], mplr_fit, app.time, color="gray", label="Modified PLR")
axes[1] = plot_relaxation_fn(axes[1], kww_fit, app.time, color="gray", label="KWW")

axes[1].legend(loc="upper right")
#%%
# %%timeit
f_sls_fit_app = eqx.filter_jit(force_approach)(sls_fit, app, tip)
f_mplr_fit_app = eqx.filter_jit(force_approach)(mplr_fit, app, tip)
f_kww_fit_app = eqx.filter_jit(force_approach)(kww_fit, app, tip)
#%%
## Latinhypercube sampling & draw histogram for parameter space(example)
sampler = qmc.LatinHypercube(d=3, seed=10)
sls_range = [(1e-3, 1e2), (1e-3, 1e2), (1e-3, 1e2)]
sls_range = np.asarray(sls_range)

sample_scale = ["log", "log", "log"]
is_logscale = [s == "log" for s in sample_scale]

sls_range[is_logscale, :] = np.log10(sls_range[is_logscale, :])

samples_sls = sampler.random(20)
samples_sls = qmc.scale(samples_sls, sls_range[:, 0], sls_range[:, 1])
samples_sls[:, is_logscale] = 10 ** samples_sls[:, is_logscale]

fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):

    if is_logscale[i] is True:
        ax.hist(np.log10(samples_sls[:, i]))
    else:
        ax.hist(samples_sls[:, i])
#%%
sampler_mplr = qmc.LatinHypercube(d=3, seed=10)
mplr_range = [(1e-5, 1e5), (1e-10, 1.0), (1e-5, 1e5)]
mplr_range = np.asarray(mplr_range)

sample_scale = ["log", "log", "log"]
is_logscale = [s == "log" for s in sample_scale]

mplr_range[is_logscale, :] = np.log10(mplr_range[is_logscale, :])

samples_mplr = sampler_mplr.random(20)
samples_mplr = qmc.scale(samples_mplr, mplr_range[:, 0], mplr_range[:, 1])
samples_mplr[:, is_logscale] = 10 ** samples_mplr[:, is_logscale]
# %%
sampler_kww = qmc.LatinHypercube(d=4, seed=10)
kww_range = [(1e-5, 1e5), (1e-5, 1e5), (1e-5, 1e5), (1e-10, 1e1),]
kww_range = np.asarray(kww_range)

sample_scale = ["log", "log", "log", "log"]
is_logscale = [s == "log" for s in sample_scale]

kww_range[is_logscale, :] = np.log10(kww_range[is_logscale, :])

samples_kww = sampler_kww.random(20)
samples_kww = qmc.scale(samples_kww, kww_range[:, 0], kww_range[:, 1])
samples_kww[:, is_logscale] = 10 ** samples_kww[:, is_logscale]
#%%
## 
sls_fits, sls_results = [], []
mplr_fits, mplr_results = [], []
kww_fits, kww_results = [], []

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
#%%




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

#%%
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

#%%
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















#%%
samples_fit = []
for r in results:
    samples_fit.append(list(r.params.valuesdict().values()))
samples_fit = np.asarray(samples_fit)
# %%
fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):
    if is_logscale[i] is True:
        s_ = samples_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(samples_fit[:, i])
# %%
results[0].params.valuesdict()
# %%
constit_fit = sls_fits[8]
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
