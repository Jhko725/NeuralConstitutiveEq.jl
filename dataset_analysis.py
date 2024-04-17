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
    Hertzian
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
## Fit only the approach portion

dt = app.time[1] - app.time[0]

constit_sls = StandardLinearSolid(10.0, 10.0, 10.0)
bounds_sls = [(1e-7, 1e7)] * 3

constit_mplr = ModifiedPowerLaw(10.0, 10.0, 10.0)
bounds_mplr = [(1e-7, 1e7), (0.0, 1.0), (1e-7, 1e7)]

constit_kww = KohlrauschWilliamsWatts(10.0, 10.0, 10.0, 10.0)
bounds_kww = [(1e-7, 1e7)] * 3 + [(0.0, 1.0)]

constit_htz = Hertzian(10.0)
bounds_htz = [(1e-7, 1e7)]
#%%
# %%timeit
sls_fit, result_sls = fit_approach_lmfit(constit_sls, bounds_sls, tip, app, f_app)
mplr_fit, result_mplr = fit_approach_lmfit(constit_mplr, bounds_mplr, tip, app, f_app)
kww_fit, result_kww = fit_approach_lmfit(constit_kww, bounds_kww, tip, app, f_app)
htz_fit, result_htz = fit_approach_lmfit(constit_htz, bounds_htz, tip, app, f_app)
# %%
display(result_sls)
display(result_mplr)
display(result_kww)
display(result_htz)
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
# %%
## Graph of 3 different viscoelastic model for approach params
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
axes[1] = plot_relaxation_fn(axes[1], mplr_fit, app.time, color="blue", label="Modified PLR")
axes[1] = plot_relaxation_fn(axes[1], kww_fit, app.time, color="green", label="KWW")
axes[1] = plot_relaxation_fn(axes[1], htz_fit, app.time, color="yellow", label="Hertzian")

axes[1].legend(loc="upper right")
# %%
# %%timeit
f_sls_fit_app = eqx.filter_jit(force_approach)(sls_fit, app, tip)
f_mplr_fit_app = eqx.filter_jit(force_approach)(mplr_fit, app, tip)
f_kww_fit_app = eqx.filter_jit(force_approach)(kww_fit, app, tip)
f_htz_fit_app = eqx.filter_jit(force_approach)(htz_fit, app, tip)
# %%
## Latinhypercube sampling & draw histogram for parameter space(example)

num = 5
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
# %%
sampler_kww = qmc.LatinHypercube(d=4, seed=seed)
kww_range = [(1e-5, 1e5), (1e-5, 1e5), (1e-7, 1e7), (1e-10, 1e1),]
kww_range = np.asarray(kww_range)

sample_scale = ["log", "log", "log", "log"]
is_logscale = [s == "log" for s in sample_scale]

kww_range[is_logscale, :] = np.log10(kww_range[is_logscale, :])

samples_kww = sampler_kww.random(num)
samples_kww = qmc.scale(samples_kww, kww_range[:, 0], kww_range[:, 1])
samples_kww[:, is_logscale] = 10 ** samples_kww[:, is_logscale]
#%%
sampler_htz = qmc.LatinHypercube(d=1, seed=seed)
htz_range = [(1e-5, 1e5)]
htz_range = np.asarray(htz_range)

sample_scale = ["log"]
is_logscale = [s == "log" for s in sample_scale]

htz_range[is_logscale, :] = np.log10(htz_range[is_logscale, :])

samples_htz = sampler_htz.random(num)
samples_htz = qmc.scale(samples_htz, htz_range[:, 0], htz_range[:, 1])
samples_htz[:, is_logscale] = 10 ** samples_htz[:, is_logscale]
# %%
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
#%%
# BIC > 0 truncation

sls_bic = np.array([sls_results[i].bic for i in np.arange(num)])
mplr_bic = np.array([mplr_results[i].bic for i in np.arange(num)])
kww_bic = np.array([kww_results[i].bic for i in np.arange(num)])
htz_bic = np.array([htz_results[i].bic for i in np.arange(num)])

sls_fits = np.array(sls_fits)[np.where(sls_bic<0)]
mplr_fits = np.array(mplr_fits)[np.where(mplr_bic<0)]
kww_fits = np.array(kww_fits)[np.where(kww_bic<0)]
htz_fits = np.array(htz_fits)[np.where(htz_bic<0)]

sls_results = np.array(sls_results)[np.where(sls_bic<0)]
mplr_results = np.array(mplr_results)[np.where(mplr_bic<0)]
kww_results = np.array(kww_results)[np.where(kww_bic<0)]
htz_results = np.array(htz_results)[np.where(htz_bic<0)]

print(f"SLS = {len(np.where(sls_bic>0))} truncation")
print(f"MPLR = {len(np.where(mplr_bic>0))} truncation")
print(f"KWW = {len(np.where(kww_bic>0))} truncation")
print(f"HTZ = {len(np.where(htz_bic>0))} truncation")
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
#%%
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
#%%
## For SLS
sls_samples_fit = []
for r in sls_results:
    sls_samples_fit.append(list(r.params.valuesdict().values()))
sls_samples_fit = np.asarray(sls_samples_fit)
# %%
# fig, axes = plt.subplots(1, 3, figsize=(7, 3))
# for i, ax in enumerate(axes):
#     if is_logscale[i] is True:
#         print(i)
#         s_ = sls_samples_fit[:, i]
#         s_[s_ > 0] = np.log10(s_[s_ > 0])
#         ax.hist(s_)
#     else:
#         ax.hist(sls_samples_fit[:, i])
# %%
sls_results[0].params.valuesdict()
# %%
## For MPLR
mplr_samples_fit = []
for r in mplr_results:
    mplr_samples_fit.append(list(r.params.valuesdict().values()))
mplr_samples_fit = np.asarray(mplr_samples_fit)
# %%
# fig, axes = plt.subplots(1, 3, figsize=(7, 3))
# for i, ax in enumerate(axes):
#     if is_logscale[i] is True:
#         s_ = mplr_samples_fit[:, i]
#         s_[s_ > 0] = np.log10(s_[s_ > 0])
#         ax.hist(s_)
#     else:
#         ax.hist(mplr_samples_fit[:, i])
# %%
mplr_results[0].params.valuesdict()
# %%
# For KWW
kww_samples_fit = []
for r in kww_results:
    kww_samples_fit.append(list(r.params.valuesdict().values()))
kww_samples_fit = np.asarray(kww_samples_fit)
# %%
# fig, axes = plt.subplots(1, 4, figsize=(7, 3))
# for i, ax in enumerate(axes):
#     if is_logscale[i] is True:
#         s_ = kww_samples_fit[:, i]
#         s_[s_ > 0] = np.log10(s_[s_ > 0])
#         ax.hist(s_)
#     else:
#         ax.hist(kww_samples_fit[:, i])
# %%
kww_results[0]
#%%
## For Hertz
htz_samples_fit = []
for r in htz_results:
    htz_samples_fit.append(list(r.params.valuesdict().values()))
htz_samples_fit = np.asarray(htz_samples_fit)
# %%
# fig, axes = plt.subplots(1, 1, figsize=(7, 3))
# for i, ax in enumerate(axes):
#     if is_logscale[i] is True:
#         s_ = htz_samples_fit[:, i]
#         s_[s_ > 0] = np.log10(s_[s_ > 0])
#         ax.hist(s_)
#     else:
#         ax.hist(htz_samples_fit[:, i])
# %%
htz_results[0]
#%%









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
#%%
## check parameter
for i in range(num):
    print(kww_results[i].params.valuesdict())
#%%
## Parameter space 
sls_params = ['E1', 'E_inf', 'tau']
mplr_params = ['E0', 't0', 'alpha']
kww_params = ['E1', 'E_inf', 'tau']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

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

#%%




  
#######################################################################################################################
# %%
## Fit the entire approach-retract curve for SLS
sls_fit, result = fit_all_lmfit(constit_sls, bounds_sls, tip, (app, ret), (f_app, f_ret))
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
mplr_fit, result = fit_all_lmfit(constit_mplr, bounds_mplr, tip, (app, ret), (f_app, f_ret))
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
kww_fit, result = fit_all_lmfit(constit_kww, bounds_kww, tip, (app, ret), (f_app, f_ret))
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
htz_fit, result = fit_all_lmfit(constit_htz, bounds_htz, tip, (app, ret), (f_app, f_ret))
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
#%%



#%%
## Latinhypercube sampling

sls_tot_fits, sls_tot_results = [], []
mplr_tot_fits, mplr_tot_results = [], []
kww_tot_fits, kww_tot_results = [], []
htz_tot_fits, htz_tot_results = [], []

for i in tqdm(range(samples_sls.shape[0])):
    sample = samples_sls[i]
    constit_i = type(constit_sls)(*sample)
    constit_fit, result = fit_all_lmfit(constit_i, bounds_sls, tip, (app, ret), (f_app, f_ret))
    sls_tot_fits.append(constit_fit)
    sls_tot_results.append(result)

for i in tqdm(range(samples_mplr.shape[0])):
    sample = samples_mplr[i]
    constit_i = type(constit_mplr)(*sample)
    constit_fit, result = fit_all_lmfit(constit_i, bounds_mplr, tip, (app, ret), (f_app, f_ret))
    mplr_tot_fits.append(constit_fit)
    mplr_tot_results.append(result)

for i in tqdm(range(samples_kww.shape[0])):
    sample = samples_kww[i]
    constit_i = type(constit_kww)(*sample)
    constit_fit, result = fit_all_lmfit(constit_i, bounds_kww, tip, (app, ret), (f_app, f_ret))
    kww_tot_fits.append(constit_fit)
    kww_tot_results.append(result)

for i in tqdm(range(samples_htz.shape[0])):
    sample = samples_htz[i]
    constit_i = type(constit_htz)(*sample)
    constit_fit, result = fit_all_lmfit(constit_i, bounds_htz, tip, (app, ret), (f_app, f_ret))
    htz_tot_fits.append(constit_fit)
    htz_tot_results.append(result)
# %%
## BIC > 0 truncation

sls_tot_bic = np.array([sls_tot_results[i].bic for i in np.arange(num)])
mplr_tot_bic = np.array([mplr_tot_results[i].bic for i in np.arange(num)])
kww_tot_bic = np.array([kww_tot_results[i].bic for i in np.arange(num)])
htz_tot_bic = np.array([htz_tot_results[i].bic for i in np.arange(num)])

sls_tot_fits = np.array(sls_tot_fits)[np.where(sls_tot_bic<0)]
mplr_tot_fits = np.array(mplr_tot_fits)[np.where(mplr_tot_bic<0)]
kww_tot_fits = np.array(kww_tot_fits)[np.where(kww_tot_bic<0)]
htz_tot_fits = np.array(htz_tot_fits)[np.where(htz_tot_bic<0)]

sls_tot_results = np.array(sls_tot_results)[np.where(sls_tot_bic<0)]
mplr_tot_results = np.array(mplr_tot_results)[np.where(mplr_tot_bic<0)]
kww_tot_results = np.array(kww_tot_results)[np.where(kww_tot_bic<0)]
htz_tot_results = np.array(htz_tot_results)[np.where(htz_tot_bic<0)]

print(f"SLS = {len(np.where(sls_tot_bic>0))} truncation")
print(f"MPLR = {len(np.where(mplr_tot_bic>0))} truncation")
print(f"KWW = {len(np.where(kww_tot_bic>0))} truncation")
print(f"HTZ = {len(np.where(htz_tot_bic>0))} truncation")
#%%
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
#%%
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
#%%
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
#%%
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
#%%
sls_samples_tot_fit = []
for r in sls_tot_results:
    sls_samples_tot_fit.append(list(r.params.valuesdict().values()))
sls_samples_tot_fit = np.asarray(sls_samples_tot_fit)
# %%
fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):
    if is_logscale[i] is True:
        s_ = sls_samples_tot_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(sls_samples_tot_fit[:, i])
# %%
sls_tot_results[0].params.valuesdict()
#%%
mplr_samples_tot_fit = []
for r in mplr_tot_results:
    mplr_samples_tot_fit.append(list(r.params.valuesdict().values()))
mplr_samples_tot_fit = np.asarray(mplr_samples_tot_fit)
#%%
fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):
    if is_logscale[i] is True:
        s_ = mplr_samples_tot_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(mplr_samples_tot_fit[:, i])
# %%
mplr_tot_results[0].params.valuesdict()
#%%
kww_samples_tot_fit = []
for r in kww_tot_results:
    kww_samples_tot_fit.append(list(r.params.valuesdict().values()))
kww_samples_tot_fit = np.asarray(kww_samples_tot_fit)
#%%
fig, axes = plt.subplots(1, 4, figsize=(7, 3))
for i, ax in enumerate(axes):
    if is_logscale[i] is True:
        s_ = kww_samples_tot_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(kww_samples_tot_fit[:, i])
# %%
kww_tot_results[0].params.valuesdict()
#%%
htz_samples_tot_fit = []
for r in htz_tot_results:
    htz_samples_tot_fit.append(list(r.params.valuesdict().values()))
htz_samples_tot_fit = np.asarray(htz_samples_tot_fit)
#%%
fig, axes = plt.subplots(1, 4, figsize=(7, 3))
for i, ax in enumerate(axes):
    if is_logscale[i] is True:
        s_ = htz_samples_tot_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(htz_samples_tot_fit[:, i])
# %%
htz_tot_results[0].params.valuesdict()
#%%
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
#%%
## Parameter space 
sls_params = ['E1', 'E_inf', 'tau']
mplr_params = ['E0', 't0', 'alpha']
kww_params = ['E1', 'E_inf', 'tau']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in np.arange(len(sls_tot_results)):
    params = sls_params
    xs = sls_results[i].params.valuesdict()[params[0]]
    ys = sls_results[i].params.valuesdict()[params[1]]
    zs = sls_results[i].params.valuesdict()[params[2]]

    xs_tot = sls_tot_results[i].params.valuesdict()[params[0]]
    ys_tot = sls_tot_results[i].params.valuesdict()[params[1]]
    zs_tot = sls_tot_results[i].params.valuesdict()[params[2]]


    ax.scatter(xs, ys, zs, '^')
    ax.scatter(xs_tot, ys_tot, zs_tot, 'o')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_zscale('log')
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
fig, ax = plt.subplots(1, 1, figsize=(10,5))

x=np.arange(2)

ax.bar(x, height=sls_bic, width=0.25, label = 'SLS')
ax.bar(x+0.25, height=mplr_bic, width=0.25, label='MPLR')
ax.bar(x+0.5, height=kww_bic, width=0.25, label='MPLR')
ax.bar(x+0.75, height=htz_bic, width=0.25, label='Hertz')

ax.bar(x, height=sls_tot_bic, width=0.25, label = 'SLS')
ax.bar(x+0.25, height=mplr_tot_bic, width=0.25, label='MPLR')
ax.bar(x+0.5, height=kww_tot_bic, width=0.25, label='KWW')
ax.bar(x+0.75, height=htz_tot_bic, width=0.25, label='Hertz')
ax.set_ylabel("BIC")
ax.legend()

# %%
display(sls_tot_results[0])
# %%
sls_tot_results[0].aborted
#%%