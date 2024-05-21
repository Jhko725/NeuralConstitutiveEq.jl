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
#%%
# %%timeit
sls_fit, result_sls = fit_approach_lmfit(constit_sls, bounds_sls, tip, app, f_app)
# %%
display(result_sls)
# %%
## Calculation of approach & retraction curve for approach params
f_sls_fit_app = force_approach(sls_fit, app, tip)
f_sls_fit_ret = force_retract(sls_fit, (app, ret), tip)
#%%
## Graph of 3 different viscoelastic model for approach params
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(app.time, f_app, ".", color="black", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="black", alpha=0.5)

axes[0].legend(loc="upper right")

axes[1] = plot_relaxation_fn(axes[1], sls_fit, app.time, color="red", label="SLS")

axes[1].legend(loc="upper right")
# %%
# %%timeit
f_sls_fit_app = eqx.filter_jit(force_approach)(sls_fit, app, tip)
# %%
## Latinhypercube sampling & draw histogram for parameter space(example)

num = 30

sampler = qmc.LatinHypercube(d=3, seed=10)
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
##
sls_fits, sls_results = [], []

for i in tqdm(range(samples_sls.shape[0])):
    sample = samples_sls[i]
    constit_i = type(constit_sls)(*sample)
    constit_fit, result = fit_approach_lmfit(constit_i, bounds_sls, tip, app, f_app)
    sls_fits.append(constit_fit)
    sls_results.append(result)
#%%
threshold = 2000

def valid_inds(model_bic, threshold):
    return model_bic<np.min(model_bic)+threshold

sls_bic = np.array([sls_results[i].bic for i in np.arange(num)])
sls_fits = np.array(sls_fits)[valid_inds(sls_bic, threshold)]
sls_results = np.array(sls_results)[valid_inds(sls_bic, threshold)]
print(f"SLS = {np.sum(valid_inds(sls_bic, threshold)==False)} truncation")
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
# %%
sls_samples_fit = []
for r in sls_results:
    sls_samples_fit.append(list(r.params.valuesdict().values()))
sls_samples_fit = np.asarray(sls_samples_fit)
# %%
fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):
    if is_logscale[i] is True:
        s_ = sls_samples_fit[:, i]
        s_[s_ > 0] = np.log10(s_[s_ > 0])
        ax.hist(s_)
    else:
        ax.hist(sls_samples_fit[:, i])
# %%
sls_results[0].params.valuesdict()
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

#%%
## check parameter
for i in range(len(sls_results)):
    print(sls_results[i].params.valuesdict())
#%%
## Parameter space 
sls_params = ['E1', 'E_inf', 'tau']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in np.arange(len(sls_results)):
    params = sls_params
    xs = sls_results[i].params.valuesdict()[params[0]]
    ys = sls_results[i].params.valuesdict()[params[1]]
    zs = sls_results[i].params.valuesdict()[params[2]]

    ax.scatter(xs, ys, zs)

ax.set_xlabel(params[0])
ax.set_ylabel(params[1])
ax.set_zlabel(params[2])

plt.show()
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
## Latinhypercube sampling

sls_tot_fits, sls_tot_results = [], []

for i in tqdm(range(samples_sls.shape[0])):
    sample = samples_sls[i]
    constit_i = type(constit_sls)(*sample)
    constit_fit, result = fit_all_lmfit(constit_i, bounds_sls, tip, (app, ret), (f_app, f_ret))
    sls_tot_fits.append(constit_fit)
    sls_tot_results.append(result)
#%%

#%%
sls_tot_bic = np.array([sls_tot_results[i].bic for i in np.arange(num)])
sls_tot_fits = np.array(sls_tot_fits)[valid_inds(sls_tot_bic, threshold)]
sls_tot_results = np.array(sls_tot_results)[valid_inds(sls_tot_bic, threshold)]
print(f"SLS = {np.sum(valid_inds(sls_tot_bic, threshold)==False)} truncation")
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
# %%
sls_tot_results[0].params.valuesdict()
#%%
## Parameter space 
sls_params = ['E1', 'E_inf', 'tau']

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(projection='3d')

for i in np.arange(len(sls_results)):
    params = sls_params
    xs = sls_results[i].params.valuesdict()[params[0]]
    ys = sls_results[i].params.valuesdict()[params[1]]
    zs = sls_results[i].params.valuesdict()[params[2]]

    ax.scatter(xs, ys, zs, marker='^', alpha=0.7)

for i in np.arange(len(sls_tot_results)):
    xs_tot = sls_tot_results[i].params.valuesdict()[params[0]]
    ys_tot = sls_tot_results[i].params.valuesdict()[params[1]]
    zs_tot = sls_tot_results[i].params.valuesdict()[params[2]]

    ax.scatter(xs_tot, ys_tot, zs_tot, marker='*', alpha=0.7)

ax.set_xlabel(params[0])
ax.set_ylabel(params[1])
ax.set_zlabel(params[2])

plt.show()
#%%
sls_bic = np.min(sls_bic)
sls_tot_bic = np.min(sls_tot_bic)
# %%
section = ("Approach","Total")
model_bic = {
    'SLS': (sls_bic, sls_tot_bic),
}

x = np.arange(len(section))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in model_bic.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('BIC')
ax.set_title('SLS model by section')
ax.set_xticks(x, section)
ax.legend(loc="lower left", ncols=2)

plt.show()
# %%
display(sls_tot_results[0])
# %%
sls_tot_results[0].aborted
# %%
print(f'operation time : {time.time()-t_start}')
#%%