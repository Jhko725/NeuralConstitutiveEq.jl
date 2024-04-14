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
from neuralconstitutive.ting import force_approach, force_retract
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
#%%
%%timeit
constit_fit, result = fit_approach_lmfit(constit, bounds, tip, app, f_app)
#%%
display(result)
#%%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, f_app, ".", color="royalblue", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="royalblue", label="Data", alpha=0.5)

f_fit_app = force_approach(constit_fit, app, tip)
f_fit_ret = force_retract(constit_fit, (app, ret), tip)

axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")
#%%
import equinox as eqx
#%%
%%timeit
f_fit_app = eqx.filter_jit(force_approach)(constit_fit, app, tip)

# %%
import numpy as np
from scipy.stats import qmc

sampler = qmc.LatinHypercube(d=3, seed=10)
sample_range = [(1e-3, 1e2), (1e-3, 1e2), (1e-3, 1e2)]
sample_range = np.asarray(sample_range)

sample_scale = ["log", "log", "log"]
is_logscale = [s == "log" for s in sample_scale]

sample_range[is_logscale, :] = np.log10(sample_range[is_logscale, :])

samples = sampler.random(20)
samples = qmc.scale(samples, sample_range[:, 0], sample_range[:, 1])
samples[:, is_logscale] = 10 ** samples[:, is_logscale]


fig, axes = plt.subplots(1, 3, figsize=(7, 3))
for i, ax in enumerate(axes):

    if is_logscale[i] is True:
        ax.hist(np.log10(samples[:, i]))
    else:
        ax.hist(samples[:, i])
# %%
from tqdm import tqdm

constit_fits, results = [], []
for i in tqdm(range(samples.shape[0])):
    sample = samples[i]
    constit_i = type(constit)(*sample)
    constit_fit, result = fit_approach_lmfit(constit_i, bounds, tip, app, f_app)
    constit_fits.append(constit_fit)
    results.append(result)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, f_app, ".", color="royalblue", label="Data", alpha=0.5)
axes[0].plot(ret.time, f_ret, ".", color="royalblue", label="Data", alpha=0.5)

for i, (constit_fit, result) in enumerate(zip(constit_fits, tqdm(results))):
    f_fit_app = force_approach(constit_fit, app, tip)
    f_fit_ret = force_retract(constit_fit, (app, ret), tip)

    axes[0].plot(app.time, f_fit_app, color="gray", alpha=0.7)
    axes[0].plot(ret.time, f_fit_ret, color="gray", alpha=0.7)

    axes[1] = plot_relaxation_fn(axes[1], constit_fit, app.time, color="gray")
# %%
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
constit_fit = constit_fits[8]
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
