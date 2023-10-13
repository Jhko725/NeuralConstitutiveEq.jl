# %%
from typing import Callable, Literal
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from jhelabtoolkit.io.nanosurf import nanosurf
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults
from scipy.interpolate import interp1d

from neuralconstitutive.preprocessing import (
    calc_tip_distance,
    estimate_derivative,
    get_sampling_rate,
    get_z_and_defl,
    ratio_of_variances,
    fit_baseline_polynomial,
)
from neuralconstitutive.tipgeometry import Spherical, TipGeometry
from neuralconstitutive.constitutive import PowerLawRheology, StandardLinearSolid
from neuralconstitutive.ting import force_approach, force_retract
from neuralconstitutive.fitting import (
    fit_approach,
    fit_retract,
    fit_total,
    split_approach_retract,
)
from neuralconstitutive.utils import squared_error


configure_matplotlib_defaults()

filepath = (
    "data/230927_onion/onion_pbs_3_Image01396.nid"
)
config, data = nanosurf.read_nid(filepath)

# %%
forward, backward = data["spec forward"], data["spec backward"]

# %%
z_fwd, defl_fwd = get_z_and_defl(forward)
z_bwd, defl_bwd = get_z_and_defl(backward)
dist_fwd = calc_tip_distance(z_fwd, defl_fwd)
dist_bwd = calc_tip_distance(z_bwd, defl_bwd)
#%%

# %%
# ROV method
N = 10
# tuple unpacking
rov_fwd, idx_fwd = ratio_of_variances(defl_fwd, N)
rov_bwd, idx_bwd = ratio_of_variances(defl_bwd, N)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(dist_fwd, rov_fwd)
ax.set_xlabel("Distance(forward)")
ax.set_ylabel("ROV")
plt.axvline(
    dist_fwd[idx_fwd],
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="maximum point",
)
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd * 1e6, defl_fwd * 1e9, label="forward")
ax.plot(dist_bwd * 1e6, defl_bwd * 1e9, label="backward")
ax.set_xlabel("Distance(μm)")
ax.set_ylabel("Force(nN)")
ax.legend()
# %%
# Find contact point
cp_fwd = dist_fwd[idx_fwd]
cp_bwd = dist_bwd[idx_bwd]
print(cp_fwd, cp_bwd)
# %%
# Translation
dist_fwd = dist_fwd - cp_fwd
dist_bwd = dist_bwd - cp_bwd

# %%
# Polynomial fitting
baseline_poly_fwd = fit_baseline_polynomial(dist_fwd, defl_fwd)
defl_processed_fwd = defl_fwd - baseline_poly_fwd(dist_fwd)
baseline_poly_bwd = fit_baseline_polynomial(dist_bwd, defl_bwd)
defl_processed_bwd = defl_bwd - baseline_poly_bwd(dist_bwd)
baseline_poly_bwd
# baseline_poly_bwd
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd * 1e6, defl_fwd * 1e9, label="forward")
ax.plot(dist_bwd * 1e6, defl_bwd * 1e9, label="backward")
ax.set_xlabel("Distance(μm)")
ax.set_ylabel("Force(nN)")
plt.axvline(cp_fwd, color="grey", linestyle="--", linewidth=1)
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
plot_kwargs = {"markersize": 2.0, "alpha": 0.7}
ax.plot(dist_fwd * 1e6, defl_processed_fwd * 1e9, ".-", label="Approach", **plot_kwargs)
ax.plot(dist_bwd * 1e6, defl_processed_bwd * 1e9, ".-", label="Retract", **plot_kwargs)
ax.set_xlabel("Distance(μm)")
ax.set_ylabel("Force(nN)")
plt.axvline(cp_fwd, color="grey", linestyle="--", linewidth=1)
ax.legend()
# %%
fig, axes = plt.subplots(2, 1, figsize=(5, 3), sharex=True, sharey=True)
plot_kwargs = {"markersize": 2.0, "alpha": 0.7}
axes[0].plot(dist_fwd * 1e6, defl_fwd * 1e9, ".-", label="Approach", **plot_kwargs)
axes[0].plot(dist_bwd * 1e6, defl_bwd * 1e9, ".-", label="Retract", **plot_kwargs)
axes[1].plot(dist_fwd * 1e6, defl_processed_fwd * 1e9, ".-", label="Approach", **plot_kwargs)
axes[1].plot(dist_bwd * 1e6, defl_processed_bwd * 1e9, ".-", label="Retract", **plot_kwargs)
for ax in axes:
    ax.set_ylabel("Force(nN)")
    ax.axvline(cp_fwd, color="grey", linestyle="--", linewidth=1)
axes[-1].set_xlabel("Distance(μm)")
axes[0].legend()
#%%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd, z_fwd, label="forward")
ax.plot(dist_bwd, z_bwd, label="backward")
ax.legend()

# %%
dist_total = np.concatenate((dist_fwd, dist_bwd[::-1]), axis=-1)
defl_total = np.concatenate((defl_fwd, defl_bwd[::-1]), axis=-1)
is_contact = dist_total >= 0
indentation = dist_total[is_contact]
# k = 0.2  # N/m
force = defl_total[is_contact]
force -= force[0]
sampling_rate = get_sampling_rate(config)
time = np.arange(len(indentation)) / sampling_rate
print(len(time))
#%%
plot_kwargs = {"markersize": 2.0, "alpha": 0.7}
fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
axes[0].plot(time, indentation * 1e6, ".", color="darkred", **plot_kwargs)
axes[1].set_xlabel("Time(s)")
axes[0].set_ylabel("Indentation(μm)")
axes[1].set_ylabel("Force(nN)")
axes[1].plot(time, force * 1e9, ".", color="navy", **plot_kwargs)
# %%
max_ind = np.argmax(indentation)
t_max = time[max_ind]
indent_max = indentation[max_ind]
# %%
indentation *= 1e6
force *= 1e6
F_app = force[: max_ind + 1]
F_ret = force[max_ind:]
# %%
# t_max 부분을 겹치게 해야 문제가 안생김

indentation_app = indentation[: max_ind + 1]
indentation_ret = indentation[max_ind:]

time_app = time[: max_ind + 1]
time_ret = time[max_ind:]

velocity_app = estimate_derivative(time_app, indentation_app)
velocity_ret = estimate_derivative(time_ret, indentation_ret)

indentation_app_func = interp1d(time_app, indentation_app)
indentation_ret_func = interp1d(time_ret, indentation_ret)
velocity_app_func = interp1d(time_app, velocity_app)
velocity_ret_func = interp1d(time_ret, velocity_ret)
# %%
# Truncate negetive force region
# negative_idx = np.where(F_ret < 0)[0]
# negative_idx = negative_idx[0]
# F_ret = F_ret[:negative_idx]
# time_ret = time_ret[:negative_idx]
# %%
# %%
## PLR model fitting ##

# Determination of Variable
tip = Spherical(0.8)

# Fit to approach, retract, total
plr = PowerLawRheology(0.562, 0.2, 1e-5)
#%%
%%time
plr_fit = [
    fit_func(plr, time, indentation, force, tip, t0=1e-5)
    for fit_func in (fit_approach, fit_retract, fit_total)
]
#%%
plr_fit
# %%
f_fit = [
    np.concatenate(
        [
            force_approach(
                time_app,
                plr_,
                indentation_app_func,
                velocity_app_func,
                tip,
            )[:-1],
            force_retract(
                time_ret,
                plr_,
                indentation_app_func,
                velocity_app_func,
                velocity_ret_func,
                tip,
            ),
        ],
        axis=0,
    )
    for plr_, _ in plr_fit
]

# %%
labels = ("Approach", "Retract", "Both")
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(time, force, ".", color="black", label="Data")
for f, lab in zip(f_fit, labels):
    ax.plot(time, f, label=lab)
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)") 
ax.legend()

# %%
E0s = [plr_.E0 for plr_, _ in plr_fit]
gammas = [plr_.gamma for plr_, _ in plr_fit]
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].plot(labels, E0s)
axes[1].plot(labels, gammas)


# %%


selections = (slice(0, len(time_app)), slice(len(time_app), None), slice(None, None))
fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True, sharey=True)
for ax, sel, lab in zip(axes, selections, labels):
    mses = [squared_error(force[sel], f[sel]) for f in f_fit]
    ax.bar(labels, mses / np.sum(force**2))
    ax.set_ylabel(f"Error: {lab}")
axes[0].set_title("Normalized squared error for the PLR curve fits")
# %%
## SLS model fitting ##

sls = StandardLinearSolid(10, 1, 2.0)
sls_fit = [
    fit_func(sls, time, indentation, force, tip)
    for fit_func in (fit_approach, fit_retract, fit_total)
]
f_fit = [
    np.concatenate(
        [
            force_approach(
                time_app,
                sls_,
                indentation_app_func,
                velocity_app_func,
                tip,
            )[:-1],
            force_retract(
                time_ret,
                sls_,
                indentation_app_func,
                velocity_app_func,
                velocity_ret_func,
                tip,
            ),
        ],
        axis=0,
    )
    for sls_, _ in sls_fit
]

# %%
labels = ("Approach", "Retract", "Both")
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(time, force, ".", color="black", label="Data")
for f, lab in zip(f_fit, labels):
    ax.plot(time, f, label=lab)
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")
ax.legend()

# %%
