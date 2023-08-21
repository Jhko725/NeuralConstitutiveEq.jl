# %%
from functools import partial

from numpy import random
import numpy as np
from scipy.interpolate import interp1d
import scipy.special as sc
import matplotlib.pyplot as plt

from neuralconstitutive.preprocessing import estimate_derivative
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.constitutive import PowerLawRheology
from neuralconstitutive.ting import force_approach, force_retract
from neuralconstitutive.fitting import fit_approach, fit_retract, fit_total
from neuralconstitutive.utils import squared_error

# %%
E_0 = 1614  # Pa
eta = 25  # Pa.s
tau = 3 * (eta / E_0)
v = 10  # um/s
theta = (18.0 / 180.0) * np.pi
alpha = 8.0 / (3.0 * np.pi) * np.tan(theta)
beta = 2.0


# %%
def SLS_Force_app(E_0, eta, t, v, alpha, beta):
    force = alpha * (v**beta) * t * (6 * eta + E_0 * t)
    return np.array(force)


def SLS_Force_ret(E_0, t, v, alpha, beta):
    force = alpha * E_0 * (v * t) ** beta
    return np.array(force)


# %%
def t1(t, t_max, tau):
    t1 = 2 * t_max - t - tau
    t1 = np.clip(t1, 0, np.inf)
    return t1


# %%
space = 201  # odd number
idx = int((space - 1) / 2)
t_array = np.linspace(0, 0.4, space)
t_app = t_array[: idx + 1]
t_ret = t_array[idx:]
t_max = t_array[idx]
t_1 = t1(t_ret, t_max, tau)
# %%
F_app = SLS_Force_app(E_0, eta, t_app, v, alpha, beta)
F_ret = SLS_Force_ret(E_0, t_1, v, alpha, beta)
F_app[np.isnan(F_app)] = 0
F_ret[np.isnan(F_ret)] = 0
# %%
F_total = np.append(F_app, F_ret[1:])
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.plot(t_array, F_total, ".")
# %%
# Generate Gaussian Noise
mu, sigma = 0, F_ret[0] * 0.001  # average = 0, sigma = 0.1%
noise = np.random.normal(mu, sigma, np.shape(F_total))
F_total_noise = F_total + noise
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.plot(t_array, F_total_noise * 1e9, ".")
# %%
v = 10  # um/s
indentation_app = v * t_app
indentation_ret = indentation_app[idx] - v * (t_ret - t_app[idx])
# %%
indentation = np.append(indentation_app, indentation_ret[1:])
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.plot(t_array, indentation, ".")
# %%
mu, sigma = 0, indentation_ret[0] * 0.001  # average = 0, sigma = 0.1%
noise = np.random.normal(mu, sigma, np.shape(indentation))
indentation_noise = indentation + noise
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.plot(t_array, indentation_noise, ".")
# %%
force = F_total
indentation = indentation_noise
time = t_array
# %%
max_ind = np.argmax(indentation)
t_max = time[max_ind]
indent_max = indentation[max_ind]
# %%
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
# Truncation negrative force region
# negative_idx = np.where(F_ret < 0)[0]
# negative_idx = negative_idx[0]
# # %%
# F_ret = F_ret[:negative_idx]
# time_ret = time_ret[:negative_idx]
# %%

# %%
# Determination of Variable
tip = Spherical(0.8)

# Fit to approach, retract, total
plr = PowerLawRheology(0.562, 0.2, 1e-5)
plr_fit = [
    fit_func(plr, time, indentation, force, tip, t0=1e-5)
    for fit_func in (fit_approach, fit_retract, fit_total)
]
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
axes[0].set_title("Normalized squared error for the PLR curve fits")  # %%
