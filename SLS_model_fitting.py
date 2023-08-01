# %%
from configparser import ConfigParser
from functools import partial
import numpy as np
from numpy import ndarray
import xarray as xr
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from jhelabtoolkit.io.nanosurf import nanosurf
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults
from neuralconstitutive.preprocessing import process_approach_data, estimate_derivative
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.preprocessing import (
    calc_tip_distance,
    estimate_derivative,
    get_sampling_rate,
    get_z_and_defl,
    ratio_of_variances,
    fit_baseline_polynomial,
)


configure_matplotlib_defaults()

filepath = (
    "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(10nN, 1s, liquid).nid"
)
config, data = nanosurf.read_nid(filepath)

forward, backward = data["spec forward"], data["spec backward"]


# %%
z_fwd, defl_fwd = get_z_and_defl(forward)
z_bwd, defl_bwd = get_z_and_defl(backward)
dist_fwd = calc_tip_distance(z_fwd, defl_fwd)
dist_bwd = calc_tip_distance(z_bwd, defl_bwd)
# cp = find_contact_point(dist_fwd, defl_fwd)
# %%
# ROV method
N = 10
rov_fwd, idx_fwd = ratio_of_variances(defl_fwd, N)
rov_bwd, idx_bwd = ratio_of_variances(defl_bwd, N)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(dist_fwd, rov_fwd)
ax.set_xlabel("Distance(forward)")
ax.set_ylabel("ROV")
plt.axvline(
    dist_fwd[N],
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
cp_fwd = dist_fwd[N + idx_fwd]
cp_bwd = dist_bwd[N + idx_bwd]
print(cp_fwd, cp_bwd)
# %%
# Translation
dist_fwd = dist_fwd - cp_fwd
dist_bwd = dist_bwd - cp_fwd

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
ax.plot(dist_fwd * 1e6, defl_processed_fwd * 1e9, label="forward")
ax.plot(dist_bwd * 1e6, defl_processed_bwd * 1e9, label="backward")
ax.set_xlabel("Distance(μm)")
ax.set_ylabel("Force(nN)")
plt.axvline(cp_fwd, color="grey", linestyle="--", linewidth=1)
ax.legend()
# %%
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
fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
axes[0].plot(time, indentation * 1e6)
axes[0].set_xlabel("Time(s)")
axes[1].set_xlabel("Time(s)")
axes[0].set_ylabel("Indentation(μm)")
axes[1].set_ylabel("Force(nN)")
axes[1].plot(time, force * 1e9)
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
negative_idx = np.where(F_ret < 0)[0]
negative_idx = negative_idx[0]
# %%
F_ret = F_ret[:negative_idx]
time_ret = time_ret[:negative_idx]


# %%
# PLR model fitting
def SLS_constit_integand(t_, t, E0, E_inf, tau, velocity, indentation, tip):
    a = tip.alpha
    b = tip.beta
    return (
        a
        * b
        * (E_inf + (E0 - E_inf) * np.exp(-(t - t_) / tau))
        * velocity(t_)
        * indentation(t_) ** (b - 1)
    )


def F_app_integral(t__, E0_, E_inf_, tau_, velocity_, indentation_, tip_):
    F = []
    for i in t__:
        integrand_ = partial(
            SLS_constit_integand,
            t=i,
            E0=E0_,
            E_inf=E_inf_,
            tau=tau_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F


def F_ret_integral(t__, t___, E0_, E_inf_, tau_, velocity_, indentation_, tip_):
    F = []
    for i, j in zip(t__, t___):
        integrand_ = partial(
            SLS_constit_integand,
            t=j,
            E0=E0_,
            E_inf=E_inf_,
            tau=tau_,
            velocity=velocity_,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, i)[0])
    return F


# %%
# Determination of Variable
tip = Spherical(0.8 * 1e-6)
E0 = 1e8
E_inf = 1e5
tau = 2
# %%
Force = F_app_integral(
    t__=time_app,
    E0_=E0,
    E_inf_=E_inf,
    tau_=tau,
    indentation_=indentation_app_func,
    velocity_=velocity_app_func,
    tip_=tip,
)

# Curve Fitting(PLR model) Approach Region
F_app_func = partial(
    F_app_integral,
    indentation_=indentation_app_func,
    velocity_=velocity_app_func,
    tip_=tip,
)
popt_app, pcov_app = curve_fit(F_app_func, time_app, F_app)
F_app_curvefit = np.array(F_app_func(time_app, *popt_app))
print(popt_app)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(time_app, F_app * 1e9, color="red")
ax.plot(time_app, F_app_curvefit * 1e9, color="blue")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")


# %%
# Find t1
def Integrand(t_, E0, E_inf, t, tau, velocity):
    return (E_inf + (E0 - E_inf) * np.exp(-(t - t_) / tau)) * velocity(t_)


# %%
def Quadrature(t1, t_, t_max_, E0_, E_inf_, tau_, velocity_app, velocity_ret):
    integrand_app = partial(
        Integrand, E0=E0_, t=t_, E_inf=E_inf_, tau=tau_, velocity=velocity_app
    )
    integrand_ret = partial(
        Integrand, E0=E0_, t=t_, E_inf=E_inf_, tau=tau_, velocity=velocity_ret
    )
    quad_app = quad(integrand_app, t1, t_max_)
    quad_ret = quad(integrand_ret, t_max_, t_)
    return quad_app[0] + quad_ret[0]


# %%
def Calculation_t1(
    time_ret, t_max__, E0__, E_inf__, tau__, velocity_app__, velocity_ret__
):
    t1 = []
    for i in time_ret:
        quadrature = partial(
            Quadrature,
            t_=i,
            t_max_=t_max__,
            E0_=E0__,
            E_inf_=E_inf__,
            tau_=tau__,
            velocity_app=velocity_app__,
            velocity_ret=velocity_ret__,
        )
        if quadrature(t1=0) * quadrature(t1=t_max__) > 0:
            t1.append(0)
        else:
            t_1 = root_scalar(quadrature, method="bisect", bracket=(0, t_max))
            t1.append(t_1.root)
    return t1


# %%
t1 = Calculation_t1(
    time_ret, t_max, E0, E_inf, tau, velocity_app_func, velocity_ret_func
)
print(t1)
# %%
# Test Retraction Region
tip = Spherical(0.8 * 1e-6)
E0 = popt_app[0]
E_inf = popt_app[1]
tau = popt_app[2]

# velocity_ret, indentation_ret은 t1에 해당하는 값을 못먹음(interpolation 범위를 넘어섬)
Force = F_ret_integral(
    t__=t1,
    t___=time_ret,
    E0_=E0,
    E_inf_=E_inf,
    tau_=tau,
    indentation_=indentation_app_func,
    velocity_=velocity_app_func,
    tip_=tip,
)
Force = np.array(Force)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(time_ret, Force * 1e9, color="red")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")


# %%
# Curve Fitting(PLR model) Retraction Region
def F_ret_integral_test(
    t__, E0_, E_inf_, tau_, indentation_, tip_, velocity_app, velocity_ret
):
    t1 = Calculation_t1(t__, t__[0], E0_, E_inf_, tau_, velocity_app, velocity_ret)
    F = []
    for i, j in zip(t__, t1):
        integrand_ = partial(
            SLS_constit_integand,
            t=i,
            E0=E0_,
            E_inf=E_inf_,
            tau=tau_,
            velocity=velocity_app,
            indentation=indentation_,
            tip=tip_,
        )
        F.append(quad(integrand_, 0, j)[0])
    return F


# %%
F_ret_integral_test(
    t__=time_ret,
    E0_=E0,
    E_inf_=E_inf,
    tau_=tau,
    indentation_=indentation_app_func,
    velocity_app=velocity_app_func,
    velocity_ret=velocity_ret_func,
    tip_=tip,
)
# %%
F_ret_func = partial(
    F_ret_integral_test,
    indentation_=indentation_app_func,
    velocity_app=velocity_app_func,
    velocity_ret=velocity_ret_func,
    tip_=tip,
)
popt_ret, pcov_ret = curve_fit(F_ret_func, time_ret, F_ret)
F_ret_curvefit = np.array(F_ret_func(time_ret, *popt_ret))
print(popt_ret)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(time_ret, F_ret * 1e9, color="red")
ax.plot(time_ret, F_ret_curvefit * 1e9, color="blue")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")


# %%
def F_total_integral(
    time, E0_, E_inf_, tau_, max_ind, indentation_, tip_, velocity_app, velocity_ret
):
    time_app = time[: max_ind + 1]
    time_ret = time[max_ind:]
    t1 = Calculation_t1(
        time_ret, time_ret[0], E0_, E_inf_, tau_, velocity_app, velocity_ret
    )
    F_app = []
    for i in time_app:
        integrand_ = partial(
            SLS_constit_integand,
            t=i,
            E0=E0_,
            E_inf=E_inf_,
            tau=tau_,
            velocity=velocity_app,
            indentation=indentation_,
            tip=tip_,
        )
        F_app.append(quad(integrand_, 0, i)[0])

    F_ret = []
    for i, j in zip(time_ret, t1):
        integrand_ = partial(
            SLS_constit_integand,
            t=i,
            E0=E0_,
            E_inf=E_inf_,
            tau=tau_,
            velocity=velocity_app,
            indentation=indentation_,
            tip=tip_,
        )
        F_ret.append(quad(integrand_, 0, j)[0])

    F = F_app[:-1] + F_ret
    return F


# %%
tip = Spherical(0.8 * 1e-6)
length = len(time_app) + len(time_ret) - 1
time = time[:length]
force = force[:length]
force_app_parameter = F_total_integral(
    time=time,
    max_ind=max_ind,
    E0_=popt_app[0],
    E_inf_=popt_app[1],
    tau_=popt_app[2],
    indentation_=indentation_app_func,
    velocity_app=velocity_app_func,
    velocity_ret=velocity_ret_func,
    tip_=tip,
)
force_ret_parameter = F_total_integral(
    time=time,
    max_ind=max_ind,
    E0_=popt_ret[0],
    E_inf_=popt_ret[1],
    tau_=popt_ret[2],
    indentation_=indentation_app_func,
    velocity_app=velocity_app_func,
    velocity_ret=velocity_ret_func,
    tip_=tip,
)
force_app_parameter = np.array(force_app_parameter)
force_ret_parameter = np.array(force_ret_parameter)
# %%
F_total_func = partial(
    F_total_integral,
    max_ind=max_ind,
    indentation_=indentation_app_func,
    velocity_app=velocity_app_func,
    velocity_ret=velocity_ret_func,
    tip_=tip,
)
# %%
popt_total, pcov_total = curve_fit(F_total_func, time, force)
F_total_curvefit = np.array(F_total_func(time, *popt_total))
print(popt_total)
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.plot(time, F_total_curvefit * 1e9, label="total curvefit")
ax.plot(time, force * 1e9, ".", label="experiment data")
ax.plot(time, force_app_parameter * 1e9, label="approach curvefit")
ax.plot(time, force_ret_parameter * 1e9, label="retraction curvefit")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")
ax.legend()
# %%
x = ["Approach", "Retraction", "Total"]
modulus_0 = [popt_app[0], popt_ret[0], popt_total[0]]
modulus_inf = [popt_app[1], popt_ret[1], popt_total[1]]
tau = [popt_app[2], popt_ret[2], popt_total[2]]
fig, axes = plt.subplots(1, 3, figsize=(15, 10))
axes[0].plot(x, modulus_0)
axes[1].plot(x, modulus_inf)
axes[2].plot(x, tau)
# %%
F_at_rp = F_app_integral(
    t__=time_app,
    E0_=popt_ret[0],
    E_inf_=popt_ret[1],
    tau_=popt_ret[2],
    indentation_=indentation_app_func,
    velocity_=velocity_app_func,
    tip_=tip,
)
F_at_tp = F_app_integral(
    t__=time_app,
    E0_=popt_total[0],
    E_inf_=popt_total[1],
    tau_=popt_total[2],
    indentation_=indentation_app_func,
    velocity_=velocity_app_func,
    tip_=tip,
)

t1_rt_ap = Calculation_t1(
    time_ret,
    t_max,
    popt_app[0],
    popt_app[1],
    popt_app[2],
    velocity_app_func,
    velocity_ret_func,
)
t1_rt_tp = Calculation_t1(
    time_ret,
    t_max,
    popt_total[0],
    popt_total[1],
    popt_total[2],
    velocity_app_func,
    velocity_ret_func,
)
F_rt_ap = F_ret_integral(
    t__=t1_rt_ap,
    t___=time_ret,
    E0_=popt_app[0],
    E_inf_=popt_app[1],
    tau_=popt_app[2],
    indentation_=indentation_app_func,
    velocity_=velocity_app_func,
    tip_=tip,
)
F_rt_tp = F_ret_integral(
    t__=t1_rt_tp,
    t___=time_ret,
    E0_=popt_total[0],
    E_inf_=popt_total[1],
    tau_=popt_total[2],
    indentation_=indentation_app_func,
    velocity_=velocity_app_func,
    tip_=tip,
)
# %%
MSE_apptime_appparams = np.sum(np.square(F_app_curvefit - F_app))
MSE_apptime_retparams = np.sum(np.square(F_at_rp - F_app))
MSE_apptime_totparams = np.sum(np.square(F_at_tp - F_app))

MSE_rettime_appparams = np.sum(np.square(F_rt_ap - F_ret))
MSE_rettime_retparams = np.sum(np.square(F_ret_curvefit - F_ret))
MSE_rettime_totparams = np.sum(np.square(F_rt_tp - F_ret))

MSE_tottime_totparams = np.sum(np.square(F_total_curvefit - force))
MSE_tottime_appparams = np.sum(np.square(force_app_parameter - force))
MSE_tottime_retparams = np.sum(np.square(force_ret_parameter - force))
MSE_tottime_totparams = np.sum(np.square(F_total_curvefit - force))
# %%
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
tot_time = ["app time - app params", "app time - ret params", "app time - tot params"]
MSE_app = [MSE_apptime_appparams, MSE_apptime_retparams, MSE_apptime_totparams]
ax.bar(tot_time, MSE_app)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
tot_time = ["ret time - app params", "ret time - ret params", "ret time - tot params"]
MSE_ret = [MSE_rettime_appparams, MSE_rettime_retparams, MSE_rettime_totparams]
ax.bar(tot_time, MSE_ret)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
tot_time = ["tot time - app params", "tot time - ret params", "tot time - tot params"]
MSE_tot = [MSE_tottime_appparams, MSE_tottime_retparams, MSE_tottime_totparams]
ax.bar(tot_time, MSE_tot)
