# %%
from configparser import ConfigParser
from functools import partial
import numpy as np
from numpy import ndarray
import xarray as xr
import kneed
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from jhelabtoolkit.io.nanosurf import nanosurf
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults
from neuralconstitutive.preprocessing import process_approach_data, estimate_derivative
from neuralconstitutive.tipgeometry import Spherical
from scipy.optimize import curve_fit
configure_matplotlib_defaults()

filepath = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(10nN, 1s, liquid).nid"
config, data = nanosurf.read_nid(filepath)
# %%
def get_sampling_rate(nid_config: ConfigParser) -> float:
    spec_config = dict(config[r"DataSet\DataSetInfos\Spec"])
    num_points = int(spec_config["data points"])
    # May later use the pint library to parse unitful quantites
    modulation_time = float(spec_config["modulation time"].split(" ")[0])
    return num_points / modulation_time

get_sampling_rate(config)
# %%
forward, backward = data["spec forward"], data["spec backward"]

forward
# %%
def get_z_and_defl(spectroscopy_data: xr.DataArray) -> tuple[ndarray, ndarray]:
    piezo_z = spectroscopy_data["z-axis sensor"].to_numpy()
    defl = spectroscopy_data["deflection"].to_numpy()
    return piezo_z.squeeze(), defl.squeeze()


def calc_tip_distance(piezo_z_pos: ndarray, deflection: ndarray) -> ndarray:
    return piezo_z_pos - deflection


def find_contact_point(distance: ndarray, deflection: ndarray) -> float:
    # Right now, only support 1D arrays of tip_distance and tip_deflection
    locator = kneed.KneeLocator(
        distance,
        deflection,
        S=10,
        curve="convex",
        direction="increasing",
        interp_method="polynomial",
        polynomial_degree=7,
    )
    return locator.knee


def find_contact_point1(deflection: ndarray, N: int) -> ndarray:
    # Ratio of Variance
    rov = np.array([])
    length = np.arange(np.size(deflection))
    rov = np.array(
        [
            np.append(
                rov,
                np.array(
                    [
                        np.var(deflection[i + 1 : i + N])
                        / np.var(deflection[i - N : i - 1])
                    ]
                ),
            )
            for i in length
        ]
    ).flatten()
    rov = rov[N : np.size(rov) - N]
    idx = np.argmax(rov)
    return rov, idx, rov[idx]


def fit_baseline_polynomial(
    distance: ndarray, deflection: ndarray, contact_point: float = 0.0, degree: int = 1
) -> Polynomial:
    pre_contact = distance < contact_point
    domain = (np.amin(distance), np.amax(distance))
    return Polynomial.fit(
        distance[pre_contact], deflection[pre_contact], deg=degree, domain=domain
    )
# %%
z_fwd, defl_fwd = get_z_and_defl(forward)
z_bwd, defl_bwd = get_z_and_defl(backward)
dist_fwd = calc_tip_distance(z_fwd, defl_fwd)
dist_bwd = calc_tip_distance(z_bwd, defl_bwd)
cp = find_contact_point(dist_fwd, defl_fwd)
# %%
cp
# %%
# ROV method
N = 10
rov_fwd = find_contact_point1(defl_fwd, N)[0]
idx_fwd = find_contact_point1(defl_fwd, N)[1]
rov_fwd_max = find_contact_point1(defl_fwd, N)[2]

rov_bwd = find_contact_point1(defl_bwd, N)[0]
idx_bwd = find_contact_point1(defl_bwd, N)[1]
rov_bwd_max = find_contact_point1(defl_bwd, N)[2]
# %%
fig, ax = plt.subplots(1, 1, figsize = (5, 3))
ax.plot(dist_fwd[N:np.size(dist_fwd)-N], find_contact_point1(defl_fwd, N)[0])
ax.set_xlabel("Distance(forward)")
ax.set_ylabel("ROV")
plt.axvline(dist_fwd[N+idx_fwd], color="black", linestyle="--", linewidth=1.5, label="maximum point")
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd*1e6, defl_fwd*1e9, label="forward")
ax.plot(dist_bwd*1e6, defl_bwd*1e9, label="backward")
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
ax.plot(dist_fwd*1e6, defl_fwd*1e9, label="forward")
ax.plot(dist_bwd*1e6, defl_bwd*1e9, label="backward")
ax.set_xlabel("Distance(μm)")
ax.set_ylabel("Force(nN)")
plt.axvline(cp_fwd, color="grey", linestyle="--", linewidth=1)
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd*1e6, defl_processed_fwd*1e9, label="forward")
ax.plot(dist_bwd*1e6, defl_processed_bwd*1e9, label="backward")
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
sampling_rate = get_sampling_rate(config)
time = np.arange(len(indentation)) / sampling_rate
print(len(time))
fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
axes[0].plot(time, indentation*1e6)
axes[0].set_xlabel("Time(s)")
axes[1].set_xlabel("Time(s)")
axes[0].set_ylabel("Indentation(μm)")
axes[1].set_ylabel("Force(nN)")
axes[1].plot(time, force*1e9)
# %%
max_ind = np.argmax(indentation)
t_max = time[max_ind]
indent_max = indentation[max_ind]
#%%
F_app = force[:max_ind+1]
F_ret = force[max_ind:]
#%%
# t_max 부분을 겹치게 해야 문제가 안생김
indentation_app = indentation[:max_ind+1]
indentation_ret = indentation[max_ind:]

time_app = time[:max_ind+1]
time_ret = time[max_ind:]

velocity_app = estimate_derivative(time_app, indentation_app)
velocity_ret = estimate_derivative(time_ret, indentation_ret)

indentation_app_func = interp1d(time_app, indentation_app)
indentation_ret_func = interp1d(time_ret, indentation_ret)
velocity_app_func = interp1d(time_app, velocity_app)
velocity_ret_func = interp1d(time_ret, velocity_ret)
# %%
# PLR model fitting
def PLR_constit_integand(t_, t, E0, alpha, t_prime, velocity, indentation, tip):
    a = tip.alpha
    b = tip.beta
    return a * b * E0 * (1 + (t-t_)/t_prime)**(-alpha) * velocity(t_) * indentation(t_)**(b-1)

def F_app_integral(t__, E0_, alpha_, t_prime_, velocity_, indentation_, tip_):
    F = []
    for i in t__:
        integrand_ = partial(
            PLR_constit_integand, 
            t=i,
            E0=E0_, 
            alpha=alpha_, 
            t_prime=t_prime_, 
            velocity=velocity_, 
            indentation=indentation_,
            tip=tip_)
        F.append(quad(integrand_, 0, i)[0])
    return F

def F_ret_integral(t__, t___, E0_, alpha_, t_prime_, velocity_, indentation_, tip_):
    F = []
    for i, j in zip(t__, t___):
        integrand_ = partial(
            PLR_constit_integand, 
            t=j,
            E0=E0_, 
            alpha=alpha_, 
            t_prime=t_prime_, 
            velocity=velocity_, 
            indentation=indentation_,
            tip=tip_)
        F.append(quad(integrand_, 0, i)[0])
    return F
#%%
# Determination of Variable
tip = Spherical(0.8*1e-6)
E0 = 0.562
alpha = 0.2
time = time_app
Force = F_app_integral(t__= time, E0_= E0, alpha_=alpha, t_prime_=1e-5, indentation_=indentation_app_func, velocity_=velocity_app_func, tip_=tip)

# Curve Fitting(PLR model) Approach Region
F_app_func = partial(F_app_integral, t_prime_=1e-5, indentation_=indentation_app_func, velocity_=velocity_app_func, tip_=tip)
popt, pcov = curve_fit(F_app_func, time_app, F_app)
F_app_curvefit = np.array(F_app_func(time, *popt))
print(popt)
# %%
fig, ax = plt.subplots(1,1, figsize=(7,5))
ax.plot(time, F_app*1e9, color="red")
ax.plot(time, F_app_curvefit*1e9, color="blue")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")
# %%
# Find t1
def Integrand(t_, E0, t, t_prime, alpha, velocity):
    return (E0 * (1 + (t-t_)/t_prime) ** (-alpha)) * velocity(t_)
#%%
def Quadrature(t1, t_, t_max_, E0_, t_prime_, alpha_, velocity_app, velocity_ret):
    integrand_app = partial(Integrand, E0=E0_, t=t_, t_prime=t_prime_, alpha=alpha_, velocity=velocity_app)
    integrand_ret = partial(Integrand, E0=E0_, t=t_, t_prime=t_prime_, alpha=alpha_, velocity=velocity_ret)
    quad_app = quad(integrand_app, t1, t_max_)
    quad_ret = quad(integrand_ret, t_max_, t_)
    return quad_app[0] + quad_ret[0]
#%%
def Calculation_t1(time_ret, t_max__, E0__, t_prime__, alpha__, velocity_app__, velocity_ret__):
    t1 = []
    for i in time_ret:
        quadrature = partial(Quadrature, 
                        t_=i, 
                        t_max_=t_max__,
                        E0_=E0__,
                        t_prime_ = t_prime__,
                        alpha_ = alpha__,
                        velocity_app=velocity_app__,
                        velocity_ret=velocity_ret__)
        if quadrature(t1=0)*quadrature(t1=t_max__) > 0:
            t1.append(0)       
        else :
            t_1 = root_scalar(quadrature, method='bisect', bracket=(0, t_max))
            t1.append(t_1.root)
    return t1
# %%
t1 = Calculation_t1(time_ret, t_max, E0, 1e-5, 0.2, velocity_app_func, velocity_ret_func)
print(t1)
# %%
# Test Retraction Region
tip = Spherical(0.8*1e-6)
E0 = popt[0]
alpha = popt[1]
time = time_ret

# velocity_ret, indentation_ret은 t1에 해당하는 값을 못먹음(interpolation 범위를 넘어섬)
Force = F_ret_integral(t__= t1, t___ = time_ret, E0_= E0, alpha_=alpha, t_prime_=1e-5, indentation_=indentation_app_func, velocity_=velocity_app_func, tip_=tip)
Force = np.array(Force)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot(time_ret, Force*1e9, color="red")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Force(nN)")
#%%
# Curve Fitting(PLR model) Retraction Region
def F_ret_integral_test(t__, E0_, alpha_, t_prime_, indentation_, tip_, velocity_app, velocity_ret):
    t1 = Calculation_t1(t__, t__[0], E0_, t_prime_, alpha_, velocity_app, velocity_ret)
    F = []
    for i, j in zip(t__, t1):
        integrand_ = partial(
            PLR_constit_integand, 
            t=i,
            E0=E0_, 
            alpha=alpha_, 
            t_prime=t_prime_, 
            velocity=velocity_app, 
            indentation=indentation_,
            tip=tip_)
        F.append(quad(integrand_, 0, j)[0])
    return F
#%%
F_ret_integral_test(
    t__ = time_ret, 
    E0_= E0, 
    alpha_=alpha, 
    t_prime_=1e-5, 
    indentation_=indentation_app_func,
    velocity_app=velocity_app_func, 
    velocity_ret=velocity_ret_func, 
    tip_=tip) 
# %%
F_ret_func = partial(F_ret_integral_test, t_prime_=1e-5, indentation_=indentation_app_func, velocity_app=velocity_app_func, velocity_ret=velocity_ret_func, tip_=tip)
popt, pcov = curve_fit(F_ret_func, time_ret, F_ret)
F_ret_curvefit = np.array(F_ret_func(time, *popt))
print(popt)
# %%
