#%%
from configparser import ConfigParser
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import integrate
from numpy import ndarray
import xarray as xr
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from jhelabtoolkit.io.nanosurf import nanosurf
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults

configure_matplotlib_defaults()

filepath = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(10nN, 1s, liquid).nid"
config, data = nanosurf.read_nid(filepath)
# %%
def get_sampling_rate(nid_config: ConfigParser) -> float:
    spec_config = dict(nid_config[r"DataSet\DataSetInfos\Spec"])
    num_points = int(spec_config["data points"])
    # May later use the pint library to parse unitful quantites
    modulation_time = float(spec_config["modulation time"].split(" ")[0])
    return num_points / modulation_time
#%%
get_sampling_rate(config)
#%%
forward, backward = data["spec forward"],data["spec backward"]
forward
# %%
def get_z_and_defl(spectroscopy_data: xr.DataArray) -> tuple[ndarray, ndarray]:
    piezo_z = spectroscopy_data["z-axis sensor"].to_numpy()
    defl = spectroscopy_data["deflection"].to_numpy()
    return piezo_z.squeeze(), defl.squeeze()

def calc_tip_distance(piezo_z_pos: ndarray, deflection: ndarray) -> ndarray:
    return piezo_z_pos - deflection

def contact_point_ROV(deflection: ndarray, N: int) -> ndarray:
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

def hysteresis(
        dist_fwd: ndarray, dist_bwd: ndarray, defl_fwd: ndarray, defl_bwd: ndarray, ROV_window: int
        ) -> float:
    cp = contact_point_ROV(defl_fwd, ROV_window)
    cp_idx = cp[1]
    cp_fwd = dist_fwd[ROV_window+cp_idx]
    baseline_poly_fwd = fit_baseline_polynomial(dist_fwd, defl_fwd, cp_fwd)
    defl_processed_fwd = defl_fwd - baseline_poly_fwd(dist_fwd)
    baseline_poly_bwd = fit_baseline_polynomial(dist_bwd, defl_bwd, cp_fwd)
    defl_processed_bwd = defl_bwd - baseline_poly_bwd(dist_bwd)
    idx_fwd = dist_fwd.argsort()
    dist_fwd = dist_fwd[idx_fwd]
    defl_fwd = defl_fwd[idx_fwd]
    idx_bwd = dist_bwd.argsort()
    dist_bwd = dist_bwd[idx_bwd]
    defl_bwd = defl_bwd[idx_bwd]
    interp_start = max(np.min(dist_fwd), np.min(dist_bwd))
    interp_end = min(np.max(dist_fwd), np.max(dist_bwd))
    cubic_spline_fwd = CubicSpline(dist_fwd, defl_processed_fwd)
    cubic_spline_bwd = CubicSpline(dist_bwd, defl_processed_bwd)
    dist_interp = np.linspace(interp_start, interp_end, 1000)
    interp_fwd=cubic_spline_fwd(dist_interp)
    interp_bwd=cubic_spline_bwd(dist_interp)
    area_functinon = cubic_spline_fwd(dist_bwd)-defl_processed_bwd
    area = integrate.simpson(area_functinon, dist_bwd)
    area_fwd = integrate.simpson(cubic_spline_fwd(dist_bwd), dist_bwd)
    area_bwd = integrate.simpson(defl_processed_bwd, dist_bwd)
    return area, area_fwd, area_bwd
#%%
z_fwd, defl_fwd = get_z_and_defl(forward)
z_bwd, defl_bwd = get_z_and_defl(backward)
dist_fwd = calc_tip_distance(z_fwd, defl_fwd)
dist_bwd = calc_tip_distance(z_bwd, defl_bwd)
#%%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd, defl_fwd, label="forward")
ax.plot(dist_bwd, defl_bwd, label="backward")
ax.legend()
#%%
# hysteresis(dist_fwd, dist_bwd, defl_fwd, defl_bwd, 4)
#%%
# Find contact point
N = 4
cp_fwd = contact_point_ROV(defl_fwd, N)
cp_bwd = contact_point_ROV(defl_bwd, N)
cp_fwd_idx = cp_fwd[1]
cp_bwd_idx = cp_bwd[1]
cp_fwd = dist_fwd[N+cp_fwd_idx]
cp_bwd = dist_bwd[N+cp_bwd_idx]
print(len(z_fwd),len(defl_fwd),len(z_bwd),len(defl_bwd))
print(cp_fwd_idx)
# %%
# Polynomial fitting
baseline_poly_fwd = fit_baseline_polynomial(dist_fwd, defl_fwd, cp_fwd)
defl_processed_fwd = defl_fwd - baseline_poly_fwd(dist_fwd)
baseline_poly_bwd = fit_baseline_polynomial(dist_bwd, defl_bwd, cp_bwd)
defl_processed_bwd = defl_bwd - baseline_poly_bwd(dist_bwd)
#%%
# Data align
idx_fwd = dist_fwd.argsort()
dist_fwd = dist_fwd[idx_fwd]
defl_fwd = defl_fwd[idx_fwd]
idx_bwd = dist_bwd.argsort()
dist_bwd = dist_bwd[idx_bwd]
difl_bwd = defl_bwd[idx_bwd]
#%%
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.plot(dist_fwd, defl_processed_fwd, label="forward")
ax.plot(dist_bwd, defl_processed_bwd, label="backward")
plt.axvline(cp_fwd, color="grey", linestyle="--", linewidth=1)
plt.axvline(cp_bwd, color="grey", linestyle="--", linewidth=1)
ax.legend()
#%%
# barycentric_fwd = BarycentricInterpolator(dist_fwd, defl_processed_fwd)
# barycentric_bwd = BarycentricInterpolator(dist_bwd, defl_processed_bwd)
cubic_spline_fwd = CubicSpline(dist_fwd, defl_processed_fwd)
cubic_spline_bwd = CubicSpline(dist_bwd, defl_processed_bwd)
# pchip_fwd = PchipInterpolator(dist_fwd, defl_processed_fwd)
# pchip_bwd = PchipInterpolator(dist_bwd, defl_processed_bwd)
# akima_fwd = Akima1DInterpolator(dist_fwd, defl_processed_fwd)
#%%
fig, ax = plt.subplots(1, 1, figsize = (15, 10))
interp_fwd = cubic_spline_fwd(dist_bwd)
ax.plot(dist_bwd, interp_fwd, label="interpolation_fwd")
# ax.plot(dist_fwd, defl_processed_fwd, label="origianl_fwd")
ax.plot(dist_bwd, defl_processed_bwd, label = "original_bwd")
ax.legend()
print(len(defl_processed_fwd), len(defl_processed_bwd), len(cubic_spline_fwd(dist_bwd)))
#%%
#%%
interp_start = max(np.min(dist_fwd), np.min(dist_bwd))
interp_end = min(np.max(dist_fwd), np.max(dist_bwd))
dist_interp = np.linspace(interp_start, interp_end, 1000)
interp_fwd=cubic_spline_fwd(dist_interp)
interp_bwd=cubic_spline_bwd(dist_interp)
fig, ax = plt.subplots(1, 1, figsize = (15, 10))
ax.plot(dist_interp, interp_fwd, label = "forward (interp)")
ax.plot(dist_interp, interp_bwd, label = "backward (interp)")
ax.legend()
#%%
area_functinon = cubic_spline_fwd(dist_bwd)-defl_processed_bwd
area = integrate.simpson(area_functinon, dist_bwd)
area_fwd = integrate.simpson(cubic_spline_fwd(dist_bwd), dist_bwd)
area_bwd = integrate.simpson(defl_processed_bwd, dist_bwd)
print(area, area_fwd, area_bwd)
#%%

#%%








# Truncation
dist_fwd = dist_fwd[cp_idx+N:]
dist_bwd = dist_bwd[cp_idx+N:]
defl_fwd = defl_fwd[cp_idx+N:]
defl_bwd = defl_bwd[cp_idx+N:]
print(len(dist_fwd), len(dist_bwd), len(defl_fwd), len(defl_bwd))
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd, defl_fwd, label="forward")
ax.plot(dist_bwd, defl_bwd, label="backward")
ax.legend()
# %%
# Translation
dist_fwd = dist_fwd - dist_fwd[cp_idx]
dist_bwd = dist_bwd - dist_bwd[cp_idx]
# %%
