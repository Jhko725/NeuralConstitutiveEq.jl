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

def read_nid_file(filepath):
    config, data = nanosurf.read_nid(filepath)
    return config, data

def get_sampling_rate(nid_config: ConfigParser) -> float:
    spec_config = dict(config[r"DataSet\DataSetInfos\Spec"])
    num_points = int(spec_config["data points"])
    # May later use the pint library to parse unitful quantites
    modulation_time = float(spec_config["modulation time"].split(" ")[0])
    return num_points / modulation_time

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
    distance: np.ndarray, deflection: np.ndarray, contact_point: float = 0.0, degree: int = 1
) -> Polynomial:
    pre_contact = distance < contact_point
    domain = (np.amin(distance), np.amax(distance))
    return Polynomial.fit(
        distance[pre_contact], deflection[pre_contact], deg=degree, domain=domain)

def hysteresis(filepath: str, ROV_window: int) -> float:
    config, data = read_nid_file(filepath)
    forward, backward = data["spec forward"],data["spec backward"]
    z_fwd, defl_fwd = get_z_and_defl(forward) 
    z_bwd, defl_bwd = get_z_and_defl(backward) 
    dist_fwd = calc_tip_distance(z_fwd, defl_fwd) 
    dist_bwd = calc_tip_distance(z_bwd, defl_bwd) 
    cp_fwd = contact_point_ROV(defl_fwd, ROV_window)
    cp_bwd = contact_point_ROV(defl_bwd, ROV_window)
    cp_fwd_idx = cp_fwd[1]
    cp_bwd_idx = cp_bwd[1]
    cp_fwd = dist_fwd[ROV_window+cp_fwd_idx]
    cp_bwd = dist_bwd[ROV_window+cp_bwd_idx]
    baseline_poly_fwd = fit_baseline_polynomial(dist_fwd, defl_fwd, cp_fwd)
    defl_processed_fwd = defl_fwd - baseline_poly_fwd(dist_fwd)
    baseline_poly_bwd = fit_baseline_polynomial(dist_bwd, defl_bwd, cp_bwd)
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
    area_functinon = interp_fwd - interp_bwd
    area = integrate.simpson(area_functinon, dist_interp)
    fig, ax = plt.subplots(1, 1, figsize = (15, 10))
    ax.plot(dist_interp * 1e6, interp_fwd * 1e9, label = "forward (interpolation)")
    ax.plot(dist_interp * 1e6, interp_bwd * 1e9, label = "backward (interpolation)")
    ax.set_xlabel("Distance(μm)")
    ax.set_ylabel("Force(nN)")
    plt.axvline(cp_fwd * 1e6, color="grey", linestyle="--", linewidth=1)
    plt.axvline(cp_bwd * 1e6, color="grey", linestyle="--", linewidth=1)
    ax.legend()
    return area