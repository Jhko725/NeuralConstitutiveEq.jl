from configparser import ConfigParser
from pathlib import Path

import bottleneck
import numpy as np
import scipy
import xarray as xr
from jhelabtoolkit.io.nanosurf import nanosurf
from numpy import ndarray
from numpy.polynomial.polynomial import Polynomial


def process_approach_data(nid_file: str | Path, contact_point: float, k: float):
    config, data = nanosurf.read_nid(nid_file)
    fs = get_sampling_rate(config)
    approach_data = data["spec forward"]
    z, defl = get_z_and_defl(approach_data)
    dist = z - defl
    baseline_poly = fit_baseline_polynomial(dist, defl, contact_point=contact_point)
    defl_corrected = defl - baseline_poly(dist)
    is_contact = dist >= contact_point
    indent, force = dist[is_contact], k * defl_corrected[is_contact]
    indent = indent - indent[0]
    force = force - force[0]
    time = np.arange(len(indent)) / fs
    return time, indent, force


def get_sampling_rate(nid_config: ConfigParser) -> float:
    spec_config = dict(nid_config[r"DataSet\DataSetInfos\Spec"])
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


def ratio_of_variances_numpy(
    deflection: ndarray, window_size: int
) -> tuple[ndarray, int]:
    """Numpy version of the ratio_of_variances function.

    Note that the performance difference between this function and the ratio_of_variances function is very large (~1000x).
    Therefore, this function is kept only for pedagogical purposes.

    Args:
        deflection: 1D numpy array containing the deflection data.
        window_size: size of the window used for the variance calculations.

    Returns:
        rov: 1D numpy with the same length as deflection, containing the ratio of variances for each point in the deflection data.
         The first and last N locations are set to NaN, as they correspond to regions with less samples than the window size.
        idx: Index corresponding to the maximum of the rov array.
    """
    defl, N = deflection.flatten(), window_size
    rov = np.full(defl.shape, np.nan)
    for i in range(N, len(defl) - N):
        rov[i] = np.var(defl[i + 1 : i + N + 1]) / np.var(defl[i - N : i])
    idx = np.nanargmax(rov)
    return rov, idx


def ratio_of_variances(deflection: ndarray, window_size: int) -> tuple[ndarray, int]:
    """Returns the ratio of variances of the given deflection array, as well as the index corresponding to the maximum value.

    Implements the ratio of variances method for the contact point determination of AFM force indentation data.

    Args:
        deflection: 1D numpy array containing the deflection data.
        window_size: size of the window used for the variance calculations.

    Returns:
        rov: 1D numpy with the same length as deflection, containing the ratio of variances for each point in the deflection data.
         The first and last N locations are set to NaN, as they correspond to regions with less samples than the window size.
        idx: Index corresponding to the maximum of the rov array.
    """
    defl, N = deflection.flatten(), window_size
    vars_ = bottleneck.move_var(defl, N)[N - 1 :]
    rov = np.full(defl.shape, np.nan)
    rov[N:-N] = vars_[N + 1 :] / vars_[: -N - 1]
    idx = bottleneck.nanargmax(rov)
    return rov, idx


def fit_baseline_polynomial(
    distance: ndarray, deflection: ndarray, contact_point: float = 0.0, degree: int = 1
) -> Polynomial:
    pre_contact = distance <= contact_point
    domain = (np.amin(distance), np.amax(distance))
    return Polynomial.fit(
        distance[pre_contact], deflection[pre_contact], deg=degree, domain=domain
    )


def estimate_derivative(x: ndarray, y: ndarray) -> ndarray:
    smoothing_spline = scipy.interpolate.make_smoothing_spline(x, y)
    Dspline = smoothing_spline.derivative()
    return Dspline(x)
