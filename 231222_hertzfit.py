# %%
from configparser import ConfigParser
from functools import partial

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np
from numpy import ndarray
import xarray as xr
import kneed
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from jhelabtoolkit.io.nanosurf import nanosurf
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults

configure_matplotlib_defaults()

filepath = "/home/paul/Projects/NeuralConstitutiveEq.jl/data/231214_collagen/Image00076.nid"
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
#%%

# %%
def get_z_and_defl(spectroscopy_data: xr.DataArray) -> tuple[ndarray, ndarray]:
    piezo_z = spectroscopy_data["z-axis sensor"].to_numpy()
    defl = spectroscopy_data["deflection"].to_numpy()
    return piezo_z.squeeze(), defl.squeeze()


def calc_tip_distance(piezo_z_pos: ndarray, deflection: ndarray) -> ndarray:
    return piezo_z_pos - deflection

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
# %%
# ROV method
N = 8
rov_fwd = find_contact_point1(defl_fwd, N)[0]
idx_fwd = find_contact_point1(defl_fwd, N)[1]
rov_fwd_max = find_contact_point1(defl_fwd, N)[2]

rov_bwd = find_contact_point1(defl_bwd, N)[0]
idx_bwd = find_contact_point1(defl_bwd, N)[1]
rov_bwd_max = find_contact_point1(defl_bwd, N)[2]
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd[N : np.size(dist_fwd) - N], find_contact_point1(defl_fwd, N)[0])
ax.set_xlabel("Distance(forward)")
ax.set_ylabel("ROV")
# %%
# Find contact point
cp_fwd = dist_fwd[N + idx_fwd]
cp_bwd = dist_bwd[N + idx_bwd]
print(cp_fwd, cp_bwd)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd* 1e6, defl_fwd * 1e9, label="forward")
ax.plot(dist_bwd* 1e6, defl_bwd * 1e9, label="backward")
plt.axvline(cp_fwd * 1e6, color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("Distance[μm]")
ax.set_ylabel("Force[nN]")
ax.legend()
#%%
# Translation
dist_fwd = dist_fwd - cp_fwd
dist_bwd = dist_bwd - cp_fwd
# %%
# Polynomial fitting
baseline_poly_fwd = fit_baseline_polynomial(dist_fwd, defl_fwd)
defl_processed_fwd = defl_fwd - baseline_poly_fwd(dist_fwd)
baseline_poly_bwd = fit_baseline_polynomial(dist_bwd, defl_bwd)
defl_processed_bwd = defl_bwd - baseline_poly_bwd(dist_bwd)

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd * 1e6, defl_fwd * 1e9, label="forward")
ax.plot(dist_bwd * 1e6, defl_bwd * 1e9, label="backward")
ax.set_xlabel("Distance[μm]")
ax.set_ylabel("Force[nN]")
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd * 1e6, defl_processed_fwd * 1e9, label="forward")
ax.plot(dist_bwd * 1e6, defl_processed_bwd * 1e9, label="backward")
plt.axvline(0, color="grey", linestyle="--", linewidth=1)
ax.set_xlabel("Distance[μm]")
ax.set_ylabel("Force[nN]")
ax.legend()

#%%
indentation_idx = dist_fwd >= 0
indentation_fwd = dist_fwd[indentation_idx] 
# k = 0.2
force_fwd = defl_fwd[indentation_idx]
#%%
def Hertzian_para(I, E, v, R):
    return 4.0/3.0*E*R**(1/2)*I**(3/2)/(1-v**2)

def Hertzian_cone(I, E, I0, F0, v, theta):
    I_ = np.clip(I-I0, 0, np.inf)
    return 2.0/np.pi*E*np.tan(theta)*I_**2 - F0


hertzian_fit_para = partial(
    Hertzian_para, 
    v=0.4,
    R=10.0*1e-9,)
popt_para, pocv_para = curve_fit(hertzian_fit_para, indentation_fwd, force_fwd)


hertzian_fit_cone = partial(
    Hertzian_cone,
    v=0.4,
    theta=20/180*np.pi,
)
popt_cone, pcov_cone = curve_fit(hertzian_fit_cone, indentation_fwd, force_fwd, p0=[1,0,0])
r2 = r2_score(force_fwd, hertzian_fit_cone(indentation_fwd, *popt_cone))


print(popt_para, popt_cone)

#%%
fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot(indentation_fwd * 1e6, force_fwd * 1e9, label="Experimental Data")
ax.plot(indentation_fwd * 1e6, hertzian_fit_cone(indentation_fwd, *popt_cone) * 1e9, label="Hertzian Fit") 
#ax.plot(indentation_fwd * 1e6, hertzian_fit_para(indentation_fwd, *popt_para) *1e9, label="Hertzian Fit_para")
ax.set_xlabel("Indentation[μm]")
ax.set_ylabel("Force[nN]")
ax.legend()
ax.set_title(f"r^2 = {r2}")
#%%