#%%
import pandas as pd
import matplotlib.pyplot as plt
from configparser import ConfigParser

import numpy as np
from numpy import ndarray
import xarray as xr
import kneed
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from jhelabtoolkit.io.nanosurf import nanosurf
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults

configure_matplotlib_defaults()

filepath = "data/20231110_onion/Image02676.nid"
config, data = nanosurf.read_nid(filepath)

# %%
onion = pd.read_csv("stress relaxation.csv")
# %%
onion.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# %%
onion
# %%
fig, ax = plt.subplots(1,1, figsize=(10,7))
ax.plot(onion[0], onion[1])
ax.plot(onion[3], onion[4])
ax.plot(onion[6] ,onion[7])
# %%

# %%
def get_z_and_defl(spectroscopy_data: xr.DataArray) -> tuple[ndarray, ndarray]:
    piezo_z = spectroscopy_data["z-axis sensor"].to_numpy()
    defl = spectroscopy_data["deflection"].to_numpy()
    return piezo_z.squeeze(), defl.squeeze()

def get_sampling_rate(nid_config: ConfigParser) -> float:
    spec_config = dict(config[r"DataSet\DataSetInfos\Spec"])
    num_points = int(spec_config["data points"])
    # May later use the pint library to parse unitful quantites
    modulation_time = float(spec_config["modulation time"].split(" ")[0])
    return num_points / modulation_time

get_sampling_rate(config)
# %%
forward, backward = data["spec forward"], data["spec backward"]
pause = data["spec fwd pause"]
#%%
pause

# %%
z_pause, defl_pause = get_z_and_defl(pause)

time = np.linspace(0,300,65534)
time += 100
time

# %%
fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot(time, defl_pause/defl_pause[0])
ax.plot(onion[0], onion[1])
ax.plot(onion[3], onion[4])
ax.plot(onion[6] ,onion[7])
# ax.plot(time, z_pause)
# %%
