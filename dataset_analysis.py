#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
df_raw = pd.read_csv('open_data/PAAM hydrogel/PAA_speed 5_4nN.tab', sep='\t', skiprows=34)
df_meta = pd.read_csv('open_data/PAAM hydrogel/PAA_speed 5_4nN.tsv', sep='\t')
# %%
# Assign raw data
cp = -df_meta['Contact Point [nm]'].to_numpy() * 1e-9

tip_position = -df_raw['tip position'].to_numpy()
tip_height_pz = df_raw['height (piezo)'].to_numpy()
tip_height_measured = df_raw['height (measured)'].to_numpy()

force = df_raw['force'].to_numpy()
time = df_raw['time'].to_numpy()
#%%
# Visualization of force curve
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(tip_position*1e6, force*1e9)
axes[0].axvline(cp*1e6, linestyle='--', color="grey", label="contact point")
axes[0].set_xlabel("Indentation[μm]")
axes[0].set_ylabel("Force[nN]")
axes[0].legend()

axes[1].plot(time, tip_position*1e6)
axes[1].set_xlabel("Time[s]")
axes[1].set_ylabel("Indentation[μm]")

axes[2].plot(time, force*1e9)
axes[2].set_xlabel("Time[s]")
axes[2].set_ylabel("Force[nN]")
#%%
tip_position = tip_position-cp
idx = np.where(0<=tip_position)
indentation = tip_position[idx]
force = force[idx]
time = time[idx]
#%%
time = time - time[0]
#%%
fig, axes= plt.subplots(1, 2, figsize=(15,10))
axes[0].plot(time, force*1e9)
axes[0].set_xlabel("Time[s]")
axes[0].set_ylabel("Force[nN]")

axes[1].plot(time, indentation*1e6)
axes[1].set_xlabel("Time[s]")
axes[1].set_ylabel("Indentation[μm]")

# %%
idx_max = np.argmax(force)
force_app = force[:idx_max]
force_ret = force[idx_max:]

time_app = time[:idx_max]
time_ret = time[idx_max:]

indentation_app = indentation[:idx_max]
indentation_ret = indentation[idx_max:]

#%%
fig, axes= plt.subplots(1, 2, figsize=(15,10))
axes[0].plot(time_app, force_app*1e9)
axes[0].set_xlabel("Time[s]")
axes[0].set_ylabel("Force[nN]")

axes[1].plot(time_app, indentation_app*1e6)
axes[1].set_xlabel("Time[s]")
axes[1].set_ylabel("Indentation[μm]")

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(indentation_app*1e6, force_app*1e9)
ax.set_xlabel("Indentation[μm]")
ax.set_ylabel("Force[nN]")
# %%
