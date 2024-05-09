#%%
import matplotlib.pyplot as plt
from hysteresis_final import hysteresis
#%%
# 10nN(1s, 5s, 10s, 20s, 60s)
filepath1 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(10nN, 1s, liquid).nid"
filepath2 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(10nN, 5s, liquid).nid"
filepath3 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(10nN, 10s, liquid).nid"
filepath4 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(10nN, 20s, liquid).nid"
filepath5 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(10nN, 60s, liquid).nid"
setpoint_10nN_1s = hysteresis(filepath1, 4)
setpoint_10nN_5s = hysteresis(filepath2, 4)
setpoint_10nN_10s = hysteresis(filepath3, 4)
setpoint_10nN_20s = hysteresis(filepath4, 4)
setpoint_10nN_60s = hysteresis(filepath5, 4)
print(setpoint_10nN_1s, setpoint_10nN_5s, setpoint_10nN_10s, setpoint_10nN_20s, setpoint_10nN_60s)
#%%
modulation = [1, 5, 10, 20, 60]
#%%
setpoint_10nN = []
setpoint_10nN.append(setpoint_10nN_1s)
setpoint_10nN.append(setpoint_10nN_5s)
setpoint_10nN.append(setpoint_10nN_10s)
setpoint_10nN.append(setpoint_10nN_20s)
setpoint_10nN.append(setpoint_10nN_60s)
setpoint_10nN
#%%
# 30nN(1s, 5s, 10s, 20s, 60s)
filepath1 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(30nN, 1s, liquid).nid"
filepath2 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(30nN, 5s, liquid).nid"
filepath3 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(30nN, 10s, liquid).nid"
filepath4 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(30nN, 20s, liquid).nid"
filepath5 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(30nN, 60s, liquid).nid"
setpoint_30nN_1s = hysteresis(filepath1, 4)
setpoint_30nN_5s = hysteresis(filepath2, 4)
setpoint_30nN_10s = hysteresis(filepath3, 4)
setpoint_30nN_20s = hysteresis(filepath4, 4)
setpoint_30nN_60s = hysteresis(filepath5, 4)
print(setpoint_30nN_1s, setpoint_30nN_5s, setpoint_30nN_10s, setpoint_30nN_20s, setpoint_30nN_60s)
#%%
setpoint_30nN = []
setpoint_30nN.append(setpoint_30nN_1s)
setpoint_30nN.append(setpoint_30nN_5s)
setpoint_30nN.append(setpoint_30nN_10s)
setpoint_30nN.append(setpoint_30nN_20s)
setpoint_30nN.append(setpoint_30nN_60s)
setpoint_30nN
#%%
# 50nN(1s, 5s, 10s, 20s, 60s)
filepath1 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(50nN, 1s, liquid).nid"
filepath2 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(50nN, 5s, liquid).nid"
filepath3 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(50nN, 10s, liquid).nid"
filepath4 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(50nN, 20s, liquid).nid"
filepath5 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(50nN, 60s, liquid).nid"
setpoint_50nN_1s = hysteresis(filepath1, 4)
setpoint_50nN_5s = hysteresis(filepath2, 4)
setpoint_50nN_10s = hysteresis(filepath3, 4)
setpoint_50nN_20s = hysteresis(filepath4, 4)
setpoint_50nN_60s = hysteresis(filepath5, 4)
print(setpoint_50nN_1s, setpoint_50nN_5s, setpoint_50nN_10s, setpoint_50nN_20s, setpoint_50nN_60s)
#%%
setpoint_50nN = []
setpoint_50nN.append(setpoint_50nN_1s)
setpoint_50nN.append(setpoint_50nN_5s)
setpoint_50nN.append(setpoint_50nN_10s)
setpoint_50nN.append(setpoint_50nN_20s)
setpoint_50nN.append(setpoint_50nN_60s)
setpoint_50nN
#%%
# 80nN(1s, 5s, 10s, 20s, 60s)
filepath1 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(80nN, 1s, liquid).nid"
filepath2 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(80nN, 5s, liquid).nid"
filepath3 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(80nN, 10s, liquid).nid"
filepath4 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(80nN, 20s, liquid).nid"
filepath5 = "Hydrogel AFM data/SD-Sphere-CONT-S/Highly Entangled Hydrogel(80nN, 60s, liquid).nid"
setpoint_80nN_1s = hysteresis(filepath1, 4)
setpoint_80nN_5s = hysteresis(filepath2, 4)
setpoint_80nN_10s = hysteresis(filepath3, 4)
setpoint_80nN_20s = hysteresis(filepath4, 4)
setpoint_80nN_60s = hysteresis(filepath5, 4)
print(setpoint_80nN_1s, setpoint_80nN_5s, setpoint_80nN_10s, setpoint_80nN_20s, setpoint_80nN_60s)
#%%
setpoint_80nN = []
setpoint_80nN.append(setpoint_80nN_1s)
setpoint_80nN.append(setpoint_80nN_5s)
setpoint_80nN.append(setpoint_80nN_10s)
setpoint_80nN.append(setpoint_80nN_20s)
setpoint_80nN.append(setpoint_80nN_60s)
setpoint_80nN
#%%
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.plot(modulation, setpoint_10nN, label="10nN")
ax.plot(modulation, setpoint_30nN, label="30nN")
ax.plot(modulation, setpoint_50nN, label="50nN")
ax.plot(modulation, setpoint_80nN, label="80nN")
ax.set_xlabel("Modulation Time(s)")
ax.set_ylabel("Hysteresis(nm^2)")
ax.legend()
# %%
