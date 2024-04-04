# %%
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuralconstitutive.constitutive import FromLogDiscreteSpectrum
from neuralconstitutive.relaxation_spectrum import HonerkampWeeseBimodal

jax.config.update("jax_enable_x64", True)
# %%
bimodal = FromLogDiscreteSpectrum(HonerkampWeeseBimodal())
t_i, h_i = bimodal.discrete_spectrum
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_i, h_i, ".")
ax.set_xscale("log", base=10)
ax.set_xlabel("Relaxation Time τ[s]")
ax.set_ylabel("Relaxation Spectrum H(τ)[Pa]")
# %%
print(len(t_i))
# %%
number = 1000
t_min = 1e-4
t_max = 1e3
t = np.linspace(t_min, t_max, number)
g = bimodal.relaxation_function(t)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t, g, ".")
ax.set_xlabel("Time[s]")
ax.set_ylabel("Relaxation Function[Pa]")

# %%
g_t = pd.DataFrame([t, g])
# %%
g_t = g_t.T
# %%
g_t.to_csv("test", index=False, header=False, sep=" ")
# %%
# Move test file
h_test = pd.read_csv("pyRespect/output/H.dat", header=None, sep=" ")
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(h_test[0], np.exp(h_test[1]), ".")
ax.plot(t_i, h_i, ".")
ax.set_title(f"t=({t_min}, {t_max}), {number} points")
ax.set_xscale("log")
ax.set_xlabel("Time[s]")
ax.set_ylabel("Relaxation Spectrum[Pa]")
# %%


# %%
h_test1 = pd.read_csv("pyRespect/output/H.dat", header=None, sep=" ")
g_test1 = pd.read_csv("pyRespect/tests/test1.dat", header=None, sep=" ")
t = np.linspace(1e-3, 5 * 1e2, 100)
g = bimodal.relaxation_function(t)
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t, g, ".")
ax.plot(g_test1[0], g_test1[1], ".")
ax.set_xlabel("Time[s]")
ax.set_ylabel("Relaxation Function[Pa]")
ax.set_xscale("log")
ax.set_yscale("log")
# %%
# %%
h_test1[0]
# %%
h_test1
# %%
