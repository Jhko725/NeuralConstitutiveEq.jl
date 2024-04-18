# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import (
    FromLogDiscreteSpectrum,
    StandardLinearSolid,
    ModifiedPowerLaw,
    AbstractConstitutiveEqn,
)
from neuralconstitutive.relaxation_spectrum import HonerkampWeeseBimodal
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.ting import force_approach, force_retract
from neuralconstitutive.integrate import integrate
from neuralconstitutive.fitting import fit_approach_lmfit, fit_all_lmfit
from neuralconstitutive.nn import FullyConnectedNetwork
from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.plotting import plot_relaxation_fn

jax.config.update("jax_enable_x64", True)

# Create the spectrum to generate simulated data
# Visualize it
bimodal = FromLogDiscreteSpectrum(
    HonerkampWeeseBimodal(t_x=5e-3, t_y=0.5, t_a=1e-4, t_b=10)
)
t_i, h_i = bimodal.discrete_spectrum

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_i, h_i, ".")
ax.set_xscale("log", base=10)
ax.set_xlabel("Relaxation Time τ[s]")
ax.set_ylabel("Relaxation Spectrum H(τ)[Pa]")
# %%
# "Experimental" time points and indentation
# Set it so that Dt and max(t_exp) more or less span the interesting parts of the spectrum
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
h = 1.0 * t
len(t)
# %%
app = Indentation(t, h)
t_ret = jnp.arange(1.0, 2.0 + 1e-3, 1e-3)
h_ret = 2.0 - t_ret
ret = Indentation(t_ret, h_ret)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, app.depth, ".", label="approach")
ax.plot(ret.time, ret.depth, ".", label="retract")

# %%
y = bimodal.relaxation_function(app.time)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, y, ".")
ax.set_xlabel("Time")
ax.set_ylabel("Relaxation function")

# %%
tip = Spherical(1.0)
f_app = force_approach(bimodal, app, tip)
f_ret = force_retract(bimodal, (app, ret), tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, ".", label="approach")
ax.plot(ret.time, f_ret, ".", label="retract")
ax.legend()
fig

# %%
# %%
sls = StandardLinearSolid(10.0, 10.0, 10.0)
bounds = [(1e-7, 1e7)] * 3
sls_fit, result = fit_approach_lmfit(sls, bounds, tip, app, f_app)
# %%
f_fit_app = force_approach(sls_fit, app, tip)
f_fit_ret = force_retract(sls_fit, (app, ret), tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, ".", color="black", alpha=0.5, label="Data")
ax.plot(ret.time, f_ret, ".", color="black", alpha=0.5, label="Data")

ax.plot(app.time, f_fit_app, ".", color="orangered", alpha=0.8, label="SLS")
ax.plot(ret.time, f_fit_ret, ".", color="orangered", alpha=0.8, label="SLS")
ax.legend()
fig

# %%


fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax = plot_relaxation_fn(
    ax, bimodal, app.time, marker=".", color="black", alpha=0.5, label="Data"
)
ax = plot_relaxation_fn(
    ax, sls_fit, app.time, marker=".", color="orangered", alpha=0.8, label="Data"
)
# %%
## Fit all
sls_fit, result = fit_all_lmfit(sls, bounds, tip, (app, ret), (f_app, f_ret))
# %%
result
# %%
f_fit_app = force_approach(sls_fit, app, tip)
f_fit_ret = force_retract(sls_fit, (app, ret), tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, ".", color="black", alpha=0.5, label="Data")
ax.plot(ret.time, f_ret, ".", color="black", alpha=0.5, label="Data")

ax.plot(app.time, f_fit_app, ".", color="orangered", alpha=0.8, label="SLS")
ax.plot(ret.time, f_fit_ret, ".", color="orangered", alpha=0.8, label="SLS")
ax.legend()
fig

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax = plot_relaxation_fn(
    ax, bimodal, app.time, marker=".", color="black", alpha=0.5, label="Data"
)
ax = plot_relaxation_fn(
    ax, sls_fit, app.time, marker=".", color="orangered", alpha=0.8, label="Data"
)

# %%
