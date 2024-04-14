# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import FromLogDiscreteSpectrum
from neuralconstitutive.relaxation_spectrum import HonerkampWeeseBimodal
from neuralconstitutive.jax.tipgeometry import Spherical
from neuralconstitutive.trajectory import make_triangular
from neuralconstitutive.simulate import simulate_data

jax.config.update("jax_enable_x64", True)

# Create the spectrum to generate simulated data
# Visualize it
bimodal = FromLogDiscreteSpectrum(HonerkampWeeseBimodal())
t_i, h_i = bimodal.discrete_spectrum

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_i, h_i, ".")
ax.set_xscale("log", base=10)
ax.set_xlabel("Relaxation Time τ[s]")
ax.set_ylabel("Relaxation Spectrum H(τ)[Pa]")
# %%
# "Experimental" time points and indentation
# Set it so that Dt and max(t_exp) more or less span the interesting parts of the spectrum
app, ret = make_triangular(0.5, 1e-3, 1.0)

y = bimodal.relaxation_function(app.t)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.t, y, ".")
ax.set_xlabel("Time")
ax.set_ylabel("Relaxation function")

# %%
tip = Spherical(1.0)
f_app, f_ret = simulate_data(app, ret, bimodal.relaxation_function, tip, 0.0, 10)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.t, f_app, ".", label="approach")
ax.plot(ret.t, f_ret, ".", label="retract")
ax.legend()
fig
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.t, app.v(app.t), ".", label="approach")
ax.plot(ret.t, ret.v(ret.t), ".", label="retract")
# %%
