# %%
import jax
from jax import Array
import jax.numpy as jnp
import jaxopt
import optimistix as optx
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
    LogDiscretizedSpectrum,
    ModifiedPowerLaw,
    StandardLinearSolid,
    KohlrauschWilliamsWatts,
    Fung,
)
from neuralconstitutive.relaxation_spectrum import HonerkampWeeseBimodal

jax.config.update("jax_enable_x64", True)

bimodal = LogDiscretizedSpectrum(HonerkampWeeseBimodal())

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(bimodal.t_grid, bimodal.h_grid, ".")
ax.set_xscale("log", base=10)
# %%
t = jnp.arange(1e-3, 100.0, 0.01)
y = bimodal.relaxation_function(t)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t, y, ".")
ax.set_xlabel("Time")
ax.set_ylabel("Relaxation function")
# %%
constit = Fung(1.0, 1e-2, 10.0, 1.0)


# %%
def relaxation_residual(constit: AbstractConstitutiveEqn, data: tuple[Array, Array]):
    t_data, y_data = data
    return y_data - constit.relaxation_function(t_data)


solver = optx.LevenbergMarquardt(
    rtol=1e-8, atol=1e-8, verbose=frozenset({"step", "accepted", "loss", "step_size"})
)
sol = optx.least_squares(
    relaxation_residual, solver, Fung(1.0, 1e-2, 10.0, 1.0), (t, y)
)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t, y, ".", label="Data")
ax.set_xlabel("Time")
ax.set_ylabel("Relaxation function")
ax.plot(t, sol.value.relaxation_function(t), ".", label="SLS (fit)")
# %%
