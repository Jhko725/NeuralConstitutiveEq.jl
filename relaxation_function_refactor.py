# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import (
    relaxation_function,
    ModifiedPowerLaw,
    StandardLinearSolid,
    KohlrauschWilliamsWatts,
    Fung,
)

t = jnp.linspace(1e-2, 1e2, 1000)
plr = ModifiedPowerLaw(572.0, 0.2, 1e-5)
phi_plr = relaxation_function(t, plr)

plt.plot(t, phi_plr)
# %%
jax.grad(relaxation_function, argnums=1)(t, plr)
# %%
