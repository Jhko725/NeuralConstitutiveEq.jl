# %%
from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import (
    relaxation_function,
    ModifiedPowerLaw,
    StandardLinearSolid,
    KohlrauschWilliamsWatts,
    Fung,
    LogDiscretizedSpectrum,
)
from neuralconstitutive.relaxation_spectrum import HonerkampWeeseBimodal
from neuralconstitutive.jax.ting import force_approach, force_retract, find_t1

t = jnp.linspace(1e-2, 1e2, 1000)
plr = ModifiedPowerLaw(572.0, 0.2, 1e-5)
# phi_plr = relaxation_function(t, plr)
honerkamp = LogDiscretizedSpectrum(HonerkampWeeseBimodal()())
plt.plot(t, honerkamp(t))
# %%
Dt = 1e-2
t_app = jnp.arange(0, 101) * Dt
t_ret = jnp.arange(100, 201) * Dt
v_app = 10.0 * jnp.ones_like(t_app)
v_ret = -v_app
d_app = v_app * t_app
d_ret = v_app * (2 * t_ret[0] - t_ret)
# %%
phi = honerkamp
F_app_pred = force_approach(t_app, phi, t_app, d_app, v_app, 1.0, 1.5)
t1 = find_t1(t_ret, phi, t_app, t_ret, v_app, v_ret)
F_ret_pred = force_retract(t_ret, t1, phi, t_app, d_app, v_app, 1.0, 1.5)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_app, F_app_pred)
ax.plot(t_ret, F_ret_pred)
# %%
