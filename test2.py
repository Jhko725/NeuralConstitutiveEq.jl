# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import (
    PowerLaw,
)
from neuralconstitutive.tipgeometry import Conical
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.ting import force_ting

jax.config.update("jax_enable_x64", True)


# %%
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
h = 1.0 * t
app = Indentation(t, h)
t_ret = jnp.arange(1.0, 2.0 + 1e-3, 1e-3)
h_ret = 2.0 - t_ret
ret = Indentation(t_ret, h_ret)
plr = PowerLaw(1.0, 0.5, 1.0)
# plr = ModifiedPowerLaw(1.0, -0.5, 1.0)
tip = Conical(jnp.pi / 18)
del t, h, t_ret, h_ret

# %%
f_app, f_ret = force_ting(plr, tip, app, ret)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, ".")
ax.plot(ret.time, f_ret, ".")
# %%
