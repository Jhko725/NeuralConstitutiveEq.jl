# %%
# ruff: noqa: F722
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import optimistix as optx
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
    ModifiedPowerLaw,
    StandardLinearSolid,
    KohlrauschWilliamsWatts,
)
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.tipgeometry import AbstractTipGeometry, Spherical
from neuralconstitutive.ting import force_approach


@eqx.filter_jit
def fit_approach(
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
    approach: Indentation,
    force: Float[Array, " {len(approach)}"],
    solver=optx.LevenbergMarquardt(atol=1e-5, rtol=1e-5),
    **least_squares_kwargs,
):
    args = (approach, tip, force)

    def residual(constitutive, args):
        approach, tip, f_true = args
        del args
        f_pred = force_approach(constitutive, approach, tip)
        return f_pred - f_true

    result = optx.least_squares(
        residual, solver, constitutive, args, throw=False, **least_squares_kwargs
    )
    return result


# %%
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)  # Normalized time
h = 1.0 * t  # Normalized indentation
indent_app = Indentation(t, h)  # Create an indentation object to store data
del t, h  # Delete unnecessary parameters to keep the namespace a bit cleaner
# %%
tip = Spherical(0.5)  # Normalize tip radius by max indentation
plr = ModifiedPowerLaw(5.0, 0.5, 1.0)
f_app = force_approach(plr, indent_app, tip)
f_app = jax.lax.stop_gradient(
    f_app
)  # Will not be computing gradients through the data generation process

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(indent_app.time, f_app, ".")
# %%
t
# %%
