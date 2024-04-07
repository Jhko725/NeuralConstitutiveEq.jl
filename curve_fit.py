# %%
# ruff: noqa: F722
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import optimistix as optx
import diffrax
import matplotlib.pyplot as plt

from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
    ModifiedPowerLaw,
    StandardLinearSolid,
    KohlrauschWilliamsWatts,
)
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.tipgeometry import AbstractTipGeometry, Spherical
from neuralconstitutive.integrate import integrate
from neuralconstitutive.plotting import plot_relaxation_fn

jax.config.update("jax_enable_x64", True)


def force_approach(constitutive, approach, tip):
    return eqx.filter_vmap(force_approach_auto, in_axes=(None, 0, None, None))(
        constitutive, approach.time, tip, approach
    )


def force_approach_auto(
    constitutive: AbstractConstitutiveEqn,
    t: FloatScalar,
    tip: AbstractTipGeometry,
    approach: Indentation,
):
    app_interp = diffrax.LinearInterpolation(approach.time, approach.depth)
    a, b = tip.a(), tip.b()

    def dF(s, t):
        g = constitutive.relaxation_function(jnp.clip(t - s, 0.0))
        dh_b = b * app_interp.derivative(s) * app_interp.evaluate(s) ** (b - 1)
        return a * g * dh_b

    dF = eqx.filter_closure_convert(dF, approach.time[0], approach.time[1])

    return integrate(dF, (0, t), (t,))


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
        return (f_pred - f_true) / f_true

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
plr = ModifiedPowerLaw(5.0, 0.5, 0.1)
f_app = force_approach(plr, indent_app, tip)
f_app = jax.lax.stop_gradient(
    f_app
)  # Will not be computing gradients through the data generation process

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(indent_app.time, f_app, ".")
# %%
sls = StandardLinearSolid(2.0, 0.5, 1.5)
result = fit_approach(sls, tip, indent_app, f_app)
# %%
fig, axes = plt.subplots(1, 2, figsize=(5, 3))
axes[0].plot(indent_app.time, f_app, label="Data")
f_fit = force_approach(result.value, indent_app, tip)
axes[0].plot(indent_app.time, f_fit, label="Curve fit")

axes[1] = plot_relaxation_fn(axes[1], plr, indent_app.time, label="Ground truth")
axes[1] = plot_relaxation_fn(axes[1], result.value, indent_app.time, label="Curve fit")
for ax in axes:
    ax.legend()

# %%
result.value.E_inf
# %%
