# ruff: noqa: F722
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import optimistix as optx
import diffrax

from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
)
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.tipgeometry import AbstractTipGeometry
from neuralconstitutive.integrate import integrate


@eqx.filter_jit
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
        return f_pred - f_true

    result = optx.least_squares(
        residual, solver, constitutive, args, throw=False, **least_squares_kwargs
    )
    return result
