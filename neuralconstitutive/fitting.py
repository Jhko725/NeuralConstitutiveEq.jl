# ruff: noqa: F722
from typing import Callable

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


def make_force_integrand(constitutive, approach, tip):

    a, b = tip.a(), tip.b()

    def dF(s, t):
        g = constitutive.relaxation_function(jnp.clip(t - s, 0.0))
        dh_b = b * approach.derivative(s) * approach.evaluate(s) ** (b - 1)
        return a * g * dh_b

    return dF


def force_approach_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    approach: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:

    dF = make_force_integrand(constitutive, approach, tip)

    return integrate(dF, (0, t), (t,))


def make_t1_integrands(constitutive, indentations) -> tuple[Callable, Callable]:
    app_interp, ret_interp = indentations

    def t1_integrand_lower(s, t):
        return constitutive.relaxation_function(t - s) * app_interp.derivative(s)

    def t1_integrand_upper(s, t):
        return constitutive.relaxation_function(t - s) * ret_interp.derivative(s)

    return t1_integrand_lower, t1_integrand_upper


def t1_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
) -> FloatScalar:
    t1_integrand_lower, t1_integrand_upper = make_t1_integrands(
        constitutive, indentations
    )
    t_m = indentations[0].t1  # Unfortunate mixup of names between my code and diffrax
    # In diffrax.AbstractPath, t1 attribute returns the final timepoint of the path
    const = integrate(t1_integrand_upper, (t_m, t), (t,))

    def residual(t1, t):
        return integrate(t1_integrand_lower, (t1, t_m), (t,)) + const

    solver = optx.Bisection(rtol=1e-3, atol=1e-3, flip=True)

    cond = residual(0.0, t) <= 0
    t_ = jnp.where(cond, t_m, t)
    return jnp.where(
        cond,
        jnp.zeros_like(t),
        optx.root_find(
            residual,
            solver,
            t_m,
            args=t_,
            options={"lower": 0.0, "upper": t_m},
            throw=False,
        ).value,
    )


def force_retract_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    tip: AbstractTipGeometry,
) -> FloatScalar:

    t1 = t1_scalar(t, constitutive, indentations)
    dF = make_force_integrand(constitutive, indentations[0], tip)

    return integrate(dF, (0, t1), (t,))


@eqx.filter_jit
def force_approach(constitutive, approach, tip):
    app_interp = diffrax.LinearInterpolation(approach.time, approach.depth)
    _force_approach = eqx.filter_vmap(
        force_approach_scalar, in_axes=(0, None, None, None)
    )
    return _force_approach(approach.time, constitutive, app_interp, tip)


@eqx.filter_jit
def force_retract(constitutive, indentations, tip):
    app, ret = indentations
    app_interp = diffrax.LinearInterpolation(app.time, app.depth)
    ret_interp = diffrax.LinearInterpolation(ret.time, ret.depth)
    _force_retract = eqx.filter_vmap(
        force_retract_scalar, in_axes=(0, None, None, None)
    )
    return _force_retract(ret.time, constitutive, (app_interp, ret_interp), tip)


def force_integrand(
    t_constit: tuple[FloatScalar, AbstractConstitutiveEqn],
    s: FloatScalar,
    indent_app: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
):
    t, constit = t_constit
    del t_constit
    a, b = tip.a(), tip.b()
    G = constit.relaxation_function(jnp.clip(t - s, 0.0))
    dh_b = b * indent_app.derivative(s) * indent_app.evaluate(s) ** (b - 1)
    return a * dh_b * G


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
