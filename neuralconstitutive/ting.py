# ruff: noqa: F722
from typing import Callable

import jax.numpy as jnp
import equinox as eqx
import diffrax

from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.constitutive import AbstractConstitutiveEqn
from neuralconstitutive.tipgeometry import AbstractTipGeometry
from neuralconstitutive.indentation import interpolate_indentation
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
    *,
    newton_iterations: int = 5,
) -> FloatScalar:
    t1_integrand_lower, t1_integrand_upper = make_t1_integrands(
        constitutive, indentations
    )
    app_interp = indentations[0]
    t_m = app_interp.t1  # Unfortunate mixup of names between my code and diffrax
    # In diffrax.AbstractPath, t1 attribute returns the final timepoint of the path
    const = integrate(t1_integrand_upper, (t_m, t), (t,))

    def residual(t1, t):
        return integrate(t1_integrand_lower, (t1, t_m), (t,)) + const

    def Dresidual(t1, t):
        return -constitutive.relaxation_function(t - t1) * app_interp.derivative(t1)

    t1 = t_m
    for _ in range(newton_iterations):
        t1 = jnp.clip(t1 - residual(t1, t) / Dresidual(t1, t), 0.0)

    return t1


def force_retract_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    tip: AbstractTipGeometry,
) -> FloatScalar:

    t1 = t1_scalar(t, constitutive, indentations)
    dF = make_force_integrand(constitutive, indentations[0], tip)

    return integrate(dF, (0, t1), (t,))


_force_approach = eqx.filter_vmap(force_approach_scalar, in_axes=(0, None, None, None))
_force_retract = eqx.filter_vmap(force_retract_scalar, in_axes=(0, None, None, None))


@eqx.filter_jit
def force_approach(constitutive, approach, tip, *, interp_method: str = "cubic"):
    app_interp = interpolate_indentation(approach, method=interp_method)
    return _force_approach(approach.time, constitutive, app_interp, tip)


@eqx.filter_jit
def force_retract(constitutive, indentations, tip, *, interp_method: str = "cubic"):
    app, ret = indentations
    app_interp = interpolate_indentation(app, method=interp_method)
    ret_interp = interpolate_indentation(ret, method=interp_method)
    return _force_retract(ret.time, constitutive, (app_interp, ret_interp), tip)
