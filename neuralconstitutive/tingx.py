# ruff: noqa: F722

import diffrax
import equinox as eqx
import jax.numpy as jnp

from integrax.integrate import integrate
from integrax.solvers import AdaptiveTrapezoid, AdaptiveSimpson
from neuralconstitutive.constitutive import AbstractConstitutiveEqn
from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.indentation import interpolate_indentation
from neuralconstitutive.tipgeometry import AbstractTipGeometry


def force_integrand(s, args):
    t, constit, app, tip = args
    a, b = tip.a(), tip.b()
    g = constit.relaxation_function(jnp.clip(t - s, 0.0))
    dh_b = b * app.derivative(s) * app.evaluate(s) ** (b - 1)
    return a * g * dh_b


def t1_integrand(s, args):
    t, constit, indent = args
    return constit.relaxation_function(t - s) * indent.derivative(s)


def force_approach_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    approach: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    method = AdaptiveSimpson(1e-4, 1e-4)
    args = (t, constitutive, approach, tip)
    return integrate(force_integrand, method, 0, t, args)


def t1_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    newton_iterations: int = 5,
) -> FloatScalar:
    app_interp, ret_interp = indentations
    t_m = app_interp.t1  # Unfortunate mixup of names between my code and diffrax
    # In diffrax.AbstractPath, t1 attribute returns the final timepoint of the path
    method = AdaptiveSimpson(1e-4, 1e-4)

    const = integrate(t1_integrand, method, t_m, t, (t, constitutive, ret_interp))

    def residual(t1, t):
        return (
            integrate(t1_integrand, method, t1, t_m, (t, constitutive, app_interp))
            + const
        )

    def Dresidual(t1, t):
        return -t1_integrand(t1, (t, constitutive, app_interp))

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
    method = AdaptiveSimpson(1e-4, 1e-4)
    args = (t, constitutive, indentations[0], tip)
    return integrate(force_integrand, method, 0, t1, args)


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
