# ruff: noqa: F722

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from neuralconstitutive.constitutive import AbstractConstitutiveEqn
from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.indentation import interpolate_indentation
from neuralconstitutive.integrate import integrate
from neuralconstitutive.tipgeometry import AbstractTipGeometry
from neuralconstitutive.tree import tree_to_array1d


def force_integrand(
    s: FloatScalar,
    t: FloatScalar,
    constit: AbstractConstitutiveEqn,
    app: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    a, b = tip.a(), tip.b()
    g = constit.relaxation_function(jnp.clip(t - s, 0.0))
    dh_b = b * app.derivative(s) * app.evaluate(s) ** (b - 1)
    return a * g * dh_b


def Dforce_integrand(
    s: FloatScalar,
    t: FloatScalar,
    constit: AbstractConstitutiveEqn,
    app: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    s, t = jnp.asarray(s), jnp.asarray(t)

    @eqx.filter_grad
    def _Dforce_integrand(inputs, s, app, tip):
        return force_integrand(s, *inputs, app, tip)

    grad_t_constit = _Dforce_integrand((t, constit), s, app, tip)
    return tree_to_array1d(grad_t_constit)


def t1_integrand(
    s: FloatScalar,
    t: FloatScalar,
    constit: AbstractConstitutiveEqn,
    indent: diffrax.AbstractPath,
) -> FloatScalar:
    return constit.relaxation_function(t - s) * indent.derivative(s)


def Dt1_integrand(s, t, constit, indent):
    s, t = jnp.asarray(s), jnp.asarray(t)

    @eqx.filter_grad
    def _Dt1_integrand(inputs, s, indent):
        return t1_integrand(s, *inputs, indent)

    grad_t_constit = _Dt1_integrand((t, constit), s, indent)
    return tree_to_array1d(grad_t_constit)


@eqx.filter_custom_jvp
def force_approach_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    approach: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    args = (t, constitutive, approach, tip)
    return integrate(force_integrand, (0, t), args)


@force_approach_scalar.def_jvp
def _force_approach_scalar_jvp(primals, tangents):
    t, constit, app, tip = primals
    t_dot, constit_dot, _, _ = tangents
    primal_out = force_approach_scalar(t, constit, app, tip)
    tangents_out = integrate(Dforce_integrand, (0, t), primals)
    tangents_out = tangents_out.at[0].add(force_integrand(t, *primals))
    t_constit_dot = tree_to_array1d((t_dot, constit_dot))
    return primal_out, jnp.dot(tangents_out, t_constit_dot)


@eqx.filter_custom_jvp
def t1_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    *,
    newton_iterations: int = 5,
) -> FloatScalar:

    app_interp, ret_interp = indentations
    t_m = app_interp.t1  # Unfortunate mixup of names between my code and diffrax
    # In diffrax.AbstractPath, t1 attribute returns the final timepoint of the path
    const = integrate(t1_integrand, (t_m, t), (t, constitutive, ret_interp))

    def residual(t1, t):
        return integrate(t1_integrand, (t1, t_m), (t, constitutive, app_interp)) + const

    def Dresidual(t1, t):
        return -t1_integrand(t1, t, constitutive, app_interp)

    t1 = t_m
    for _ in range(newton_iterations):
        f_t1, Df_t1 = residual(t1, t), Dresidual(t1, t)
        t1 = jnp.clip(t1 - f_t1 / Df_t1, 0.0)

    return t1


@t1_scalar.def_jvp
def _t1_scalar_jvp(primals, tangents, *, newton_iterations: int = 5):
    # Take into account when t1=0
    t, constit, indentations = primals

    t1 = t1_scalar(t, constit, indentations, newton_iterations=newton_iterations)

    def _tangents_nonzero(t1, t, constit, indentations, tangents):
        app, ret = indentations
        t_m = app.t1
        tangents_out = integrate(Dt1_integrand, (t1, t_m), (t, constit, app))
        tangents_out = tangents_out + integrate(
            Dt1_integrand, (t_m, t), (t, constit, ret)
        )
        tangents_out = tangents_out.at[0].add(t1_integrand(t, t, constit, ret))
        tangents_out = tangents_out / constit.relaxation_function(t - t1)

        t_dot, constit_dot, _ = tangents
        t_constit_dot = tree_to_array1d(t_dot, constit_dot)
        return jnp.dot(tangents_out, t_constit_dot)

    def _tangents_zero(t1, *_):
        return jnp.zeros_like(t1)

    tangents_out = jax.lax.cond(
        t1 > 0,
        _tangents_nonzero,
        _tangents_zero,
        t1,
        t,
        constit,
        indentations,
        tangents,
    )
    return t1, tangents_out


def force_retract_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    tip: AbstractTipGeometry,
) -> FloatScalar:
    t1 = t1_scalar(t, constitutive, indentations)
    return _force_retract_scalar(t1, t, constitutive, indentations, tip)


@eqx.filter_custom_jvp
def _force_retract_scalar(
    t1: FloatScalar,
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    tip: AbstractTipGeometry,
) -> FloatScalar:
    args = (t, constitutive, indentations[0], tip)
    return integrate(force_integrand, (0, t1), args)


@_force_retract_scalar.def_jvp
def _force_retract_jvp(primals, tangents):
    t1, t, constit, indentations, tip = primals
    app = indentations[0]
    t1_dot, t_dot, constit_dot, _, _ = tangents
    primal_out = _force_retract_scalar(t1, t, constit, indentations, tip)
    tangents_t_constit = integrate(Dforce_integrand, (0, t1), (t, constit, app, tip))
    t_constit_dot = tree_to_array1d(t_dot, constit_dot)
    tangents_t1 = force_integrand(t1, t, constit, app, tip)
    tangents_out = jnp.dot(tangents_t_constit, t_constit_dot) + tangents_t1 * t1_dot
    return primal_out, tangents_out


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
