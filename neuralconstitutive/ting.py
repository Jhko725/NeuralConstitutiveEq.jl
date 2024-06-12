# ruff: noqa: F722
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

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


def t1_integrand(
    s: FloatScalar,
    t: FloatScalar,
    constit: AbstractConstitutiveEqn,
    indent: diffrax.AbstractPath,
) -> FloatScalar:
    return constit.relaxation_function(t - s) * indent.derivative(s)


@eqx.filter_custom_jvp
def force_approach_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    approach: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    args = (t, constitutive, approach, tip)
    return integrate(force_integrand, (0, t), args)


def _is_none(x: Any) -> bool:
    return x is None


def _zeros_like_arg1(x: Array, *_) -> Array:
    return jnp.zeros_like(x)


@force_approach_scalar.def_jvp
def _force_approach_scalar_jvp(primals, tangents):
    t, constit, app, tip = primals
    t_dot, constit_dot, _, _ = tangents
    t_constit, t_constit_dot = (t, constit), (t_dot, constit_dot)

    primal_out = force_approach_scalar(t, constit, app, tip)

    is_nondiff = jtu.tree_map(_is_none, t_constit_dot, is_leaf=_is_none)
    t_constit_nondiff, t_constit_diff = eqx.partition(
        t_constit, is_nondiff, is_leaf=_is_none
    )

    @eqx.filter_grad
    def _Dforce_integrand(_t_constit_diff, s):
        _t_constit = eqx.combine(_t_constit_diff, t_constit_nondiff)
        return force_integrand(s, *_t_constit, app, tip)

    def Dforce_integrand(s, _t_constit_diff):
        return tree_to_array1d(_Dforce_integrand(_t_constit_diff, s))

    Df = integrate(Dforce_integrand, (0, t), (t_constit_diff,))
    Df_boundary = jax.lax.cond(
        t_dot is None, _zeros_like_arg1, force_integrand, t, *primals
    )
    Df = Df.at[0].add(Df_boundary)
    tangents_out = jnp.dot(Df, tree_to_array1d(t_constit_dot))
    return primal_out, tangents_out


@eqx.filter_custom_jvp
def t1_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    newton_iterations: int = 3,
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
        t1 = jnp.clip(t1 - f_t1 / Df_t1, 0.0, t_m)

    return t1


@t1_scalar.def_jvp
def _t1_scalar_jvp(primals, tangents, *, newton_iterations: int = 5):
    # Take into account when t1=0
    t, constit, indentations = primals
    t_dot, constit_dot, _ = tangents
    t_constit, t_constit_dot = (t, constit), (t_dot, constit_dot)

    t1 = t1_scalar(t, constit, indentations, newton_iterations=newton_iterations)

    is_nondiff = jtu.tree_map(_is_none, t_constit_dot, is_leaf=_is_none)
    t_constit_nondiff, t_constit_diff = eqx.partition(
        t_constit, is_nondiff, is_leaf=_is_none
    )

    @eqx.filter_grad
    def _Dt1_integrand(_t_constit_diff, s, indent):
        _t_constit = eqx.combine(_t_constit_diff, t_constit_nondiff)
        return t1_integrand(s, *_t_constit, indent)

    def Dt1_integrand(s, _t_constit_diff, indent):
        return tree_to_array1d(_Dt1_integrand(_t_constit_diff, s, indent))

    app, ret = indentations
    t_m = app.t1

    Dt1 = integrate(Dt1_integrand, (t1, t_m), (t_constit_diff, app))
    Dt1 = Dt1 + integrate(Dt1_integrand, (t_m, t), (t_constit_diff, ret))
    Dt1_boundary = jax.lax.cond(
        t_dot is None, _zeros_like_arg1, t1_integrand, t, t, constit, ret
    )
    Dt1 = Dt1.at[0].add(Dt1_boundary)
    Dt1 = Dt1 / constit.relaxation_function(t - t1)
    tangents_out = jnp.dot(Dt1, tree_to_array1d(t_constit_dot))
    tangents_out = jnp.where(t1 <= app.t0, jnp.zeros_like(tangents_out), tangents_out)
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

    t_constit, t_constit_dot = (t, constit), (t_dot, constit_dot)
    is_nondiff = jtu.tree_map(_is_none, t_constit_dot, is_leaf=_is_none)
    t_constit_nondiff, t_constit_diff = eqx.partition(
        t_constit, is_nondiff, is_leaf=_is_none
    )

    @eqx.filter_grad
    def _Dforce_integrand(_t_constit_diff, s):
        _t_constit = eqx.combine(_t_constit_diff, t_constit_nondiff)
        return force_integrand(s, *_t_constit, app, tip)

    def Dforce_integrand(s, _t_constit_diff):
        return tree_to_array1d(_Dforce_integrand(_t_constit_diff, s))

    Df = integrate(Dforce_integrand, (0, t1), (t_constit_diff,))
    tangents_out = jnp.dot(Df, tree_to_array1d(t_constit_dot))

    def tangents_out_t1(t1, t1_dot, constit, app, tip):
        Dt1 = force_integrand(t1, t, constit, app, tip)
        return t1_dot * Dt1

    tangents_out = tangents_out + jax.lax.cond(
        t1_dot is None, _zeros_like_arg1, tangents_out_t1, t1, t1_dot, constit, app, tip
    )

    return primal_out, tangents_out


_force_approach = eqx.filter_jit(eqx.filter_vmap(force_approach_scalar, in_axes=(0, None, None, None)))
_force_retract = eqx.filter_jit(eqx.filter_vmap(force_retract_scalar, in_axes=(0, None, None, None)))


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
