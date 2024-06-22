# ruff: noqa: F722
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

from neuralconstitutive.constitutive import AbstractConstitutive
from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.indentation import interpolate_indentation, AbstractIndentation
from neuralconstitutive.integrate import integrate
from neuralconstitutive.tipgeometry import AbstractTipGeometry
from neuralconstitutive.tree import tree_to_array1d


def force_integrand(
    s: FloatScalar,
    t: FloatScalar,
    constit: AbstractConstitutive,
    indent: AbstractIndentation,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    a, b = tip.a(), tip.b()
    g = constit.relaxation_function(jnp.clip(t - s, 0.0))
    dh_b = b * indent.v_app(s) * indent.h_app(s) ** (b - 1)
    return a * g * dh_b


@eqx.filter_custom_jvp
def force_approach_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutive,
    indent: AbstractIndentation,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    args = (t, constitutive, indent, tip)
    return integrate(force_integrand, (0, t), args)


def _is_none(x: Any) -> bool:
    return x is None


def _zeros_like_arg1(x: Array, *_) -> Array:
    return jnp.zeros_like(x)


@force_approach_scalar.def_jvp
def _force_approach_scalar_jvp(primals, tangents):
    t, constit, indent, tip = primals
    t_dot, constit_dot, _, _ = tangents
    t_constit, t_constit_dot = (t, constit), (t_dot, constit_dot)

    primal_out = force_approach_scalar(t, constit, indent, tip)

    is_nondiff = jtu.tree_map(_is_none, t_constit_dot, is_leaf=_is_none)
    t_constit_nondiff, t_constit_diff = eqx.partition(
        t_constit, is_nondiff, is_leaf=_is_none
    )

    @eqx.filter_grad
    def _Dforce_integrand(_t_constit_diff, s):
        _t_constit = eqx.combine(_t_constit_diff, t_constit_nondiff)
        return force_integrand(s, *_t_constit, indent, tip)

    def Dforce_integrand(s, _t_constit_diff):
        return tree_to_array1d(_Dforce_integrand(_t_constit_diff, s))

    Df = integrate(Dforce_integrand, (0, t), (t_constit_diff,))
    Df_boundary = jax.lax.cond(
        t_dot is None, _zeros_like_arg1, force_integrand, t, *primals
    )
    Df = Df.at[0].add(Df_boundary)
    tangents_out = jnp.dot(Df, tree_to_array1d(t_constit_dot))
    return primal_out, tangents_out


def t1_integrand_app(
    s: FloatScalar,
    t: FloatScalar,
    constit: AbstractConstitutive,
    indent: AbstractIndentation,
) -> FloatScalar:
    return constit.relaxation_function(t - s) * indent.v_app(s)


def t1_integrand_ret(
    s: FloatScalar,
    t: FloatScalar,
    constit: AbstractConstitutive,
    indent: AbstractIndentation,
) -> FloatScalar:
    return constit.relaxation_function(t - s) * indent.v_ret(s)


@eqx.filter_custom_jvp
def t1_scalar(
    t: FloatScalar,
    constit: AbstractConstitutive,
    indent: AbstractIndentation,
    newton_iterations: int = 3,
) -> FloatScalar:

    t_m = indent.t_m
    args = (t, constit, indent)

    const = integrate(t1_integrand_ret, (t_m, t), args)

    def residual(t1):
        return integrate(t1_integrand_app, (t1, t_m), args) + const

    def Dresidual(t1):
        return -t1_integrand_app(t1, *args)

    t1 = t_m
    for _ in range(newton_iterations):
        f_t1, Df_t1 = residual(t1), Dresidual(t1)
        t1 = jnp.clip(t1 - f_t1 / Df_t1, 0.0, t_m)

    return t1


@t1_scalar.def_jvp
def _t1_scalar_jvp(primals, tangents, *, newton_iterations: int = 5):
    # Take into account when t1=0
    t, constit, indent = primals
    t_dot, constit_dot, _ = tangents
    t_constit, t_constit_dot = (t, constit), (t_dot, constit_dot)

    t1 = t1_scalar(t, constit, indent, newton_iterations=newton_iterations)

    is_nondiff = jtu.tree_map(_is_none, t_constit_dot, is_leaf=_is_none)
    t_constit_nondiff, t_constit_diff = eqx.partition(
        t_constit, is_nondiff, is_leaf=_is_none
    )

    @eqx.filter_grad
    def _Dt1_integrand_app(_t_constit_diff, s, indent):
        _t_constit = eqx.combine(_t_constit_diff, t_constit_nondiff)
        return t1_integrand_app(s, *_t_constit, indent)

    def Dt1_integrand_app(s, _t_constit_diff, indent):
        return tree_to_array1d(_Dt1_integrand_app(_t_constit_diff, s, indent))

    @eqx.filter_grad
    def _Dt1_integrand_ret(_t_constit_diff, s, indent):
        _t_constit = eqx.combine(_t_constit_diff, t_constit_nondiff)
        return t1_integrand_ret(s, *_t_constit, indent)

    def Dt1_integrand_ret(s, _t_constit_diff, indent):
        return tree_to_array1d(_Dt1_integrand_ret(_t_constit_diff, s, indent))

    t_m = indent.t_m

    Dt1 = integrate(Dt1_integrand_app, (t1, t_m), (t_constit_diff, indent))
    Dt1 = Dt1 + integrate(Dt1_integrand_ret, (t_m, t), (t_constit_diff, indent))
    Dt1_boundary = jax.lax.cond(
        t_dot is None, _zeros_like_arg1, t1_integrand_ret, t, t, constit, indent
    )
    Dt1 = Dt1.at[0].add(Dt1_boundary)
    Dt1 = Dt1 / constit.relaxation_function(t - t1)
    tangents_out = jnp.dot(Dt1, tree_to_array1d(t_constit_dot))
    tangents_out = jnp.where(t1 <= 0.0, jnp.zeros_like(tangents_out), tangents_out)
    return t1, tangents_out


def force_retract_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutive,
    indent: AbstractIndentation,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    t1 = t1_scalar(t, constitutive, indent)
    return _force_retract_scalar(t1, t, constitutive, indent, tip)


@eqx.filter_custom_jvp
def _force_retract_scalar(
    t1: FloatScalar,
    t: FloatScalar,
    constitutive: AbstractConstitutive,
    indent: AbstractIndentation,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    args = (t, constitutive, indent, tip)
    return integrate(force_integrand, (0, t1), args)


@_force_retract_scalar.def_jvp
def _force_retract_jvp(primals, tangents):
    t1, t, constit, indent, tip = primals
    t1_dot, t_dot, constit_dot, _, _ = tangents
    primal_out = _force_retract_scalar(t1, t, constit, indent, tip)

    t_constit, t_constit_dot = (t, constit), (t_dot, constit_dot)
    is_nondiff = jtu.tree_map(_is_none, t_constit_dot, is_leaf=_is_none)
    t_constit_nondiff, t_constit_diff = eqx.partition(
        t_constit, is_nondiff, is_leaf=_is_none
    )

    @eqx.filter_grad
    def _Dforce_integrand(_t_constit_diff, s):
        _t_constit = eqx.combine(_t_constit_diff, t_constit_nondiff)
        return force_integrand(s, *_t_constit, indent, tip)

    def Dforce_integrand(s, _t_constit_diff):
        return tree_to_array1d(_Dforce_integrand(_t_constit_diff, s))

    Df = integrate(Dforce_integrand, (0, t1), (t_constit_diff,))
    tangents_out = jnp.dot(Df, tree_to_array1d(t_constit_dot))

    def tangents_out_t1(t1, t1_dot, constit, indent, tip):
        Dt1 = force_integrand(t1, t, constit, indent, tip)
        return t1_dot * Dt1

    tangents_out = tangents_out + jax.lax.cond(
        t1_dot is None,
        _zeros_like_arg1,
        tangents_out_t1,
        t1,
        t1_dot,
        constit,
        indent,
        tip,
    )

    return primal_out, tangents_out


force_approach = eqx.filter_vmap(force_approach_scalar, in_axes=(0, None, None, None))
force_retract = eqx.filter_vmap(force_retract_scalar, in_axes=(0, None, None, None))
t1_ting = eqx.filter_vmap(t1_scalar, in_axes=(0, None, None))
