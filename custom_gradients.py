# %%
# ruff: noqa: F722
from typing import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import diffrax

from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.constitutive import AbstractConstitutiveEqn, StandardLinearSolid
from neuralconstitutive.tipgeometry import AbstractTipGeometry, Conical
from neuralconstitutive.indentation import interpolate_indentation, Indentation
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


@eqx.filter_jit
def f_app_grad(t, constit, approach, tip):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _f_app_grad(inputs):
        return force_approach_scalar(*inputs)

    return _f_app_grad((t, constit, approach, tip))


# %%
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)

t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
h = 1.0 * t
app = Indentation(t, h)
t_ret = jnp.arange(1.0, 2.0 + 1e-3, 1e-3)
h_ret = 2.0 - t_ret
ret = Indentation(t_ret, h_ret)
# plr = PowerLaw(2.0, 0.5, 1.0)
# fung = Fung(10.0, 0.1, 10.0, 1.0)
# plr = ModifiedPowerLaw(1.0, 0.5, 1.0)
plr = StandardLinearSolid(1.0, 1.0, 10.0)
tip = Conical(jnp.pi / 18)
del t, h, t_ret, h_ret
app_interp = interpolate_indentation(app)
ret_interp = interpolate_indentation(ret)
# %%
dF_app = f_app_grad(0.3, plr, app_interp, tip)
print(dF_app[0], dF_app[1].E1, dF_app[1].E_inf, dF_app[1].tau)
# %%
dF_app[3]


# %%
@eqx.filter_custom_jvp
def force_approach_scalar2(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    approach: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    args = (t, constitutive, approach, tip)
    return integrate(force_integrand, (0, t), args)


@force_approach_scalar2.def_jvp
def force_app_jvp(primals, tangents):
    t, constit, app, tip = primals
    t_dot, constit_dot, _, _ = tangents
    primal_out = force_approach_scalar2(t, constit, app, tip)
    tangents_out = integrate(Dforce_integrand, (0, t), primals)
    tangents_out = tangents_out.at[0].add(force_integrand(t, *primals))
    t_constit_dot = jnp.asarray(jtu.tree_flatten((t_dot, constit_dot))[0])
    return primal_out, jnp.dot(tangents_out, t_constit_dot)


@eqx.filter_jit
def f_app_grad2(t, constit, approach, tip):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _f_app_grad2(inputs):
        return force_approach_scalar2(*inputs)

    return _f_app_grad2((t, constit, approach, tip))


# %%


def force_integrand(s, t, constit, app, tip):
    a, b = tip.a(), tip.b()
    g = constit.relaxation_function(jnp.clip(t - s, 0.0))
    dh_b = b * app.derivative(s) * app.evaluate(s) ** (b - 1)
    return a * g * dh_b


def Dforce_integrand(s, t, constit, app, tip):
    s, t = jnp.asarray(s), jnp.asarray(t)

    @eqx.filter_grad
    def _Dforce_integrand(inputs, s, app, tip):
        return force_integrand(s, *inputs, app, tip)

    grad_t_constit = _Dforce_integrand((t, constit), s, app, tip)
    return jnp.asarray(jtu.tree_flatten(grad_t_constit)[0])


def t1_integrand(s, t, constit, indent):
    return constit.relaxation_function(t - s) * indent.derivative(s)


def Dt1_integrand(s, t, constit, indent):
    s, t = jnp.asarray(s), jnp.asarray(t)

    @eqx.filter_grad
    def _Dt1_integrand(inputs, s, indent):
        return t1_integrand(s, *inputs, indent)

    grad_t_constit = _Dt1_integrand((t, constit), s, indent)
    return jnp.asarray(jtu.tree_flatten(grad_t_constit)[0])


# %%
Dforce_integrand(0.2, 0.5, plr, app_interp, tip)
# %%
dF_app2 = f_app_grad2(0.3, plr, app_interp, tip)
print(dF_app2[0], dF_app2[1].E1, dF_app2[1].E_inf, dF_app2[1].tau)
# %%
eps = 1e-3
plr1 = StandardLinearSolid(1.0, 1.0, 10.0 + 0.5 * eps)
plr2 = StandardLinearSolid(1.0, 1.0, 10.0 - 0.5 * eps)
args = (app_interp, tip)

(
    force_approach_scalar2(0.3, plr1, *args) - force_approach_scalar2(0.3, plr2, *args)
) / eps


# %%
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


@eqx.filter_jit
def t1_grad(t, constit, indentations):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _t1_grad(inputs):
        return t1_scalar(*inputs)

    return _t1_grad((t, constit, indentations))


@eqx.filter_custom_jvp
def t1_scalar2(
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
        t1 = jnp.clip(t1 - residual(t1, t) / Dresidual(t1, t), 0.0)

    return t1


@t1_scalar2.def_jvp
def t1_scalar_jvp(primals, tangents, *, newton_iterations: int = 5):
    # Take into account when t1=0
    t, constit, indentations = primals

    t1 = t1_scalar2(t, constit, indentations, newton_iterations=newton_iterations)

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
        t_constit_dot = jnp.asarray(jtu.tree_flatten((t_dot, constit_dot))[0])
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


@eqx.filter_jit
def t1_grad2(t, constit, indentations):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _t1_grad2(inputs):
        return t1_scalar2(*inputs)

    return _t1_grad2((t, constit, indentations))


# %%
dt1 = t1_grad(1.99, plr, (app_interp, ret_interp))
print(dt1[0], dt1[1].E1, dt1[1].E_inf, dt1[1].tau)
# %%
t_test = 1.99
eps = 1e-3
plr1 = StandardLinearSolid(1.0 + 0.5 * eps, 1.0, 10.0)
plr2 = StandardLinearSolid(1.0 - 0.5 * eps, 1.0, 10.0)
args = (app_interp, ret_interp)

(t1_scalar(t_test, plr1, args) - t1_scalar(t_test, plr2, args)) / eps

# %%
dt12 = t1_grad2(1.99, plr, (app_interp, ret_interp))
print(dt12[0], dt12[1].E1, dt12[1].E_inf, dt12[1].tau)


# %%
def force_retract_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    tip: AbstractTipGeometry,
) -> FloatScalar:

    t1 = t1_scalar(t, constitutive, indentations)
    dF = make_force_integrand(constitutive, indentations[0], tip)

    return integrate(dF, (0, t1), (t,))


def force_retract_scalar2(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    tip: AbstractTipGeometry,
) -> FloatScalar:
    t1 = t1_scalar2(t, constitutive, indentations)
    return _force_retract_scalar2(t1, t, constitutive, indentations, tip)


@eqx.filter_custom_jvp
def _force_retract_scalar2(
    t1: FloatScalar,
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    tip: AbstractTipGeometry,
) -> FloatScalar:
    args = (t, constitutive, indentations[0], tip)
    return integrate(force_integrand, (0, t1), args)


@_force_retract_scalar2.def_jvp
def force_ret_jvp(primals, tangents):
    t1, t, constit, indentations, tip = primals
    app = indentations[0]
    t1_dot, t_dot, constit_dot, _, _ = tangents
    primal_out = _force_retract_scalar2(t1, t, constit, indentations, tip)
    tangents_t_constit = integrate(Dforce_integrand, (0, t1), (t, constit, app, tip))
    t_constit_dot = jnp.asarray(jtu.tree_flatten((t_dot, constit_dot))[0])
    tangents_t1 = force_integrand(t1, t, constit, app, tip)
    tangents_out = jnp.dot(tangents_t_constit, t_constit_dot) + tangents_t1 * t1_dot
    return primal_out, tangents_out


@eqx.filter_jit
def f_ret_grad(t, constit, indentations, tip):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _f_ret_grad(inputs):
        return force_retract_scalar(*inputs)

    return _f_ret_grad((t, constit, indentations, tip))


@eqx.filter_jit
def f_ret_grad2(t, constit, indentations, tip):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _f_ret_grad2(inputs):
        return force_retract_scalar2(*inputs)

    return _f_ret_grad2((t, constit, indentations, tip))


# %%
args = ((app_interp, ret_interp), tip)
dF_ret = f_ret_grad(1.6, plr, *args)
print(dF_ret[0], dF_ret[1].E1, dF_ret[1].E_inf, dF_ret[1].tau)
# %%
eps = 1e-3
plr1 = StandardLinearSolid(1.0, 1.0, 10.0 + 0.5 * eps)
plr2 = StandardLinearSolid(1.0, 1.0, 10.0 - 0.5 * eps)
(force_retract_scalar(1.6, plr1, *args) - force_retract_scalar(1.6, plr2, *args)) / eps

# %%
dF_ret2 = f_ret_grad2(1.6, plr, *args)
print(dF_ret2[0], dF_ret2[1].E1, dF_ret2[1].E_inf, dF_ret2[1].tau)
# %%
