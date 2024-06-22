# %%
# ruff: noqa: F722
from typing import Any, Callable
from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array
import matplotlib.pyplot as plt

from integrax.integrate import integrate as integratex
from integrax.solvers import AdaptiveTrapezoid
from neuralconstitutive.constitutive import AbstractConstitutive, StandardLinearSolid
from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.indentation import Indentation, interpolate_indentation
from neuralconstitutive.integrate import integrate
from neuralconstitutive.tipgeometry import AbstractTipGeometry, Conical
from neuralconstitutive.tree import tree_to_array1d

jax.config.update("jax_enable_x64", True)


def force_integrand(s, t, constit, app, tip):
    a, b = tip.a(), tip.b()
    g = constit.relaxation_function(jnp.clip(t - s, 0.0))
    dh_b = b * app.derivative(s) * app.evaluate(s) ** (b - 1)
    return a * g * dh_b


def force_integrandx(s, args):
    t, constit, app, tip = args
    a, b = tip.a(), tip.b()
    g = constit.relaxation_function(jnp.clip(t - s, 0.0))
    dh_b = b * app.derivative(s) * app.evaluate(s) ** (b - 1)
    return a * g * dh_b


def force_approach_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutive,
    approach: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    args = (t, constitutive, approach, tip)
    return integrate(force_integrand, (0, t), args)


def force_approach_scalarx(
    t: FloatScalar,
    constitutive: AbstractConstitutive,
    approach: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    method = AdaptiveTrapezoid(1e-4, 1e-4)
    args = (t, constitutive, approach, tip)
    return integratex(force_integrandx, method, 0, t, args)


@eqx.filter_jit
def f_app_grad(t, constit, approach, tip):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _f_app_grad(inputs):
        return force_approach_scalar(*inputs)

    return _f_app_grad((t, constit, approach, tip))


@eqx.filter_jit
def f_app_gradx(t, constit, approach, tip):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _f_app_gradx(inputs):
        return force_approach_scalarx(*inputs)

    return _f_app_gradx((t, constit, approach, tip))


def force_approach_conv(constit, t_app, dIb_app, a: float = 1.0):
    G = eqx.filter_vmap(constit.relaxation_function)(t_app)
    dt = t_app[1] - t_app[0]
    return a * jnp.convolve(G, dIb_app)[0 : len(G)] * dt


@partial(eqx.filter_vmap, in_axes=(None, 0, 0, None, None, None))
def _force_retract_conv(constit, t, t1, t_app, dIb_app, a: float = 1.0):
    G = eqx.filter_vmap(constit.relaxation_function)(t - t_app)
    dt = t_app[1] - t_app[0]
    dIb_masked = jnp.where(t_app <= t1, dIb_app, 0.0)
    return a * jnp.dot(G, dIb_masked) * dt


@partial(jax.vmap, in_axes=(None, None, 0))
def mask_by_time_lower(t, x, t1):
    return jnp.where(t > t1, x, 0.0)


@eqx.filter_jit
def find_t1(constit, t_ret, v_ret, t_app, v_app):
    relax_fn = eqx.filter_vmap(constit.relaxation_function)
    G_ret = relax_fn(t_ret)
    dt = t_ret[1] - t_ret[0]
    t1_obj_const = jnp.convolve(G_ret, v_ret)[0 : len(G_ret)]

    G_matrix = eqx.filter_vmap(relax_fn)(jnp.expand_dims(t_ret, -1) - t_app)

    def t1_objective(t1):
        v_app_ = mask_by_time_lower(t_app, v_app, t1)
        return (jnp.sum(G_matrix * v_app_, axis=-1) + t1_obj_const) * dt

    def Dt1_objective(t1):
        ind_t1 = jnp.rint(t1 / dt).astype(jnp.int_)
        return -relax_fn(t_ret - t1) * v_app.at[ind_t1].get()

    t1 = jnp.linspace(t_app[-1], 0.0, len(t_ret))

    for _ in range(3):
        t1 = jnp.clip(t1 - t1_objective(t1) / Dt1_objective(t1), 0.0)

    return jnp.rint(t1 / dt) * dt


@eqx.filter_jit
def force_retract_conv(constit, t_ret, t_app, v_ret, v_app, dIb_app, a: float = 1.0):
    t1 = find_t1(constit, t_ret, v_ret, t_app, v_app)
    return _force_retract_conv(constit, t_ret, t1, t_app, dIb_app, a)


# %%

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

dIb = (
    tip.b()
    * app_interp.derivative(app.time)
    * app_interp.evaluate(app.time) ** (tip.b() - 1)
)
# %%
f_app = eqx.filter_vmap(force_approach_scalar, in_axes=(0, None, None, None))(
    app.time, plr, app_interp, tip
)
f_appx = eqx.filter_vmap(force_approach_scalarx, in_axes=(0, None, None, None))(
    app.time, plr, app_interp, tip
)
f_app_conv = eqx.filter_jit(force_approach_conv)(plr, app.time, dIb, tip.a())
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, label="quadax")
ax.plot(app.time, f_appx, label="integrax")
ax.plot(app.time, f_app_conv, label="Conv")
ax.legend()
# %%
ind = 10
dF_app = f_app_grad(app.time[ind], plr, app_interp, tip)
print(dF_app[0], dF_app[1].E1, dF_app[1].E_inf, dF_app[1].tau)
# %%
dF_appx = f_app_gradx(app.time[ind], plr, app_interp, tip)
print(dF_appx[0], dF_appx[1].E1, dF_appx[1].E_inf, dF_appx[1].tau)
# %%
dF_app_conv = eqx.filter_grad(
    lambda constit: force_approach_conv(constit, app.time, dIb, tip.a())[ind]
)(plr)
print(dF_app_conv.E1, dF_app_conv.E_inf, dF_app_conv.tau)
# %%
eps = 1e-4
plr1 = StandardLinearSolid(1.0 + 0.5 * eps, 1.0, 10.0)
plr2 = StandardLinearSolid(1.0 - 0.5 * eps, 1.0, 10.0)
args = (app_interp, tip)

(
    force_approach_scalar(app.time[ind], plr1, *args)
    - force_approach_scalar(app.time[ind], plr2, *args)
) / eps


# %%


def t1_integrand(s, t, constit, indent):
    return constit.relaxation_function(t - s) * indent.derivative(s)


def t1_integrandx(s, args):
    t, constit, indent = args
    return constit.relaxation_function(t - s) * indent.derivative(s)


def t1_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutive,
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


@eqx.filter_jit
def t1_grad(t, constit, indentations):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _t1_grad(inputs):
        return t1_scalar(*inputs)

    return _t1_grad((t, constit, indentations))


def t1_scalarx(
    t: FloatScalar,
    constitutive: AbstractConstitutive,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    *,
    newton_iterations: int = 5,
) -> FloatScalar:
    app_interp, ret_interp = indentations
    t_m = app_interp.t1  # Unfortunate mixup of names between my code and diffrax
    # In diffrax.AbstractPath, t1 attribute returns the final timepoint of the path
    method = AdaptiveTrapezoid(1e-4, 1e-4)

    const = integratex(t1_integrandx, method, t_m, t, (t, constitutive, ret_interp))

    def residual(t1, t):
        return (
            integratex(t1_integrandx, method, t1, t_m, (t, constitutive, app_interp))
            + const
        )

    def Dresidual(t1, t):
        return -t1_integrandx(t1, (t, constitutive, app_interp))

    t1 = t_m
    for _ in range(newton_iterations):
        t1 = jnp.clip(t1 - residual(t1, t) / Dresidual(t1, t), 0.0)

    return t1


@eqx.filter_jit
def t1_gradx(t, constit, indentations):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _t1_gradx(inputs):
        return t1_scalarx(*inputs)

    return _t1_gradx((t, constit, indentations))


# %%
t1 = eqx.filter_vmap(t1_scalar, in_axes=(0, None, None))(
    ret.time, plr, (app_interp, ret_interp)
)
t1x = eqx.filter_vmap(t1_scalarx, in_axes=(0, None, None))(
    ret.time, plr, (app_interp, ret_interp)
)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(ret.time, t1, label="quadax")
ax.plot(ret.time, t1x, label="integrax")
ax.legend()

# %%
dt1 = t1_grad(1.4, plr, (app_interp, ret_interp))
print(dt1[0], dt1[1].E1, dt1[1].E_inf, dt1[1].tau)
# %%
dt1x = t1_gradx(1.4, plr, (app_interp, ret_interp))
print(dt1x[0], dt1x[1].E1, dt1x[1].E_inf, dt1x[1].tau)
# %%
t_test = 1.4
eps = 1e-3
plr1 = StandardLinearSolid(1.0 + 0.5 * eps, 1.0, 10.0)
plr2 = StandardLinearSolid(1.0 - 0.5 * eps, 1.0, 10.0)
args = (app_interp, ret_interp)

(t1_scalar(t_test, plr1, args) - t1_scalar(t_test, plr2, args)) / eps


# %%
def force_retract_scalar(
    t: FloatScalar,
    constitutive: AbstractConstitutive,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    tip: AbstractTipGeometry,
) -> FloatScalar:
    t1 = t1_scalar(t, constitutive, indentations)
    args = (t, constitutive, indentations[0], tip)
    return integrate(force_integrand, (0, t1), args)


@eqx.filter_jit
def f_ret_grad(t, constit, indentations, tip):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _f_ret_grad(inputs):
        return force_retract_scalar(*inputs)

    return _f_ret_grad((t, constit, indentations, tip))


def force_retract_scalarx(
    t: FloatScalar,
    constitutive: AbstractConstitutive,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    tip: AbstractTipGeometry,
) -> FloatScalar:
    t1 = t1_scalarx(t, constitutive, indentations)
    method = AdaptiveTrapezoid(1e-4, 1e-4)
    args = (t, constitutive, indentations[0], tip)
    return integratex(force_integrandx, method, 0, t1, args)


@eqx.filter_jit
def f_ret_gradx(t, constit, indentations, tip):
    t = jnp.asarray(t)

    @eqx.filter_grad
    def _f_ret_gradx(inputs):
        return force_retract_scalarx(*inputs)

    return _f_ret_gradx((t, constit, indentations, tip))


v_app = app_interp.derivative(app.time)
v_ret = ret_interp.derivative(ret.time)
# %%
f_ret = eqx.filter_vmap(force_retract_scalar, in_axes=(0, None, None, None))(
    ret.time, plr, (app_interp, ret_interp), tip
)
f_retx = eqx.filter_vmap(force_retract_scalarx, in_axes=(0, None, None, None))(
    ret.time, plr, (app_interp, ret_interp), tip
)
f_ret_conv = eqx.filter_jit(force_retract_conv)(
    plr, ret.time, app.time, v_ret, v_app, dIb, tip.a()
)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(ret.time, f_ret, label="quadax")
ax.plot(ret.time, f_retx, label="integrax")
ax.plot(ret.time, f_ret_conv, label="conv")
ax.legend()

# %%
ind = 10
args = ((app_interp, ret_interp), tip)
dF_ret = f_ret_grad(ret.time[ind], plr, *args)
print(dF_ret[0], dF_ret[1].E1, dF_ret[1].E_inf, dF_ret[1].tau)
# %%
args = ((app_interp, ret_interp), tip)
dF_retx = f_ret_gradx(ret.time[ind], plr, *args)
print(dF_retx[0], dF_retx[1].E1, dF_retx[1].E_inf, dF_retx[1].tau)
# %%
dF_conv = eqx.filter_grad(
    lambda constit: force_retract_conv(
        constit, ret.time, app.time, v_ret, v_app, dIb, tip.a()
    )[ind]
)(plr)
print(dF_conv.E1, dF_conv.E_inf, dF_conv.tau)
# %%
eps = 1e-3
plr1 = StandardLinearSolid(1.0, 1.0, 10.0 + 0.5 * eps)
plr2 = StandardLinearSolid(1.0, 1.0, 10.0 - 0.5 * eps)
(
    force_retract_scalar(ret.time[ind], plr1, *args)
    - force_retract_scalar(ret.time[ind], plr2, *args)
) / eps

# %%
# %%
