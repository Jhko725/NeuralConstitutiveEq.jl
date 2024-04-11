# %%
# ruff: noqa: F722
from typing import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Float, Array
import equinox as eqx
import diffrax
import optimistix as optx
import matplotlib.pyplot as plt

from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
    ModifiedPowerLaw,
    StandardLinearSolid,
    PowerLaw,
    Fung,
)
from neuralconstitutive.tipgeometry import AbstractTipGeometry, Conical
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.integrate import integrate

# from neuralconstitutive.ting import force_ting, force_approach

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
plr = ModifiedPowerLaw(1.0, 0.5, 1.0)
tip = Conical(jnp.pi / 18)
del t, h, t_ret, h_ret
app_interp = diffrax.LinearInterpolation(app.time, app.depth)
ret_interp = diffrax.LinearInterpolation(ret.time, ret.depth)


# %%
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
    approach: Indentation,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    app_interp = diffrax.LinearInterpolation(approach.time, approach.depth)

    dF = make_force_integrand(constitutive, app_interp, tip)
    dF = eqx.filter_closure_convert(dF, approach.time[0], approach.time[1])

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
#%%
def t1_scalar2(
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
        return -constitutive.relaxation_function(t-t1)*app_interp.derivative(t1)

    t1 = t_m
    for _ in range(newton_iterations):
        t1 = jnp.clip(t1-residual(t1, t)/Dresidual(t1, t), 0.0)

    return t1

def force_retract_scalar2(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    tip: AbstractTipGeometry,
    *,
    newton_iterations:int = 5
) -> FloatScalar:

    t1 = t1_scalar2(t, constitutive, indentations, newton_iterations=newton_iterations)
    dF = make_force_integrand(constitutive, indentations[0], tip)

    return integrate(dF, (0, t1), (t,))
#%%
@eqx.filter_jit
def force_approach(constitutive, approach, tip):
    app_interp = diffrax.LinearInterpolation(approach.time, approach.depth)
    _force_approach = eqx.filter_vmap(force_approach_scalar, in_axes=(0, None, None, None))
    return _force_approach(approach.time, constitutive, app_interp, tip)

@eqx.filter_jit
def force_retract(constitutive, indentations, tip):
    app, ret = indentations
    app_interp = diffrax.LinearInterpolation(app.time, app.depth)
    ret_interp = diffrax.LinearInterpolation(ret.time, ret.depth)
    _force_retract = eqx.filter_vmap(force_retract_scalar, in_axes=(0, None, None, None))
    return _force_retract(ret.time, constitutive, (app_interp, ret_interp), tip)

@eqx.filter_jit
def force_retract2(constitutive, indentations, tip):
    app, ret = indentations
    app_interp = diffrax.LinearInterpolation(app.time, app.depth)
    ret_interp = diffrax.LinearInterpolation(ret.time, ret.depth)
    _force_retract = eqx.filter_vmap(force_retract_scalar2, in_axes=(0, None, None, None))
    return _force_retract(ret.time, constitutive, (app_interp, ret_interp), tip)
# %%
%%timeit
f_ret = force_retract(plr, (app, ret), tip)
#%%
%%timeit
f_ret2 = force_retract2(plr, (app, ret), tip)
#%%
fig, ax = plt.subplots(1, 1, figsize = (5, 3))
ax.plot(ret.time, f_ret, ".", label = "optx")
ax.plot(ret.time, f_ret2, ".", label = "newton")
ax.legend()
#%%
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


# %%

t1_ret = eqx.filter_vmap(t1_scalar, in_axes=(0, None, None))(
    ret.time, plr, (app_interp, ret_interp)
)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(
    ret.time,
    t1_ret,
)
#%%
%%timeit
f_ret = force_retract(plr, (app, ret), tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(
    ret.time,
    f_ret,
)


# %%
args = (jnp.asarray(0.8), plr)
force_integrand(args, jnp.asarray(0.1), app_interp, tip)
# %%
dt, d_constit = eqx.filter_grad(force_integrand)(
    args, jnp.asarray(0.1), app_interp, tip
)
out, _ = jtu.tree_flatten((dt, d_constit))
out


# %%
@eqx.filter_custom_jvp
def force_approach(
    t: FloatScalar,
    constit: AbstractConstitutiveEqn,
    app_interp: Indentation,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    def _integrand(s: FloatScalar, t: FloatScalar) -> FloatScalar:
        return force_integrand((t, constit), s, app_interp, tip)

    args = (t,)
    bounds = (jnp.zeros_like(t), t)
    return integrate(_integrand, bounds, args)


@force_approach.def_jvp
def force_approach_jvp(primals, tangents):
    t, constit, app_interp, tip = primals
    Dt, Dconstit, _, _ = (
        tangents  # Will not propagate gradients through app_interp and tip
    )
    primal_out = force_approach(t, constit, app_interp, tip)

    def _Dintegrand(s: FloatScalar, t: FloatScalar) -> Float[Array, " N+1"]:
        grad_pytree = eqx.filter_grad(force_integrand)((t, constit), s, app_interp, tip)
        grad_array, _ = jtu.tree_flatten(grad_pytree)
        return jnp.asarray(grad_array)

    args = (t,)
    bounds = (jnp.zeros_like(t), t)

    grad_force_array = integrate(_Dintegrand, bounds, args)
    grad_force_array = grad_force_array.at[0].add(
        force_integrand((t, constit), t, app_interp, tip)
    )
    tangent_array, _ = jtu.tree_flatten(
        jtu.tree_map(_none_to_zero, (Dt, Dconstit), is_leaf=_is_none)
    )
    tangent_out = jnp.dot(jnp.asarray(tangent_array), grad_force_array)
    return primal_out, tangent_out


def _is_none(x):
    return x is None


def _none_to_zero(x):
    if x is None:
        return jnp.asarray(0.0)
    else:
        return x


# %%
def relaxation_fn(t_constit: tuple[float, AbstractConstitutiveEqn]):
    t, constitutive = t_constit
    return constitutive.relaxation_function(t)


def Drelaxation_fn(t_constit: tuple[float, AbstractConstitutiveEqn]):
    dG_pytree = eqx.filter_grad(relaxation_fn)(t_constit)
    dG_array, tree_def = jtu.tree_flatten(dG_pytree)
    dG_array = jnp.asarray(dG_array)
    return dG_array, tree_def


def force_integrand(s, t, constit, app_interp, tip):
    g = relaxation_fn((t - s, constit))
    a, b = tip.a(), tip.b()
    d_hb = b * app_interp.derivative(s) * app_interp.evaluate(s) ** (b - 1)
    return a * g * d_hb


def Dforce_integrand(s, t, constit, app_interp, tip):
    dG, _ = Drelaxation_fn((t - s, constit))
    a, b = tip.a(), tip.b()
    d_hb = b * app_interp.derivative(s) * app_interp.evaluate(s) ** (b - 1)
    return a * dG * d_hb


def make_force_integrand(constit, app_interp, tip):
    a, b = tip.a(), tip.b()

    def _integrand(s, t):
        g = relaxation_fn((jnp.clip(t - s, 0.0), constit))
        dh_b = b * app_interp.derivative(s) * app_interp.evaluate(s) ** (b - 1)
        return a * g * dh_b

    return _integrand


def make_dforce_integrand(constit, app_interp, tip):
    a, b = tip.a(), tip.b()

    def _integrand(s, t):
        dG, _ = Drelaxation_fn((jnp.clip(t - s, 0.0), constit))
        dh_b = b * app_interp.derivative(s) * app_interp.evaluate(s) ** (b - 1)
        return a * dG * dh_b

    return _integrand


# %%
## Approach 1. Automatic gradient
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


# %%
## Approach 2. Manual gradient
def force_approach_manual(
    constitutive: AbstractConstitutiveEqn,
    t: FloatScalar,
    tip: AbstractTipGeometry,
    approach: Indentation,
):
    app_interp = diffrax.LinearInterpolation(approach.time, approach.depth)
    return force_approach_((t, constitutive), app_interp, tip)


@eqx.filter_custom_vjp
def force_approach_(t_constit, app_interp, tip):
    t, constitutive = t_constit
    del t_constit
    force_integrand = make_force_integrand(constitutive, app_interp, tip)
    args = (t,)
    bounds = (jnp.zeros_like(t), t)
    return integrate(force_integrand, bounds, args)


@force_approach_.def_fwd
def force_approach_fwd(perturbed, t_constit, app_interp, tip):
    del perturbed
    f_app = force_approach_(t_constit, app_interp, tip)
    return f_app, None  # other stuff


@force_approach_.def_bwd
def force_approach_bwd(residuals, grad_obj, perturbed, t_constit, app_interp, tip):
    del residuals, perturbed
    t, constitutive = t_constit
    del t_constit
    force_integrand = make_force_integrand(constitutive, app_interp, tip)
    dforce_integrand = make_dforce_integrand(constitutive, app_interp, tip)
    args = (t,)
    bounds = (jnp.zeros_like(t), t)
    out = integrate(dforce_integrand, bounds, args)
    out = out.at[0].add(force_integrand(t, t))
    out = out * grad_obj
    _, tree_def = Drelaxation_fn((t, constitutive))
    return jtu.tree_unflatten(tree_def, out)


# %%
plr = ModifiedPowerLaw(1.0, 0.5, 0.2)
f_app_auto = eqx.filter_vmap(force_approach_auto, in_axes=(None, 0, None, None))(
    plr, app.time, tip, app
)
# %%
plr = ModifiedPowerLaw(1.0, 0.5, 0.2)
f_app_jvp = eqx.filter_vmap(force_approach, in_axes=(0, None, None, None))(
    app.time, plr, app_interp, tip
)
# %%
plr = ModifiedPowerLaw(1.0, 0.5, 0.2)
f_app_manual = eqx.filter_vmap(force_approach_manual, in_axes=(0, None, None, None))(
    app.time, plr, tip, app
)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_auto, ".", label="Automatic diff")
ax.plot(app.time, f_app_jvp, ".", label="Manual diff")
# %%
# %%timeit
plr = ModifiedPowerLaw(1.0, 0.5, 0.2)
grad_f_auto = eqx.filter_jit(
    eqx.filter_vmap(eqx.filter_grad(force_approach_auto), in_axes=(None, 0, None, None))
)(plr, app.time, tip, app)
# %%
plr = ModifiedPowerLaw(1.0, 0.5, 0.3)


def force_approach_2(constit, t, app, tip):
    return force_approach(t, constit, app, tip)


grad_f_jvp = eqx.filter_vmap(
    eqx.filter_grad(force_approach_2), in_axes=(None, 0, None, None)
)(plr, app.time, app_interp, tip)
# %%
eqx.filter_grad(force_approach)(app.time[10], plr, app_interp, tip)
# %%timeit
app.time[10]
# %%
plr = ModifiedPowerLaw(1.0, 0.5, 0.2)
grad_f_manual = eqx.filter_jit(
    eqx.filter_vmap(
        eqx.filter_grad(force_approach_manual), in_axes=(None, 0, None, None)
    )
)(plr, app.time, tip, app)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, grad_f_auto.E0, ".", label="Automatic diff")
axes[0].plot(app.time, grad_f_jvp.E0, ".", label="Manual diff")
axes[1].plot(app.time, grad_f_auto.alpha, ".", label="Automatic diff")
axes[1].plot(app.time, grad_f_manual.alpha, ".", label="Manual diff")

# %%
grad_f_manual


# %%
@eqx.filter_jit
def find_t1_auto(
    constitutive: AbstractConstitutiveEqn,
    t: FloatScalar,
    approach: Indentation,
    retract: Indentation,
) -> FloatScalar:
    t_m = approach.time[-1]

    app_interp = diffrax.LinearInterpolation(approach.time, approach.depth)
    ret_interp = diffrax.LinearInterpolation(retract.time, retract.depth)

    def t1_integrand_lower(s, t):
        return constitutive.relaxation_function(t - s) * app_interp.derivative(s)

    def t1_integrand_upper(s, t):
        return constitutive.relaxation_function(t - s) * ret_interp.derivative(s)

    def residual(t1, t):
        return integrate(t1_integrand_lower, (t1, t_m), (t,)) + integrate(
            t1_integrand_upper, (t_m, t), (t,)
        )

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
            max_steps=30,
            throw=False,
            adjoint=optx.RecursiveCheckpointAdjoint(),
        ).value,
    )

    # _find_t1 = eqx.filter_closure_convert(_find_t1, retract.time[0])
    # return eqx.filter_vmap(_find_t1)(retract.time)


# %%
@eqx.filter_custom_vjp
def _find_t1(t_constit, app_interp, tip):
    t, constitutive = t_constit
    del t_constit
    args = (t, constitutive, app_interp, tip)
    bounds = (jnp.zeros_like(t), t)
    return integrate(force_integrand, bounds, args)


# %%
res = find_t1(plr, app, ret)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(ret.time, res)


# %%
def test(constitutive, approach, retract):
    return find_t12(constitutive, approach, retract)[10]


# %%
val, dval = eqx.filter_value_and_grad(test)(plr, app, ret)
print(val, dval.E0, dval.alpha, dval.t0)
# print(dval.time, dval.depth)
# %%
val2, dval2 = eqx.filter_value_and_grad(test2)(plr, tip, app, ret)
print(val2, dval2.E0, dval2.alpha, dval2.t0)
# %%
#@eqx.filter_jit
def residual(t1, t, constitutive, indentations):
    app_interp, ret_interp = indentations
    t_m = app_interp.t1
    def _t1_integrand_lower(s, t):
        return constitutive.relaxation_function(t-s)*app_interp.derivative(s)
    
    def _t1_integrand_upper(s, t):
        return constitutive.relaxation_function(t-s)*ret_interp.derivative(s)
    
    return integrate(_t1_integrand_lower, (t1, t_m), (t,)) + integrate(_t1_integrand_upper, (t_m, t), (t,))
#@eqx.filter_jit
def Dresidual(t1, t, constitutive, indentations):
    app_interp, _ = indentations
    return -constitutive.relaxation_function(t-t1)*app_interp.derivative(t1)
# %%
t = 1.8
t1 = 1.0
#%%
f_t1 = residual(t1, t, plr, (app_interp, ret_interp))
Df_t1 = Dresidual(t1, t, plr, (app_interp, ret_interp))
t1 = jnp.clip(t1-f_t1/Df_t1, 0.0)
print(t1)
#%%
residual_vec = eqx.filter_vmap(residual, in_axes = (0, 0, None, None))
Dresidual_vec = eqx.filter_vmap(Dresidual, in_axes = (0, 0, None, None))
residual_vec = eqx.filter_jit(residual_vec)
Dresidual_vec = eqx.filter_jit(Dresidual_vec)
# %%
residual(t1, t, plr, (app_interp, ret_interp))
# %%
%%timeit
find_t1_auto(plr, 1.7, app, ret)
# %%
find_t1_vec = eqx.filter_vmap(find_t1_auto, in_axes = (None, 0, None, None))
find_t1_vec = eqx.filter_jit(find_t1_vec)
#%%
%%timeit
t1_vec_optx = find_t1_vec(plr, ret.time, app, ret)
#%%
#%%timeit
t_vec = ret.time
t1_vec = jnp.ones_like(ret.time)*ret.time[0]
for i in range(3):
    f_t1_vec = residual_vec(t1_vec, ret.time, plr, (app_interp, ret_interp))
    Df_t1_vec = Dresidual_vec(t1_vec, ret.time, plr, (app_interp, ret_interp))
    t1_vec = t1_vec-f_t1_vec/Df_t1_vec
# %%
fig, axes = plt.subplots(2, 1, figsize = (5, 3), sharex=True)
axes[0].plot(t_vec, jnp.clip(t1_vec, 0.0))
axes[0].plot(t_vec, t1_vec_optx)
axes[1].plot(t_vec, f_t1_vec)
# %%
t1_vec
# %%
print(f_t1_vec)

# %%
print(Df_t1_vec)
# %%
Dresidual(1.0, 1.0, plr, (app_interp, ret_interp))
# %%
def residual2(t1, t, constitutive, indentations):
    app_interp, ret_interp = indentations
    t_m = app_interp.t1
    def _t1_integrand_lower(u, t, t1):
        s = t1*(1-u)+t_m*u
        return constitutive.relaxation_function(t-s)*app_interp.derivative(s)
    
    def _t1_integrand_upper(u, t):
        s = t_m*(1-u)+t*u
        return constitutive.relaxation_function(t-s)*ret_interp.derivative(s)
    
    return integrate(_t1_integrand_lower, (0.0, 1.0), (t,t1)) + integrate(_t1_integrand_upper, (0.0, 1.0), (t,))
# %%
t = 1.4
t1 = 0.5
res = residual(t1, t, plr, (app_interp, ret_interp))
res2 = residual2(t1, t, plr, (app_interp, ret_interp))
print(res, res2)
# %%
