# %%
# ruff: noqa: F722
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


# %%
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
dval.depth
# %%
