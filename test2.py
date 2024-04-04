# %%
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
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


# %%
## Approach 1. Automatic gradient
def force_approach_auto(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
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
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
    approach: Indentation,
):
    app_interp = diffrax.LinearInterpolation(approach.time, approach.depth)
    return force_approach_((t, constitutive), app_interp, tip)


@eqx.filter_custom_vjp
def force_approach_(t_constit, app_interp, tip):
    t, constitutive = t_constit
    del t_constit
    args = (t, constitutive, app_interp, tip)
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
    args = (t, constitutive, app_interp, tip)
    bounds = (jnp.zeros_like(t), t)
    out = integrate(Dforce_integrand, bounds, args)
    out = out.at[0].add(force_integrand(t, t, constitutive, app_interp, tip))
    out = out * grad_obj
    _, tree_def = Drelaxation_fn((t, constitutive))
    return jtu.tree_unflatten(tree_def, out)


# %%
plr = ModifiedPowerLaw(1.0, 0.5, 0.1)
f_app_auto = eqx.filter_vmap(force_approach_auto, in_axes=(0, None, None, None))(
    app.time, plr, tip, app
)

# %%
plr = ModifiedPowerLaw(1.0, 0.5, 0.1)
f_app_manual = eqx.filter_vmap(force_approach_manual, in_axes=(0, None, None, None))(
    app.time, plr, tip, app
)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_auto, ".", label="Automatic diff")
ax.plot(app.time, f_app_manual, ".", label="Manual diff")
# %%
# %%timeit
grad_f_auto = eqx.filter_jit(
    eqx.filter_vmap(eqx.filter_grad(force_approach_auto), in_axes=(0, None, None, None))
)(app.time, plr, tip, app)
# %%
# %%timeit
grad_f_manual = eqx.filter_jit(
    eqx.filter_vmap(
        eqx.filter_grad(force_approach_manual), in_axes=(0, None, None, None)
    )
)(app.time, plr, tip, app)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].plot(app.time, grad_f_auto, ".", label="Automatic diff")
axes[0].plot(app.time, grad_f_manual, ".", label="Manual diff")
# axes[1].plot(app.time, grad_f_auto.alpha, ".", label="Automatic diff")
# axes[1].plot(app.time, grad_f_manual.alpha, ".", label="Manual diff")

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
