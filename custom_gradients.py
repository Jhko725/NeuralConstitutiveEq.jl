# %%
# ruff: noqa: F722
from typing import Callable

import jax
import jax.numpy as jnp
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
import jax.tree_util as jtu


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
