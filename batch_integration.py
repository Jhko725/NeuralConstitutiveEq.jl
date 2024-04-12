# %%
# ruff: noqa: F722
from typing import Callable
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import diffrax
import matplotlib.pyplot as plt

from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.indentation import Indentation, interpolate_indentation
from neuralconstitutive.integrate import integrate
from neuralconstitutive.constitutive import StandardLinearSolid, AbstractConstitutiveEqn
from neuralconstitutive.tipgeometry import AbstractTipGeometry, Conical, Spherical
from neuralconstitutive.io import import_data
from neuralconstitutive.utils import (
    smooth_data,
    normalize_forces,
    normalize_indentations,
)

jax.config.update("jax_enable_x64", True)

datadir = Path("open_data/PAAM hydrogel")
(app, ret), (f_app, f_ret) = import_data(
    datadir / "PAA_speed 5_4nN.tab", datadir / "PAA_speed 5_4nN.tsv"
)
app, ret = smooth_data(app), smooth_data(ret)

# %%
(f_app, f_ret), _ = normalize_forces(f_app, f_ret)
(app, ret), (_, h_m) = normalize_indentations(app, ret)
# %%
tip = Spherical(2.5e-6 / h_m)  # Scale tip radius by the length scale we are using
constit = StandardLinearSolid(10.0, 10.0, 10.0)
app_interp = interpolate_indentation(app)
ret_interp = interpolate_indentation(ret)


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
    approach: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:

    dF = make_force_integrand(constitutive, approach, tip)

    return integrate(dF, (0, t), (t,))


def make_force_integrand2(constitutive, approach, tip):

    a, b = tip.a(), tip.b()

    def dF(u, t):
        s = t * u
        g = constitutive.relaxation_function(jnp.clip(t - s, 0.0))
        dh_b = b * approach.derivative(s) * approach.evaluate(s) ** (b - 1)
        return a * g * dh_b * t  # from ds = tdu

    return dF


def force_approach_scalar2(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    approach: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:

    dF = make_force_integrand2(constitutive, approach, tip)

    return integrate(dF, (0, 1), (t,))


@eqx.filter_jit
def force_approach_vec(
    t: Float[Array, " time"],
    constitutive: AbstractConstitutiveEqn,
    approach: diffrax.AbstractPath,
    tip: AbstractTipGeometry,
) -> FloatScalar:
    a, b = tip.a(), tip.b()

    @partial(eqx.filter_vmap, in_axes=(None, 0))
    def _dF(u: FloatScalar, t: FloatScalar) -> FloatScalar:
        s = t * u
        g = constitutive.relaxation_function(jnp.clip(t - s, 0.0))
        dh_b = b * approach.derivative(s) * approach.evaluate(s) ** (b - 1)
        return g * dh_b  # from ds = tdu

    return a * t * integrate(_dF, (0, 1), (t,))


# %%
f_app = eqx.filter_jit(
    eqx.filter_vmap(force_approach_scalar, in_axes=(0, None, None, None))
)(app.time, constit, app_interp, tip)

f_app2 = eqx.filter_jit(
    eqx.filter_vmap(force_approach_scalar2, in_axes=(0, None, None, None))
)(app.time, constit, app_interp, tip)
f_vec = eqx.filter_jit(force_approach_vec)(app.time, constit, app_interp, tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, ".", label="Original")
ax.plot(app.time, f_app2, ".", label="Rescaled")
ax.plot(app.time, f_vec, ".", label="Vectorized")
ax.legend()


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


def make_t1_integrands2(constitutive, indentations) -> tuple[Callable, Callable]:
    app_interp, ret_interp = indentations
    t_m = app_interp.t1  # Unfortunate mixup of names between my code and diffrax
    # In diffrax.AbstractPath, t1 attribute returns the final timepoint of the path

    def t1_integrand_lower(u, t, t1):
        s = t1 + (t_m - t1) * u
        return (
            (t_m - t1)
            * constitutive.relaxation_function(t - s)
            * app_interp.derivative(s)
        )

    def t1_integrand_upper(u, t):
        s = t_m + (t - t_m) * u
        return (
            (t - t_m)
            * constitutive.relaxation_function(t - s)
            * ret_interp.derivative(s)
        )

    return t1_integrand_lower, t1_integrand_upper


def t1_scalar2(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    newton_iterations: int = 5,
) -> FloatScalar:
    t1_integrand_lower, t1_integrand_upper = make_t1_integrands2(
        constitutive, indentations
    )
    app_interp = indentations[0]
    t_m = app_interp.t1
    const = integrate(t1_integrand_upper, (0.0, 1.0), (t,))

    def residual(t1, t):
        return integrate(t1_integrand_lower, (0.0, 1.0), (t, t1)) + const

    def Dresidual(t1, t):
        return -constitutive.relaxation_function(t - t1) * app_interp.derivative(t1)

    t1 = t_m
    for _ in range(newton_iterations):
        t1 = jnp.clip(t1 - residual(t1, t) / Dresidual(t1, t), 0.0)

    return t1


@eqx.filter_jit
def t1_vec(
    t: Float[Array, " time"],
    constitutive: AbstractConstitutiveEqn,
    indentations: tuple[diffrax.AbstractPath, diffrax.AbstractPath],
    newton_iterations: int = 5,
) -> Float[Array, " time"]:
    app_interp, ret_interp = indentations
    t_m = app_interp.t1

    #@partial(eqx.filter_vmap, in_axes=(None, 0))
    def _integrand_upper(u, t):
        s = t_m + (t - t_m) * u
        return constitutive.relaxation_function(t - s) * ret_interp.derivative(s)

    const = (t - t_m) * integrate(_integrand_upper, (0.0, 1.0), (t,))

    @partial(eqx.filter_vmap, in_axes=(None, 0, 0))
    def _integrand_lower(u, t, t1):
        s = t1 + (t_m - t1) * u
        return constitutive.relaxation_function(t - s) * app_interp.derivative(s)

    def residual(t1, t):
        return (t_m - t1) * integrate(_integrand_lower, (0.0, 1.0), (t, t1)) + const

    def Dresidual(t1, t):
        return -_integrand_lower(0.0, t1, t)

    t1 = t
    for _ in range(newton_iterations):
        t1 = jnp.clip(t1 - residual(t1, t) / Dresidual(t1, t), 0.0)

    return t1


# %%
%%timeit
t1_1 = eqx.filter_jit(eqx.filter_vmap(t1_scalar, in_axes=(0, None, None, None)))(
    ret.time, constit, (app_interp, ret_interp)
)
#%%
%%timeit
t1_2 = eqx.filter_jit(eqx.filter_vmap(t1_scalar2, in_axes=(0, None, None, None)))(
    ret.time, constit, (app_interp, ret_interp)
)
#%%
%%timeit
t1_3 = t1_vec(ret.time, constit, (app_interp, ret_interp))

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(ret.time, t1_1, ".", label="Original")
ax.plot(ret.time, t1_2, ".", label="Rescaled")
ax.plot(ret.time, t1_3, ".", label="Vectorized")
ax.legend()
# %%
