# %%
# ruff: noqa: F722
from functools import partial

import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import Float
import equinox as eqx
import diffrax
import optimistix as optx
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
    PowerLaw,
    ModifiedPowerLaw,
)
from neuralconstitutive.jax.tipgeometry import AbstractTipGeometry, Conical
from neuralconstitutive.indentation import Indentation

# from neuralconstitutive.jax.ting import force_approach as force_approach2

jax.config.update("jax_enable_x64", True)


def make_force_integrand(
    indentation: Indentation,
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
):
    a, b = tip.a(), tip.b()
    interp = interpolate_indentation(indentation)

    @jax.jit
    def dF(u: float, t: float) -> float:
        return (
            a
            * b
            * constitutive.relaxation_function(u)
            * interp.derivative(t - u)
            * interp.evaluate(t - u) ** (b - 1)
        )

    return dF


def make_force_integrand_vec(
    indentation: Indentation,
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
):
    a, b = tip.a(), tip.b()
    interp = interpolate_indentation(indentation)

    @eqx.filter_vmap
    def d_hb(s: float) -> float:
        return interp.derivative(s) * interp.evaluate(s) ** (b - 1)

    @jax.jit
    def dF(u: float, t: Float[Array, " N"]) -> Float[Array, " N"]:
        s = jnp.clip(t - u, 0.0, None)
        out = jnp.where(s >= 0, d_hb(s), 0.0)
        return a * b * constitutive.relaxation_function(u) * out

    return dF


def make_force_integrand_vec2(
    indentation: Indentation,
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
):
    a, b = tip.a(), tip.b()
    interp = interpolate_indentation(indentation)

    def d_hb(s: float) -> float:
        return interp.derivative(s) * interp.evaluate(s) ** (b - 1)

    @partial(eqx.filter_vmap, in_axes=(0, None))
    @partial(eqx.filter_vmap, in_axes=(None, 0))
    def dF(u: float, t: float) -> float:
        s = jnp.clip(t - u, 0.0, None)
        out = jnp.where(s >= 0, d_hb(s), 0.0)
        return a * b * constitutive.relaxation_function(u) * out

    return dF


def force_approach(
    approach: Indentation,
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
) -> Float[Array, " {len(approach)}"]:

    dF = make_force_integrand(approach, constitutive, tip)
    force = []

    for t in approach.time:

        F_i, _ = scipy.integrate.quad(
            dF,
            0,
            t,
            args=(t,),
            points=[
                0.0,
            ],
        )

        force.append(F_i)

    force = jnp.asarray(force)
    return force


def force_approach_vec(
    approach: Indentation,
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
) -> Float[Array, " {len(approach)}"]:

    dF_vec = make_force_integrand_vec(approach, constitutive, tip)
    t = approach.time
    force, _ = scipy.integrate.quad_vec(
        dF_vec,
        0,
        t[-1],
        args=(t,),
        points=t,
    )
    force = jnp.asarray(force)
    return force


def force_approach_midpoint(
    approach: Indentation,
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
) -> Float[Array, " {len(approach)}"]:
    t = approach.time
    du = t[1] - t[0]
    u = t[0:-1] + 0.5 * du
    dF_vec = make_force_integrand_vec2(approach, constitutive, tip)
    return jnp.sum(dF_vec(u, t), axis=0) * du


def force_powerlaw(
    indentation: Indentation, constitutive: PowerLaw, tip: AbstractTipGeometry
) -> Float[Array, " {len(approach)}"]:
    a, b = tip.a(), tip.b()
    E0, alpha, t0 = constitutive.E0, constitutive.alpha, constitutive.t0
    v = (indentation.depth[-1] - indentation.depth[0]) / (
        indentation.time[-1] - indentation.time[0]
    )
    Beta = jnp.exp(jax.scipy.special.betaln(1 - alpha, b))
    coeff = a * b * E0 * (v**b) * (t0 ** (alpha)) * Beta
    return coeff * (indentation.time ** (b - alpha))


# %%
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
h = 1.0 * t

app = Indentation(t, h)
plr = PowerLaw(1.0, 0.5, 1.0)
# plr = ModifiedPowerLaw(1.0, -0.5, 1.0)
tip = Conical(jnp.pi / 18)
# %%
# %%time
f_app_mid = force_approach_midpoint(app, plr, tip)

# %%
# %%time
f_app = force_approach(app, plr, tip)
# %%
# %%time
f_app_vec = force_approach_vec(app, plr, tip)
# %%
f_true = force_powerlaw(app, plr, tip)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# ax.plot(app.time, f_app, ".", markersize=1.0, label="Ting")
ax.plot(app.time, f_true, ".", markersize=1.0, label="Analytic")
ax.plot(app.time, f_app_vec, ".", markersize=1.0, label="Ting_vectorized")
ax.plot(app.time, f_app_mid, ".", markersize=1.0, label="Ting_midpoint")
ax.legend()


# %%
def find_t1(
    indentations: tuple[Indentation, Indentation], constitutive: AbstractConstitutiveEqn
):
    approach, retract = indentations
    t_m = approach.time[-1]
    d_res = make_t1_integrand(indentations, constitutive)

    def residual(t1: float, t: float) -> float:
        return scipy.integrate.quad(
            d_res,
            0.0,
            t - t1,
            args=(t,),
            points=(
                0.0,
                t - t_m,
            ),
        )[0]

    t1 = []

    for t in tqdm(retract.time):
        try:
            t1_i = scipy.optimize.root_scalar(
                residual, args=(t,), method="bisect", bracket=(0, t_m)
            )
        except ValueError:
            t1_i = 0.0
    t1.append(t1_i)
    t1 = jnp.asarray(t1)
    return t1


def make_t1_integrand(
    indentations: tuple[Indentation, Indentation], constitutive: AbstractConstitutiveEqn
):
    approach, retract = indentations
    interp_app = interpolate_indentation(approach)
    interp_ret = interpolate_indentation(retract)
    t_m = approach.time[-1]
    assert t_m == retract.time[0]

    @jax.jit
    def v(t: float) -> float:
        return jax.lax.cond(
            t <= t_m,
            lambda u: interp_app.derivative(u),
            lambda u: interp_ret.derivative(u),
            t,
        )

    @jax.jit
    def integrand(u: float, t: float) -> float:
        return constitutive.relaxation_function(u) * v(t - u)

    return integrand


def make_t1_integrand_midpoint(
    indentations: tuple[Indentation, Indentation],
    constitutive: AbstractConstitutiveEqn,
):
    approach, retract = indentations
    interp_app = interpolate_indentation(approach)
    interp_ret = interpolate_indentation(retract)
    t_m = approach.time[-1]
    assert t_m == retract.time[0]

    @partial(eqx.filter_vmap, in_axes=(0, None, None))
    def d_residual(u: float, t: float, t1: float) -> float:

        out = jnp.where(
            u > t - t1,
            0.0,
            jnp.where(
                u < t - t_m, interp_ret.derivative(t - u), interp_app.derivative(t - u)
            ),
        )
        return out * constitutive.relaxation_function(u)

    return d_residual


def find_t1_midpoint(
    indentations: tuple[Indentation, Indentation], constitutive: AbstractConstitutiveEqn
):
    approach, retract = indentations
    t_m = approach.time[-1]
    d_res = make_t1_integrand_midpoint(indentations, constitutive)

    du = approach.time[1] - approach.time[0]
    u = jnp.arange(approach.time[0], retract.time[-1], du) + 0.5 * du

    @jax.jit
    def residual(t1: float, t: float) -> float:
        return jnp.sum(d_res(u, t, t1), axis=0) * du

    t1 = []

    for t in tqdm(retract.time):
        try:
            t1_i = scipy.optimize.root_scalar(
                residual, args=(t,), method="bisect", bracket=(0, t_m)
            ).root
        except ValueError:
            t1_i = 0.0
        t1.append(t1_i)
    t1 = jnp.asarray(t1)
    return t1


def find_t1_midpoint_optx(
    indentations: tuple[Indentation, Indentation], constitutive: AbstractConstitutiveEqn
):
    approach, retract = indentations
    t_m = approach.time[-1]
    d_res = make_t1_integrand_midpoint(indentations, constitutive)

    du = approach.time[1] - approach.time[0]
    u = jnp.arange(approach.time[0], retract.time[-1], du) + 0.5 * du

    @eqx.filter_jit
    def residual(t1: float, t: tuple[float]) -> float:
        t = t[0]
        return jnp.sum(d_res(u, t, t1), axis=0) * du

    solver = optx.Bisection(rtol=1e-3, atol=1e-3, flip=True)

    @eqx.filter_vmap
    def _find_t1(t: float) -> float:
        t1 = jax.lax.cond(
            residual(0.0, (t,)) <= 0,
            lambda _: 0.0,
            lambda t: optx.root_find(
                residual,
                solver,
                t_m,
                args=(t,),
                options={"lower": 0.0, "upper": t_m},
                max_steps=30,
                throw=False,
            ).value,
            t,
        )

        return t1

    return _find_t1(retract.time)


# %%
t_ret = jnp.arange(1.0, 2.0 + 1e-3, 1e-3)
h_ret = 2.0 - t_ret
ret = Indentation(t_ret, h_ret)

t1 = find_t1_midpoint((app, ret), plr)
# %%
t1_quad = find_t1((app, ret), plr)
# %%

t1_vec = find_t1_midpoint_optx((app, ret), plr)
# %%
t1_finder = find_t1_midpoint_optx((app, ret), plr)
# %%

t1_finder(jnp.asarray([1.0, 1.0, 1.0]))
# %%

t1_finder(jnp.asarray([1.5]))
# %%

t1_finder(jnp.asarray([1.2, 1.2, 1.2]))
# %%
du = app.time[1] - app.time[0]
u = jnp.arange(app.time[0], ret.time[-1], du) + 0.5 * du
d_resid(u, ret.time, 0.5).shape


# %%
def t1_powerlaw(
    indentations: tuple[Indentation, Indentation], constitutive: PowerLaw
) -> Float[Array, " {len(indentations[1])}"]:
    approach, retract = indentations
    t_m = approach.time[-1]
    alpha = constitutive.alpha
    t = retract.time
    t1 = t - (2 ** (1 / (1 - alpha))) * (t - t_m)
    return jnp.clip(t1, approach.time[0], None)


def retract_powerlaw(
    indentations: tuple[Indentation, Indentation],
    constitutive: PowerLaw,
    tip: AbstractTipGeometry,
) -> Float[Array, " {len(indentations[1])}"]:
    a, b = tip.a(), tip.b()
    approach, retract = indentations
    E0, alpha, t0 = constitutive.E0, constitutive.alpha, constitutive.t0
    v = (approach.depth[-1] - approach.depth[0]) / (
        approach.time[-1] - approach.time[0]
    )
    t1 = t1_powerlaw(indentations, constitutive)
    Beta = jnp.exp(jax.scipy.special.betaln(1 - alpha, b))
    # BetaInc = Beta * jax.scipy.special.betainc(1 - alpha, b, t1 / retract.time)
    # print(t1 / retract.time)
    # coeff = a * b * E0 * (v**b) * (t0 ** (alpha)) * BetaInc
    return a * b * E0 * (v**b) * (t0 ** (alpha)) * Beta * t1 ** (b - alpha)
    # return coeff * (retract.time ** (b - alpha))


# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(ret.time, t1, label="Numerical (JAX, midpoint, scipy)")
ax.plot(ret.time, t1_vec, label="Numerical (JAX, midpoint, optimistix)")
ax.plot(ret.time, t1_powerlaw((app, ret), plr), label="Analytic result")
ax.legend()
# %%
t1_vec


# %%
def make_retract_integrand_vec2(
    indentation: Indentation,
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
):
    a, b = tip.a(), tip.b()
    interp = interpolate_indentation(indentation)

    def d_hb(s: float) -> float:
        return interp.derivative(s) * interp.evaluate(s) ** (b - 1)

    @partial(eqx.filter_vmap, in_axes=(0, None, None))
    @partial(eqx.filter_vmap, in_axes=(None, 0, 0))
    def dF(u: float, t: float, t1: float) -> float:
        out = jnp.where(
            u < t - t1,
            0.0,
            jnp.where(u < t, d_hb(t - u), 0.0),
        )
        return a * b * constitutive.relaxation_function(u) * out

    return dF


def force_retract_midpoint(
    indentations: tuple[Indentation, Indentation],
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
) -> Float[Array, " {len(approach)}"]:
    approach, retract = indentations

    t1 = find_t1_midpoint_optx(indentations, constitutive)
    t_ret = retract.time
    t_app = approach.time
    du = t_app[1] - t_app[0]
    u = t_app[0:-1] + 0.5 * du
    dF_vec = make_retract_integrand_vec2(approach, constitutive, tip)
    return jnp.sum(dF_vec(u, t_ret, t1), axis=0) * du


# %%

f_ret = force_retract_midpoint((app, ret), plr, tip)
# %%
f_ret_true = retract_powerlaw((app, ret), plr, tip)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(ret.time, f_ret, label="JAX, midpoint")
ax.plot(ret.time, f_ret_true, label="Analytic")
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_mid, label="JAX, midpoint")
ax.plot(ret.time, f_ret, label="JAX, midpoint")
# %%
import jax.numpy as jnp
import quadax
import equinox as eqx


def sine(x, w):
    return jnp.sin(w * x)


def integrate(func, lower, upper, args):
    return quadax.quadgk(func, (lower, upper), args)


x = jnp.arange(0.0, 3.0, 0.2)
y = eqx.filter_vmap(integrate, in_axes=(None, None, 0, None))(sine, 0.0, x, (1.0,))
# %%
import matplotlib.pyplot as plt

plt.plot(x, y[0])
# %%
