# %%
from typing import Callable
from functools import partial

from jax import Array
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import diffrax
import jaxopt
import optimistix as optx

from neuralconstitutive.jax.integrate import integrate_to, integrate_from
from neuralconstitutive.jax.tipgeometry import AbstractTipGeometry, Spherical
from neuralconstitutive.constitutive import ModifiedPowerLaw


class Indentation(eqx.Module):
    time: Array
    indent: Array
    velocity: Array


@partial(eqx.filter_vmap, in_axes=(0, None, None, None))
def force_approach(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Indentation,
    tip: AbstractTipGeometry,
):
    phi_app = relaxation(t - approach.time)
    integrand = phi_app * approach.velocity * approach.indent ** (tip.b() - 1)
    return integrate_to(t, approach.time, integrand) * tip.a()


# %%
plr = ModifiedPowerLaw(572.0, 0.2, 1e-5)
tip = Spherical(1.0)
Dt = 1e-2
t_app = jnp.arange(0, 101) * Dt
t_ret = jnp.arange(100, 201) * Dt
v_app = 10.0 * jnp.ones_like(t_app)
v_ret = -v_app
d_app = v_app * t_app
d_ret = v_app * (2 * t_ret[0] - t_ret)
app = Indentation(t_app, d_app, v_app)
ret = Indentation(t_ret, d_ret, v_ret)
# %%
f_app = force_approach(app.time, plr.relaxation_function, app, tip)

# %%
plr.relaxation_spectrum()
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app)
# %%
app2 = diffrax.LinearInterpolation(t_app, t_app * v_app)

# %%
app2.derivative(t_app)


# %%
@partial(eqx.filter_vmap, in_axes=(0, None, None, None))
def force_approach2(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: diffrax.AbstractGlobalInterpolation,
    tip: AbstractTipGeometry,
):
    ts = approach.ts
    b = tip.b()

    @partial(eqx.filter_vmap, in_axes=(0, None))
    def force_integrand(t_, t):
        return (
            relaxation(t - t_)
            * approach.derivative(t_)
            * approach.evaluate(t_) ** (b - 1)
        )

    return integrate_to(t, ts, force_integrand(ts, t)) * tip.a()


# %%
f_app2 = force_approach2(app2.ts, plr.relaxation_function, app2, tip)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app2)


# %%
def t1_constraint(
    t1: float,
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Indentation,
    retract: Indentation,
) -> float:
    t_app, t_ret = approach.time, retract.time
    v_app, v_ret = approach.velocity, retract.velocity
    phi_app = relaxation(t - t_app)
    phi_ret = relaxation(t - t_ret)
    return integrate_from(t1, t_app, phi_app * v_app) + integrate_to(
        t, t_ret, phi_ret * v_ret
    )


@partial(eqx.filter_vmap, in_axes=(0, None, None, None))
def find_t1(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Indentation,
    retract: Indentation,
) -> Array:
    sol_exists = t1_constraint(approach.time[0], t, relaxation, approach, retract) > 0.0
    return jnp.where(
        sol_exists, _find_t1(t, relaxation, approach, retract), jnp.asarray(0.0)
    )


def _find_t1(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Indentation,
    retract: Indentation,
) -> Array:
    root_finder = jaxopt.Bisection(
        optimality_fun=t1_constraint,
        lower=approach.time[0],
        upper=approach.time[-1],
        check_bracket=False,
    )
    return root_finder.run(
        t=t,
        relaxation=relaxation,
        approach=approach,
        retract=retract,
    ).params


# %%
find_t1(ret.time, plr.relaxation_function, app, ret)


# %%
# @partial(eqx.filter_vmap, in_axes=(0, None, None, None))
def find_t12(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Indentation,
    retract: Indentation,
) -> Array:
    t_app, t_ret = approach.time, retract.time
    v_app, v_ret = approach.velocity, retract.velocity
    phi_app = relaxation(t - t_app)
    phi_ret = relaxation(t - t_ret)
    constant = integrate_to(t, t_ret, phi_ret * v_ret)

    def t1_constraint(t1, args=None):
        return integrate_from(t1, t_app, phi_app * v_app) + constant

    def _find_t1():
        solver = optx.Bisection(rtol=1e-2, atol=1e-2)
        sol = optx.root_find(
            t1_constraint,
            solver,
            0.5 * (t_app[0] + t_app[-1]),
            options={"lower": t_app[0], "upper": t_app[-1]},
        )
        return sol.value

    sol_exists = t1_constraint(t_app[0]) >= 0.0

    return jnp.where(sol_exists, _find_t1(), jnp.asarray(t_app[0]))


# %%
find_t12(ret.time[10], plr.relaxation_function, app, ret)
# %%
