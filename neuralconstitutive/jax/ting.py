from typing import Callable
from functools import partial

from jax import Array
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx

from neuralconstitutive.jax.integrate import integrate
from neuralconstitutive.jax.tipgeometry import AbstractTipGeometry
from neuralconstitutive.trajectory import Trajectory


def force_approach(
    approach: Trajectory, relaxation: Callable[[Array], Array], tip: AbstractTipGeometry
) -> Array:
    return _force_approach(approach.t, relaxation, approach, tip)


def force_retract(
    approach: Trajectory,
    retract: Trajectory,
    relaxation: Callable[[Array], Array],
    tip: AbstractTipGeometry,
) -> Array:
    t1 = find_t1(retract.t, relaxation, approach, retract)
    return _force_retract(retract.t, t1, relaxation, approach, tip)


@partial(eqx.filter_vmap, in_axes=(0, None, None, None))
def _force_approach(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Trajectory,
    tip: AbstractTipGeometry,
) -> Array:
    t_app = approach.t
    dx = t_app[1] - t_app[0]
    a, b = tip.a(), tip.b()

    def force_integrand(t_: Array) -> Array:
        return relaxation(t - t_) * approach.v(t_) * approach.z(t_) ** (b - 1)

    return a * integrate(force_integrand, t_app[0], t, dx)


@partial(eqx.filter_vmap, in_axes=(0, None, None, None))
def find_t1(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Trajectory,
    retract: Trajectory,
) -> Array:
    def app_integrand(t_: Array) -> Array:
        return relaxation(t - t_) * approach.v(t_)

    def ret_integrand(t_: Array) -> Array:
        return relaxation(t - t_) * retract.v(t_)

    t_app, t_ret = approach.t, retract.t
    dt = t_app[1] - t_app[0]
    constant = integrate(ret_integrand, t_ret[0], t, dt)

    def t1_constraint(t1, args=None):
        out = integrate(app_integrand, t1, t_app[-1], dt) + constant
        return out

    solver = optx.Bisection(rtol=1e-3, atol=1e-3, flip=True)
    sol = optx.root_find(
        t1_constraint,
        solver,
        0.5 * (t_app[0] + t_app[-1]),
        options={"lower": t_app[0], "upper": t_app[-1]},
        max_steps=30,
        throw=False,
    )

    condlist = jnp.asarray(
        [
            t == t_ret[0],
            sol.result != optx.RESULTS.successful,
            sol.result == optx.RESULTS.successful,
        ]
    )
    choicelist = jnp.asarray([t_ret[0], t_app[0], sol.value])

    return jnp.select(condlist, choicelist)


@partial(eqx.filter_vmap, in_axes=(0, 0, None, None, None))
def _force_retract(
    t: float,
    t1: float,
    relaxation: Callable[[Array], Array],
    approach: Trajectory,
    tip: AbstractTipGeometry,
) -> Array:
    t_app = approach.t
    dx = t_app[1] - t_app[0]
    a, b = tip.a(), tip.b()

    def force_integrand(t_: Array) -> Array:
        return relaxation(t - t_) * approach.v(t_) * approach.z(t_) ** (b - 1)

    return a * integrate(force_integrand, t_app[0], t1, dx)
