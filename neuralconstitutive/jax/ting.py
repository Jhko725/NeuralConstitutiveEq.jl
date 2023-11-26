from typing import Callable
from functools import partial

from jax import Array
import jax.numpy as jnp
import equinox as eqx
import jaxopt

from neuralconstitutive.jax.integrate import integrate_to, integrate_from
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
    t_s = approach.t
    b = tip.b()

    @partial(eqx.filter_vmap, in_axes=(0, None))
    def force_integrand(t_, t):
        return relaxation(t - t_) * approach.v(t_) * approach.z(t_) ** (b - 1)

    return integrate_to(t, t_s, force_integrand(t_s, t)) * tip.a()


# %%
def t1_constraint(
    t1: float,
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Trajectory,
    retract: Trajectory,
) -> float:
    @partial(eqx.filter_vmap, in_axes=(0, None))
    def app_integrand(t_, t):
        return relaxation(t - t_) * approach.v(t_)

    @partial(eqx.filter_vmap, in_axes=(0, None))
    def ret_integrand(t_, t):
        return relaxation(t - t_) * retract.v(t_)

    t_app, t_ret = approach.t, retract.t
    return integrate_from(t1, t_app, app_integrand(t_app, t)) + integrate_to(
        t, t_ret, ret_integrand(t_ret, t)
    )


@partial(eqx.filter_vmap, in_axes=(0, None, None, None))
def find_t1(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Trajectory,
    retract: Trajectory,
) -> Array:
    sol_exists = t1_constraint(approach.t[0], t, relaxation, approach, retract) > 0.0
    return jnp.where(
        sol_exists, _find_t1(t, relaxation, approach, retract), jnp.asarray(0.0)
    )


def _find_t1(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Trajectory,
    retract: Trajectory,
) -> Array:
    root_finder = jaxopt.Bisection(
        optimality_fun=t1_constraint,
        lower=approach.t[0],
        upper=approach.t[-1],
        check_bracket=False,
    )
    return root_finder.run(
        t=t,
        relaxation=relaxation,
        approach=approach,
        retract=retract,
    ).params


@partial(eqx.filter_vmap, in_axes=(0, 0, None, None, None))
def _force_retract(
    t: float,
    t1: float,
    relaxation: Callable[[Array], Array],
    approach: Trajectory,
    tip: AbstractTipGeometry,
) -> Array:
    t_s = approach.t
    b = tip.b()

    @partial(eqx.filter_vmap, in_axes=(0, None))
    def force_integrand(t_, t):
        return relaxation(t - t_) * approach.v(t_) * approach.z(t_) ** (b - 1)

    return integrate_to(t1, t_s, force_integrand(t_s, t)) * tip.a()
