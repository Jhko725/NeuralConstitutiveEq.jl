from typing import Callable
from functools import partial

from jax import Array, vmap
import jax.numpy as jnp
import jaxopt

from .integrate import integrate_from, integrate_to


@partial(vmap, in_axes=(0, None, None, None, None, None, None))
def force_approach(
    t: float,
    model: Callable,
    t_app: Array,
    d_app: Array,
    v_app: Array,
    a: float,
    b: float,
):
    phi_app: Array = model(t - t_app)
    integrand = phi_app * v_app * d_app ** (b - 1)
    return integrate_to(t, t_app, integrand) * a


@partial(vmap, in_axes=(0, 0, None, None, None, None, None, None))
def force_retract(
    t: float,
    t1: float,
    model: Callable,
    t_app: Array,
    d_app: Array,
    v_app: Array,
    a: float,
    b: float,
):
    phi_app: Array = model(t - t_app)
    integrand = phi_app * v_app * d_app ** (b - 1)
    return integrate_to(t1, t_app, integrand) * a


@partial(vmap, in_axes=(0, None, None, None, None, None))
def find_t1(
    t: float,
    model: Callable,
    t_app: Array,
    t_ret: Array,
    v_app: Array,
    v_ret: Array,
) -> Array:
    sol_exists = t1_constraint(t_app[0], t, model, t_app, t_ret, v_app, v_ret) > 0.0
    return jnp.where(
        sol_exists, _find_t1(t, model, t_app, t_ret, v_app, v_ret), jnp.asarray(0.0)
    )


def _find_t1(
    t: float,
    model: Callable,
    t_app: Array,
    t_ret: Array,
    v_app: Array,
    v_ret: Array,
) -> Array:
    root_finder = jaxopt.Bisection(
        optimality_fun=t1_constraint,
        lower=t_app[0],
        upper=t_app[-1],
        check_bracket=False,
    )
    return root_finder.run(
        t=t,
        model=model,
        t_app=t_app,
        t_ret=t_ret,
        v_app=v_app,
        v_ret=v_ret,
    ).params


def t1_constraint(
    t1: float,
    t: float,
    model: Callable,
    t_app: Array,
    t_ret: Array,
    v_app: Array,
    v_ret: Array,
) -> Array:
    phi_app, phi_ret = model(t - t_app), model(t - t_ret)
    return integrate_from(t1, t_app, phi_app * v_app) + integrate_to(
        t, t_ret, phi_ret * v_ret
    )
