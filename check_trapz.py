# %%
from typing import Callable
from functools import partial

import jax
from jax import Array, jit
import jax.numpy as jnp

import jaxopt
import matplotlib.pyplot as plt

from neuralconstitutive.jax.constitutive import SimpleLinearSolid
from neuralconstitutive.jax.integrate import integrate_from, integrate_to

jax.config.update("jax_enable_x64", True)


sls = SimpleLinearSolid(E0=8.0, E_inf=2.0, tau=0.01)
t_app = jnp.linspace(0, 0.2, 100)
t_ret = jnp.linspace(0.2, 0.4, 100)
sls(t_app)
d_app = 10.0 * t_app
v_app = 10.0 * jnp.ones_like(t_app)
v_ret = -10.0 * jnp.ones_like(t_ret)
# %%
grads = jax.grad(lambda t, model: model(t), argnums=1)(0.1, sls)
grads.E0


# %%
def objective(
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


@partial(jax.vmap, in_axes=(0, None, None, None, None, None))
def find_t1(
    t: float,
    model: Callable,
    t_app: Array,
    t_ret: Array,
    v_app: Array,
    v_ret: Array,
) -> Array:
    sol_exists = objective(0.0, t, model, t_app, t_ret, v_app, v_ret) > 0.0
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
        optimality_fun=objective, lower=0.0, upper=0.2, check_bracket=False
    )
    return root_finder.run(
        t=t,
        model=model,
        t_app=t_app,
        t_ret=t_ret,
        v_app=v_app,
        v_ret=v_ret,
    ).params


# %%
find_t1(t_ret, sls, t_app, t_ret, v_app, v_ret)


# %%
@partial(jax.vmap, in_axes=(0, None, None, None, None, None, None))
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


@partial(jax.vmap, in_axes=(0, 0, None, None, None, None, None, None))
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


# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
F_app = force_approach(t_app, sls, t_app, d_app, v_app, 1.0, 2.0)
t1 = find_t1(t_ret, sls, t_app, t_ret, v_app, v_ret)
F_ret = force_retract(t_ret, t1, sls, t_app, d_app, v_app, 1.0, 2.0)
ax.plot(t_app, F_app, label="approach")
ax.plot(t_ret, F_ret, label="retract")
ax.legend()
fig
