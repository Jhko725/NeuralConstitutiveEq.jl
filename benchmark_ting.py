# %%
from typing import Callable
from functools import partial

from jax import Array
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import jaxopt
import optimistix as optx

from neuralconstitutive.jax.integrate import integrate_to, integrate_from
from neuralconstitutive.jax.tipgeometry import AbstractTipGeometry, Spherical
from neuralconstitutive.constitutive import ModifiedPowerLaw
from neuralconstitutive.trajectory import Trajectory, make_triangular


@partial(eqx.filter_vmap, in_axes=(0, None, None, None))
def force_approach(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Trajectory,
    tip: AbstractTipGeometry,
):
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
def force_retract(
    t: float,
    t1: float,
    relaxation: Callable[[Array], Array],
    approach: Trajectory,
    tip: AbstractTipGeometry,
):
    t_s = approach.t
    b = tip.b()

    @partial(eqx.filter_vmap, in_axes=(0, None))
    def force_integrand(t_, t):
        return relaxation(t - t_) * approach.v(t_) * approach.z(t_) ** (b - 1)

    return integrate_to(t1, t_s, force_integrand(t_s, t)) * tip.a()


# %%
## Attempt to use optimistix: doesn't work
## Bisection does not properly converge at the moment...
# @partial(eqx.filter_vmap, in_axes=(0, None, None, None))
def find_t12(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Trajectory,
    retract: Trajectory,
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
plr = ModifiedPowerLaw(572.0, 0.2, 1e-5)
tip = Spherical(1.0)


app, ret = make_triangular(1.0, 1e-2, 10.0)
# %%
f_app = force_approach(app.t, plr.relaxation_function, app, tip)
t1 = find_t1(ret.t, plr.relaxation_function, app, ret)
f_ret = force_retract(ret.t, t1, plr.relaxation_function, app, tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.t, f_app, ".")
ax.plot(ret.t, f_ret, ".")

# %%
