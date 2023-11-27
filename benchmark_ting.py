# %%
from typing import Callable
from functools import partial

import jax
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

jax.config.update("jax_enable_x64", True)


@partial(eqx.filter_vmap, in_axes=(0, None, None, None))
def force_approach(
    t: float,
    relaxation: Callable[[Array], Array],
    approach: Trajectory,
    tip: AbstractTipGeometry,
) -> Array:
    t_s = approach.t
    b = tip.b()

    def force_integrand(t_: Array, t: float) -> Array:
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
    def app_integrand(t_: Array, t: float) -> Array:
        return relaxation(t - t_) * approach.v(t_)

    def ret_integrand(t_: Array, t: float) -> Array:
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
) -> Array:
    t_s = approach.t
    b = tip.b()

    def force_integrand(t_: Array, t: float) -> Array:
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
    def app_integrand(t_: Array, t: float) -> Array:
        return relaxation(t - t_) * approach.v(t_)

    def ret_integrand(t_: Array, t: float) -> Array:
        return relaxation(t - t_) * retract.v(t_)

    t_app, t_ret = approach.t, retract.t
    # constant = integrate_to(t, t_ret, ret_integrand(t_ret, t))

    def t1_constraint(t1, args=None):
        val = integrate_from(t1, t_app, app_integrand(t_app, t)) + integrate_to(
            t, t_ret, ret_integrand(t_ret, t)
        )
        return val

    def _find_t1():
        solver = optx.Newton(rtol=1e-5, atol=1e-5)
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
find_t12(ret.t[2], plr.relaxation_function, app, ret)


# %%
def fn(y, args):
    x = jnp.linspace(-0.5, 1.0, 100)
    h = x - 1
    val = _integrate_from(y, x, h)
    return val


solver = optx.Newton(rtol=1e-4, atol=1e-4)
y0 = 0.0 * jnp.pi
sol = optx.root_find(fn, solver, y0)
# %%
sol
# %%
x = jnp.linspace(-jnp.pi, 2 * jnp.pi, 100)
h = jnp.sin(x)
integrate_from(0, x, h)
# %%
from neuralconstitutive.jax.integrate import _integrate_to, _integrate_from

# %%
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import optimistix as optx


# Seek `y` such that `y - tanh(y + 1) = 0`.
@eqx.filter_jit
def fn2(y, args):
    relaxation, approach, retract, t = args

    def app_integrand(t_: Array, t: float) -> Array:
        return relaxation(t - t_) * approach.v(t_)

    def ret_integrand(t_: Array, t: float) -> Array:
        return relaxation(t - t_) * retract.v(t_)

    t_app, t_ret = approach.t, retract.t
    # constant = integrate_to(t, t_ret, ret_integrand(t_ret, t))

    out = integrate_from(y, t_app, app_integrand(t_app, t)) + integrate_to(
        t, t_ret, ret_integrand(t_ret, t)
    )
    aux = None
    return out, aux


@eqx.filter_jit
def fn(y, args):
    relaxation, approach, retract, t = args

    def app_integrand(t_: Array) -> Array:
        return relaxation(t - t_) * approach.v(t_)

    def ret_integrand(t_: Array) -> Array:
        return relaxation(t - t_) * retract.v(t_)

    t_app, t_ret = approach.t, retract.t
    dt = t_app[1] - t_app[0]
    # constant = integrate_to(t, t_ret, ret_integrand(t_ret, t))

    out = integrate(app_integrand, y, t_app[-1], dt) + integrate(
        ret_integrand, t_ret[0], t, dt
    )
    aux = None
    return out, aux


solver = optx.Bisection(rtol=1e-3, atol=1e-3)
# The initial guess for the solution
y = jnp.array(0.5)
# Any auxiliary information to pass to `fn`.
args = (plr.relaxation_function, app, ret, ret.t[10])
# The interval to search over. Required for `optx.Bisection`.
options = dict(lower=0.0, upper=1.0)
# The shape+dtype of the output of `fn`
f_struct = jax.ShapeDtypeStruct((), jnp.float64)
aux_struct = None
# Any Lineax tags describing the structure of the Jacobian matrix d(fn)/dy.
# (In this scale it's a scalar, so these don't matter.)
tags = frozenset()


def solve(y, solver):
    # These arguments are always fixed throughout interactive solves.
    step = eqx.filter_jit(
        eqx.Partial(solver.step, fn=fn, args=args, options=options, tags=tags)
    )
    terminate = eqx.filter_jit(
        eqx.Partial(solver.terminate, fn=fn, args=args, options=options, tags=tags)
    )

    # Initial state before we start solving.
    state = solver.init(fn, y, args, options, f_struct, aux_struct, tags)
    done, result = terminate(y=y, state=state)

    # Alright, enough setup. Let's do the solve!
    while not done:
        print(f"Evaluating point {y} with value {fn(y, args)[0]}.")
        y, state, aux = step(y=y, state=state)
        done, result = terminate(y=y, state=state)
    if result != optx.RESULTS.successful:
        print(f"Oh no! Got error {result}.")
    y, _, _ = solver.postprocess(fn, y, aux, args, options, state, tags, result)
    print(f"Found solution {y} with value {fn(y, args)[0]}.")


solve(y, solver)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
y_array = jnp.linspace(0.7, 0.9, 10000)
errors = jax.vmap(fn, in_axes=(0, None))(y_array, args)[0]
ax.plot(y_array, errors, ".", markersize=1.0)
# %%
y_array.shape
# %%
errors2 = jax.vmap(t1_constraint, in_axes=(0, None, None, None, None))(
    y_array, ret.t[10], plr.relaxation_function, app, ret
)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(y_array, errors2, ".")
# %%
test = plr.relaxation_function(ret.t[10] - y_array) * app.v(y_array)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(y_array, test, "-")
# %%
x = jnp.linspace(-0.5, 1.0, 100)
h = x - 1
y = jnp.linspace(-0.5, 0.5, 10000)
val = eqx.filter_vmap(integrate_from, in_axes=(0, None, None))(y, x, h)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(y, val, ".")


# %%
def integrate(f: Callable[[float], Array], x_lower: float, x_upper: float, dx: float):
    def _body_fun(val: tuple[float, Array]) -> tuple[float, Array]:
        x, I_x = val
        x_next = jnp.clip(x + dx, x_lower, x_upper)
        dx_ = x_next - x
        I_x += 0.5 * (f(x) + f(x_next)) * dx_
        return x_next, I_x

    def _cond_fun(val: tuple[float, Array]) -> bool:
        x, _ = val
        return x < x_upper

    _init_val = (x_lower, jnp.zeros_like(f(x_lower)))

    _, I_x = jax.lax.while_loop(_cond_fun, _body_fun, _init_val)
    return I_x


# %%
val2 = eqx.filter_vmap(integrate, in_axes=(None, 0, None, None))(
    lambda x: x - 1, y, 1.0, x[1] - x[0]
)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(y, val2, ".", markersize=0.1)

# %%
