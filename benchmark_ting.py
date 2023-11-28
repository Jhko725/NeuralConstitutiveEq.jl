# %%
import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import optimistix as optx

from neuralconstitutive.jax.integrate import integrate
from neuralconstitutive.jax.ting import force_approach, force_retract
from neuralconstitutive.jax.tipgeometry import Spherical
from neuralconstitutive.constitutive import ModifiedPowerLaw
from neuralconstitutive.trajectory import make_triangular

jax.config.update("jax_enable_x64", True)

# %%
plr = ModifiedPowerLaw(572.0, 0.2, 1e-5)
tip = Spherical(1.0)

app, ret = make_triangular(1.0, 1e-2, 10.0)
# %%
f_app = force_approach(app, plr.relaxation_function, tip)
f_ret = force_retract(app, ret, plr.relaxation_function, tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.t, f_app, ".")
ax.plot(ret.t, f_ret, ".")


# %%
# testing out interactive solve
# may need later when I decide to short circuit bisection evaluation for trivial inputs
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
args = (plr.relaxation_function, app, ret, ret.t[0])
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
