from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from integrax.adjoint import AbstractAdjoint, ImplicitAdjoint
from integrax.custom_types import Args, X, Y
from integrax.solvers import AbstractIntegration
from integrax.type_util import ReturnsArrays, as_inexact_array


def _solve_integral(inputs):
    fn, method, lower, upper, args, options, while_loop = inputs

    init_carry = method.init(fn, lower, upper, args)

    def cond_fun(carry):
        integral, num_steps, state = carry
        terminate, _ = method.terminate(
            integral, num_steps, fn, lower, upper, args, state
        )
        return jnp.invert(terminate)

    def body_fun(carry):
        integral, num_steps, state = carry
        new_integral, new_num_steps, new_state = method.step(
            integral, num_steps, fn, lower, upper, args, state
        )
        return new_integral, new_num_steps, new_state

    final_carry = while_loop(cond_fun, body_fun, init_carry)
    final_integral, num_steps, final_state = final_carry

    return final_integral


def integrate(
    fn: Callable[[X, Args], Y],
    method: AbstractIntegration,
    lower: X,
    upper: X,
    args: Args,
    options=None,
    *,
    max_steps: int | None = 64,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
):
    # Cast leaf nodes of (lower, upper) to be inexact arrays
    lower, upper = jtu.tree_map(as_inexact_array, (lower, upper))
    # Wrap fn into a callable PyTree that is guaranteed to return PyTree with floating point array leaf nodes.
    fn = ReturnsArrays(fn)
    # Closure over any captured values in fn
    fn = eqx.filter_closure_convert(
        fn, lower, args
    )  # May hit a singularity at x=lower, might have to reconsider logic

    inputs = (fn, method, lower, upper, args, options)
    out = adjoint.apply(_solve_integral, inputs, max_steps)
    return out
