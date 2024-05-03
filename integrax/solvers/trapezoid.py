from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import PyTree
from lineax.internal import rms_norm

from integrax.custom_types import BoolScalar, FloatScalar, IntScalar
from integrax.solvers.base import AbstractIntegration


class TrapezoidState(eqx.Module):
    num_points: IntScalar
    terminate: BoolScalar


class TrapezoidBase(AbstractIntegration):
    def init(self, fn, lower, upper, args) -> tuple[PyTree, IntScalar, TrapezoidState]:
        dx = upper - lower
        integral = (0.5 * dx * (fn(upper, args) ** ω - fn(lower, args) ** ω)).ω
        init_state = TrapezoidState(jnp.array(1), jnp.array(False))
        return integral, jnp.array(1), init_state

    def step_trapezoid(
        self, integral: PyTree, num_steps, fn, lower, upper, args, state: TrapezoidState
    ) -> PyTree:
        num_points = state.num_points * 2
        dx = (upper - lower) / num_points
        x_init = lower + 0.5 * dx
        sum_init = jtu.tree_map(jnp.zeros_like, integral)
        carry_init = (sum_init, x_init)

        def _body_fun(_, carry):
            sum_, x = carry
            sum_ = (sum_**ω + fn(x, args) ** ω).ω
            return (sum_, x + dx)

        sum_, _ = jax.lax.fori_loop(0, num_points, _body_fun, carry_init)
        integral_new = (0.5 * (integral**ω + dx * sum_**ω)).ω
        state_new = TrapezoidState(num_points, jnp.array(False))
        return integral_new, num_steps + 1, state_new

    def terminate(
        self, integral, num_steps, fn, lower, upper, args, state: TrapezoidState
    ):
        return state.terminate, None


class ExtendedTrapezoid(TrapezoidBase):
    num_refine: int

    def step(self, integral, num_steps, fn, lower, upper, args, state):
        integral_new, num_steps_new, state_new = self.step_trapezoid(
            integral, num_steps, fn, lower, upper, args, state
        )
        terminate = num_steps_new < self.num_refine
        state_new = TrapezoidState(state_new.num_points, terminate)
        return integral_new, num_steps_new, state_new


class AdaptiveTrapezoid(TrapezoidBase):
    rtol: float
    atol: float
    norm: Callable[[PyTree], FloatScalar] = rms_norm
    min_refines: int = 5

    def step(self, integral, num_steps, fn, lower, upper, args, state):
        integral_new, num_steps_new, state_new = self.step_trapezoid(
            integral, num_steps, fn, lower, upper, args, state
        )

        terminate = (num_steps_new > self.min_refines) & reached_tolerance(
            integral_new, integral, self.rtol, self.atol, self.norm
        )
        state_new = TrapezoidState(state_new.num_points, terminate)
        return integral_new, num_steps_new, state_new


def reached_tolerance(
    value_new: PyTree,
    value_old: PyTree,
    rtol: float,
    atol: float,
    norm: Callable[[PyTree], FloatScalar],
):
    value_diff = (value_new**ω - value_old**ω).ω
    value_scale = (atol + rtol * ω(value_old).call(jnp.abs)).ω
    tol_satisfied = norm((ω(value_diff).call(jnp.abs) / value_scale**ω).ω) < 1
    return tol_satisfied
