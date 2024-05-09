from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import PyTree
from lineax.internal import rms_norm

from integrax.custom_types import BoolScalar, FloatScalar, IntScalar
from integrax.solvers.base import AbstractIntegration, reached_tolerance


class TrapezoidState(eqx.Module):
    num_points: IntScalar
    terminate: BoolScalar


def init_trapezoid(fn, lower, upper, args):
    dx = upper - lower
    S_init = (0.5 * dx * (fn(upper, args) ** ω - fn(lower, args) ** ω)).ω
    return S_init


def refine_trapezoid(S: PyTree, num_refine: IntScalar, fn, lower, upper, args):
    """Refine the integral of fn(x, args) from [lower, upper] using num_refine additional equi-spaced interior points."""
    dx = (upper - lower) / num_refine
    x_init = lower + 0.5 * dx
    sum_init = jtu.tree_map(jnp.zeros_like, S)
    carry_init = (sum_init, x_init)

    def _body_fun(_, carry):
        sum_, x = carry
        sum_ = (sum_**ω + fn(x, args) ** ω).ω
        return (sum_, x + dx)

    sum_, _ = jax.lax.fori_loop(0, num_refine, _body_fun, carry_init)
    S_refined = (0.5 * (S**ω + dx * sum_**ω)).ω
    return S_refined


def refine_trapezoid_batch(
    S: PyTree, num_refine: IntScalar, fn, lower, upper, args, batch_size=512
):
    """Refine the integral of fn(x, args) from [lower, upper] using num_refine additional equi-spaced interior points."""
    dx = (upper - lower) / num_refine
    num_batch, num_remainder = jnp.divmod(num_refine, batch_size)
    inds_remainder = jnp.arange(batch_size)

    x_init = lower + 0.5 * dx
    sum_init = jtu.tree_map(jnp.zeros_like, S)
    carry_init = (sum_init, x_init)

    fn_vec = eqx.filter_vmap(in_axes=(0, None))(fn)

    def _body_fun_batch(i, carry):
        sum_, x = carry
        x_batch = x + dx * inds_remainder
        sum_ = (sum_**ω + jtu.tree_map(jnp.sum, fn_vec(x_batch, args)) ** ω).ω
        return (sum_, x + dx * batch_size)

    def _body_fun(_, carry):
        sum_, x = carry
        sum_ = (sum_**ω + fn(x, args) ** ω).ω
        return (sum_, x + dx)

    carry_out = jax.lax.fori_loop(0, num_batch, _body_fun_batch, carry_init)
    sum_, _ = jax.lax.fori_loop(0, num_remainder, _body_fun, carry_out)
    S_refined = (0.5 * (S**ω + dx * sum_**ω)).ω
    return S_refined


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
