import abc
from typing import Any, Callable

import equinox as eqx
from equinox.internal import ω
import jax.numpy as jnp
from jaxtyping import PyTree

from integrax.custom_types import Args, BoolScalar, IntScalar, FloatScalar, SolverState, X, Y


class AbstractIntegration(eqx.Module):
    @abc.abstractmethod
    def init(
        self, fn: Callable[[X, Args], Y], lower: X, upper: Y, args: Args
    ) -> tuple[PyTree, IntScalar, SolverState]:
        pass

    @abc.abstractmethod
    def step(
        self,
        integral: PyTree,
        num_steps: IntScalar,
        fn: Callable[[X, Args], Y],
        lower: X,
        upper: Y,
        args: Args,
        state: Any,
    ) -> tuple[PyTree, IntScalar, SolverState]:
        pass

    @abc.abstractmethod
    def terminate(
        self,
        integral: PyTree,
        num_steps: IntScalar,
        fn: Callable[[X, Args], Y],
        lower: X,
        upper: Y,
        args: Args,
        state: Any,
    ) -> tuple[BoolScalar, Any]:
        pass


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
