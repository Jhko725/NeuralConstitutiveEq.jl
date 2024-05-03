import abc
from typing import Any, Callable

import equinox as eqx
from jaxtyping import PyTree

from integrax.custom_types import Args, BoolScalar, IntScalar, SolverState, X, Y


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
