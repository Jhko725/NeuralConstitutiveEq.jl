from typing import Callable
import abc
from functools import cached_property

from torch import nn, Tensor
from torch.func import vmap, grad


class Indentation(nn.Module, abc.ABC):
    @abc.abstractmethod
    def indent_approach(self, t: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def indent_retract(self, t: Tensor) -> Tensor:
        pass

    @property
    @abc.abstractmethod
    def t_max(self) -> float:
        return None

    @cached_property
    def i_app(self) -> Callable[[Tensor], Tensor]:
        def _inner(t):
            return self.indent_approach(t)

        return vmap(_inner)

    @cached_property
    def i_ret(self) -> Callable[[Tensor], Tensor]:
        def _inner(t):
            return self.indent_retract(t)

        return vmap(_inner)

    @cached_property
    def v_app(self) -> Callable[[Tensor], Tensor]:
        def _inner(t):
            return self.indent_approach(t)

        return vmap(grad(_inner))

    @cached_property
    def v_ret(self) -> Callable[[Tensor], Tensor]:
        def _inner(t):
            return self.indent_retract(t)

        return vmap(grad(_inner))


class Triangular(Indentation):
    def __init__(self, v: float, t_max: float):
        super().__init__()
        self.v = v
        self._t_max = t_max

    @property
    def t_max(self) -> float:
        return self._t_max

    def indent_approach(self, t: Tensor) -> Tensor:
        return self.v * t

    def indent_retract(self, t: Tensor) -> Tensor:
        return self.v * (2 * self.t_max - t)
