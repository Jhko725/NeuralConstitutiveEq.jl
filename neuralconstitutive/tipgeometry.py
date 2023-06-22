import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TipGeometry(abc.ABC):
    @abc.abstractproperty
    def max_order(self) -> int:
        pass

    @abc.abstractproperty
    def alpha(self) -> float:
        pass

    @abc.abstractproperty
    def beta(self) -> float:
        pass

    @abc.abstractmethod
    def alpha_corrections(self, h: float, order: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def beta_corrections(self, order: int) -> np.ndarray:
        pass

    def _validate_expansion_order(self, order: int) -> None:
        if order > self.max_order:
            raise ValueError(
                f"Maximum order of expansion supported is {self.max_order}!"
            )


@dataclass(frozen=True)
class Spherical(TipGeometry):
    R: float
    max_order: int = field(default=4, init=False)

    @property
    def alpha(self) -> float:
        return (16 / 9) * np.sqrt(self.R)

    @property
    def beta(self) -> float:
        return 1.5

    def alpha_corrections(self, h: float, order: int) -> np.ndarray:
        self._validate_expansion_order(order)
        coeffs = np.array([1, 1.133, 1.497, 1.469, 0.755])
        coeffs = coeffs[: order + 1]
        higher_orders = (np.sqrt(self.R) / h) ** np.arange(len(coeffs))
        return self.alpha * coeffs * higher_orders

    def beta_corrections(self, order: int) -> np.ndarray:
        self._validate_expansion_order(order)
        return self.beta + 0.5 * np.arange(order + 1)


@dataclass(frozen=True)
class Conical(TipGeometry):
    theta: float
    max_order: int = field(default=4, init=False)

    @property
    def alpha(self) -> float:
        return (8 / (3 * np.pi)) * np.tan(self.theta)

    @property
    def beta(self) -> float:
        return 2.0

    def alpha_corrections(self, h: float, order: int) -> np.ndarray:
        self._validate_expansion_order(order)
        coeffs = np.array([1, 0.721, 0.650, 0.491, 0.225])
        coeffs = coeffs[: order + 1]
        higher_orders = (np.tan(self.theta) / h) ** np.arange(len(coeffs))
        return self.alpha * coeffs * higher_orders

    def beta_corrections(self, order: int) -> np.ndarray:
        self._validate_expansion_order(order)
        return self.beta + np.arange(order + 1)
