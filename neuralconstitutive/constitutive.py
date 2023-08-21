from dataclasses import dataclass

import numpy as np
from numpy import ndarray


@dataclass(frozen=True, slots=True)
class PowerLawRheology:
    E0: float
    gamma: float
    t0: float

    def __call__(self, t: ndarray) -> ndarray:
        return self.E0 * (1 + t / self.t0) ** (-self.gamma)


@dataclass(frozen=True, slots=True)
class StandardLinearSolid:
    E0: float
    E_inf: float
    tau: float

    def __call__(self, t: ndarray) -> ndarray:
        return self.E_inf + (self.E0 - self.E_inf) * np.exp(-t / self.tau)
