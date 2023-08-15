from dataclasses import dataclass

from numpy import ndarray


@dataclass(frozen=True, slots=True)
class PowerLawRheology:
    E0: float
    gamma: float
    t0: float

    def __call__(self, t: ndarray) -> ndarray:
        return self.E0 * (1 + t / self.t0) ** (-self.gamma)
