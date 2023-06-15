import abc
import torch
from torch import nn, Tensor
from .utils import to_parameter


class ConstitutiveEqn(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def stress_relaxation(self, t: Tensor) -> Tensor:
        pass

    def forward(self, t: Tensor) -> Tensor:
        return self.stress_relaxation(t)


class PowerLawRheology(ConstitutiveEqn):
    def __init__(self, E0: float, gamma: float, t0: float = 1):
        super().__init__()
        self.E0 = to_parameter(E0)
        self.gamma = to_parameter(gamma)
        self.t0 = to_parameter(t0)

    def stress_relaxation(self, t: Tensor) -> Tensor:
        return self.E0 * (t / self.t0) ** (-self.gamma)


class StandardLinearSolid(ConstitutiveEqn):
    def __init__(self, E0: float, tau: float, E_inf: float = 0):
        super().__init__()
        self.E0 = to_parameter(E0)
        self.tau = to_parameter(tau)
        self.E_inf = to_parameter(E_inf)

    def stress_relaxation(self, t: Tensor) -> Tensor:
        return self.E_inf + (self.E0 - self.E_inf) * torch.exp(-t / self.tau)
