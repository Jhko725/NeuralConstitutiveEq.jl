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
    def __init__(
        self, E0: float, gamma: float, E_inf: float = 0.0, t_offset: float = 1e-3
    ):
        super().__init__()
        self.E0 = to_parameter(E0)
        self.gamma = to_parameter(gamma)
        self.E_inf = to_parameter(E_inf)
        self.t_offset = to_parameter(t_offset)

    def stress_relaxation(self, t: Tensor) -> Tensor:
        return self.E_inf + (self.E0 - self.E_inf) * (1 + t / self.t_offset) ** (
            -self.gamma
        )


class StandardLinearSolid(ConstitutiveEqn):
    def __init__(self, E0: float, tau: float, E_inf: float = 0):
        super().__init__()
        self.E0 = to_parameter(E0)
        self.tau = to_parameter(tau)
        self.E_inf = to_parameter(E_inf)

    def stress_relaxation(self, t: Tensor) -> Tensor:
        return self.E_inf + (self.E0 - self.E_inf) * torch.exp(-t / self.tau)
