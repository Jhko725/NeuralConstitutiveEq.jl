import abc

import jax
from jax import Array
import jax.numpy as jnp
from jax.scipy.special import exp1
import equinox as eqx

from .relaxation_spectrum import AbstractLogDiscreteSpectrum


class AbstractConstitutiveEqn(eqx.Module):
    """Abstract base class for all constitutive equations"""

    @abc.abstractmethod
    def relaxation_function(self, t: Array) -> Array:
        pass

    def relaxation_spectrum(self, t: Array) -> Array | None:
        return None


class ModifiedPowerLaw(AbstractConstitutiveEqn):
    E0: float
    alpha: float
    t0: float = eqx.field(static=True)

    def relaxation_function(self, t):
        return self.E0 * (1 + t / self.t0) ** (-self.alpha)


class StandardLinearSolid(AbstractConstitutiveEqn):
    E0: float
    E_inf: float
    tau: float

    def relaxation_function(self, t):
        return self.E_inf + (self.E0 - self.E_inf) * jnp.exp(-t / self.tau)


class KohlrauschWilliamsWatts(AbstractConstitutiveEqn):
    E0: float
    E_inf: float
    tau: float
    beta: float

    def relaxation_function(self, t):
        exponent = (t / self.tau) ** self.beta
        return self.E_inf + (self.E0 - self.E_inf) * jnp.exp(-exponent)


class Fung(AbstractConstitutiveEqn):
    E0: float
    tau1: float
    tau2: float
    C: float

    def relaxation_function(self, t):
        def _exp1_diff(t_i: float) -> float:
            return exp1(t_i / self.tau2) - exp1(t_i / self.tau1)

        # A workaround for https://github.com/google/jax/issues/13543
        # Note that the use of jax.lax.map causes the time complexity of this function
        # to scale with the length of array t
        # Tested on an array of length 100, this is 25 times (~550ms vs 13.5s)
        # faster than the commented out vectorized version

        numerator = 1 + self.C * jax.lax.map(_exp1_diff, t)
        # numerator = 1 + self.C * (exp1(t / self.tau2) - exp1(t / self.tau1))
        denominator = 1 + self.C * jnp.log(self.tau2 / self.tau1)
        return self.E0 * numerator / denominator


class LogDiscretizedSpectrum(AbstractConstitutiveEqn):
    """
    Assume that log_t_grid is equispaced
    """

    log10_t_grid: Array
    h_grid: Array

    def __init__(self, relaxation_spectrum: AbstractLogDiscreteSpectrum):
        self.log10_t_grid, self.h_grid = relaxation_spectrum.discrete_spectrum()

    def relaxation_function(self, t: Array) -> Array:
        h0 = self.log10_t_grid[1] - self.log10_t_grid[0]
        t_grid = 10**self.log10_t_grid
        return jnp.matmul(jnp.exp(-jnp.expand_dims(t, -1) / t_grid), self.h_grid * h0)

    @property
    def t_grid(self) -> Array:
        return 10**self.log10_t_grid
