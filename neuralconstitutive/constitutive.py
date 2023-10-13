import abc

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax.scipy.special import exp1
import equinox as eqx


class AbstractConstitutiveEqn(eqx.Module):
    """Abstract base class for all constitutive equations"""

    @abc.abstractmethod
    def relaxation_function(self, t: Array) -> Array:
        pass

    def relaxation_spectrum(self, t: Array) -> Array | None:
        return None


@jax.jit
def relaxation_function(t: ArrayLike, constit: AbstractConstitutiveEqn) -> Array:
    t_array = jnp.asarray(t)
    return constit.relaxation_function(t_array)


class ModifiedPowerLaw(AbstractConstitutiveEqn):
    E0: float
    alpha: float
    t0: float

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
        numerator = 1 + self.C * (exp1(t / self.tau2) - exp1(t / self.tau1))
        denominator = 1 + self.C * jnp.log(self.tau2 / self.tau1)
        return self.E0 * numerator / denominator


class LogDiscretizedSpectrum(AbstractConstitutiveEqn):
    """
    Assume that log_t_grid is equispaced
    """

    log10_t_grid: Array
    h_grid: Array

    def relaxation_function(self, t: Array) -> Array:
        h0 = self.log10_t_grid[1] - self.log10_t_grid[0]
        t_grid = 10**self.log10_t_grid
        return jnp.dot(self.h_grid * h0, jnp.exp(-t / t_grid))
