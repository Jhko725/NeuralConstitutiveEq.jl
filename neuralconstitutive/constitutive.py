import abc
from functools import partial

from jax import Array
import jax.numpy as jnp
from jax.scipy.special import exp1
import equinox as eqx

from neuralconstitutive.custom_types import FloatScalar, as_floatscalar
from neuralconstitutive.relaxation_spectrum import AbstractLogDiscreteSpectrum

floatscalar_field = partial(eqx.field, converter=as_floatscalar)


class AbstractConstitutiveEqn(eqx.Module):
    """Abstract base class for all constitutive equations"""

    @abc.abstractmethod
    def relaxation_function(self, t: Array) -> Array:
        pass

    def relaxation_spectrum(self, t: Array) -> Array | None:
        return None

class Hertzian(AbstractConstitutiveEqn):
    E0: FloatScalar = floatscalar_field()

    def relaxation_function(self, _: Array) -> Array:
        return self.E0

class PowerLaw(AbstractConstitutiveEqn):
    E0: FloatScalar = floatscalar_field()
    alpha: FloatScalar = floatscalar_field()
    t0: float

    def relaxation_function(self, t):
        return self.E0 * (t / self.t0) ** (-self.alpha)


class ModifiedPowerLaw(AbstractConstitutiveEqn):
    E0: FloatScalar = floatscalar_field()
    alpha: FloatScalar = floatscalar_field()
    t0: FloatScalar = floatscalar_field()

    def relaxation_function(self, t):
        return self.E0 * (1 + t / self.t0) ** (-self.alpha)


class StandardLinearSolid(AbstractConstitutiveEqn):
    E1: FloatScalar = floatscalar_field()
    E_inf: FloatScalar = floatscalar_field()
    tau: FloatScalar = floatscalar_field()

    @property
    def E0(self) -> FloatScalar:
        return self.E_inf + self.E1

    def relaxation_function(self, t):
        return self.E_inf + self.E1 * jnp.exp(-t / self.tau)


class KohlrauschWilliamsWatts(AbstractConstitutiveEqn):
    E1: FloatScalar = floatscalar_field()
    E_inf: FloatScalar = floatscalar_field()
    tau: FloatScalar = floatscalar_field()
    beta: FloatScalar = floatscalar_field()

    @property
<<<<<<< HEAD
    def E0(self) -> FloatScalar:
        return self.E_inf + self.E1
=======
    def E0(self):
        return self.E1 + self.E_inf
>>>>>>> 37a4350ff3a7780acdbd93f7bef617d3becd29c7

    def relaxation_function(self, t):
        exponent = (t / self.tau) ** self.beta
        return self.E_inf + self.E1 * jnp.exp(-exponent)


class Fung(AbstractConstitutiveEqn):
    E0: FloatScalar = floatscalar_field()
    tau1: FloatScalar = floatscalar_field()
    tau2: FloatScalar = floatscalar_field()
    C: FloatScalar = floatscalar_field()

    def relaxation_function(self, t):
        def _exp1_diff(t_i: float) -> float:
            return exp1(t_i / self.tau2) - exp1(t_i / self.tau1)

        # A workaround for https://github.com/google/jax/issues/13543
        # Note that the use of jax.lax.map causes the time complexity of this function
        # to scale with the length of array t
        # Tested on an array of length 100, this is 25 times (~550ms vs 13.5s)
        # faster than the commented out vectorized version

        numerator = 1 + self.C * _exp1_diff(t)
        # numerator = 1 + self.C * (exp1(t / self.tau2) - exp1(t / self.tau1))
        denominator = 1 + self.C * jnp.log(self.tau2 / self.tau1)
        return self.E0 * numerator / denominator


class FromLogDiscreteSpectrum(AbstractConstitutiveEqn):
    """
    Assume that log_t_grid is equispaced
    """

    log10_t_grid: Array
    h_grid: Array

    def __init__(self, relaxation_spectrum: AbstractLogDiscreteSpectrum):
        self.log10_t_grid, self.h_grid = relaxation_spectrum.discrete_spectrum()

    def relaxation_function(self, t: Array) -> Array:
        h0 = jnp.log(self.t_grid[1]) - jnp.log(self.t_grid[0])
        return jnp.matmul(
            jnp.exp(-jnp.expand_dims(t, -1) / self.t_grid), self.h_grid * h0
        )

    @property
    def t_grid(self) -> Array:
        return 10**self.log10_t_grid

    @property
    def discrete_spectrum(self) -> tuple[Array, Array]:
        return self.t_grid, self.h_grid
