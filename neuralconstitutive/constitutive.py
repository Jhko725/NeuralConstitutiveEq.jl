# ruff: noqa: F722
import abc
from functools import partial

from jaxtyping import ArrayLike, Array, Float
import equinox as eqx
import jax.numpy as jnp
from jax.scipy.special import exp1

from neuralconstitutive.custom_types import FloatScalar, as_floatscalar
from neuralconstitutive.relaxation_spectrum import AbstractLogDiscreteSpectrum
from neuralconstitutive.misc import stretched_exp

from mittag_leffler_master.mittag_leffler import ml

from jax.scipy.special import gamma


floatscalar_field = partial(eqx.field, converter=as_floatscalar)


class AbstractConstitutiveEqn(eqx.Module):
    """Abstract base class for all constitutive equations"""

    def relaxation_function(
        self, t: Float[ArrayLike, "*dims"]
    ) -> Float[Array, "*dims"]:
        """Given an array of timepoints, return the corresponding array of stress relaxation function values.

        The shape of the output will match that of the input. Internally, this is done by flattening the input into a 1D vector,
        performing the calculation, then reshaping the result.

        This function can also be called through its alias, self.G."""

        t = jnp.asarray(t)
        out_shape = t.shape
        return self._relaxation_function_1D(jnp.ravel(t)).reshape(out_shape)

    G = relaxation_function  # An alias for self.relaxation_function

    @abc.abstractmethod
    def _relaxation_function_1D(self, t: Float[Array, " N"]) -> Float[Array, " N"]:
        """Abstract method corresponding to the actual implementation of the stress relaxation function.

        Note that unlike self.relaxation_function, the inputs and the corresponding outputs are 1D arrays.
        """
        pass

    def relaxation_spectrum(self, t: Array) -> Array | None:
        return None


## Constitutive equations that are nonsingular at t=0
class Hertzian(AbstractConstitutiveEqn):
    E0: FloatScalar = floatscalar_field()

    def _relaxation_function_1D(self, t: Float[Array, " N"]) -> Float[Array, " N"]:
        return self.E0 * jnp.ones_like(t)


class ModifiedPowerLaw(AbstractConstitutiveEqn):
    E0: FloatScalar = floatscalar_field()
    alpha: FloatScalar = floatscalar_field()
    t0: FloatScalar = floatscalar_field()

    def _relaxation_function_1D(self, t: Float[Array, " N"]) -> Float[Array, " N"]:
        return self.E0 * (1 + t / self.t0) ** (-self.alpha)


class StandardLinearSolid(AbstractConstitutiveEqn):
    E1: FloatScalar = floatscalar_field()
    E_inf: FloatScalar = floatscalar_field()
    tau: FloatScalar = floatscalar_field()

    @property
    def E0(self) -> FloatScalar:
        return self.E_inf + self.E1

    def _relaxation_function_1D(self, t: Float[Array, " N"]) -> Float[Array, " N"]:
        return self.E_inf + self.E1 * jnp.exp(-t / self.tau)

class GeneralizedMaxwellmodel(AbstractConstitutiveEqn):
    E1: FloatScalar = floatscalar_field()
    E2: FloatScalar = floatscalar_field()
    E_inf: FloatScalar = floatscalar_field()
    tau1: FloatScalar = floatscalar_field()
    tau2: FloatScalar = floatscalar_field()

    @property
    def E0(self) -> FloatScalar:
        return self.E_inf + self.E1 + self.E2

    def _relaxation_function_1D(self, t):
        return self.E_inf + self.E1 * jnp.exp(-t / self.tau1) + self.E2 * jnp.exp(-t / self.tau2)


class KohlrauschWilliamsWatts(AbstractConstitutiveEqn):
    E1: FloatScalar = floatscalar_field()
    E_inf: FloatScalar = floatscalar_field()
    tau: FloatScalar = floatscalar_field()
    beta: FloatScalar = floatscalar_field()

    @property
    def E0(self) -> FloatScalar:
        return self.E_inf + self.E1

    def _relaxation_function_1D(self, t: Float[Array, " N"]) -> Float[Array, " N"]:
        return self.E_inf + self.E1 * stretched_exp(t, self.tau, self.beta)


class Fung(AbstractConstitutiveEqn):
    E0: FloatScalar = floatscalar_field()
    tau1: FloatScalar = floatscalar_field()
    tau2: FloatScalar = floatscalar_field()
    C: FloatScalar = floatscalar_field()

    def _relaxation_function_1D(self, t: Float[Array, " N"]) -> Float[Array, " N"]:
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

class FractionalKelvinVoigt(AbstractConstitutiveEqn):
    E1: FloatScalar = floatscalar_field()
    E_inf: FloatScalar = floatscalar_field()
    alpha: FloatScalar = floatscalar_field()
    
    def _relaxation_function_1D(self, t: Array) -> Array:
        return self.E_inf+self.E1*(t**(-self.alpha)/gamma(1-self.alpha))


class FromLogDiscreteSpectrum(AbstractConstitutiveEqn):
    """
    Assume that log_t_grid is equispaced
    """

    log10_t_grid: Array
    h_grid: Array

    def __init__(self, relaxation_spectrum: AbstractLogDiscreteSpectrum):
        self.log10_t_grid, self.h_grid = relaxation_spectrum.discrete_spectrum()

    def _relaxation_function_1D(self, t: Float[Array, " N"]) -> Float[Array, " N"]:
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


## Constitutive equations that are singular ar t=0
class PowerLaw(AbstractConstitutiveEqn):
    E0: FloatScalar = floatscalar_field()
    alpha: FloatScalar = floatscalar_field()
    t0: float

    def relaxation_function(self, t):
        return self.E0 * (t / self.t0) ** (-self.alpha)
