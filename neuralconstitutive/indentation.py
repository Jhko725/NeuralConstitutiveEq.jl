# ruff: noqa: F722
import abc
from typing import Literal

import diffrax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from neuralconstitutive.custom_types import (
    FloatScalar,
    FloatScalarLike,
    FloatScalarOr1D,
    floatscalar_field,
)


class AbstractIndentation(eqx.Module):
    """Interface for the indentation of the tip in an indentation experiment.

    Attributes:
    t_m: Time at which the tip is at its maximum indentation depth.
        t<=t_m corresponds to approach and t>=t_m corresponds to retract."""

    t_m: eqx.AbstractVar[FloatScalar]

    @abc.abstractmethod
    def h_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        pass

    @abc.abstractmethod
    def h_ret(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        pass

    @abc.abstractmethod
    def v_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        pass

    @abc.abstractmethod
    def v_ret(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        pass


class ConstantVelocityIndentation(AbstractIndentation):
    v: FloatScalar = floatscalar_field()
    t_m: FloatScalar = floatscalar_field(default=1.0)

    def h_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.v * t

    def h_ret(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return

    def v_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.v.reshape(t.shape)

    def v_ret(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return -self.v.reshape(t.shape)


class Indentation(eqx.Module):
    time: Float[Array, " N"] = eqx.field(converter=jnp.asarray)
    depth: Float[Array, " N"] = eqx.field(converter=jnp.asarray)

    # Might want to check monotonicity of time here
    # Also might want to check monotonicity of depth, and assign whether approach or retract
    def __len__(self) -> int:
        return len(self.time)


InterpolationMethod = Literal["linear", "cubic"]


def interpolate_indentation(
    indentation: Indentation, *, method: InterpolationMethod = "cubic"
) -> diffrax.AbstractPath:
    ts, ys = indentation.time, indentation.depth
    match method:
        case "linear":
            interp = diffrax.LinearInterpolation(ts, ys)
        case "cubic":
            coeffs = diffrax.backward_hermite_coefficients(ts, ys)
            interp = diffrax.CubicInterpolation(ts, coeffs)
    return interp
