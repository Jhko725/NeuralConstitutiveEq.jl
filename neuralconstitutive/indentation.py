# ruff: noqa: F722
from typing import Literal

import diffrax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


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
