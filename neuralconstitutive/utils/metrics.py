# ruff: noqa: F722
import jax.numpy as jnp
from jaxtyping import Float, Array

from neuralconstitutive.custom_types import FloatScalar


def SMAPE(y_pred: Float[Array, " N"], y_true: Float[Array, " N"]) -> FloatScalar:
    """Calculates the standard mean absolute percentage error between the two arrays.

    Note that this quantity ranges from 0 to 200."""
    numer = jnp.abs(y_pred - y_true)
    denom = 0.5 * (jnp.abs(y_pred) + jnp.abs(y_true))
    denom = jnp.where(
        denom == 0.0, 1.0, denom
    )  # Replace zero denominator with dummy value
    return 100 * jnp.mean(numer / denom)


def NMSE(y_pred: Float[Array, " N"], y_true: Float[Array, " N"]) -> FloatScalar:
    """Calculates the normalized mean squared error between the two arrays.

    This function returns its outputs in percentages of the mean of the squared true array.
    """
    return 100 * jnp.mean((y_pred - y_true) ** 2) / jnp.mean(y_true**2)
