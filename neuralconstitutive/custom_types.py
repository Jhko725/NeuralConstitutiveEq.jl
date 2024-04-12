"""
Contains custom type definitions used throughout the codebase, as well as some utility functions to work with them
"""

import os

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

FileName = str | bytes | os.PathLike

# Jax-related types
FloatScalar = Float[Array, ""]
FloatScalarLike = Float[ArrayLike, ""]


## From lineax/lineax/_misc.py
def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def as_floatscalar(x: FloatScalarLike) -> FloatScalar:
    return jnp.asarray(x, dtype=default_floating_dtype())
