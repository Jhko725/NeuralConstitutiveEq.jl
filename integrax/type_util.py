# ruff: noqa: F722
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Float, Shaped


def default_floating_dtype() -> jnp.dtype:
    """Returns the default floating type for JAX

    Inspired from lineax.misc.default_floating_dtype"""
    if jax.config.jax_enable_x64:
        return jnp.float64
    else:
        return jnp.float32


def as_inexact_array(x: Shaped[ArrayLike, "*dims"]) -> Float[Array, "*dims"]:
    """Given an array-like input x, cast it to a floating point JAX array.

    Inspired from optimistix.misc.inexact_asarray"""
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = default_floating_dtype()
    return jnp.asarray(x, dtype)


class ReturnsArrays(eqx.Module):
    """Wraps the given function so that it always returns PyTree of floating point arrays

    Inspired from optimistix.misc.OutAsArray"""

    fn: Callable

    def __call__(self, *args, **kwargs):
        y = self.fn(*args, **kwargs)
        y = jtu.tree_map(as_inexact_array, y)
        return y
