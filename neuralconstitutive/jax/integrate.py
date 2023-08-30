import jax
from jax import Array
import jax.numpy as jnp


@jax.jit
def integrate_to(x_upper: float, x: Array, y: Array) -> Array:
    return jax.lax.cond(
        x_upper > x[0], _integrate_to, lambda *_: jnp.asarray(0.0), x_upper, x, y
    )


def _integrate_to(x_upper: float, x: Array, y: Array) -> Array:
    ind = jnp.searchsorted(x, x_upper)
    mask = jnp.arange(x.shape[0]) < ind
    y_, x_ = jnp.where(mask, y, 0), jnp.where(mask, x, x[ind - 1])
    y2, y1 = y[ind], y[ind - 1]
    x2, x1 = x[ind], x[ind - 1]
    y_upper = ((x_upper - x1) * y2 + (x2 - x_upper) * y1) / (x2 - x1)
    return jnp.trapz(y_, x=x_) + 0.5 * (y1 + y_upper) * (x_upper - x1)


@jax.jit
def integrate_from(x_lower: float, x: Array, y: Array) -> Array:
    return jax.lax.cond(
        x_lower < x[-1], _integrate_from, lambda *_: jnp.asarray(0.0), x_lower, x, y
    )


def _integrate_from(x_lower: float, x: Array, y: Array) -> Array:
    ind = jnp.searchsorted(x, x_lower, side="right")
    mask = jnp.arange(x.shape[0]) > ind
    y_, x_ = jnp.where(mask, y, 0), jnp.where(mask, x, x[ind])
    y2, y1 = y[ind], y[ind - 1]
    x2, x1 = x[ind], x[ind - 1]
    y_lower = ((x_lower - x1) * y2 + (x2 - x_lower) * y1) / (x2 - x1)
    return jnp.trapz(y_, x=x_) + 0.5 * (y1 + y_lower) * (x_lower - x1)
