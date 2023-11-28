from typing import Callable

import jax
from jax import Array
import jax.numpy as jnp
from equinox.internal import while_loop


def integrate(
    f: Callable[[float], Array], x_lower: float, x_upper: float, dx: float
) -> Array:
    ## TODO: change implementation to jax.lax.foriloop
    def _body_fun(val: tuple[float, Array]) -> tuple[float, Array]:
        x, I_x = val
        x_next = jnp.clip(x + dx, x_lower, x_upper)
        dx_ = x_next - x
        I_x += 0.5 * (f(x) + f(x_next)) * dx_
        return x_next, I_x

    def _cond_fun(val: tuple[float, Array]) -> bool:
        x, _ = val
        return x < x_upper

    _init_val = (x_lower, jnp.zeros_like(f(x_lower)))

    _, I_x = while_loop(_cond_fun, _body_fun, _init_val, kind="bounded", max_steps=1000)
    return I_x
