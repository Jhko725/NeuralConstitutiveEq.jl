from typing import Callable, Any

import jax.numpy as jnp
from jaxtyping import PyTree
import quadax


def integrate(fn: Callable, lower: float, upper: float, args: PyTree[Any]):
    return jnp.where(
        lower == upper,
        0.0,
        quadax.quadgk(fn, (lower, upper), (args,), epsabs=1e-4, epsrel=1e-4)[0],
    )
