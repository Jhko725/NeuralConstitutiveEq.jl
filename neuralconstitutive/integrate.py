from typing import Callable

import jax.numpy as jnp
import quadax


def integrate(fn: Callable, bounds: tuple[float, float], args: tuple):
    """Wrap the integrator from quadax

    This is to handle the case when lower == upper, in which quadax returns nan for the integral
    Obviously, the true value is zero
    """
    lower, upper = bounds
    cond = lower == upper
    # Add some random small value to upper so that upper==lower never happens
    # This is to avoid the nan gradient problem outlined in https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    upper_ = upper + 1e-3  # jnp.where(cond, upper + 1.0, upper)
    return jnp.where(
        cond,
        0.0,
        quadax.quadgk(fn, (lower, upper_), args)[0],
    )
