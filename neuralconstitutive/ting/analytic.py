# ruff: noqa: F722
import jax.numpy as jnp

from jax.scipy.special import beta  # , betainc
from tensorflow_probability.substrates import jax as tfp
from jaxtyping import Array, Float

from neuralconstitutive.constitutive import PowerLaw
from neuralconstitutive.indentation import (
    ApproachRetract,
)
from neuralconstitutive.tipgeometry import AbstractTipGeometry


def t1_powerlaw(
    t: Float[Array, " N"],
    constit: PowerLaw,
    indentation: ApproachRetract,
) -> Float[Array, " N"]:
    """Computes the $t_1(t) $ function for a powerlaw sample indented with constant velocity."""
    const = 2 ** (1 / (1 - constit.alpha))
    return jnp.clip(t - const * (t - indentation.t_ret), 0.0)


def force_powerlaw(
    t: Float[Array, " N"],
    constit: PowerLaw,
    indentation: ApproachRetract,
    tip: AbstractTipGeometry,
) -> Float[Array, " N"]:
    """Computes the force response of a powerlaw sample indented with constant velocity."""
    v = indentation.approach.velocity_
    a, b = tip.a(), tip.b()
    coeff = a * b * constit.E0 * (v**b) * beta(b, 1 - constit.alpha)
    force = coeff * (t ** (b - constit.alpha))
    t1 = t1_powerlaw(t, constit, indentation)
    correction_factor = jnp.where(
        t <= indentation.t_ret, 1.0, tfp.math.betainc(b, 1 - constit.alpha, t1 / t)
    )
    return force * correction_factor
