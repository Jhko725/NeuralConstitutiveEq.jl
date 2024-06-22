from jax.scipy.special import beta, betainc
import jax.numpy as jnp

from neuralconstitutive.constitutive import AbstractConstitutive, PowerLaw
from neuralconstitutive.indentation import (
    AbstractIndentation,
    ConstantVelocityIndentation,
)
from neuralconstitutive.tipgeometry import AbstractTipGeometry
from neuralconstitutive.custom_types import FloatScalarOr1D


def force_approach_analytic(
    t: FloatScalarOr1D,
    constit: AbstractConstitutive,
    indent: AbstractIndentation,
    tip: AbstractTipGeometry,
) -> FloatScalarOr1D:
    if isinstance(constit, PowerLaw) and isinstance(
        indent, ConstantVelocityIndentation
    ):
        a, b = tip.a(), tip.b()
        coeff = a * b * constit.E0 * (indent.v**b) * beta(1 - constit.alpha, b)
        return coeff * (t ** (b - constit.alpha))
    else:
        raise ValueError(
            "Analytic solution unknown for the given combination of constitutive model and indentation!"
        )


def t1_analytic(
    t: FloatScalarOr1D,
    constit: AbstractConstitutive,
    indent: AbstractIndentation,
) -> FloatScalarOr1D:
    if isinstance(constit, PowerLaw) and isinstance(
        indent, ConstantVelocityIndentation
    ):
        const = 2 ** (1 / (1 - constit.alpha))
        return jnp.clip(t - const * (t - indent.t_m), 0.0)
    else:
        raise ValueError(
            "Analytic solution unknown for the given combination of constitutive model and indentation!"
        )


def force_retract_analytic(
    t: FloatScalarOr1D,
    constit: AbstractConstitutive,
    indent: AbstractIndentation,
    tip: AbstractTipGeometry,
) -> FloatScalarOr1D:
    if isinstance(constit, PowerLaw) and isinstance(
        indent, ConstantVelocityIndentation
    ):
        a, b = tip.a(), tip.b()
        coeff = a * b * constit.E0 * (indent.v**b) * beta(b, 1 - constit.alpha)
        t1 = t1_analytic(t, constit, indent)
        return (
            coeff * (t ** (b - constit.alpha)) * betainc(b, 1 - constit.alpha, t1 / t)
        )
    else:
        raise ValueError(
            "Analytic solution unknown for the given combination of constitutive model and indentation!"
        )
