# ruff: noqa: F722
from typing import TypeVar, Sequence
import dataclasses

import jax.numpy as jnp
from jaxtyping import Float, Array
import lmfit

from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
)
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.tipgeometry import AbstractTipGeometry
from neuralconstitutive.ting import force_approach, force_retract

ConstitEqn = TypeVar("ConstitEqn", bound=AbstractConstitutiveEqn)


def constitutive_to_params(
    constit, bounds: Sequence[tuple[float, float] | None]
) -> lmfit.Parameters:
    params = lmfit.Parameters()

    constit_dict = dataclasses.asdict(constit)  # Equinox modules are dataclasses
    assert len(constit_dict) == len(
        bounds
    ), "Length of bounds should match the number of parameters in consitt"

    for (k, v), bound in zip(constit_dict.items(), bounds):
        if bound is None:
            max_, min_ = None, None
        else:
            max_, min_ = bound

        params.add(k, value=float(v), min=min_, max=max_)

    return params


def params_to_constitutive(params: lmfit.Parameters, constit: ConstitEqn) -> ConstitEqn:
    return type(constit)(**params.valuesdict())


def fit_approach_lmfit(
    constitutive: AbstractConstitutiveEqn,
    bounds: Sequence[tuple[float, float] | None],
    tip: AbstractTipGeometry,
    approach: Indentation,
    force: Float[Array, " {len(approach)}"],
):
    params = constitutive_to_params(constitutive, bounds)

    def residual(
        params: lmfit.Parameters, indentation: Indentation, force: Float[Array, " N"]
    ) -> Float[Array, " N"]:
        constit = params_to_constitutive(params, constitutive)
        f_pred = force_approach(constit, indentation, tip)
        return f_pred - force

    result = lmfit.minimize(residual, params, args=(approach, force))
    constit_fit = params_to_constitutive(result.params, constitutive)
    return constit_fit, result


def fit_all_lmfit(
    constitutive: AbstractConstitutiveEqn,
    bounds: Sequence[tuple[float, float] | None],
    tip: AbstractTipGeometry,
    indentations: tuple[Indentation, Indentation],
    force: Float[Array, " {len(approach)}"],
):
    params = constitutive_to_params(constitutive, bounds)

    def residual(params: lmfit.Parameters, indentations, forces) -> Float[Array, " N"]:
        constit = params_to_constitutive(params, constitutive)
        f_pred_app = force_approach(constit, indentations[0], tip)
        f_pred_ret = force_retract(constit, indentations, tip)
        return jnp.concatenate((f_pred_app - forces[0], f_pred_ret - forces[1]))

    result = lmfit.minimize(residual, params, args=(indentations, force))
    constit_fit = params_to_constitutive(result.params, constitutive)
    return constit_fit, result
