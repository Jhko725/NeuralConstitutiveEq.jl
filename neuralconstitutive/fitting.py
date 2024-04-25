# ruff: noqa: F722
import dataclasses
from typing import Sequence, TypeVar, Literal

import equinox as eqx
import jax.numpy as jnp
import lmfit
import numpy as np
from jaxtyping import Array, Float
from scipy.stats import qmc
from tqdm import tqdm

from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
)
from neuralconstitutive.indentation import Indentation, interpolate_indentation
from neuralconstitutive.ting import (
    _force_approach,
    force_approach,
    force_retract,
    _force_retract,
)
from neuralconstitutive.tipgeometry import AbstractTipGeometry

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
    app_interp = interpolate_indentation(approach)

    @eqx.filter_jit
    def _residual_jax(constit):
        f_pred = _force_approach(approach.time, constit, app_interp, tip)
        return f_pred - force

    def residual(params: lmfit.Parameters, args) -> Float[Array, " N"]:
        constit = params_to_constitutive(params, constitutive)
        return _residual_jax(constit)

    minimizer = lmfit.Minimizer(residual, params, fcn_args=(None,))
    result = minimizer.minimize()
    constit_fit = params_to_constitutive(result.params, constitutive)
    return constit_fit, result, minimizer


def fit_all_lmfit(
    constitutive: AbstractConstitutiveEqn,
    bounds: Sequence[tuple[float, float] | None],
    tip: AbstractTipGeometry,
    indentations: tuple[Indentation, Indentation],
    forces: tuple[Array, Array],
):
    params = constitutive_to_params(constitutive, bounds)
    app, ret = indentations
    app_interp = interpolate_indentation(app)
    ret_interp = interpolate_indentation(ret)

    @eqx.filter_jit
    def _residual_jax(constit):
        f_pred_app = _force_approach(app.time, constit, app_interp, tip)
        f_pred_ret = _force_retract(ret.time, constit, (app_interp, ret_interp), tip)

        return jnp.concatenate((f_pred_app, f_pred_ret)) - jnp.concatenate(forces)

    def residual(params: lmfit.Parameters, args) -> Float[Array, " N"]:
        constit = params_to_constitutive(params, constitutive)
        return _residual_jax(constit)

    minimizer = lmfit.Minimizer(residual, params, fcn_args=(None,))
    result = minimizer.minimize()
    constit_fit = params_to_constitutive(result.params, constitutive)
    return constit_fit, result, minimizer


class LatinHypercubeSampler:

    def __init__(self, sample_range, sample_scale, random_seed: int = 20):
        self.random_seed = random_seed
        self.sample_range = np.asarray(sample_range)
        self.sample_scale = sample_scale

        self.sampler = qmc.LatinHypercube(
            d=self.sample_range.shape[0], seed=self.random_seed
        )

    @property
    def n_dim(self) -> int:
        return self.sample_range.shape[0]

    def sample(self, n_samples: int) -> Float[np.ndarray, "{n_samples} {self.n_dim}"]:
        samples_norm = self.sampler.random(n_samples)
        is_logscale = [s == "log" for s in self.sample_scale]

        sample_range = self.sample_range
        sample_range[is_logscale, :] = np.log10(sample_range[is_logscale, :])

        samples = qmc.scale(samples_norm, sample_range[:, 0], sample_range[:, 1])
        samples[:, is_logscale] = 10 ** samples[:, is_logscale]
        return samples


FitType = Literal["approach", "both"]


def fit_indentation_data(
    constit,
    bounds,
    indentations,
    forces,
    tip,
    fit_type: FitType = "approach",
    init_val_sampler=None,
    n_samples: int = 1,
):
    if fit_type == "approach":
        fit_func = fit_approach_lmfit
        fit_data = (indentations[0], forces[0])

    else:
        fit_func = fit_all_lmfit
        fit_data = (indentations, forces)

    constit_fits = []
    results = []
    minimizers = []

    init_vals = None
    if init_val_sampler is not None:
        init_vals = init_val_sampler.sample(n_samples)

    for i in tqdm(range(n_samples)):
        if init_vals is not None:
            constit_ = type(constit)(*init_vals[i])
        else:
            constit_ = constit

        constit_fit, result, minimizer = fit_func(constit_, bounds, tip, *fit_data)
        constit_fits.append(constit_fit)
        results.append(result)
        minimizers.append(minimizer)

    constit_fits = np.array(constit_fits)
    results = np.array(results)
    init_vals = np.array(init_vals)
    minimizers = np.array(minimizers)
    return constit_fits, results, init_vals, minimizers
