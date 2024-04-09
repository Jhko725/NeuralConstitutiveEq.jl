# %%
# ruff: noqa: F722]
from typing import Sequence, TypeVar
from pathlib import Path
import dataclasses

import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from scipy.interpolate import make_smoothing_spline
import lmfit

from neuralconstitutive.constitutive import AbstractConstitutiveEqn
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.plotting import plot_indentation, plot_relaxation_fn
from neuralconstitutive.constitutive import (
    StandardLinearSolid,
    ModifiedPowerLaw,
    KohlrauschWilliamsWatts,
)
from neuralconstitutive.tipgeometry import AbstractTipGeometry, Spherical
from neuralconstitutive.fitting import fit_approach, force_approach
from neuralconstitutive.ting import force_ting

jax.config.update("jax_enable_x64", True)


def to_jax_numpy(series: pd.Series) -> Array:
    return jnp.asarray(series.to_numpy())


def smooth_data(indentation: Indentation) -> Indentation:
    t = indentation.time
    spl = make_smoothing_spline(t, indentation.depth)
    return Indentation(t, spl(t))


def import_data(rawdata_file, metadata_file):
    # Read csv files
    df_raw = pd.read_csv(rawdata_file, sep="\t", skiprows=34)
    df_meta = pd.read_csv(metadata_file, sep="\t")

    # Extract relevant raw data
    force = to_jax_numpy(df_raw["force"])
    time = to_jax_numpy(df_raw["time"])
    tip_position = -to_jax_numpy(df_raw["tip position"])
    contact_point = -to_jax_numpy(df_meta["Contact Point [nm]"]) * 1e-9

    # Retain only the indenting part of the data
    depth = tip_position - contact_point
    in_contact = depth >= 0
    force, time, depth = force[in_contact], time[in_contact], depth[in_contact]
    force = force - force[0]
    time = time - time[0]

    # Split into approach and retract
    idx_max = jnp.argmax(depth)
    approach = Indentation(time[: idx_max + 1], depth[: idx_max + 1])
    retract = Indentation(time[idx_max:], depth[idx_max:])
    force_app, force_ret = force[: idx_max + 1], force[idx_max:]

    approach, retract = smooth_data(approach), smooth_data(retract)
    return (approach, retract), (force_app, force_ret)


# %%

sls = StandardLinearSolid(1.0, 1.0, 10.0)


# %%
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


ConstitEqn = TypeVar("ConstitEqn", bound=AbstractConstitutiveEqn)


def params_to_constitutive(params: lmfit.Parameters, constit: ConstitEqn) -> ConstitEqn:
    return type(constit)(**params.valuesdict())


# %%
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
    return result


# %%
datadir = Path("open_data/PAAM hydrogel")
(app, ret), (f_app, f_ret) = import_data(
    datadir / "PAA_speed 5_4nN.tab", datadir / "PAA_speed 5_4nN.tsv"
)
# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0] = plot_indentation(axes[0], app, marker=".")
axes[0] = plot_indentation(axes[0], ret, marker=".")

axes[1].plot(app.time, f_app, ".")
axes[1].plot(ret.time, f_ret, ".")

axes[2].plot(app.depth, f_app, ".")
axes[2].plot(ret.depth, f_ret, ".")
# %%


def normalize_indentations(approach: Indentation, retract: Indentation):
    t_m, h_m = approach.time[-1], approach.depth[-1]
    t_app, t_ret = approach.time / t_m, retract.time / t_m
    h_app, h_ret = approach.depth / h_m, retract.depth / h_m
    app_norm = Indentation(t_app, h_app)
    ret_norm = Indentation(t_ret, h_ret)
    return (app_norm, ret_norm), (t_m, h_m)


def normalize_forces(force_app, force_ret):
    f_m = force_app[-1]
    return (force_app / f_m, force_ret / f_m), f_m


(f_app, f_ret), _ = normalize_forces(f_app, f_ret)
(app, ret), (_, h_m) = normalize_indentations(app, ret)
# %%
tip = Spherical(2.5e-6 / h_m)  # Scale tip radius by the length scale we are using

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0] = plot_indentation(axes[0], app, marker=".")
axes[0] = plot_indentation(axes[0], ret, marker=".")

axes[1].plot(app.time, f_app, ".")
axes[1].plot(ret.time, f_ret, ".")

axes[2].plot(app.depth, f_app, ".")
axes[2].plot(ret.depth, f_ret, ".")
# %%
bounds = [(0.0, jnp.inf)] * 3
result = fit_approach_lmfit(sls, bounds, tip, app, f_app)

# %%
sls_fit = params_to_constitutive(result.params, sls)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, label="Data")
f_fit = force_approach(sls_fit, app, tip)
ax.plot(app.time, f_fit, label="Curve fit")
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
plot_relaxation_fn(ax, sls_fit, app.time)
# %%
import diffrax

# axes[1].plot(app.time, app_interp.derivative(app.time))
# %%

sls_fit
# %%
spl = make_smoothing_spline(app.time, app.depth)
dspl = spl.derivative()
# %%
app_interp2 = diffrax.LinearInterpolation(app.time, spl(app.time))
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(app.time, app_interp.derivative(app.time))
ax.plot(app.time, dspl(app.time))
ax.plot(app.time, app_interp2.derivative(app.time))
# %%
app_interp2 = diffrax.LinearInterpolation(app.time, spl(app.time))
# %%

# %%
