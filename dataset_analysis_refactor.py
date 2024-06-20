# %%
# ruff: noqa: F722
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
from IPython.display import display

from neuralconstitutive.constitutive import (
    FractionalKelvinVoigt,
    GeneralizedMaxwellmodel,
    Hertzian,
    KohlrauschWilliamsWatts,
    ModifiedPowerLaw,
    StandardLinearSolid,
)
from neuralconstitutive.fitting import (
    LatinHypercubeSampler,
    fit_all_lmfit,
    fit_approach_lmfit,
    fit_indentation_data,
)
from neuralconstitutive.io import import_data, truncate_adhesion, normalize_dataset
from neuralconstitutive.plotting import plot_relaxation_fn, plot_forceindent
from neuralconstitutive.smoothing import make_smoothed_cubic_spline
from neuralconstitutive.ting import (
    force_approach_scalar,
    force_retract_scalar,
    _force_approach,
    _force_retract,
)
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.utils import (
    normalize_forces,
    normalize_indentations,
)

# %%
jax.config.update("jax_enable_x64", True)

# datadir = Path("open_data/PAAM hydrogel/speed 5")
# name = "PAA_speed 5_4nN"

#datadir = Path("data/abuhattum_iscience_2022/Interphase rep 2")
#name = "interphase_speed 2_2nN"
datadir = Path("data/abuhattum_iscience_2022/Agarose/speed 10")
name = "Agarose_speed 10_2nN"
dataset = import_data(datadir / f"{name}.tab", datadir / f"{name}.tsv")


# app, ret = smooth_data(app), smooth_data(ret)
# %%
fig = plot_forceindent(dataset)
# %%
dataset = truncate_adhesion(dataset)
dataset, scale = normalize_dataset(dataset)
fig = plot_forceindent(dataset)
# %%
tip = Spherical(
    2.5e-6 / scale.depth
)  # Scale tip radius by the length scale we are using

# %%
app_interp = make_smoothed_cubic_spline(dataset.approach)
ret_interp = make_smoothed_cubic_spline(dataset.retract)

f_app_func = eqx.Partial(force_approach_scalar, approach=app_interp, tip=tip)
f_ret_func = eqx.Partial(
    force_retract_scalar, indentations=(app_interp, ret_interp), tip=tip
)
#%%
import bayeux as bx

@eqx.filter_jit
def log_likelihood_unnormed(constit):
    f_app_pred = eqx.filter_vmap(force_approach_scalar, in_axes=(0, None, None, None))(dataset.t_app, constit, app_interp, tip)
    f_ret_pred = eqx.filter_vmap(force_retract_scalar, in_axes=(0, None, None, None))(dataset.t_ret, constit, (app_interp, ret_interp), tip)
    return -jnp.sum((f_app_pred-dataset.f_app)**2)-jnp.sum((f_ret_pred-dataset.f_ret)**2)

#log_likelihood_fn = eqx.filter_jit(eqx.Partial(log_likelihood_unnormed, dataset))
#%%
def transform_fn(sls: StandardLinearSolid):
    return StandardLinearSolid(jnp.exp(sls.E1), jnp.exp(sls.E_inf), jnp.exp(sls.tau))

model = bx.Model(log_density=log_likelihood_unnormed, test_point=StandardLinearSolid(10.0, 10.0, 10.0), transform_fn=transform_fn)

#%%
import time
t_start = time.time()
seed = jax.random.key(0)
opt = model.optimize.optimistix_bfgs(seed = seed, num_particles = 10)
print(f"Time elapsed: {time.time()-t_start}")
#%%
import time
model.optimize.methods
#%%
sls_test = StandardLinearSolid(10.0, 10.0, 10.0)
log_likelihood_unnormed(sls_test)
#%%
#%%
from tqdm import tqdm
f_app_fn = eqx.filter_jit(eqx.filter_vmap(f_app_func, in_axes = (0, None)))
f_ret_fn = eqx.filter_jit(eqx.filter_vmap(f_ret_func, in_axes = (0, None)))

f_app_list = []
f_ret_list = []
for i in tqdm(range(10)):
    sls_fit = jtu.tree_map(lambda leaf: leaf[i], opt.params)
    f_app_list.append(f_app_fn(dataset.t_app, sls_fit))
    f_ret_list.append(f_ret_fn(dataset.t_ret, sls_fit))
#%%
fig, axes = plt.subplots(2, 1, figsize = (7, 3))

ax = axes[0]
for f_app_fit, f_ret_fit in zip(f_app_list, f_ret_list):
    ax.plot(dataset.t_app, f_app_fit, color = "royalblue", linewidth = 1.0, alpha = 0.8)
    ax.plot(dataset.t_ret, f_ret_fit, color = "royalblue", linewidth = 1.0, alpha = 0.8)

    ax.plot(dataset.t_app, dataset.f_app, ".", color="black", alpha = 0.8)
    ax.plot(dataset.t_ret, dataset.f_ret, ".", color="black", alpha = 0.8)

ax1 = axes[1]
for i in tqdm(range(10)):
    sls_fit = jtu.tree_map(lambda leaf: leaf[i], opt.params)
    ax1 = plot_relaxation_fn(ax1, sls_fit, dataset.t_app)
#%%
opt.params
#%%
model.mcmc.numpyro_nuts.get_kwargs()
#%%
idata = model.mcmc.numpyro_nuts(seed, dense_mass = True)
#%%
jax.grad(log_likelihood_unnormed)(sls_test).E1
#%%
@eqx.filter_jit
def force_app_map(t, constit):
    def _f_app_func(t_):
        return f_app_func(t_, constit)
    return jax.lax.map(_f_app_func, t)


@eqx.filter_jit
def force_app_vmap(t, constit):
    return eqx.filter_vmap(f_app_func, in_axes=(0, None))(t, constit)


@eqx.filter_jit
def force_ret_map(t, constit):
    def _f_ret_func(t_):
        return f_ret_func(t_, constit)
    return jax.lax.map(_f_ret_func, t)


@eqx.filter_jit
def force_ret_vmap(t, constit):
    return eqx.filter_vmap(f_ret_func, in_axes=(0, None))(t, constit)


constit_test = StandardLinearSolid(10.0, 10.0, 10.0)
# %%
%%timeit
f_app_pred1 = force_app_map(dataset.t_app, constit_test)
f_app_pred1.block_until_ready()
# %%
%%timeit
f_app_pred2 = force_app_vmap(dataset.t_app, constit_test)
f_app_pred2.block_until_ready()


# %%
def residual_approach(constit, args):
    t_data, f_data, interp, tip = args
    constit = jtu.tree_map(lambda x: x**10, constit)
    f_app_pred = _force_approach(t_data, constit, interp, tip)
    return f_app_pred - f_data


def residual_all(constit, args):
    t_data, f_data, interps, tip = args
    t_app, t_ret = t_data
    f_app_pred = _force_approach(t_app, constit, interps[0], tip)
    f_ret_pred = _force_retract(t_ret, constit, interps, tip)
    return jnp.concatenate((f_app_pred, f_ret_pred)) - jnp.concatenate(f_data)


def fit_approach_optx(constit, t_data, f_data, interp, tip, bounds):
    solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-6)
    args = (t_data, f_data, interp, tip)
    sol = optx.least_squares(residual_approach, solver, constit, args, max_steps=1000)
    return sol.value


def fit_all_optx(constit, t_data, f_data, interp, tip, bounds):
    solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-6)
    args = (t_data, f_data, interp, tip)
    sol = optx.least_squares(residual_all, solver, constit, args, max_steps=1000)
    return sol.value


# %%
# %%
## Fit using Latin hypercube sampling
N_SAMPLES = 100
fit_type = "all"
### Hertzian model

constit_htz = Hertzian(10.0)
bounds_htz = [(0, 1e3)]

sampler = LatinHypercubeSampler(
    sample_range=[(1e-2, 1e2)],
    sample_scale=["log"],
)

htz_fits, htz_results, htz_initvals, htz_minimizers = fit_indentation_data(
    constit_htz,
    bounds_htz,
    dataset,
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)
# %%
# %%
### SLS model
constit_sls = StandardLinearSolid(10.0, 10.0, 10.0)
bounds_sls = [(0, 1e3), (0, 1e3), (1e-6, 1e3)]
# %%
sampler = LatinHypercubeSampler(
    sample_range=[(1e-2, 1e2), (1e-2, 1e2), (1e-5, 1e2)],
    sample_scale=["log", "log", "log"],
)
sls_fits, sls_results, sls_initvals, sls_minimizers = fit_indentation_data(
    constit_sls,
    bounds_sls,
    dataset,
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)
# %%
constit_sls = StandardLinearSolid(1.0, 1.0, 1.0)
out = fit_approach_optx(constit_sls, app.time, f_app, app_interp, tip, None)
# %%
out.E1
# %%
out2, _, _ = fit_approach_lmfit(constit_sls, bounds_sls, tip, app, f_app)
# %%
out2.E1
# %%
constit_sls = StandardLinearSolid(1.0, 1.0, 1.0)
out = fit_all_optx(
    constit_sls,
    (app.time, ret.time),
    (f_app, f_ret),
    (app_interp, ret_interp),
    tip,
    None,
)
# %%
out = jtu.tree_map(lambda x: 10**x, out)
out.E1
# %%
out2, _, _ = fit_all_lmfit(constit_sls, bounds_sls, tip, (app, ret), (f_app, f_ret))
out2.E1
# %%
### Modified PLR model

constit_mplr = ModifiedPowerLaw(10.0, 10.0, 10.0)
bounds_mplr = [(0, 1e3), (0.0, 1.0), (1e-6, 1e3)]

sampler = LatinHypercubeSampler(
    sample_range=[(1e-2, 1e2), (0.0, 1.0), (1e-5, 1e2)],
    sample_scale=["log", "linear", "log"],
)
mplr_fits, mplr_results, mplr_initvals, mplr_minimizers = fit_indentation_data(
    constit_mplr,
    bounds_mplr,
    dataset,
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)

# %%
### KWW model

constit_kww = KohlrauschWilliamsWatts(10.0, 10.0, 10.0, 10.0)
bounds_kww = [(0, 1e3), (0, 1e3), (1e-6, 1e3), (0.0, 1.0)]

sampler = LatinHypercubeSampler(
    sample_range=[
        (1e-2, 1e2),
        (1e-2, 1e2),
        (1e-5, 1e2),
        (0.0, 1.0),
    ],
    sample_scale=["log", "log", "log", "linear"],
)

kww_fits, kww_results, kww_initvals, kww_minimizers = fit_indentation_data(
    constit_kww,
    bounds_kww,
    dataset,
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)
# %%
### Fractional Kelvin Voigt model

constit_fkv = FractionalKelvinVoigt(10.0, 10.0, 10.0)
bounds_fkv = [(0, 1e3), (0, 1e3), (0.0, 1.0)]

sampler = LatinHypercubeSampler(
    sample_range=[(1e-2, 1e2), (1e-2, 1e2), (0.0, 1.0)],
    sample_scale=["log", "log", "linear"],
)

fkv_fits, fkv_results, fkv_initvals, fkv_minimizers = fit_indentation_data(
    constit_fkv,
    bounds_fkv,
    dataset,
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)
# %%
### Generalized Maxwell model
constit_gm = GeneralizedMaxwellmodel(10.0, 10.0, 10.0, 10.0, 10.0)
bounds_gm = [(0, 1e3), (0, 1e3), (0, 1e3), (0, 1e3), (0, 1e3)]

sampler = LatinHypercubeSampler(
    sample_range=[(1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2), (1e-2, 1e2)],
    sample_scale=["log", "log", "log", "log", "log"],
)

gm_fits, gm_results, gm_initvals, gm_minimizers = fit_indentation_data(
    constit_gm,
    bounds_gm,
    dataset,
    tip,
    fit_type=fit_type,
    init_val_sampler=sampler,
    n_samples=N_SAMPLES,
)
# %%
for r in mplr_results:
    display(r)

# %%


def get_best_model(results: list[lmfit.minimizer.MinimizerResult]):
    bic = np.array([res.bic if res is not None else np.inf for res in results])
    ind_best = np.argmin(bic)
    return results[ind_best], ind_best


# %%
## Assess results

results_best = {}
fits_best = {}
for results, fits, name in zip(
    [htz_results, sls_results],  # mplr_results, kww_results, fkv_results, gm_results],
    [htz_fits, sls_fits],  # mplr_fits, kww_fits,fkv_fits, gm_fits],
    ["Hertzian", "SLS"],  # , "MPLR", "KWW", "FKV", "GM"],
):
    res_best, ind_best = get_best_model(results)
    results_best[name] = res_best
    fits_best[name] = fits[ind_best]

# %%
for r in results_best.values():
    display(r)
    print(r.uvars)

# %%

# from neuralconstitutive.fitting import create_subsampled_interpolants

f_app_fits = {}
f_ret_fits = {}

t_app, t_ret = dataset.approach.time, dataset.retract.time
for name, constit in fits_best.items():
    f_app_fits[name] = _force_approach(t_app, constit, app_interp, tip)
    f_ret_fits[name] = _force_retract(t_ret, constit, (app_interp, ret_interp), tip)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
axes[0].plot(
    t_app,
    dataset.approach.force,
    ".",
    color="k",
    markersize=1.0,
    label="Data",
    alpha=0.7,
)
axes[0].plot(t_ret, dataset.retract.force, ".", color="k", markersize=1.0, alpha=0.7)

color_palette = np.array(
    [
        [0.86666667, 0.23137255, 0.20784314],
        [1.0, 0.3882, 0.2784],
        [0.92941176, 0.61568627, 0.24705882],
        [0.63921569, 0.85490196, 0.52156863],
        [0.19764706, 0.46431373, 0.34196078],
        [0.17254902, 0.45098039, 0.69803922],
        [0.372549, 0.596078, 1.0],
        [0.7172549, 0.48117647, 0.92313726],
        [0.64313725, 0.14117647, 0.48627451],
    ]
)
names = ["Hertzian", "SLS"]  # , "MPLR", "KWW", "FKV", "GM"]
color_inds = [0, 3]  # , 6, 8, 5, 1]
for n, c_ind in zip(names, color_inds):
    constit = fits_best[n]

    color = color_palette[c_ind]
    axes[0].plot(t_app, f_app_fits[n], color=color, linewidth=1.0, label=n, alpha=0.9)
    axes[0].plot(t_ret, f_ret_fits[n], color=color, linewidth=1.0, alpha=0.9)
    axes[0].set_xlabel("Time (norm.)")
    axes[0].set_ylabel("Force (norm.)")

    axes[1] = plot_relaxation_fn(axes[1], constit, t_app, color=color, linewidth=1.0)
    axes[1].set_ylim((0, 0.8))
    axes[1].set_xlabel("Time (norm.)")
    axes[1].set_ylabel("$G(t)$ (norm.)")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, ncols=2, loc="upper right")
for ax in axes:
    ax.grid(ls="--", color="lightgray")

# fig.suptitle("PAAM hydrogel curve fit results: Entire curve")
fig.suptitle("Agarose hydrogel curve fit results: Approach only")

# %%
bics = jnp.asarray([results_best[n].bic for n in names])
colors = color_palette[color_inds]
fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
ax.grid(ls="--", color="darkgray")
ax.bar(names, bics, color=colors)
ax.set_yscale("symlog")
ax.set_ylabel("BIC")
ax.set_title("PAAM hydrogel, Entire")


# %%
def process_uvars(uvars: dict):
    if "E0" in uvars:
        return uvars
    elif ("E1" in uvars) and ("E_inf" in uvars):
        uvars_ = {}
        for k, v in uvars.items():
            if k == "E1":
                uvars_["E0"] = uvars["E1"] + uvars["E_inf"]
            else:
                uvars_[k] = v
        return uvars_
    else:
        raise ValueError("Unexpected combination of parameters!")


def rel_error(uvar):
    return uvar.s / uvar.n


def get_param_rel_errors(uvars: dict):
    uvars = deepcopy(uvars)
    rel_errors = []
    # First element of the list is always that of E0
    E0 = uvars.pop("E0")
    rel_errors.append(rel_error(E0))
    for v in uvars.values():
        rel_errors.append(rel_error(v))
    return np.asarray(rel_errors)


relative_errors = {}
for name, res in results_best.items():
    display(res)
    uvars_proc = process_uvars(res.uvars)
    relative_errors[name] = get_param_rel_errors(uvars_proc)


fig, ax = plt.subplots(1, 1, figsize=(4, 3))
names = ("Hertzian", "MPLR", "SLS", "KWW")  # , "FKV", "GM")
for name in names:
    ax.plot(relative_errors[name], ".-", label=name, linewidth=1.0, markersize=8.0)
ax.set_yscale("log", base=10)
ax.set_xticks([0, 1, 2, 3])
ax.legend()
ax.set_ylabel("Relative error")
ax.set_xlabel("Model free parameters")
xticklabels = ax.get_xticks().tolist()
xticklabels[0] = "0 ($E_0$)"
ax.set_xticklabels(xticklabels)
ax.grid(ls="--", color="lightgray")

# %%
for n, r in results_best.items():
    print(n, r.params.valuesdict())


# %%
def residual(constit):
    f_app_pred = _force_approach(t_app, constit, app_interp, tip)
    if fit_type == "approach":
        return dataset.approach.force - f_app_pred
    else:
        f_ret_pred = _force_retract(t_ret, constit, (app_interp, ret_interp), tip)
        f_pred = jnp.concatenate((f_app_pred, f_ret_pred), axis=0)
        f = jnp.concatenate((dataset.approach.force, dataset.retract.force), axis=0)
        return f - f_pred


def sensitivity_matrix(constit):
    log_constit = jtu.tree_map(jnp.log, constit)

    def residual_from_logconstit(log_constit):
        constit = jtu.tree_map(jnp.exp, log_constit)
        return residual(constit)

    D_res = jax.jacfwd(residual_from_logconstit)(log_constit)
    D_res_array, _ = jtu.tree_flatten(D_res)
    D_res_array = jnp.stack(D_res_array, axis=-1)

    S = jax.vmap(jnp.outer)(D_res_array, D_res_array)
    return jnp.sum(S, axis=0)


S_matrix = sensitivity_matrix(fits_best["SLS"])
eigval, eigvec = jnp.linalg.eigh(S_matrix)
# %%
print(eigval)
print(eigvec)


# %%
def plot_eigval_spectrum(ax, eigvals, bar_offset=0.0, bar_length=1.0, **hlines_kwargs):
    eigvals = jnp.clip(jnp.asarray(eigvals), 1e-50)
    ax.hlines(jnp.log10(eigvals), bar_offset, bar_offset + bar_length, **hlines_kwargs)
    return ax


fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
names = ["Hertzian", "SLS"]  # , "MPLR", "KWW", "FKV", "GM"]
color_inds = [0, 3]  # , 6, 8, 5, 1]
bar_offset = 0.0
bar_length = 1.0
x_tick_positions = []
for n, c_ind in zip(names, color_inds):
    constit = fits_best[n]
    color = color_palette[c_ind]

    S_matrix = sensitivity_matrix(constit)
    eigval, eigvec = jnp.linalg.eigh(S_matrix)
    eigval = eigval / jnp.max(eigval)
    print(eigval)
    x_tick_positions.append(bar_offset + 0.5 * bar_length)
    plot_eigval_spectrum(ax, eigval, bar_offset, bar_length, color=color)

    bar_offset += 2 * bar_length

ax.set_xticks(x_tick_positions, names)
ax.set_ylabel("log (eigenvalues)")
ax.grid(ls="--", color="lightgray")
# %%
