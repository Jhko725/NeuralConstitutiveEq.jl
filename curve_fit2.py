# %%
from pathlib import Path
import operator

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optimistix as optx
from tqdm import tqdm
from jaxtyping import PyTree, Array

from neuralconstitutive.indentation import CubicSpline, IndentationBuilder
from neuralconstitutive.io import import_data, normalize_dataset, truncate_adhesion
from neuralconstitutive.ting.numerical import force_approach, force_retract, force_ting
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.constitutive import StandardLinearSolid
from neuralconstitutive.custom_types import FloatScalar

jax.config.update("jax_enable_x64", True)

speed = 5
file_index = 0

datadir = Path(f"data/abuhattum_iscience_2022/PAAM hydrogel/speed {speed}")
name_suffix = "" if file_index == 0 else f"_{file_index}"
filename = f"PAA_speed {speed}_4nN{name_suffix}"

dataset = import_data(datadir / f"{filename}.tab", datadir / f"{filename}.tsv")
dataset = truncate_adhesion(dataset)
dataset, scale = normalize_dataset(dataset)
tip = Spherical(1e-6 / scale.depth)

fig, axes = plt.subplots(2, 1, figsize=(5, 3), sharex=True)
axes[0].plot(dataset.total_time, dataset.total_force)
axes[1].plot(dataset.total_time, dataset.total_depth)
# %%
indentation = IndentationBuilder(enforce_continuity=False).build_from_dataset(
    dataset, smoothing=1e-3
)


@eqx.filter_jit
def residual(constit, args) -> PyTree[Array]:
    indentation, tip, dataset = args
    f_app_pred = force_approach(dataset.t_app, constit, indentation, tip)
    f_ret_pred = force_retract(dataset.t_ret, constit, indentation, tip)
    return (f_app_pred - dataset.f_app, f_ret_pred - dataset.f_ret)


@eqx.filter_jit
def objective(constit, args) -> FloatScalar:
    residuals = residual(constit, args)
    mses = jax.tree.map(lambda x: jnp.sum(x**2), residuals)
    return jax.tree.reduce(operator.add, mses)


solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
args = (indentation, tip, dataset)
constit = StandardLinearSolid(10.0, 10.0, 10.0)
sol = optx.least_squares(residual, solver, constit, args, max_steps=1000)
# %%
print(sol.value.E0, sol.value.E1, sol.value.tau)
print(objective(sol.value, args))
# %%
objective_fn = eqx.filter_jit(eqx.Partial(objective, args=args))
E1_array = jnp.logspace(-2, 0, 50, base=10)
tau_array = jnp.logspace(-2, 0, 50, base=10)

E1_mesh, tau_mesh = jnp.meshgrid(E1_array, tau_array)
constits = [
    StandardLinearSolid(E1_i, sol.value.E_inf, tau_i)
    for E1_i, tau_i in zip(E1_mesh.flatten(), tau_mesh.flatten())
]
loss_vals = jnp.asarray([objective_fn(c) for c in tqdm(constits)])
# %%
loss_mesh = jnp.reshape(loss_vals, E1_mesh.shape)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(jnp.log(loss_mesh))
# ax.imshow(loss_mesh)
# ax.contour(E1_mesh, tau_mesh, loss_mesh)
# ax.set_xscale("log", base = 10)
# ax.set_yscale("log", base = 10)
# %%
E_inf_array = jnp.logspace(-2, 0, 50, base=10)
tau_array = jnp.logspace(-2, 0, 50, base=10)

E_inf_mesh, tau_mesh = jnp.meshgrid(E_inf_array, tau_array)
constits = [
    StandardLinearSolid(sol.value.E1, E_inf_i, tau_i)
    for E_inf_i, tau_i in zip(E_inf_mesh.flatten(), tau_mesh.flatten())
]
loss_vals = jnp.asarray([objective_fn(c) for c in tqdm(constits)])
# %%
loss_mesh = jnp.reshape(loss_vals, E_inf_mesh.shape)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.imshow(jnp.log(loss_mesh))
ax.imshow(loss_mesh)
# ax.contour(E1_mesh, tau_mesh, loss_mesh)
# ax.set_xscale("log", base = 10)
# ax.set_yscale("log", base = 10)
# %%
E1_array = jnp.logspace(-2, 0, 50, base=10)
E_inf_array = jnp.logspace(-2, 0, 50, base=10)

E1_mesh, E_inf_mesh = jnp.meshgrid(E1_array, E_inf_array)
constits = [
    StandardLinearSolid(E1_i, E_inf_i, sol.value.tau)
    for E1_i, E_inf_i in zip(E1_mesh.flatten(), E_inf_mesh.flatten())
]
loss_vals = jnp.asarray([objective_fn(c) for c in tqdm(constits)])
# %%
loss_mesh = jnp.reshape(loss_vals, E1_mesh.shape)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.imshow(jnp.log(loss_mesh))
ax.imshow(loss_mesh)
# ax.contour(E1_mesh, tau_mesh, loss_mesh)
# ax.set_xscale("log", base = 10)
# ax.set_yscale("log", base = 10)
# %%
f_pred = force_ting(dataset.total_time, sol.value, indentation, tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(
    dataset.total_time,
    dataset.total_force,
    ".",
    markersize=1.0,
    color="k",
    label="Data",
)
ax.plot(dataset.total_time, f_pred, color="orangered", label="SLS curve fit")
# %%

hess = eqx.filter_jacfwd(eqx.filter_jacfwd)(objective_fn)(sol.value)
# %%


# %%
def grad_i(constit):
    return eqx.filter_grad(objective_fn)(constit).E1


ddE1 = eqx.filter_grad(grad_i)(sol.value)
print(ddE1.E1, ddE1.E_inf, ddE1.tau)
# eqx.filter_jacfwd(eqx.filter_grad)(objective_fn)(sol.value)
# %%
fig, axes = plt.subplots(1, 2, figsize=(5, 3))

for i in range(4):
    name_suffix = "" if i == 0 else f"_{i}"
    filename = f"PAA_speed {speed}_4nN{name_suffix}"
    dataset = import_data(datadir / f"{filename}.tab", datadir / f"{filename}.tsv")
    dataset = truncate_adhesion(dataset)
    dataset, scale = normalize_dataset(dataset)
    time = dataset.total_time
    depth = dataset.total_depth
    force = dataset.total_force
    print(time[1] - time[0])
    axes[0].plot(time, depth, markersize=0.5, label=f"{i}")
    axes[1].plot(time, force, label=f"{i}")


# %%
fig, axes = plt.subplots(1, 2, figsize=(5, 3))

for i in range(4):
    name_suffix = "" if i == 0 else f"_{i}"
    filename = f"PAA_speed {speed}_4nN{name_suffix}"
    dataset = import_data(datadir / f"{filename}.tab", datadir / f"{filename}.tsv")
    dataset, scale = normalize_dataset(dataset)
    time = dataset.total_time
    depth = dataset.total_depth
    force = dataset.total_force
    print(time[1] - time[0])
    axes[0].plot(time, depth, markersize=0.5, label=f"{i}")
    depth_resid = jnp.concatenate(
        [
            dataset.approach.depth - dataset.approach.time,
            dataset.retract.depth - 2.0 + dataset.retract.time,
        ]
    )
    axes[1].plot(time, depth_resid, label=f"{i}")
# %%

dataset, scale = normalize_dataset(dataset)
tip = Spherical(1e-6 / scale.depth)


fig, axes = plt.subplots(1, 2, figsize=(5, 3))
smoothing_values = jnp.logspace(-6, -1, 200, base=10)
for i in range(4):
    name_suffix = "" if i == 0 else f"_{i}"
    filename = f"PAA_speed {speed}_4nN{name_suffix}"
    dataset = import_data(datadir / f"{filename}.tab", datadir / f"{filename}.tsv")
    dataset, scale = normalize_dataset(dataset)
    time = dataset.total_time
    depth = dataset.total_depth
    force = dataset.total_force
    err = []
    n_knots = []
    for smoothing in tqdm(smoothing_values):
        indentation = (
            IndentationBuilder(enforce_continuity=False)
            .append(
                CubicSpline(
                    dataset.approach.time, dataset.approach.depth, smoothing=smoothing
                ),
                duration=dataset.approach.time[-1] - dataset.approach.time[0],
            )
            .append(
                CubicSpline(
                    dataset.retract.time, dataset.retract.depth, smoothing=smoothing
                ),
                duration=dataset.retract.time[-1] - dataset.retract.time[0],
            )
            .build()
        )

        n_knots.append(indentation.approach.n_knots + indentation.retract.n_knots)

        depth_resid = depth - indentation.depth(time)
        err.append(jnp.mean(depth_resid**2))
    axes[0].plot(smoothing_values, jnp.asarray(err), ".", label=f"{i}")
    axes[1].plot(smoothing_values, jnp.asarray(n_knots), ".", label=f"{i}")
for ax in axes:
    ax.set_yscale("log", base=10)
    ax.set_xscale("log", base=10)
# %%

fig


# %%
def make_residual(dataset, tip, smoothing=1e-3):
    f_app, f_ret = dataset.approach.force, dataset.retract.force
    t_app, t_ret = dataset.approach.time, dataset.retract.time

    indentation = (
        IndentationBuilder(enforce_continuity=False)
        .append(
            CubicSpline(
                dataset.approach.time, dataset.approach.depth, smoothing=smoothing
            ),
            duration=dataset.approach.time[-1] - dataset.approach.time[0],
        )
        .append(
            CubicSpline(
                dataset.retract.time, dataset.retract.depth, smoothing=smoothing
            ),
            duration=dataset.retract.time[-1] - dataset.retract.time[0],
        )
        .build()
    )

    @eqx.filter_jit
    def residual(constit, _):
        constit = jax.tree.map(lambda x: 10**x, constit)
        f_app_pred = force_approach(t_app, constit, indentation, tip)
        f_ret_pred = force_retract(t_ret, constit, indentation, tip)
        return (f_app_pred - f_app, f_ret_pred - f_ret)
        # return f_app_pred - f_app
        # return f_ret_pred - f_ret

    return residual, indentation


filename = f"PAA_speed {speed}_4nN"
dataset = import_data(datadir / f"{filename}.tab", datadir / f"{filename}.tsv")
dataset = truncate_adhesion(dataset)
dataset, scale = normalize_dataset(dataset)
tip = Spherical(1e-6 / scale.depth)

residual_func, indentation = make_residual(dataset, tip)
# %%


constit = StandardLinearSolid(-1.0, -1.0, -1.0)
residual_func(constit, None)
# %%
solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
sol = optx.least_squares(residual_func, solver, constit, max_steps=1000)
# %%
sol.value.tau
# %%
constit = jax.tree.map(lambda x: 10**x, sol.value)
f_app_fit = force_approach(dataset.t_app, constit, indentation, tip)
f_ret_fit = force_retract(dataset.t_ret, constit, indentation, tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(dataset.t_app, dataset.f_app)
ax.plot(dataset.t_ret, dataset.f_ret)
ax.plot(dataset.t_app, f_app_fit)
ax.plot(dataset.t_ret, f_ret_fit)
# %%
indentation.t_ret
# %%
plt.plot(dataset.t_app, sol.value.relaxation_function(dataset.t_app))
# %%
dataset.t_app[1] - dataset.t_app[0]
# %%
