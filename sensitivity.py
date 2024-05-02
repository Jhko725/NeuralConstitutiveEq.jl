# %%
# ruff: noqa: F722
from pathlib import Path

import equinox as eqx
import jax
import jax.flatten_util
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import (
    KohlrauschWilliamsWatts,
    StandardLinearSolid,
)
from neuralconstitutive.io import import_data
from neuralconstitutive.plotting import plot_indentation
from neuralconstitutive.tingx import force_approach, force_retract
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.utils import (
    normalize_forces,
    normalize_indentations,
    smooth_data,
)

jax.config.update("jax_enable_x64", True)


# %%

datadir = Path("open_data/PAAM hydrogel/speed 5")
(app, ret), (f_app, f_ret) = import_data(
    datadir / "PAA_speed 5_4nN.tab", datadir / "PAA_speed 5_4nN.tsv"
)
app, ret = smooth_data(app), smooth_data(ret)
# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0] = plot_indentation(axes[0], app, marker=".")
axes[0] = plot_indentation(axes[0], ret, marker=".")

axes[1].plot(app.time, f_app, ".")
axes[1].plot(ret.time, f_ret, ".")

axes[2].plot(app.depth, f_app, ".")
axes[2].plot(ret.depth, f_ret, ".")
# %%
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


# constit_sls = StandardLinearSolid(0.00911, 0.30103, 9544133.23332)
# constit_sls = ModifiedPowerLaw(0.31013634, 2.7684e-09, 19806.8520)
constit_sls = KohlrauschWilliamsWatts(0.04057453, 0.26956466, 9997384.77, 0.98299196)
# %%
import jax.numpy as jnp
import jax
from neuralconstitutive.tingx import force_approach_scalar
from neuralconstitutive.indentation import interpolate_indentation
from functools import partial

app_interp = interpolate_indentation(app)
ret_interp = interpolate_indentation(ret)


def sensitvity_scalar(t, constit, app, tip):

    @partial(eqx.filter_vmap, in_axes=(None, 0, None, None))
    @eqx.filter_grad
    def inner(log_constit, t, app, tip):
        constit = jtu.tree_map(jnp.exp, log_constit)
        return force_approach_scalar(t, constit, app, tip)

    log_constit = jtu.tree_map(jnp.log, constit)
    return inner(constit, t, app, tip)


out = sensitvity_scalar(app.time, constit_sls, app_interp, tip)
# %%
import jax.tree_util as jtu

print(jax.flatten_util.ravel_pytree(jtu.tree_map(jnp.mean, out)))


# %%
def residual(constit, app, f_app, tip):
    f_app_pred = force_approach(constit, app, tip)
    return jnp.sum((f_app - f_app_pred))


grad_func = eqx.filter_grad(residual, has_aux=False)
# %%
g = grad_func(constit_sls, app, f_app, tip)
# %%
g_array, _ = jax.flatten_util.ravel_pytree(g)
g_array
# %%
g_array = jnp.asarray([1.1981892, 1.50651879, 0, 0])
L_mat = g_array.reshape(-1, 1) * g_array.reshape(1, -1)
L_mat
# %%
eigvals, eigvecs = jnp.linalg.eigh(L_mat)
# %%
eigvals
# %%
eigvecs[:, 3]
# %%
residual(constit_sls, app, f_app, tip)
# %%
f_app_pred = force_approach(constit_sls, app, tip)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, ".")
ax.plot(app.time, f_app_pred)


# %%
def residual_all(constit, indentations, forces, tip):
    app, _ = indentations
    f_app, f_ret = forces
    f_app_pred = force_approach(constit, app, tip)
    f_ret_pred = force_retract(constit, indentations, tip)
    return jnp.sum((f_app - f_app_pred)) + jnp.sum((f_ret - f_ret_pred))


constit_sls = StandardLinearSolid(0.08490180, 0.26819740, 0.35419074)
# constit_sls = ModifiedPowerLaw(0.38759320, 0.08431627, 	0.01657088)
# constit_sls = KohlrauschWilliamsWatts(1.19798431, 0.01304129, 4.8266e-04, 0.05537683)
# %%
grad_func = eqx.filter_grad(residual_all, has_aux=False)
# %%
g = grad_func(constit_sls, (app, ret), (f_app, f_ret), tip)
# %%
g_array, _ = jax.flatten_util.ravel_pytree(g)
g_array
# %%
L_mat = g_array.reshape(-1, 1) * g_array.reshape(1, -1)
L_mat
# %%
eigvals, eigvecs = jnp.linalg.eigh(L_mat)
# %%
eigvals
# %%
eigvecs[:, 3]


# %%
from neuralconstitutive.misc import stretched_exp

# %%
stretched_exp(0.0, 9997384.77, 0.98299196)
# %%
import jax
from jax import make_jaxpr

make_jaxpr(jax.grad(stretched_exp, argnums=2))(0.1, 9997384.77, 0.98299196)
# %%
jnp.asarray(-2) ** 9997384.77
# %%
jax.grad(stretched_exp, argnums=2)(0.1, 9997384.77, 0.98299196)
# %%
0.0**0.98299196
# %%
