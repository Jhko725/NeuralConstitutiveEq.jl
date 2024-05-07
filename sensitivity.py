# %%
# ruff: noqa: F722
from pathlib import Path

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import (
    ModifiedPowerLaw,
    floatscalar_field,
    AbstractConstitutiveEqn,
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
from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.misc import stretched_exp

jax.config.update("jax_enable_x64", True)


class StandardLinearSolid(AbstractConstitutiveEqn):
    E0: FloatScalar = floatscalar_field()
    E_inf: FloatScalar = floatscalar_field()
    tau: FloatScalar = floatscalar_field()

    def relaxation_function(self, t):
        return self.E_inf + (self.E0 - self.E_inf) * jnp.exp(-t / self.tau)


class KohlrauschWilliamsWatts(AbstractConstitutiveEqn):
    E0: FloatScalar = floatscalar_field()
    E_inf: FloatScalar = floatscalar_field()
    tau: FloatScalar = floatscalar_field()
    beta: FloatScalar = floatscalar_field()

    def relaxation_function(self, t):
        return self.E_inf + (self.E0 - self.E_inf) * stretched_exp(
            t, self.tau, self.beta
        )


# %%

# datadir = Path("open_data/PAAM hydrogel/speed 5")
# name = "PAA_speed 5_4nN"

datadir = Path("open_data/Interphase rep 2")
name = "interphase_speed 2_2nN"
(app, ret), (f_app, f_ret) = import_data(
    datadir / f"{name}.tab", datadir / f"{name}.tsv"
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
f_ret = jnp.clip(f_ret, 0.0)
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

## pAAm, approach
# constit_sls = StandardLinearSolid(3.8624659026709196e-10+0.30914401077841713, 0.30914401077841713, 0.0006863434675687636)
# constit_sls = ModifiedPowerLaw(0.30914400690168486, 8.837375276016246e-14, 1.5692865062547984e-06)
# constit_sls = KohlrauschWilliamsWatts(6.396549956377839e-10+0.3091440117012345, 0.3091440117012345, 0.0009709553387991132, 0.6871551483634736)

## HeLa, approach
# constit_sls = StandardLinearSolid(2.889893879753913e-08+0.33311486553838643, 0.33311486553838643, 987.3255341470276)
# constit_sls = ModifiedPowerLaw(0.33311573822136786, 2.6413182681594982e-06, 0.6659960422297778)
# constit_sls = KohlrauschWilliamsWatts(
#     0.0002274177325900517 + 0.3328883488087353,
#     0.3328883488087353,
#     969.0153621760131,
#     0.8037062892890634,
# )

## HeLa, both
# constit_sls = StandardLinearSolid(0.2704624285342905+0.08856032985310458, 0.08856032985310458, 1.8086871935944557)
# constit_sls = ModifiedPowerLaw(0.4557845450756659, 0.5378549312669225, 0.34712280476197444)
constit_sls = KohlrauschWilliamsWatts(
    0.6208389433733918 + 0.07978665555324893,
    0.07978665555324893,
    0.3394100921127051,
    0.31797642295608725,
)
# %%
import jax.numpy as jnp
import jax
from neuralconstitutive.tingx import force_approach_scalar
from neuralconstitutive.indentation import interpolate_indentation
from functools import partial
from neuralconstitutive.utils import smooth_data

app_interp = interpolate_indentation(smooth_data(app))
ret_interp = interpolate_indentation(smooth_data(ret))


def sensitvity_scalar(t, constit, app, tip):

    @partial(eqx.filter_vmap, in_axes=(None, 0, None, None))
    @eqx.filter_grad
    def inner(log_constit, t, app, tip):
        constit = jtu.tree_map(jnp.exp, log_constit)
        return force_approach_scalar(t, constit, app, tip)

    log_constit = jtu.tree_map(jnp.log, constit)
    return inner(log_constit, t, app, tip)


out = sensitvity_scalar(app.time, constit_sls, app_interp, tip)
# %%
g_array = jax.flatten_util.ravel_pytree(jtu.tree_map(jnp.mean, out))[0]
g_array = g_array.at[jnp.isnan(g_array)].set(0.0)
print(g_array)
L_mat = g_array.reshape(-1, 1) * g_array.reshape(1, -1)
eigvals, eigvecs = jnp.linalg.eigh(L_mat)
print(f"eigvals={eigvals}")
print(f"best eigvec = {eigvecs[:,-1]}")


# %%
import numpy as np

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
fig, ax = plt.subplots(1, 1, figsize=(3, 3))


def plot_eigval_spectrum(ax, eigvals, *args, **hlines_kwargs):
    eigvals = jnp.clip(jnp.asarray(eigvals), 1e-50)
    ax.hlines(jnp.log10(eigvals), *args, **hlines_kwargs)
    return ax


ax = plot_eigval_spectrum(
    ax,
    [-1.86785584e-19, -1.72626105e-19, 1.79797253e-01],
    0.0,
    1.0,
    color=color_palette[6],
)
ax = plot_eigval_spectrum(
    ax,
    [1.34109128e-21, 9.22859914e-20, 1.25797755e-01],
    2.0,
    3.0,
    color=color_palette[3],
)
ax = plot_eigval_spectrum(
    ax,
    [0.00000000e00, 8.36262631e-20, 1.43577951e-19, 2.48371196e-01],
    4.0,
    5.0,
    color=color_palette[8],
)
ax.set_ylabel("log (eigenvalues)")
ax.set_xticks([0.5, 2.5, 4.5], ["MPLR", "SLS", "KWW"])
ax.set_title("Eigenvalue spectrum, HeLa, Both")


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
def normalize_and_threshold(x, threshold=0.02):
    x = np.asarray(x)
    x = x / np.sum(x * x)
    x[x < threshold] = 0.0

    return x


# %%
out = normalize_and_threshold([0.99426116, 0.0596164, 0.08882927, 0.0])
print(out)
# %%
