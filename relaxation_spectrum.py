# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import (
    FromLogDiscreteSpectrum,
    StandardLinearSolid,
    ModifiedPowerLaw,
    AbstractConstitutiveEqn,
)
from neuralconstitutive.relaxation_spectrum import HonerkampWeeseBimodal
from neuralconstitutive.indentation import Indentation, interpolate_indentation
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.ting import force_approach, force_retract
from neuralconstitutive.integrate import integrate
from neuralconstitutive.fitting import fit_approach_lmfit, fit_all_lmfit
from neuralconstitutive.nn import FullyConnectedNetwork
from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.plotting import plot_relaxation_fn

jax.config.update("jax_enable_x64", True)

# Create the spectrum to generate simulated data
# Visualize it
bimodal = FromLogDiscreteSpectrum(
    HonerkampWeeseBimodal(t_x=5e-3, t_y=0.5, t_a=1e-4, t_b=10)
)
t_i, h_i = bimodal.discrete_spectrum

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_i, h_i, ".")
ax.set_xscale("log", base=10)
ax.set_xlabel("Relaxation Time τ[s]")
ax.set_ylabel("Relaxation Spectrum H(τ)[Pa]")
# %%
# "Experimental" time points and indentation
# Set it so that Dt and max(t_exp) more or less span the interesting parts of the spectrum
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
h = 1.0 * t
len(t)
# %%
app = Indentation(t, h)
t_ret = jnp.arange(1.0, 2.0 + 1e-3, 1e-3)
h_ret = 2.0 - t_ret
ret = Indentation(t_ret, h_ret)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, app.depth, ".", label="approach")
ax.plot(ret.time, ret.depth, ".", label="retract")

# %%
y = bimodal.relaxation_function(app.time)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, y, ".")
ax.set_xlabel("Time")
ax.set_ylabel("Relaxation function")

# %%
tip = Spherical(1.0)
f_app = force_approach(bimodal, app, tip)
f_ret = force_retract(bimodal, (app, ret), tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, ".", label="approach")
ax.plot(ret.time, f_ret, ".", label="retract")
ax.legend()
fig

# %%
# %%
sls = StandardLinearSolid(10.0, 10.0, 10.0)
bounds = [(1e-7, 1e7)] * 3
sls_fit, result = fit_approach_lmfit(sls, bounds, tip, app, f_app)
# %%
f_fit_app = force_approach(sls_fit, app, tip)
f_fit_ret = force_retract(sls_fit, (app, ret), tip)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
axes[0].plot(app.time, f_app, ".", color="black", alpha=0.5, label="Data")
axes[0].plot(ret.time, f_ret, ".", color="black", alpha=0.5)

axes[0].plot(app.time, f_fit_app, "-", color="orangered", alpha=0.8, label="SLS")
axes[0].plot(ret.time, f_fit_ret, "-", color="orangered", alpha=0.8)


axes[0].set_xlabel("Normalized time")
axes[0].set_ylabel("Normalized force")

axes[1] = plot_relaxation_fn(
    axes[1],
    bimodal,
    app.time,
    marker=".",
    color="black",
    alpha=0.5,
    label="Ground truth",
)
axes[1] = plot_relaxation_fn(
    axes[1], sls_fit, app.time, color="orangered", alpha=0.8, label="SLS model"
)

axes[1].set_xlabel("Normalized time")
axes[1].set_ylabel("Normalized relaxation function")
axes[1].legend()

for ax in axes:
    ax.grid(ls="--", color="lightgray")

fig.suptitle("Fitting only on the approach curve")
# %%


# %%
## Fit all
sls_fit, result = fit_all_lmfit(sls, bounds, tip, (app, ret), (f_app, f_ret))
# %%
result
# %%
f_fit_app = force_approach(sls_fit, app, tip)
f_fit_ret = force_retract(sls_fit, (app, ret), tip)
# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
axes[0].plot(app.time, f_app, ".", color="black", alpha=0.5, label="Data")
axes[0].plot(ret.time, f_ret, ".", color="black", alpha=0.5)

axes[0].plot(app.time, f_fit_app, "-", color="orangered", alpha=0.8, label="SLS")
axes[0].plot(ret.time, f_fit_ret, "-", color="orangered", alpha=0.8)


axes[0].set_xlabel("Normalized time")
axes[0].set_ylabel("Normalized force")

axes[1] = plot_relaxation_fn(
    axes[1],
    bimodal,
    app.time,
    marker=".",
    color="black",
    alpha=0.5,
    label="Ground truth",
)
axes[1] = plot_relaxation_fn(
    axes[1], sls_fit, app.time, color="orangered", alpha=0.8, label="SLS model"
)

axes[1].set_xlabel("Normalized time")
axes[1].set_ylabel("Normalized relaxation function")
axes[1].legend()

for ax in axes:
    ax.grid(ls="--", color="lightgray")

fig.suptitle("Fitting on the approach+retract curve")

# %%


# %%
from functools import partial
import equinox as eqx


class NeuralConstitutive(AbstractConstitutiveEqn):
    nn: FullyConnectedNetwork

    def relaxation_function(self, t: FloatScalar) -> FloatScalar:
        @partial(eqx.filter_vmap, in_axes=(0, None))
        def integrand(k: FloatScalar, t: FloatScalar) -> FloatScalar:
            t = jnp.asarray(t)
            return self.nn(k) * jnp.exp(-t * jnp.exp(-k))

        dk = 0.01
        k_grid = jnp.arange(-5, 5, dk)
        return jnp.sum(integrand(k_grid, t)) * dk


# %%
nn = FullyConnectedNetwork(["scalar", 20, 20, "scalar"])
constit = NeuralConstitutive(nn)

# %%
f_nn_app = force_approach(constit, app, tip)
# %%
f_nn_ret = force_retract(constit, (app, ret), tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, ".", color="black", alpha=0.5, label="Data")
ax.plot(ret.time, f_ret, ".", color="black", alpha=0.5, label="Data")
ax.plot(app.time, f_nn_app, ".", color="orangered", alpha=0.8, label="NN")
ax.plot(ret.time, f_nn_ret, ".", color="orangered", alpha=0.8, label="NN")
ax.legend()
fig
# %%
from jaxtyping import Array


def l2_loss(x: Array, x_pred: Array) -> float:
    return jnp.mean((x - x_pred) ** 2)


def loss_total(
    model: AbstractConstitutiveEqn,
    trajectories,
    forces: tuple[Array, Array],
    tip,
) -> float:
    app, ret = trajectories
    f_app, f_ret = forces

    f_app_pred = force_approach(model, app, tip)
    f_ret_pred = force_retract(model, (app, ret), tip)
    return l2_loss(f_app, f_app_pred) + l2_loss(f_ret, f_ret_pred)


# %%
loss_total(constit, (app, ret), (f_app, f_ret), tip)
# %%
eqx.filter_grad(loss_total)(constit, (app, ret), (f_app, f_ret), tip)
# %%
from neuralconstitutive.ting import force_approach_scalar, force_integrand

# %%
app_interp = interpolate_indentation(app)
force_approach_scalar(jnp.asarray(0.4), constit, app_interp, tip)
# %%
constit
# %%
force_integrand(jnp.asarray(0.3), jnp.asarray(0.8), constit, app_interp, tip)
# %%
integrate(
    force_integrand, (0, jnp.asarray(0.8)), (jnp.asarray(0.8), constit, app_interp, tip)
)
# %%

# %%
