# %%
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.constitutive import AbstractConstitutiveEqn

# from neuralconstitutive.integrate import integrate
from neuralconstitutive.nn import FullyConnectedNetwork
from integrax.solvers import AbstractIntegration, AdaptiveSimpson, ExtendedTrapezoid
from integrax.integrate import integrate

from neuralconstitutive.fitting import create_subsampled_interpolants
from neuralconstitutive.constitutive import FromLogDiscreteSpectrum
from neuralconstitutive.relaxation_spectrum import HonerkampWeeseBimodal
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.tipgeometry import Spherical

from neuralconstitutive.ting import _force_approach, _force_retract
from jaxtyping import Array
import numpy as np
from neuralconstitutive.plotting import plot_relaxation_fn

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


class NeuralConstitutive(AbstractConstitutiveEqn):
    nn: FullyConnectedNetwork

    @property
    def dlogt(self):
        return np.log10(self.t_grid[1]) - np.log10(self.t_grid[0])

    def relaxation_function(self, t: FloatScalar) -> FloatScalar:
        log10t_grid = jnp.linspace(-5, 5, 50)
        t_grid = 10**log10t_grid
        dlogt = jnp.log(t_grid[1]) - jnp.log(t_grid[0])
        h_grid = eqx.filter_vmap(self.nn)(log10t_grid)
        dG = h_grid * jnp.exp(-t / t_grid) * dlogt
        return jnp.sum(dG) / 40.0


# %%
nn = FullyConnectedNetwork(["scalar", 10, 10, "scalar"])
constit = NeuralConstitutive(nn)

# %%
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
G = eqx.filter_vmap(constit.relaxation_function)(t)
# %%

# %%
bimodal = FromLogDiscreteSpectrum(
    HonerkampWeeseBimodal(t_x=5e-3, t_y=0.5, t_a=1e-4, t_b=10)
)
t_app = jnp.arange(0.0, 1.0 + 1e-2, 1e-2)
d_app = 1.0 * t_app
t_ret = jnp.arange(1.0, 2.0 + 1e-2, 1e-2)
d_ret = 2.0 - t_ret

app, ret = Indentation(t_app, d_app), Indentation(t_ret, d_ret)
del t_app, d_app, t_ret, d_ret

app_interp = create_subsampled_interpolants(app)
ret_interp = create_subsampled_interpolants(ret)

tip = Spherical(1.0)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t, G, ".", label="pred")
ax = plot_relaxation_fn(ax, bimodal, app.time, label="data")
ax.legend()
# %%


# %%timeit
f_app = eqx.filter_jit(_force_approach)(app.time, constit, app_interp, tip)
f_ret = eqx.filter_jit(_force_retract)(ret.time, constit, (app_interp, ret_interp), tip)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app)
ax.plot(ret.time, f_ret, ".")
# %%
f_app_data = eqx.filter_jit(_force_approach)(app.time, bimodal, app_interp, tip)
f_ret_data = eqx.filter_jit(_force_retract)(
    ret.time, bimodal, (app_interp, ret_interp), tip
)

# %%
import optax


@eqx.filter_value_and_grad
def compute_loss(constit):
    f_app = eqx.filter_jit(_force_approach)(app.time, constit, app_interp, tip)
    f_ret = eqx.filter_jit(_force_retract)(
        ret.time, constit, (app_interp, ret_interp), tip
    )
    return jnp.sum((f_app - f_app_data) ** 2) + jnp.sum((f_ret - f_ret_data) ** 2)


@eqx.filter_jit
def make_step(constit, opt_state):
    loss, grads = compute_loss(constit)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(constit, updates)
    return loss, model, opt_state


# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_data, label="data")
ax.plot(ret.time, f_ret_data, label="data")
f_app = eqx.filter_jit(_force_approach)(app.time, constit, app_interp, tip)
f_ret = eqx.filter_jit(_force_retract)(ret.time, constit, (app_interp, ret_interp), tip)
ax.plot(app.time, f_app, label="prediction")
ax.plot(ret.time, f_ret, label="prediction")
ax.legend()
# %%
import numpy as np

optim = optax.adam(5e-3)
opt_state = optim.init(constit)

max_epochs = 100
loss_history = np.empty(max_epochs)
for step in range(max_epochs):
    loss, constit, opt_state = make_step(constit, opt_state)
    loss = loss.item()
    loss_history[step] = loss
    print(f"step={step}, loss={loss}")
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app_data, label="data")
f_app = eqx.filter_jit(_force_approach)(app.time, constit, app_interp, tip)
ax.plot(app.time, f_app, label="prediction")
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(np.arange(max_epochs), loss_history)
ax.set_yscale("log", base=10)
# %%
