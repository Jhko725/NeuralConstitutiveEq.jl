# %%
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.constitutive import AbstractConstitutiveEqn
from neuralconstitutive.integrate import integrate
from neuralconstitutive.nn import FullyConnectedNetwork

jax.config.update("jax_enable_x64", True)


class NeuralConstitutive(AbstractConstitutiveEqn):
    nn: FullyConnectedNetwork

    def integrand(self, k: FloatScalar, t: FloatScalar) -> FloatScalar:
        return self.nn(k) * jnp.exp(-t * jnp.exp(-k))

    def relaxation_function(self, t: FloatScalar) -> FloatScalar:
        return integrate(self.integrand, (-5, 5), (t,))


# %%
nn = FullyConnectedNetwork(["scalar", 20, 20, "scalar"])
constit = NeuralConstitutive(nn)

# %%
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
G = eqx.filter_vmap(constit.relaxation_function)(t)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t, G, ".")


# %%
def Inn(t):
    def _inner(s, _):
        return nn(s)

    return integrate(_inner, (0, t), (None,))


t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
G = eqx.filter_vmap(Inn)(t)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(t, jax.nn.tanh(G), ".")
# %%
