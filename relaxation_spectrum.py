# %%
from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class HonerkampWeeseBimodal(eqx.Module):
    A: float
    B: float
    t_x: float
    t_y: float
    t_a: float
    t_b: float
    n: int

    def __init__(
        self,
        A: float = 0.1994711402,
        B: float = 0.1994711402,
        t_x: float = 5e-2,
        t_y: float = 5.0,
        t_a: float = 1e-3,
        t_b: float = 1e2,
        n: int = 100,
    ):
        # Default values for A, B correspond to 1/(2*sqrt(2*pi))
        self.A = A
        self.B = B
        self.t_x = t_x
        self.t_y = t_y
        self.t_a = t_a
        self.t_b = t_b
        self.n = n

    def __call__(self):
        t_i = jnp.logspace(jnp.log10(self.t_a), jnp.log10(self.t_b), self.n)
        h_i = self.A * jnp.exp(-0.5 * jnp.log(t_i / self.t_x) ** 2) + self.B * jnp.exp(
            -0.5 * jnp.log(t_i / self.t_y) ** 2
        )
        return t_i, h_i


# %%
spectrum = HonerkampWeeseBimodal()
t_i, h_i = spectrum()
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t_i, h_i, ".")
ax.set_xscale("log", base=10)


# %%
@partial(jax.vmap, in_axes=(0, None, None))
def function_from_discrete_spectra(t: float, t_i: jax.Array, h_i: jax.Array):
    h0 = jnp.log(t_i[1]) - jnp.log(t_i[0])
    return jnp.dot(h_i * h0, jnp.exp(-t / t_i))


# %%
t = jnp.arange(0.0, 10.0, 0.001)
g = function_from_discrete_spectra(t, *spectrum())
# %%
plt.plot(t, g, ".")
# %%
