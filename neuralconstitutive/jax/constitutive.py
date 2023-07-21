import jax
import jax.numpy as jnp
import equinox as eqx


class SimpleLinearSolid(eqx.Module):
    E0: float
    E_inf: float
    tau: float

    def __init__(self, E0: float, E_inf: float, tau: float):
        self.E0 = E0
        self.E_inf = E_inf
        self.tau = tau

    def __call__(self, t: jax.Array) -> jax.Array:
        return self.E_inf + (self.E0 - self.E_inf) * jnp.exp(-t / self.tau)
