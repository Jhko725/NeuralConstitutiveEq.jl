import abc

from jax import Array
import jax.numpy as jnp
import equinox as eqx


class AbstractLogDiscreteSpectrum(eqx.Module):
    @abc.abstractmethod
    def discrete_spectrum(self) -> tuple[Array, Array]:
        pass

    def __call__(self):
        return self.discrete_spectrum()


class HonerkampWeeseBimodal(AbstractLogDiscreteSpectrum):
    A: float
    B: float
    t_x: float
    t_y: float
    t_a: float
    t_b: float
    n: int

    def __init__(
        self,
        A: float = 1 / (2 * jnp.sqrt(2 * jnp.pi)),
        B: float = 1 / (2 * jnp.sqrt(2 * jnp.pi)),
        t_x: float = 5e-2,
        t_y: float = 5.0,
        t_a: float = 1e-3,
        t_b: float = 1e2,
        n: int = 1000,
    ):
        self.A = A
        self.B = B
        self.t_x = t_x
        self.t_y = t_y
        self.t_a = t_a
        self.t_b = t_b
        self.n = n

    def discrete_spectrum(self) -> tuple[Array, Array]:
        log10_t_a, log10_t_b = jnp.log10(self.t_a), jnp.log10(self.t_b)
        log10_t_grid = jnp.linspace(log10_t_a, log10_t_b, self.n)
        t_grid = 10**log10_t_grid
        h_grid = self.A * jnp.exp(
            -0.5 * jnp.log(t_grid / self.t_x) ** 2
        ) + self.B * jnp.exp(-0.5 * jnp.log(t_grid / self.t_y) ** 2)
        return log10_t_grid, h_grid
