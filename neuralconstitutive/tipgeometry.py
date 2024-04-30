import abc

import equinox as eqx
import jax.numpy as jnp


class AbstractTipGeometry(eqx.Module):
    @abc.abstractmethod
    def a(self) -> float:
        pass

    @abc.abstractmethod
    def b(self) -> float:
        pass


class Spherical(AbstractTipGeometry):
    R: float

    def a(self) -> float:
        return (16 / 9) * jnp.sqrt(self.R)

    def b(self) -> float:
        return 1.5


class Conical(AbstractTipGeometry):
    theta: float

    def a(self) -> float:
        return (8 / (3 * jnp.pi)) * jnp.tan(self.theta)

    def b(self) -> float:
        return 2.0
