from jax import Array
import equinox as eqx
import diffrax


class Indentation(eqx.Module):
    trajectory: diffrax.AbstractGlobalInterpolation

    def __init__(self, time: Array, indentation: Array):
        # TODO: add keyword argument that determines what interpolation to use
        # TODO: check input to see if indentation is monotone increasing or not
        self.trajectory = diffrax.LinearInterpolation(time, indentation)

    @property
    def t(self) -> Array:
        return self.trajectory.ts

    def z(self, t: Array) -> Array:
        return self.trajectory.evaluate(t)

    def v(self, t: Array) -> Array:
        return self.trajectory.derivative(t)
