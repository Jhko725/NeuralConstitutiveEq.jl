from jax import Array
import jax.numpy as jnp
import equinox as eqx
import diffrax


class Trajectory(eqx.Module):
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


def make_triangular(t_max: float, dt: float, v: float) -> tuple[Trajectory, Trajectory]:
    t_app = jnp.arange(0, t_max + dt, dt)
    t_ret = jnp.arange(t_max, 2 * t_max + dt, dt)
    d_app = v * t_app
    d_ret = v * (2 * t_ret[0] - t_ret)
    return Trajectory(t_app, d_app), Trajectory(t_ret, d_ret)
