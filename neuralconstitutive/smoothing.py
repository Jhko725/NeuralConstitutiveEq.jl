import diffrax
import jax.numpy as jnp
import scipy.interpolate as scinterp


class PiecewiseCubic(diffrax.CubicInterpolation):
    def _interpret_t(self, t, left: bool) -> tuple:
        maxlen = self.ts_size - 2
        index = jnp.searchsorted(
            self.ts, t, side="left" if left else "right", method="compare_all"
        )
        index = jnp.clip(index - 1, a_min=0, a_max=maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part


def make_smoothed_cubic_spline(indentation, s=1.5e-4):
    tck = scinterp.splrep(indentation.time, indentation.depth, s=s)
    ppoly = scinterp.PPoly.from_spline(tck)
    cubic_interp = PiecewiseCubic(ppoly.x[3:-3], tuple(ppoly.c[:, 3:-3]))
    return cubic_interp
