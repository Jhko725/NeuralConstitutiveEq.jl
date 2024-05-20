# %%
from functools import partial
from pathlib import Path

import optax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float

from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
    FromLogDiscreteSpectrum,
)
from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.indentation import Indentation, interpolate_indentation
from neuralconstitutive.io import import_data

# from neuralconstitutive.integrate import integrate
from neuralconstitutive.plotting import plot_relaxation_fn
from neuralconstitutive.relaxation_spectrum import HonerkampWeeseBimodal
from neuralconstitutive.tipgeometry import Spherical
from neuralconstitutive.utils import (
    normalize_forces,
    normalize_indentations,
    smooth_data,
)

jax.config.update("jax_enable_x64", True)
# %%

datadir = Path("data/abuhattum_iscience_2022/Interphase rep 2")
name = "interphase_speed 2_2nN"
(app, ret), (f_app_data, f_ret_data) = import_data(
    datadir / f"{name}.tab", datadir / f"{name}.tsv"
)
# f_ret_data = jnp.clip(f_ret_data, 0.0)
f_ret_data = jnp.trim_zeros(jnp.clip(f_ret_data, 0.0), "b")
ret = Indentation(ret.time[: len(f_ret_data)], ret.depth[: len(f_ret_data)])
(f_app_data, f_ret_data), _ = normalize_forces(f_app_data, f_ret_data)
(app, ret), (_, h_m) = normalize_indentations(app, ret)
# %%
import scipy.interpolate as scinterp
import diffrax

def make_smoothed_cubic_spline(indentation, s = 1.5e-4):
    tck = scinterp.splrep(indentation.time, indentation.depth, s=s)
    ppoly = scinterp.PPoly.from_spline(tck)
    cubic_interp = diffrax.CubicInterpolation(ppoly.x[3:-3], tuple(ppoly.c[:, 3:-3]))
    return cubic_interp

fig, axes = plt.subplots(1, 2, figsize=(5, 3))

for s in [1.5e-4]:
    tck = scinterp.splrep(app.time, app.depth, s=s)
    spl_app = scinterp.UnivariateSpline._from_tck(tck)
    v_app = spl_app.derivative()(app.time)
    axes[0].plot(
        app.time, spl_app(app.time) - app.depth, ".", markersize=1.0, label=f"s={s}"
    )
    axes[1].plot(app.time, v_app, ".", markersize=1.0, label=f"s={s}")

    ppoly = scinterp.PPoly.from_spline(tck)
    app_interp = diffrax.CubicInterpolation(ppoly.x[3:-3], tuple(ppoly.c[:, 3:-3]))
    axes[0].plot(
        app.time,
        app_interp.evaluate(app.time) - app.depth,
        ".",
        markersize=1.0,
        label=f"s={s}, diffrax",
    )
    axes[1].plot(
        app.time,
        app_interp.derivative(app.time),
        ".",
        markersize=1.0,
        label=f"s={s}, diffrax",
    )
    print(len(spl_app.get_knots()))

coeffs = diffrax.backward_hermite_coefficients(app.time, app.depth)
app_interp_diffrax = diffrax.CubicInterpolation(app.time, coeffs)

axes[-1].legend()
# %%
ppoly = scinterp.PPoly.from_spline(tck)
app_interp = diffrax.CubicInterpolation(ppoly.x[3:-3], tuple(ppoly.c[:, 3:-3]))

# %%
coeffs = tuple(ppoly.c[:, 3:-3])
coeffs
# %%
ppoly.c[:, 3:-3].shape
# %%
ts, *coeffs = diffrax.backward_hermite_coefficients(app.time, app.depth)
print(ts.shape)
print(coeffs[1].shape)
# %%
len(coeffs)


# %%
@eqx.filter_jit
def get_velocity(times, interp):
    return interp.derivative(times)

t_test = jnp.linspace(app.time[0], app.time[-1], 10*len(app.time))
#%%
%%timeit
v_app1 = get_velocity(t_test, app_interp_diffrax)
v_app1.block_until_ready()
# %%
%%timeit
v_app2 = get_velocity(t_test, app_interp)
v_app2.block_until_ready()
# %%
%%timeit
v_app3 = get_velocity(t_test, app_interp2)
v_app3.block_until_ready()
# %%
class MyCubic(diffrax.CubicInterpolation):

    def _interpret_t(
    self, t: RealScalarLike, left: bool
    ) -> tuple[IntScalarLike, RealScalarLike]:
        maxlen = self.ts_size - 2
        index = jnp.searchsorted(self.ts, t, side="left" if left else "right", method="compare_all")
        index = jnp.clip(index - 1, a_min=0, a_max=maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part
# %%
app_interp2 = MyCubic(ppoly.x[3:-3], tuple(ppoly.c[:, 3:-3]))
# %%
app_interp2.evaluate(app.time[0])
# %%
a