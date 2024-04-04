# %%
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from neuralconstitutive.constitutive import ModifiedPowerLaw, Fung
from neuralconstitutive.tipgeometry import Conical
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.ting import force_ting, force_approach

jax.config.update("jax_enable_x64", True)


# %%
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
h = 1.0 * t
app = Indentation(t, h)
t_ret = jnp.arange(1.0, 2.0 + 1e-3, 1e-3)
h_ret = 2.0 - t_ret
ret = Indentation(t_ret, h_ret)
# plr = PowerLaw(1.0, 0.5, 1.0)
fung = Fung(10.0, 0.1, 10.0, 1.0)
plr = ModifiedPowerLaw(1.0, -0.5, 1.0)
tip = Conical(jnp.pi / 18)
del t, h, t_ret, h_ret

# %%
f_app, f_ret = force_ting(plr, tip, app, ret)
# f_app, f_ret = force_ting(fung, tip, app, ret)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, ".")
ax.plot(ret.time, f_ret, ".")


# %%
@eqx.filter_jit
def test(const, tip, app, ret):
    f_app = force_approach(const, tip, app, ret)
    return jnp.sum(f_app)


# %%
value = eqx.filter_grad(test)(plr, tip, app, ret)
print(value)
# %%
value.E0

# %%
import diffrax
import quadax
from neuralconstitutive.integrate import integrate


# %%
def make_force_integrand(constitutive, tip, approach_interp):
    a, b = tip.a(), tip.b()

    def dF(s, t):
        g = constitutive.relaxation_function(t - s)
        dh_b = (
            b * approach_interp.derivative(s) * approach_interp.evaluate(s) ** (b - 1)
        )
        return a * g * dh_b

    return dF


@eqx.filter_jit
def force_approach2(constitutive, tip, approach, retract):
    del retract
    app_interp = diffrax.LinearInterpolation(approach.time, approach.depth)

    dF = make_force_integrand(constitutive, tip, app_interp)

    def F(t):
        return integrate(dF, (0.0, t), (t,))

    return eqx.filter_vmap(F)(approach.time)


@eqx.filter_jit
def force_approach3(constitutive, tip, approach, retract):
    del retract
    app_interp = diffrax.LinearInterpolation(approach.time, approach.depth)

    dF = make_force_integrand(constitutive, tip, app_interp)

    def F(t):
        return quadax.quadgk(dF, (0.0, t), (t,))[0]

    return eqx.filter_vmap(F)(approach.time[1:])


# %%
# f1 = force_approach(plr, tip, app, ret)
f2 = force_approach2(plr, tip, app, ret)

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# ax.plot(app.time, f1, ".", label="Impl 1")
ax.plot(app.time, f2, ".", label="Impl 2")
ax


# %%
def test1(constitutive, tip, approach, retract):
    return jnp.sum(force_approach(constitutive, tip, approach, retract))


def test2(constitutive, tip, approach, retract):
    return jnp.sum(force_approach2(constitutive, tip, approach, retract))


# %%
val, dval = eqx.filter_value_and_grad(test)(plr, tip, app, ret)
print(val, dval.E0, dval.alpha, dval.t0)
# %%
val2, dval2 = eqx.filter_value_and_grad(test2)(plr, tip, app, ret)
print(val2, dval2.E0, dval2.alpha, dval2.t0)
# %%
