# %%
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import optimistix as optx
import quadax

from neuralconstitutive.constitutive import (
    AbstractConstitutiveEqn,
    ModifiedPowerLaw,
    PowerLaw,
)
from neuralconstitutive.tipgeometry import AbstractTipGeometry, Conical
from neuralconstitutive.indentation import Indentation, interpolate_indentation

jax.config.update("jax_enable_x64", True)


class TingIndentationProblem(eqx.Module):
    constitutive: AbstractConstitutiveEqn
    tip: AbstractTipGeometry
    app_interp: diffrax.LinearInterpolation  # Change later
    ret_interp: diffrax.LinearInterpolation

    def __init__(
        self,
        constitutive: AbstractConstitutiveEqn,
        tip: AbstractTipGeometry,
        approach: Indentation,
        retract: Indentation,
    ):
        self.constitutive = constitutive
        self.tip = tip
        self.app_interp = interpolate_indentation(approach)
        self.ret_interp = interpolate_indentation(retract)

    ## Some convenience functions for improved readability
    def a(self) -> float:
        """Tip parameter a"""
        return self.tip.a()

    def b(self) -> float:
        """Tip parameter b"""
        return self.tip.b()

    def h_app(self, s: float) -> float:
        """Position of the tip during approach"""
        return self.app_interp.evaluate(s)

    def v_app(self, s: float) -> float:
        """Velocity of the tip during approach"""
        return self.app_interp.derivative(s)

    def v_ret(self, s: float) -> float:
        """Velocity of the tip during retract"""
        return self.ret_interp.derivative(s)

    def dh_b(self, s: float) -> float:
        """Calculates dh^b(s)/ds for the approach phase.

        This is used for the calculation of the force."""
        b = self.b()
        return b * self.v_app(s) * self.h_app(s) ** (b - 1)

    def relaxation_fn(self, t: float) -> float:
        return self.constitutive.relaxation_function(t)

    ## Main functions
    def force_integrand(self, s: float, t: float) -> float:
        return self.a() * self.relaxation_fn(t - s) * self.dh_b(s)

    def t1_integrand_lower(self, s: float, t: float) -> float:
        return self.relaxation_fn(t - s) * self.v_app(s)

    def t1_integrand_upper(self, s: float, t: float) -> float:
        return self.relaxation_fn(t - s) * self.v_ret(s)


def integrate(fn, lower, upper, args):
    return jnp.where(
        lower == upper,
        0.0,
        quadax.quadgk(fn, (lower, upper), (args,), epsabs=1e-4, epsrel=1e-4)[0],
    )


def force_approach(approach: Indentation, problem: TingIndentationProblem):
    t_app = approach.time
    return eqx.filter_vmap(integrate, in_axes=(None, None, 0, 0))(
        problem.force_integrand, 0.0, t_app, t_app
    )


def find_t1(retract: Indentation, problem: TingIndentationProblem):

    t_m = retract.time[0]

    def residual(t1, args):
        (t,) = args
        return integrate(problem.t1_integrand_lower, t1, t_m, t) + integrate(
            problem.t1_integrand_upper, t_m, t, t
        )

    solver = optx.Bisection(rtol=1e-3, atol=1e-3, flip=True)

    @eqx.filter_vmap
    def _find_t1(t: float) -> float:
        cond = residual(0.0, (t,)) <= 0
        t_ = jnp.where(cond, t_m, t)
        return jnp.where(
            cond,
            0.0,
            optx.root_find(
                residual,
                solver,
                t_m,
                args=(t_,),
                options={"lower": 0.0, "upper": t_m},
                max_steps=30,
                throw=False,
            ).value,
        )

    return _find_t1(retract.time)


def force_retract(retract: Indentation, problem: TingIndentationProblem):
    t1 = find_t1(retract, problem)
    t_ret = retract.time
    return eqx.filter_vmap(integrate, in_axes=(None, None, 0, 0))(
        problem.force_integrand, 0.0, t1, t_ret
    )


@eqx.filter_jit
def force(
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
    approach: Indentation,
    retract: Indentation,
):
    ting = TingIndentationProblem(constitutive, tip, approach, retract)
    f_app = force_approach(approach, ting)
    f_ret = force_retract(retract, ting)
    return f_app, f_ret


# %%
t = jnp.arange(0.0, 1.0 + 1e-3, 1e-3)
h = 1.0 * t
app = Indentation(t, h)
t_ret = jnp.arange(1.0, 2.0 + 1e-3, 1e-3)
h_ret = 2.0 - t_ret
ret = Indentation(t_ret, h_ret)
plr = PowerLaw(1.0, 0.5, 1.0)
#plr = ModifiedPowerLaw(1.0, -0.5, 1.0)
tip = Conical(jnp.pi / 18)
del t, h, t_ret, h_ret

# %%
ting = TingIndentationProblem(plr, tip, app, ret)
# %%

f_app = force_approach(app, ting)
# %%
import matplotlib.pyplot as plt

plt.plot(app.time, f_app)
# %%

t1 = find_t1(ret, ting)

# %%
plt.plot(ret.time, t1)
# %%
f_ret = force_retract(ret, ting)
# %%
plt.plot(ret.time, f_ret)
# %%
jnp.sqrt(jnp.finfo(jnp.array(1.0)).eps)
# %%
%%timeit
f_app, f_ret = force(plr, tip, app, ret)
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(app.time, f_app, ".")
ax.plot(ret.time, f_ret, ".")
# %%
