# ruff: noqa: F722
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Float, Array
import equinox as eqx
import diffrax
import optimistix as optx

from neuralconstitutive.custom_types import FloatScalar
from neuralconstitutive.constitutive import AbstractConstitutiveEqn
from neuralconstitutive.tipgeometry import AbstractTipGeometry
from neuralconstitutive.indentation import Indentation, interpolate_indentation
from neuralconstitutive.integrate import integrate


class _TingIndentationProblem(eqx.Module):
    constitutive: AbstractConstitutiveEqn
    tip: AbstractTipGeometry
    app_interp: diffrax.LinearInterpolation  # Change later
    ret_interp: diffrax.LinearInterpolation
    """A helper class containing all the relevant information to calculate Ting's solution for the force."""

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


@eqx.filter_jit
def force_ting(
    constitutive: AbstractConstitutiveEqn,
    tip: AbstractTipGeometry,
    approach: Indentation,
    retract: Indentation,
) -> tuple[Float[Array, " {len(approach)}"], Float[Array, " {len(retract)}"]]:
    ting = _TingIndentationProblem(constitutive, tip, approach, retract)

    def _force_approach(t: FloatScalar) -> FloatScalar:
        return integrate(ting.force_integrand, (0.0, t), (t,))

    f_app = eqx.filter_vmap(_force_approach)(approach.time)
    f_ret = _force_retract(retract, ting)
    return f_app, f_ret


def force_approach(
    constitutive: AbstractConstitutiveEqn,
    approach: Indentation,
    tip: AbstractTipGeometry,
) -> Float[Array, " {len(approach)}"]:
    app_interp = diffrax.LinearInterpolation(approach.time, approach.depth)

    def _force_approach(t, constitutive):
        return force_approach_((t, constitutive), app_interp, tip)

    return eqx.filter_vmap(_force_approach, in_axes=(0, None), out_axes=0)(
        approach.time, constitutive
    )


def force_approach_manual(
    t: FloatScalar,
    constitutive: AbstractConstitutiveEqn,
    approach: Indentation,
    tip: AbstractTipGeometry,
):
    app_interp = diffrax.LinearInterpolation(approach.time, approach.depth)
    return force_approach_((t, constitutive), app_interp, tip)


def relaxation_fn(t_constit: tuple[float, AbstractConstitutiveEqn]):
    t, constitutive = t_constit
    return constitutive.relaxation_function(t)


def Drelaxation_fn(t_constit: tuple[float, AbstractConstitutiveEqn]):
    dG_pytree = eqx.filter_grad(relaxation_fn)(t_constit)
    dG_array, tree_def = jtu.tree_flatten(dG_pytree)
    dG_array = jnp.asarray(dG_array)
    return dG_array, tree_def


def make_force_integrand(constit, app_interp, tip):
    a, b = tip.a(), tip.b()

    def _integrand(s, t):
        g = relaxation_fn((jnp.clip(t - s, 0.0), constit))
        dh_b = b * app_interp.derivative(s) * app_interp.evaluate(s) ** (b - 1)
        return a * g * dh_b

    return _integrand


def make_dforce_integrand(constit, app_interp, tip):
    a, b = tip.a(), tip.b()

    def _integrand(s, t):
        dG, _ = Drelaxation_fn((jnp.clip(t - s, 0.0), constit))
        dh_b = b * app_interp.derivative(s) * app_interp.evaluate(s) ** (b - 1)
        return a * dG * dh_b

    return _integrand


@eqx.filter_custom_vjp
def force_approach_(t_constit, app_interp, tip):
    t, constitutive = t_constit
    del t_constit
    force_integrand = make_force_integrand(constitutive, app_interp, tip)
    args = (t,)
    bounds = (jnp.zeros_like(t), t)
    return integrate(force_integrand, bounds, args)


@force_approach_.def_fwd
def force_approach_fwd(perturbed, t_constit, app_interp, tip):
    del perturbed
    f_app = force_approach_(t_constit, app_interp, tip)
    return f_app, None  # other stuff


@force_approach_.def_bwd
def force_approach_bwd(residuals, grad_obj, perturbed, t_constit, app_interp, tip):
    del residuals, perturbed
    t, constitutive = t_constit
    del t_constit
    force_integrand = make_force_integrand(constitutive, app_interp, tip)
    dforce_integrand = make_dforce_integrand(constitutive, app_interp, tip)
    args = (t,)
    bounds = (jnp.zeros_like(t), t)
    out = integrate(dforce_integrand, bounds, args)
    out = out.at[0].add(force_integrand(t, t))
    out = out * grad_obj
    _, tree_def = Drelaxation_fn((t, constitutive))
    return jtu.tree_unflatten(tree_def, out)


def _force_retract(retract: Indentation, problem: _TingIndentationProblem):
    t1 = _find_t1(retract, problem)
    t_ret = retract.time
    return eqx.filter_vmap(integrate, in_axes=(None, None, 0, 0))(
        problem.force_integrand, 0.0, t1, t_ret
    )


def _find_t1(retract: Indentation, problem: _TingIndentationProblem):

    t_m = retract.time[0]

    def residual(t1: float, t: float) -> float:

        return integrate(problem.t1_integrand_lower, (t1, t_m), (t,)) + integrate(
            problem.t1_integrand_upper, (t_m, t), (t,)
        )

    solver = optx.Bisection(rtol=1e-3, atol=1e-3, flip=True)

    @eqx.filter_vmap
    def _find_t1_inner(t: float) -> float:
        cond = residual(0.0, (t,)) <= 0
        t_ = jnp.where(cond, t_m, t)
        return jnp.where(
            cond,
            0.0,
            optx.root_find(
                residual,
                solver,
                t_m,
                args=t_,
                options={"lower": 0.0, "upper": t_m},
                max_steps=30,
                throw=False,
            ).value,
        )

    return _find_t1_inner(retract.time)
