import jax
import jax.numpy as jnp

from neuralconstitutive.custom_types import FloatScalar


@jax.custom_jvp
def stretched_exp(x: FloatScalar, t: FloatScalar, b: FloatScalar) -> FloatScalar:
    return jnp.exp(-((x / t) ** b))


@stretched_exp.defjvp
def _stretched_exp_jvp(primals, tangents):
    """Define custom gradients because autodiff is numerically unstable.

    For an example of such a situation, see:
    https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#numerical-stability
    """
    x, t, b = primals
    x_dot, t_dot, b_dot = tangents

    primal_out = stretched_exp(x, t, b)
    xT = x / t
    xT_b = xT**b
    grad_x = -primal_out * (b / x) * xT_b
    grad_T = primal_out * (b / t) * xT_b
    grad_b = -primal_out * xT_b * jnp.log(xT)
    tangent_out = grad_x * x_dot + grad_T * t_dot + grad_b * b_dot
    return primal_out, tangent_out
