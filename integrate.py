# %%
import abc
from typing import TypeVar, TypeAlias, Callable, Any
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree, Array, Float
import equinox as eqx
import scipy

# inspired from optimistix _custom_types.py
X = TypeVar("X")
Y = TypeVar("Y")
Args: TypeAlias = Any
Fn: TypeAlias = Callable[[X, Any], Y]


## From lineax/lineax/_misc.py
def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


## From optimistix/optimistix/_misc.py with minor alterations
## as JAX issue #15676 is resolved
def inexact_asarray(x):
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        dtype = default_floating_dtype()
    return jnp.asarray(x, dtype)


class AbstractIntegrator(eqx.Module, strict=True):
    """Abstract base class for all integration algorithms"""

    @abc.abstractmethod
    def step(self, fn, lower, upper, args):
        pass


class VectorizedMidpoint(AbstractIntegrator):
    dx: X
    buffer_size: int

    def init(self, fn, lower, upper, args):
        del fn, args
        x_grid = lower + (jnp.arange(self.buffer_size) + 0.5) * self.dx
        assert x_grid[-1] > upper

    def step(self, fn, lower, upper, args):
        n_grid, dx_f = jnp.divmod(upper - lower, self.dx)
        x_grid = (jnp.arange(n_grid) + 0.5) * self.dx
        y_grid = eqx.filter_vmap(fn, in_axes=(0, None))(x_grid, args)
        I = jnp.sum(y_grid) * self.dx
        I += fn(x_grid[-1] + 0.5 * dx_f, args) * dx_f
        return I


class AbstractAdjoint(eqx.Module, strict=True):
    """Abstract base class for adjoints - i.e. gradient computation"""

    @abc.abstractmethod
    def apply(self, primal_fn, inputs):
        pass


class DirectAdjoint(AbstractAdjoint, strict=True):

    def apply(self, primal_fn, inputs):
        return primal_fn(inputs)


def implicit_jvp():
    pass


class ImplicitAdjoint(AbstractAdjoint, strict=True):

    def apply(self, primal_fn, inputs):
        return implicit_jvp(primal_fn, inputs)


def _evaluate_integral(inputs):
    fn, solver, lower, upper, args = inputs
    del inputs
    return solver.step(fn, lower, upper, args)


@partial(eqx.filter_jit, static_argnums=(4,))
def integrate(
    fn: Fn,
    solver: AbstractIntegrator,
    lower: X,
    upper: X,
    args: PyTree[Any],
    *,
    adjoint: AbstractAdjoint
):
    r"""Numerically integrate a given function.

    Given a function `y = fn(x, args)` where x and y are pytrees representing the input and output of the function,
    this returns the solution to $\int_a^b \textrm{fn}(x, \textrm{args})$, where $\textrm{domain} = (a, b)$.
    """
    # Coerce (lower, upper) into same inexact (i.e. floating/complex) array
    # This automatically handles situations such as lower/upper being a scalar value or an integer array
    # by casting it into a float array.
    lower, upper = jtu.tree_map(inexact_asarray, (lower, upper))

    fn = eqx.filter_closure_convert(fn, lower, args)

    inputs = fn, solver, lower, upper, args
    return adjoint.apply(_evaluate_integral, inputs)


# %%
def integrate_scipy(fn, solver, lower, upper, args, *, adjoint):
    del adjoint
    fn = eqx.filter_closure_convert(fn, lower, args)
    fn = eqx.filter_jit(fn)

    return scipy.integrate.quad(fn, lower, upper, args)[0]


# %%
solver = Midpoint(dx=0.01)
out = integrate(lambda x, args: x, solver, 0.0, 2.0, None, adjoint=DirectAdjoint())

# %%

# %%

# %%
