# ruff: noqa: F722
import operator
from typing import Any
import dataclasses

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Float, Array
import jaxopt
import equinox as eqx
import optimistix as optx
import numpy as np
from scipy.stats import qmc

from neuralconstitutive.custom_types import FloatScalar


def count_free_params(params: PyTree):
    return jax.tree.reduce(operator.add, jax.tree.map(maybe_size, params))


def maybe_size(leaf: Any) -> int:
    return leaf.size if isinstance(leaf, jax.Array) else 0


def bayesian_information_criterion(residual_fn, y, args) -> FloatScalar:
    residuals = residual_fn(y, args)
    n_data = len(residuals)
    n_params = count_free_params(y)
    rss = jnp.sum(residuals**2) / n_data
    bic = n_data * jnp.log(rss / n_data) + n_params * jnp.log(n_data)
    return bic


def get_num_params(sample_range: tuple) -> int:
    lower, upper = sample_range
    n_params = len(lower)
    assert n_params == len(
        upper
    ), "Number of parameters for lower and upper bound must be the same"
    return n_params


def scale_linear(samples, lower, upper):
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)

    input_shape = samples.shape
    samples = samples.reshape((-1, len(lower)))
    scaled = qmc.scale(samples, lower, upper)
    return scaled.reshape(input_shape)


def scale_loglinear(samples, lower, upper):
    log_lower, log_upper = np.log10(lower), np.log10(upper)
    return 10 ** scale_linear(samples, log_lower, log_upper)


def scale_samples(samples, range, scale):
    samples_scaled = []
    lower, upper = range
    for i, s in enumerate(scale):
        if s == "linear":
            s_scaled = scale_linear(samples[:, i], lower[i], upper[i])
            samples_scaled.append(s_scaled)
        elif s == "log":
            s_scaled = scale_loglinear(samples[:, i], lower[i], upper[i])
            samples_scaled.append(s_scaled)
        else:
            raise ValueError(f"Unrecognized scaling: scale = {s}")
    return jnp.stack(samples_scaled, axis=-1)


def sample_params_lhs(n_samples: int, sample_range, scale=None, seed: int = 0):
    n_params = get_num_params(sample_range)

    if scale is None:
        scale = ["linear"] * n_params
    else:
        assert (
            len(scale) == n_params
        ), "The length of scale does not match the actual number of parameters."

    sampler = qmc.LatinHypercube(d=n_params, seed=seed)

    samples_normalized: Float[np.ndarray, "n_samples n_params"] = sampler.random(
        n_samples
    )
    return scale_samples(samples_normalized, sample_range, scale)


class NoBound(eqx.Module):

    def bound_value(self, x: FloatScalar) -> FloatScalar:
        return x

    def unbound_value(self, x: FloatScalar) -> FloatScalar:
        return x


class UpperBound(eqx.Module):
    upper: float

    def bound_value(self, x: FloatScalar) -> FloatScalar:
        return self.upper + 1 - jnp.sqrt(x**2 + 1)

    def unbound_value(self, x: FloatScalar) -> FloatScalar:
        return jnp.sqrt((self.upper - x + 1) ** 2 - 1)


class LowerBound(eqx.Module):
    lower: float

    def bound_value(self, x: FloatScalar) -> FloatScalar:
        return self.lower - 1 + jnp.sqrt(x**2 + 1)

    def unbound_value(self, x: FloatScalar) -> FloatScalar:
        return jnp.sqrt((x - self.lower + 1) ** 2 - 1)


class BothBound(eqx.Module):
    lower: float
    upper: float

    def bound_value(self, x: FloatScalar) -> FloatScalar:
        lo, up = self.lower, self.upper
        return lo + 0.5 * (up - lo) * (jnp.sin(x) + 1)

    def unbound_value(self, x: FloatScalar) -> FloatScalar:
        lo, up = self.lower, self.upper
        return jnp.arcsin(2 * (x - lo) / (up - lo) - 1)


def make_transformation_function(bounds: tuple[PyTree, PyTree]):
    bounds = jax.tree.map(float, bounds)
    lower, upper = bounds
    vals_l, treedef_l = jax.tree.flatten(lower)
    vals_u, treedef_u = jax.tree.flatten(upper)

    assert (
        treedef_l == treedef_u
    ), "PyTree structure of lower and upper bounds must be the same."

    def make_bounds(lower: float, upper: float):
        if lower == -jnp.inf:
            if upper == jnp.inf:
                bound = NoBound()
            else:
                bound = UpperBound(upper)
        else:
            if upper == jnp.inf:
                bound = LowerBound(lower)
            else:
                bound = BothBound(lower, upper)
        return bound

    bounds = jax.tree.map(make_bounds, lower, upper)
    bounds_list, _ = eqx.tree_flatten_one_level(bounds)

    def to_bounded(params_unbounded: PyTree):
        vals_unbounded, treedef = jax.tree.flatten(params_unbounded)
        vals_bounded = [b.bound_value(v) for b, v in zip(bounds_list, vals_unbounded)]

        return jax.tree.unflatten(treedef, vals_bounded)

    def to_unbounded(params_bounded: PyTree):
        vals_bounded, treedef = jax.tree.flatten(params_bounded)
        vals_unbounded = [b.unbound_value(v) for b, v in zip(bounds_list, vals_bounded)]

        return jax.tree.unflatten(treedef, vals_unbounded)

    return to_bounded, to_unbounded


class LeastSquaresResult(eqx.Module):
    value: PyTree
    success: bool
    n_eval: int


def least_squares(
    residual_fn,
    y0,
    args,
    bounds,
    *,
    backend: str = "optimistix",
    method: str = "levenbergmarquardt",
    tol=1e-6,
    max_iter: int = 256,
    **kwargs,
):
    if backend == "optimistix":
        return least_squares_optx(
            residual_fn,
            y0,
            args,
            bounds,
            method=method,
            tol=tol,
            max_iter=max_iter,
            **kwargs,
        )
    elif backend == "jaxopt":
        return least_squares_jaxopt(
            residual_fn,
            y0,
            args,
            bounds,
            method=method,
            tol=tol,
            max_iter=max_iter,
            **kwargs,
        )


def least_squares_optx(
    residual_fn,
    y0,
    args,
    bounds,
    *,
    method: str = "levenbergmarquardt",
    tol: float = 1e-6,
    max_iter: int = 256,
    **kwargs,
) -> LeastSquaresResult:

    to_bounded, to_unbounded = make_transformation_function(bounds)

    y0_unbounded = to_unbounded(y0)

    @eqx.filter_jit
    def objective_fn_unbounded(y, args):
        y_bounded = to_bounded(y)
        return residual_fn(y_bounded, args)

    solver = optx.LevenbergMarquardt(rtol=tol, atol=tol)
    sol = optx.least_squares(
        objective_fn_unbounded, solver, y0_unbounded, args, max_steps=max_iter, **kwargs
    )
    sol_bounded = dataclasses.replace(sol, value=to_bounded(sol.value))
    return LeastSquaresResult(
        sol_bounded.value,
        sol_bounded.result == optx.RESULTS.successful,
        sol_bounded.stats["num_steps"],
    )


def least_squares_jaxopt(
    residual_fn,
    y0,
    args,
    bounds,
    *,
    method: str = "dogbox",
    tol: float = 1e-6,
    max_iter: int = 256,
    **kwargs,
) -> LeastSquaresResult:
    options = dict(max_nfev=max_iter, xtol=tol, ftol=tol, gtol=tol)
    solver = jaxopt.ScipyBoundedLeastSquares(
        fun=residual_fn, method=method, options=options
    )
    result = solver.run(y0, bounds=bounds, args=args, **kwargs)
    return LeastSquaresResult(
        result.params, result.state.success, result.state.num_fun_eval
    )
