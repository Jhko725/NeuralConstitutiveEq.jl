import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from lineax.internal import tree_dot


@eqx.filter_custom_vjp
def leibniz_integration_rule(
    fn_lower_upper_args, *, fn_primal, method, options, while_loop
):
    fn, lower, upper, args = fn_lower_upper_args
    inputs = (fn, method, lower, upper, args, options, while_loop)
    return jtu.tree_map(jnp.asarray, fn_primal(inputs))


@leibniz_integration_rule.def_fwd
def _leibniz_integration_rule_fwd(perturbed, fn_lower_upper_args, **kwargs):
    del perturbed
    integral = leibniz_integration_rule(fn_lower_upper_args, **kwargs)
    return integral, None


@leibniz_integration_rule.def_bwd
def _leibniz_integration_rule_bwd(
    residuals,
    grad_integral,
    perturbed,
    fn_lower_upper_args,
    *,
    fn_primal,
    method,
    options,
    while_loop,
):
    fn, lower, upper, args = fn_lower_upper_args
    _, perturbed_lower, perturbed_upper, _ = perturbed

    if perturbed_lower:
        grad_lower = -tree_dot(grad_integral, fn(lower, args))
    else:
        grad_lower = jnp.zeros_like(lower)

    if perturbed_upper:
        grad_upper = tree_dot(grad_integral, fn(upper, args))
    else:
        grad_upper = jnp.zeros_like(upper)

    fn_args_diff, fn_args_nondiff = eqx.partition((fn, args), eqx.is_inexact_array)

    def fn_vjp(x, fn_vjp_args):
        fn_args_diff, grad_integral = fn_vjp_args

        def for_jvp(fn_args_diff_):
            fn, args = eqx.combine(fn_args_diff_, fn_args_nondiff)
            return fn(x, args)

        _, vjp_func = jax.vjp(for_jvp, fn_args_diff)

        # vjp_func from jax.vjp returns a tuple of cotangents and we have only one
        return vjp_func(grad_integral)[0]
        # return None

    fnvjp_lower_upper_args = (fn_vjp, lower, upper, (fn_args_diff, grad_integral))
    grad_fn_args = leibniz_integration_rule(
        fnvjp_lower_upper_args,
        fn_primal=fn_primal,
        method=method,
        options=options,
        while_loop=while_loop,
    )
    grad_fn, grad_args = grad_fn_args

    return grad_fn, grad_lower, grad_upper, grad_args
