import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω

from integrax.tree_util import assert_no_leaf_nodes, partition_nondiff_diff


def implicit_jvp(fn_primal, inputs):
    out = _implicit_impl(fn_primal, inputs)
    return out


@eqx.filter_custom_jvp
def _implicit_impl(fn_primal, inputs):
    # Do the primal computation, and ensure output leaf nodes are arrays
    return jtu.tree_map(jnp.asarray, fn_primal(inputs))


@_implicit_impl.def_jvp
def _implicit_impl_jvp(primals, tangents):
    fn_primal, inputs = primals
    t_fn_primal, t_inputs = tangents

    assert_no_leaf_nodes(t_fn_primal)
    del t_fn_primal

    fn, method, lower, upper, args, options, while_loop = inputs
    t_fn, _, t_lower, t_upper, t_args, _, _ = t_inputs

    nondiff, diff = partition_nondiff_diff((fn, args), (t_fn, t_args))

    def _jvp_func(x, jvp_args):
        fn_args_diff, t_fn_args = jvp_args

        def _for_jvp(fn_args_diff_):
            fn, args = eqx.combine(nondiff, fn_args_diff_)
            return fn(x, args)

        return jax.jvp(_for_jvp, (fn_args_diff,), (t_fn_args,))[1]
        

    jvp_inputs = (
        _jvp_func,
        method,
        lower,
        upper,
        (diff, (t_fn, t_args)),
        options,
        while_loop,
    )

    tangent_body = implicit_jvp(fn_primal, jvp_inputs)

    def for_jvp_boundary(x, t_x):
        x_nondiff, x_diff = partition_nondiff_diff(x, t_x)

        def _f_inner(x_diff_):
            x = eqx.combine(x_nondiff, x_diff_)
            return fn(x, args)

        return jax.jvp(_f_inner, (x_diff,), (t_x,))[1]

    tangent_upper = for_jvp_boundary(upper, t_upper)
    tangent_lower = for_jvp_boundary(lower, t_lower)

    tangent_boundary = (tangent_upper**ω - tangent_lower**ω).ω
    tangent_out = (tangent_body**ω + tangent_boundary**ω).ω
    primal_out = implicit_jvp(fn_primal, inputs)

    return primal_out, tangent_out
