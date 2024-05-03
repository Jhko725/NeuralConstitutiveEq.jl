from typing import Any

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree


def assert_no_leaf_nodes(tree: PyTree) -> None:
    """Raises an AssertionError if tree has any leaf nodes

    Note that this does not include None nodes, which are by default considered as part of the PyTree structure (and not as leaf nodes)"""
    jtu.tree_map(_assert_false, tree)


def _assert_false(_):
    """Helper function for assert_no_leaf_nodes

    If this function is called, an Assertion error is raised
    Note that this is not written as a lambda function since assert is a statement and lambdas cannot contain statements.
    See: https://stackoverflow.com/questions/8477346/checking-assertions-in-a-lambda-in-python"""
    assert False


def partition_nondiff_diff(primals: PyTree, tangents: PyTree):
    """Partitions the primals PyTree into nondifferentiable and differentiable parts,
    depending on whether the corresponding leaf nodes in the tangents PyTree is None or not."""
    is_none_tangent = jtu.tree_map(_is_none, tangents, is_leaf=_is_none)
    return eqx.partition(primals, is_none_tangent, is_leaf=_is_none)


def _is_none(x: Any) -> bool:
    """A helper function to manipulate PyTrees

    This is used for two purposes:
    1) Use in the is_leaf argument of jtu.tree_map to treat None values as leaf nodes
    2) tree_map on to a PyTree (with is_leaf=_is_none) to obtain a boolean mask of None positions within the PyTree"""
    return x is None
