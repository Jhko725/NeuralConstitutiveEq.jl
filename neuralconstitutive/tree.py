# ruff: noqa: F722
from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree, Float

PyTreeDef = Any


class PyTreeTopology(eqx.Module):
    treedef: PyTreeDef
    shapes: list[tuple]
    static: PyTree

    @property
    def n_leaves(self) -> int:
        return len(self.shapes)

    @property
    def sizes(self) -> list[int]:
        return jnp.asarray([jnp.prod(jnp.asarray(s)) for s in self.shapes])


def tree_to_array1d(
    tree: PyTree, filter_spec: Callable[[PyTree], bool] = eqx.is_array
) -> Float[Array, " N"]:
    params = eqx.filter(tree, filter_spec)
    params1d = jnp.concatenate([leaf.reshape(-1) for leaf in jtu.tree_leaves(params)])
    return params1d


def get_tree_topology(
    tree: PyTree, filter_spec: Callable[[PyTree], bool] = eqx.is_array
) -> PyTreeTopology:
    params, static = eqx.partition(tree, filter_spec)
    shapes = [leaf.shape for leaf in jtu.tree_leaves(params)]
    return PyTreeTopology(jtu.tree_structure(params), shapes, static)


def array1d_to_tree(arr: Float[Array, " N"], tree_topo: PyTreeTopology) -> PyTree:
    split_inds = jnp.cumsum(tree_topo.sizes)
    array_splits = jnp.split(arr, split_inds)
    params_list = [a.reshape(s) for (a, s) in zip(array_splits, tree_topo.sizes)]
    params = jtu.tree_unflatten(tree_topo.treedef, params_list)
    return eqx.combine(params, tree_topo.static)
