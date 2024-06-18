# ruff: noqa: F722
import jax
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, Float
import equinox as eqx

from neuralconstitutive.custom_types import FileName
from neuralconstitutive.indentation import Indentation


def to_jax_numpy(series: pd.Series) -> Array:
    return jnp.asarray(series.to_numpy())


def import_data(
    rawdata_file: FileName, metadata_file: FileName
) -> tuple[tuple[Indentation, Indentation], tuple[Array, Array]]:
    # Read csv files
    df_raw = pd.read_csv(rawdata_file, sep="\t", skiprows=34)
    df_meta = pd.read_csv(metadata_file, sep="\t")

    # Extract relevant raw data
    force = to_jax_numpy(df_raw["force"])
    time = to_jax_numpy(df_raw["time"])
    tip_position = -to_jax_numpy(df_raw["tip position"])
    contact_point = -to_jax_numpy(df_meta["Contact Point [nm]"]) * 1e-9

    # Retain only the indenting part of the data
    depth = tip_position - contact_point
    in_contact = depth >= 0
    force, time, depth = force[in_contact], time[in_contact], depth[in_contact]
    force = force - force[0]
    time = time - time[0]

    # Split into approach and retract
    idx_max = jnp.argmax(depth)
    slice_app, slice_ret = slice(None, idx_max + 1), slice(idx_max, None)
    approach = ForceIndentDataSegment(
        time[slice_app], depth[slice_app], force[slice_app]
    )
    retract = ForceIndentDataSegment(
        time[slice_ret], depth[slice_ret], force[slice_ret]
    )

    return ForceIndentDataset(approach, retract)


class ForceIndentDataSegment(eqx.Module):
    time: Float[Array, " N"] = eqx.field(converter=jnp.asarray)
    depth: Float[Array, " N"] = eqx.field(converter=jnp.asarray)
    force: Float[Array, " N"] = eqx.field(converter=jnp.asarray)


class ForceIndentDataset(eqx.Module):
    approach: ForceIndentDataSegment
    retract: ForceIndentDataSegment | None = None

    def __iter__(self):
        if self.retract is None:
            return iter((self.approach,))
        else:
            return iter((self.approach, self.retract))

    @property
    def t_app(self):
        return self.approach.time

    @property
    def t_ret(self):
        return self.retract.time

    @property
    def f_app(self):
        return self.approach.force

    @property
    def f_ret(self):
        return self.retract.force


def truncate_adhesion(dataset: ForceIndentDataset):
    ret = dataset.retract

    if isinstance(ret, ForceIndentDataSegment):
        f_ret = jnp.trim_zeros(jnp.clip(ret.force, 0.0), "b")
        retract_new = jax.tree.map(lambda leaf: leaf[: len(f_ret)], ret)
        dataset_new = ForceIndentDataset(dataset.approach, retract_new)
    else:
        dataset_new = dataset

    return dataset_new


def normalize_dataset(dataset: ForceIndentDataset):

    app, ret = dataset.approach, dataset.retract
    scale = jax.tree.map(lambda leaf: leaf[-1], app)

    app_new = jax.tree.map(lambda leaf, scale: leaf / scale, app, scale)
    ret_new = jax.tree.map(lambda leaf, scale: leaf / scale, ret, scale)
    return ForceIndentDataset(app_new, ret_new), scale
