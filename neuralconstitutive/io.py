import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array

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
    approach = Indentation(time[: idx_max + 1], depth[: idx_max + 1])
    retract = Indentation(time[idx_max:], depth[idx_max:])
    force_app, force_ret = force[: idx_max + 1], force[idx_max:]

    # Clip the negative adhesive force
    force_ret = jnp.clip(force_ret, 0.0)
    return (approach, retract), (force_app, force_ret)
