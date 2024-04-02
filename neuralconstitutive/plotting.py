# ruff: noqa: F722
from jaxtyping import Array, Float
from matplotlib.axes import Axes

from neuralconstitutive.constitutive import AbstractConstitutiveEqn
from neuralconstitutive.indentation import Indentation


def plot_relaxation_fn(
    ax: Axes,
    constitutive: AbstractConstitutiveEqn,
    time: Float[Array, " time"],
    **plot_kwargs
) -> Axes:
    g = constitutive.relaxation_function(time)
    ax.plot(time, g, **plot_kwargs)
    return ax


def plot_indentation(ax: Axes, indentation: Indentation, **plot_kwargs) -> Axes:
    ax.plot(indentation.time, indentation.depth, **plot_kwargs)
    return ax
