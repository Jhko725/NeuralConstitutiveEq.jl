# ruff: noqa: F722
from pathlib import Path
from jaxtyping import Array, Float
from matplotlib.axes import Axes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from neuralconstitutive.constitutive import AbstractConstitutiveEqn
from neuralconstitutive.indentation import Indentation
from neuralconstitutive.io import ForceIndentDataset

matplotlib.rc("axes", grid=True)
matplotlib.rc("axes.spines", right=False, top=False)
matplotlib.rc("grid", color="lightgray", linestyle="--")
matplotlib.rc("lines", markersize=1.0)
matplotlib.rc("figure.constrained_layout", use=True)


def plot_relaxation_fn(
    ax: Axes,
    constitutive: AbstractConstitutiveEqn,
    time: Float[Array, " time"],
    **plot_kwargs
) -> Axes:
    g = constitutive.relaxation_function(time)
    ax.plot(time, g, **plot_kwargs)
    return ax


def align_zeros(axes):

    ylims_current = {}  #  Current ylims
    ylims_mod = {}  #  Modified ylims
    deltas = {}  #  ymax - ymin for ylims_current
    ratios = {}  #  ratio of the zero point within deltas

    for ax in axes:
        ylims_current[ax] = list(ax.get_ylim())
        # Need to convert a tuple to a list to manipulate elements.
        deltas[ax] = ylims_current[ax][1] - ylims_current[ax][0]
        ratios[ax] = -ylims_current[ax][0] / deltas[ax]

    for ax in axes:  # Loop through all axes to ensure each ax fits in others.
        ylims_mod[ax] = [np.nan, np.nan]  # Construct a blank list
        ylims_mod[ax][1] = max(deltas[ax] * (1 - np.array(list(ratios.values()))))
        # Choose the max value among (delta for ax)*(1-ratios),
        # and apply it to ymax for ax
        ylims_mod[ax][0] = min(-deltas[ax] * np.array(list(ratios.values())))
        # Do the same for ymin
        ax.set_ylim(tuple(ylims_mod[ax]))


def plot_forceindent(dataset, figsize=(8, 2.5), **plot_kwargs):
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True, width_ratios=(2, 1)
    )
    ax = axes[0].twinx()
    for data in dataset:
        ax.plot(data.time, data.depth, "--", linewidth=0.8, **plot_kwargs)
        axes[0].plot(data.time, data.force, **plot_kwargs)
        axes[1].plot(data.depth, data.force, **plot_kwargs)

    align_zeros([axes[0], ax])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Depth [m]")
    ax.spines["right"].set_visible(True)

    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Force [N]")

    axes[1].set_xlabel("Depth [m]")
    axes[1].set_ylabel("Force [N]")
    return fig
