# ruff: noqa: F722
from typing import Sequence

from jaxtyping import Array, ArrayLike, Float, Real
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from neuralconstitutive.constitutive import AbstractConstitutive

matplotlib.rc("axes", grid=True)
matplotlib.rc("axes.spines", right=False, top=False)
matplotlib.rc("grid", color="lightgray", linestyle="--")
matplotlib.rc("lines", markersize=1.0)
matplotlib.rc("figure.constrained_layout", use=True)

MM = 0.0393701  # mm to inches conversion factor
CM = 10 * MM  # cm to inches conversion factor


def connected_scatter(
    ax: Axes,
    x: Real[ArrayLike, " N"],
    y: Real[ArrayLike, " N"],
    y_err: Real[ArrayLike, " N"] | None = None,
    *,
    color: str,
    label: str,
    marker: str = "o",
    markersize: float = 4.0,
    linewidth: float = 1.0,
    linealpha: float = 0.8,
    facecolor: str = "w",
    facealpha: float = 0.5,
    **plot_kwargs,
):
    """Draw a scatterplot connected by line.

    Basically a convenience wrapper around ax.plot(), but with more sensible function arguments.
    """
    plot_kwargs.update(
        dict(
            marker=marker,
            color=color,
            label=label,
            linewidth=linewidth,
            markersize=markersize,
        )
    )
    plot_kwargs["markeredgecolor"] = to_rgba(color, linealpha)
    plot_kwargs["markerfacecolor"] = to_rgba(facecolor, facealpha)

    if y_err is None:
        ax.plot(x, y, **plot_kwargs)
    else:
        ax.errorbar(x, y, y_err, **plot_kwargs)
    return ax


def ordered_legend(ax: Axes, order: Sequence[int] | None = None) -> Axes:
    """Plots the legend on the pass axis but with the items ordered according to `order`.

    If `order=None`, behaves identically to ax.legend().
    """
    handles_orig, labels_orig = ax.get_legend_handles_labels()
    if order is None:
        handles, labels = handles_orig, labels_orig
    else:
        if len(handles_orig) != len(order):
            raise ValueError(
                "Length of order must be identical to number of items in the legend!"
            )

        handles = [handles_orig[idx] for idx in order]
        labels = [labels_orig[idx] for idx in order]

    ax.legend(handles, labels)
    return ax


def plot_relaxation_fn(
    ax: Axes,
    constitutive: AbstractConstitutive,
    time: Float[Array, " time"],
    **plot_kwargs,
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
