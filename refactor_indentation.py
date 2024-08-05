# %%
# ruff: noqa: F722
import abc
import dataclasses
from typing import Callable, ClassVar, Literal, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from neuralconstitutive.custom_types import (
    FloatScalar,
    FloatScalarOr1D,
    floatscalar_field,
)


class INDENT_TYPE(eqx.Enumeration):
    """An enum representing the type of an indentation."""

    app = "approach"
    hold = "hold"
    ret = "retract"


class AbstractIndentation(eqx.Module):
    """Interface for the motion of the indenter during a particular segment of a force-indentation experiment.

    The kind of motion is one of approach / hold / retract.
    Depending on the type of motion, indentation depth as a function of time must be monotone increasing / constant / monotone decreasing.
    """

    indent_type: eqx.AbstractVar[INDENT_TYPE]
    depth_offset: eqx.AbstractVar[FloatScalar]

    @abc.abstractmethod
    def depth(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        """Indentation depth ($h(t)$) of the indenter as a function of time ($t$).

        Positive indentation corresponds to indenting into the surface and zero indentation corresponds to no indentation.
        Therefore negative depth does not hold much meaning.

        This function can be called via its alias- that is,
            indentation.depth(t) == indentation.h(t)
        where indentation is an instance of a concrete class of AbstractIndentation

        **Arguments**

        - `time`: A 0D (scalar) or 1D jax array of time points at which the indentation depth is to be computed.

        **Returns**
        - `depth`: A jax array containing the corresponding indentation depths. Has the same shape as `time`.
        """
        pass

    @abc.abstractmethod
    def velocity(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        """Indentation velocity ($v(t)$) of the indenter as a function of time ($t$).

        Positive velocity indicates approaching/indenting into and negative velocity retracting from the sample.
        Zero velocity is indentor being held in place.

        This function can be called via its alias- that is,
            indentation.velocity(t) == indentation.v(t)
        where indentation is an instance of a concrete class of AbstractIndentation

        **Arguments**

        - `time`: A 0D (scalar) or 1D jax array of time points at which the indentation depth is to be computed.

        **Returns**
        - `velocity`: A jax array containing the corresponding indentation velocities. Has the same shape as `time`.
        """
        pass

    # Function aliases for user convenience
    def h(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.depth(t)

    def v(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.velocity(t)


class ConstantVelocity(AbstractIndentation):
    velocity_: FloatScalar = floatscalar_field()
    depth_offset: FloatScalar = floatscalar_field(default=0.0)
    indent_type: Literal[INDENT_TYPE.app, INDENT_TYPE.ret] = eqx.field(init=False)

    def __post_init__(self):
        self.indent_type = INDENT_TYPE.app if self.velocity_ >= 0 else INDENT_TYPE.ret

    def __check_init__(self):
        if self.velocity_ == 0.0:
            raise ValueError("For zero velocity, use indentation.Hold instead.")

    def depth(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.depth_offset + self.velocity_ * time

    def velocity(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.velocity_ * jnp.ones_like(time)


class Hold(AbstractIndentation):
    depth_offset: FloatScalar = floatscalar_field(default=0.0)
    indent_type: ClassVar[INDENT_TYPE] = INDENT_TYPE.hold

    def depth(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.depth_offset * jnp.ones_like(time)

    def velocity(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        return jnp.zeros_like(time)


class IndentationSequence(eqx.Module):
    indentations: list[AbstractIndentation]
    breakpoints: Float[Array, " len(indentations)"] = eqx.field(converter=jnp.asarray)
    _depth_funcs: list[Callable[[FloatScalar], FloatScalar]] = eqx.field(init=False)
    _velocity_funcs: list[Callable[[FloatScalar], FloatScalar]] = eqx.field(init=False)

    def __post_init__(self):
        # Use lambda t: ind.depth(t) instead of ind.depth due to https://github.com/patrick-kidger/equinox/issues/709
        # Also define auxillary functions to create lambdas due to https://stackoverflow.com/questions/938429/scope-of-lambda-functions-and-their-parameters/938493#938493
        def _make_depth_func(indentation):
            return lambda t: indentation.depth(t)

        def _make_velocity_func(indentation):
            return lambda t: indentation.velocity(t)

        self._depth_funcs = [_make_depth_func(ind) for ind in self.indentations]
        self._velocity_funcs = [_make_velocity_func(ind) for ind in self.indentations]

    def _find_indentation_index(self, time):
        index = (
            jnp.searchsorted(self.breakpoints, time, side="right", method="compare_all")
            - 1
        )
        return index

    def depth(self, time: FloatScalar) -> FloatScalar:
        print(time)
        index = self._find_indentation_index(time)
        print(index)
        return jax.lax.switch(index, self._depth_funcs, time)

    def velocity(self, time: FloatScalar) -> FloatScalar:
        index = self._find_indentation_index(time)
        return jax.lax.switch(index, self._velocity_funcs, time)


class IndentationSequenceBuilder:
    """A class to create complicated indentatoin sequences using the builder pattern.

    This class is a regular class and not an equinox.Module as its state is intended to be mutable.
    """

    indentations: list[AbstractIndentation]
    breakpoints: list[float]
    enforce_continuity: bool

    def __init__(self, enforce_continuity: bool = True):
        self.indentations = []
        self.breakpoints = [0.0]
        self.enforce_continuity = enforce_continuity

    def append(self, indentation: AbstractIndentation, duration: float) -> Self:
        if self.enforce_continuity:
            indentation = self._modify_depth_offset(indentation)
        self.indentations.append(indentation)
        self.breakpoints.append(self.breakpoints[-1] + duration)
        return self

    def _modify_depth_offset(self, indentation: AbstractIndentation):

        if len(self.indentations) == 0:
            indentation_new = indentation
        else:
            breakpt_prev = self.breakpoints[-1]
            depth_diff = self.indentations[-1].depth(breakpt_prev) - indentation.depth(
                breakpt_prev
            )
            depth_offset_new = depth_diff + indentation.depth_offset
            indentation_new = dataclasses.replace(
                indentation, depth_offset=depth_offset_new
            )
        return indentation_new

    def build(self) -> IndentationSequence:
        return IndentationSequence(self.indentations, self.breakpoints)


# %%
indentation = (
    IndentationSequenceBuilder()
    .append(ConstantVelocity(velocity_=3.0), duration=2.0)
    .append(Hold(depth_offset=6.0), duration=4.0)
    .append(ConstantVelocity(velocity_=-3.0, depth_offset=6.0), duration=2.0)
    .build()
)

# %%
import matplotlib.pyplot as plt

t_array = jnp.arange(0.0, indentation.breakpoints[-1] + 0.1, 0.1)
d_array = jax.vmap(indentation.depth)(t_array)
v_array = jax.vmap(indentation.velocity)(t_array)

fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
axes[0].plot(t_array, d_array)
axes[1].plot(t_array, v_array)
# %%
