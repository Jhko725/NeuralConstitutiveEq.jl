# ruff: noqa: F722
import abc
import dataclasses
from typing import ClassVar, Literal, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from neuralconstitutive.custom_types import (
    FloatScalar,
    FloatScalarOr1D,
    floatscalar_field,
)
from neuralconstitutive.utils.smoothing import (
    make_smoothed_cubic_spline,
    PiecewiseCubic,
)


class INDENT_TYPE(eqx.Enumeration):
    """An enum representing the type of an indentation."""

    app = "approach"
    hold = "hold"
    ret = "retract"


class AbstractIndentationSegment(eqx.Module):
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
        where indentation is an instance of a concrete class of AbstractIndentationSegment

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
        where indentation is an instance of a concrete class of AbstractIndentationSegment

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


class ConstantVelocity(AbstractIndentationSegment):
    velocity_: FloatScalar = floatscalar_field()
    depth_offset: FloatScalar = floatscalar_field(default=0.0)
    indent_type: Literal[INDENT_TYPE.app, INDENT_TYPE.ret] = eqx.field(init=False)

    def __post_init__(self):
        self.indent_type = INDENT_TYPE.app if self.velocity_ >= 0 else INDENT_TYPE.ret

    def __check_init__(self):
        if self.velocity_ == 0.0:
            raise ValueError("For zero velocity, use indentation.Constant instead.")

    def depth(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.depth_offset + self.velocity_ * time

    def velocity(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.velocity_ * jnp.ones_like(time)


class Constant(AbstractIndentationSegment):
    depth_offset: FloatScalar = floatscalar_field(default=0.0)
    indent_type: ClassVar[INDENT_TYPE] = INDENT_TYPE.hold

    def depth(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.depth_offset * jnp.ones_like(time)

    def velocity(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        return jnp.zeros_like(time)


class CubicSpline(AbstractIndentationSegment):
    depth_offset: FloatScalar
    spline: PiecewiseCubic
    indent_type: INDENT_TYPE

    def __init__(
        self,
        time_data,
        depth_data,
        smoothing: float = 1.5e-4,
        depth_offset: float = 0.0,
    ):
        self.depth_offset = jnp.asarray(depth_offset)
        self.spline = make_smoothed_cubic_spline(time_data, depth_data, s=smoothing)
        self.indent_type = self._infer_indent_type()

    def _infer_indent_type(self) -> INDENT_TYPE:
        depth_start = self.depth(self.spline.t0)
        depth_end = self.depth(self.spline.t1)

        if depth_end > depth_start:
            indent_type = INDENT_TYPE.app
        elif depth_end < depth_start:
            indent_type = INDENT_TYPE.ret
        else:
            indent_type = INDENT_TYPE.hold
        return indent_type

    def depth(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.spline.evaluate(time)

    def velocity(self, time: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.spline.derivative(time)


class AbstractIndentation(eqx.Module):
    t_hold: eqx.AbstractVar[float]
    t_ret: eqx.AbstractVar[float]

    @abc.abstractmethod
    def h_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        """Depth of the indenter as a function of time during approach."""
        pass

    @abc.abstractmethod
    def v_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        """Velocity of the indenter as a function of time during approach."""
        pass

    @abc.abstractmethod
    def depth(self, t: FloatScalar) -> FloatScalar:
        """Depth of the indenter $h(t)$ as a function of time $t$.

        This function is useful when trying to calculate the depth of the indenter during the entire experiment,
        and not just during a particular indentation segment."""

    @abc.abstractmethod
    def velocity(self, t: FloatScalar) -> FloatScalar:
        """Depth of the indenter $v(t)$ as a function of time $t$.

        This function is useful when trying to calculate the velocity of the indenter during the entire experiment,
        and not just during a particular indentation segment."""


class Approach(AbstractIndentation):
    approach: AbstractIndentationSegment
    t_hold: ClassVar[float] = jnp.inf
    t_ret: ClassVar[float] = jnp.inf

    def __check_init__(self):
        if self.approach.indent_type != INDENT_TYPE.app:
            raise ValueError("approach does not correspond to approaching motion.")

    def h_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.approach.depth(t)

    def v_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.approach.velocity(t)

    def depth(self, time: FloatScalar) -> FloatScalar:
        return self.h_app(time)

    def velocity(self, time: FloatScalar) -> FloatScalar:
        return self.v_app(time)


class ApproachHold(AbstractIndentation):
    approach: AbstractIndentationSegment
    hold: Constant
    t_hold: float
    t_ret: ClassVar[float] = jnp.inf

    def __check_init__(self):
        if self.approach.indent_type != INDENT_TYPE.app:
            raise ValueError("approach does not correspond to approaching motion.")

        if self.t_hold <= 0:
            raise ValueError(
                "t_hold must be larger than zero - i.e., hold segment must start after approach"
            )

    def h_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.approach.depth(t)

    def v_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.approach.velocity(t)

    def h_hold(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.hold.depth(t)

    def v_hold(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.hold.velocity(t)

    def depth(self, time: FloatScalar) -> FloatScalar:
        return jax.lax.cond(time <= self.t_hold, self.h_app, self.h_hold, time)

    def velocity(self, time: FloatScalar) -> FloatScalar:
        return jax.lax.cond(time <= self.t_hold, self.v_app, self.v_hold, time)


class ApproachRetract(AbstractIndentation):
    approach: AbstractIndentationSegment
    retract: AbstractIndentationSegment
    t_hold: float = eqx.field(init=False)
    t_ret: float

    def __post_init__(self):
        self.t_hold = self.t_ret

    def __check_init__(self):
        if self.approach.indent_type != INDENT_TYPE.app:
            raise ValueError("self.approach does not correspond to approaching motion.")

        if self.retract.indent_type != INDENT_TYPE.ret:
            raise ValueError("self.retract does not correspond to retracting motion.")

        if self.t_ret <= 0:
            raise ValueError(
                "t_ret must be larger than zero - i.e., retract segment must start after approach"
            )

    def h_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.approach.depth(t)

    def v_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.approach.velocity(t)

    def h_ret(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.retract.depth(t)

    def v_ret(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.retract.velocity(t)

    def depth(self, time: FloatScalar) -> FloatScalar:
        return jnp.select(
            [time < self.t_ret],
            [self.h_app(time)],
            default=self.h_ret(time),
        )

    def velocity(self, time: FloatScalar) -> FloatScalar:
        return jnp.select(
            [time < self.t_ret],
            [self.v_app(time)],
            default=self.v_ret(time),
        )


class ApproachHoldRetract(AbstractIndentation):
    approach: AbstractIndentationSegment
    hold: Constant
    retract: AbstractIndentationSegment
    t_hold: float
    t_ret: float

    def __check_init__(self):
        if self.approach.indent_type != INDENT_TYPE.app:
            raise ValueError("self.approach does not correspond to approaching motion.")

        if self.retract.indent_type != INDENT_TYPE.ret:
            raise ValueError("self.retract does not correspond to retracting motion.")

        if self.t_hold <= 0:
            raise ValueError(
                "t_hold must be larger than zero - i.e., hold segment must start after approach"
            )

        if self.t_ret <= self.t_hold:
            raise ValueError(
                "t_ret must be larger than t_hold - i.e., retract segment must start after hold"
            )

    def h_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.approach.depth(t)

    def v_app(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.approach.velocity(t)

    def h_hold(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.hold.depth(t)

    def v_hold(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.hold.velocity(t)

    def h_ret(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.retract.depth(t)

    def v_ret(self, t: FloatScalarOr1D) -> FloatScalarOr1D:
        return self.retract.velocity(t)

    def depth(self, time: FloatScalar) -> FloatScalar:
        return jnp.select(
            [time < self.t_hold, time < self.t_ret],
            [self.h_app(time), self.h_hold(time)],
            default=self.h_ret(time),
        )

    def velocity(self, time: FloatScalar) -> FloatScalar:
        return jnp.select(
            [time < self.t_hold, time < self.t_ret],
            [self.v_app(time), self.v_hold(time)],
            default=self.v_ret(time),
        )


class IndentationBuilder:
    """A class to create complicated indentation sequences using the builder pattern.

    This class is a regular class and not an equinox.Module as its state is intended to be mutable.
    """

    indentations: list[AbstractIndentationSegment]
    breakpoints: list[float]
    enforce_continuity: bool

    def __init__(self, enforce_continuity: bool = True):
        self.indentations = []
        self.breakpoints = [0.0]
        self.enforce_continuity = enforce_continuity

    def append(self, indentation: AbstractIndentationSegment, duration: float) -> Self:
        if self.enforce_continuity:
            indentation = self._modify_depth_offset(indentation)
        self.indentations.append(indentation)
        self.breakpoints.append(self.breakpoints[-1] + duration)
        return self

    def _modify_depth_offset(self, indentation: AbstractIndentationSegment):
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

    def build(self) -> AbstractIndentation:
        match self.indentations, self.breakpoints:
            case [[app], [0.0, *_]]:
                return Approach(app)
            case [[app, Constant() as hold], [0.0, t_hold, *_]]:
                return ApproachHold(app, hold, t_hold)
            case [[app, ret], [0.0, t_ret, *_]]:
                return ApproachRetract(app, ret, t_ret)
            case [[app, hold, ret], [0.0, t_hold, t_ret, *_]]:
                return ApproachHoldRetract(app, hold, ret, t_hold, t_ret)
            case _:
                raise ValueError("Invalid combination of IndentationSegments")
