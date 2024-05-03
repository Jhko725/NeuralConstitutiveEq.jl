from typing import TypeVar, Any

from jaxtyping import Float, Integer, Bool, Array, PyTree

FloatScalar = Float[Array, ""]
IntScalar = Integer[Array, ""]
BoolScalar = Bool[Array, ""]

X = TypeVar("X", bound=FloatScalar)
Y = TypeVar("Y", bound=PyTree)
Args = Any
SolverState = TypeVar("SolverState")