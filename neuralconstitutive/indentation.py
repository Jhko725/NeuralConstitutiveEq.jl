from jaxtyping import Array, Float
import equinox as eqx
import diffrax


class Indentation(eqx.Module):
    time: Float[Array, " N"]
    depth: Float[Array, " N"]

    # Might want to check monotonicity of time here
    # Also might want to check monotonicity of depth, and assign whether approach or retract
    def __len__(self) -> int:
        return len(self.time)


def interpolate_indentation(indentation: Indentation):
    return diffrax.LinearInterpolation(indentation.time, indentation.depth)
