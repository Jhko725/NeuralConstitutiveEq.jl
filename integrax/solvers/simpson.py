from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import PyTree
from lineax.internal import rms_norm

from integrax.custom_types import BoolScalar, FloatScalar, IntScalar, X, Y, Args
from integrax.solvers.base import AbstractIntegration, reached_tolerance
from integrax.solvers.trapezoid import init_trapezoid, refine_trapezoid, refine_trapezoid_batch

class SimpsonState(eqx.Module):
    S_trapz: FloatScalar
    num_refine: IntScalar
    terminate: BoolScalar

class AdaptiveSimpson(AbstractIntegration):
    rtol: float
    atol: float
    norm: Callable[[PyTree], FloatScalar] = rms_norm
    min_refines: int = 5

    def init(self, fn: Callable[[X, Args], Y], lower: X, upper: Y, args: Args):
        S_trapz = init_trapezoid(fn, lower, upper, args)
        state_init = SimpsonState(S_trapz, jnp.array(1), jnp.array(False))
        S_init = ((4/3)*S_trapz**ω).ω
        return S_init, jnp.asarray(1), state_init

    def step(self, S_prev, num_steps, fn, lower, upper, args, state_prev: SimpsonState):
        S_trapz_prev = state_prev.S_trapz
        num_refine = state_prev.num_refine*2
        S_trapz = refine_trapezoid_batch(S_trapz_prev, num_refine, fn, lower, upper, args)

        S = ((4*S_trapz**ω - S_trapz_prev ** ω)/3).ω

        num_steps = num_steps+1
        terminate = (num_steps > self.min_refines) & reached_tolerance(
            S, S_prev, self.rtol, self.atol, self.norm
        )
        state = SimpsonState(S_trapz, num_refine, terminate)
        return S, num_steps, state
    
    def terminate(
        self, integral, num_steps, fn, lower, upper, args, state: SimpsonState
    ):
        return state.terminate, None