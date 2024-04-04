from typing import Callable

import jax
from jax import Array
import jax.numpy as jnp

from neuralconstitutive.trajectory import Trajectory
from neuralconstitutive.jax.tipgeometry import AbstractTipGeometry
from neuralconstitutive.jax.ting import force_approach, force_retract


def simulate_data(
    app: Trajectory,
    ret: Trajectory,
    relaxation: Callable[[Array], Array],
    tip: AbstractTipGeometry,
    noise_strength: float,
    random_seed: int,
) -> tuple[Array, Array]:
    key = jax.random.PRNGKey(random_seed)
    f_app = force_approach(app, relaxation, tip)
    f_ret = force_retract(app, ret, relaxation, tip)
    noise_scale = jnp.max(f_app)
    noise_app = jax.random.normal(key, f_app.shape) * noise_strength * noise_scale
    noise_ret = (
        jax.random.normal(jax.random.split(key, num=1), f_ret.shape)
        * noise_strength
        * noise_scale
    )
    return f_app + noise_app, f_ret + noise_ret
