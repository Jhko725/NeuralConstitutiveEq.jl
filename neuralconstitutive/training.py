from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from neuralconstitutive.jax.ting import force_approach, force_retract
from neuralconstitutive.jax.tipgeometry import AbstractTipGeometry
from neuralconstitutive.trajectory import Trajectory


def train_model(
    model: eqx.Module,
    x: tuple[Trajectory, Trajectory],
    y: tuple[Array, Array],
    args: Any,
    loss_function: Callable,
    optimizer: optax.GradientTransformation,
    max_epochs: int = 1000,
):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss_and_grad_func = eqx.filter_value_and_grad(loss_function)

    @eqx.filter_jit
    def make_epoch(
        model_: eqx.Module,
        opt_state_: optax.OptState,
        x_: tuple[Trajectory, Trajectory],
        y_: tuple[Array, Array],
    ) -> tuple[float, eqx.Module, optax.OptState]:
        loss, grad = loss_and_grad_func(model_, x_, y_, args)
        updates, opt_state_ = optimizer.update(grad, opt_state_)
        model_ = eqx.apply_updates(model_, updates)
        return loss, model_, opt_state_

    loss_history = np.empty(max_epochs)
    try:
        for epoch in range(max_epochs):
            loss, model, opt_state = make_epoch(model, opt_state, x, y)
            loss_val = loss.item()
            loss_history[epoch] = loss_val
            print(f"step={epoch}, loss={loss_val}")
    except KeyboardInterrupt:
        loss_history = loss_history[
            :epoch
        ]  # Truncate the unused part of the history buffer

    return model, loss_history


def l2_loss(x: Array, x_pred: Array) -> float:
    return jnp.mean((x - x_pred) ** 2)


def loss_total(
    model: eqx.Module,
    trajectories: tuple[Trajectory, Trajectory],
    forces: tuple[Array, Array],
    tip: AbstractTipGeometry,
) -> float:
    app, ret = trajectories
    f_app, f_ret = forces

    f_app_pred = force_approach(app, model, tip)
    f_ret_pred = force_retract(app, ret, model, tip)
    return l2_loss(f_app, f_app_pred) + l2_loss(f_ret, f_ret_pred)
