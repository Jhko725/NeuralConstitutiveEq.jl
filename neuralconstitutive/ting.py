from typing import Callable
import functools

import numpy as np
from numpy import ndarray
from scipy.integrate import quad
from scipy.optimize import root_scalar

from .tipgeometry import TipGeometry


def force_approach(
    t: ndarray,
    constit: Callable[[ndarray], ndarray],
    indent_app: Callable[[ndarray], ndarray],
    velocity_app: Callable[[ndarray], ndarray],
    tip: TipGeometry,
    **quad_kwargs,
) -> ndarray:
    dF = make_force_integand(constit, velocity_app, indent_app, tip)
    F = np.stack([quad(dF, 0, t_i, args=(t_i,), **quad_kwargs)[0] for t_i in t], axis=0)
    return F


def force_retract(
    t: ndarray,
    constit: Callable[[ndarray], ndarray],
    indent_app: Callable[[ndarray], ndarray],
    velocity_app: Callable[[ndarray], ndarray],
    velocity_ret: Callable[[ndarray], ndarray],
    tip: TipGeometry,
    **quad_kwargs,
) -> ndarray:
    calc_t1 = functools.partial(
        calculate_t1,
        t_max=t[0],
        constit=constit,
        vel_app=velocity_app,
        vel_ret=velocity_ret,
        **quad_kwargs,
    )
    dF = make_force_integand(constit, velocity_app, indent_app, tip)
    F = np.stack(
        [quad(dF, 0, calc_t1(t_i), args=(t_i,), **quad_kwargs)[0] for t_i in t], axis=0
    )
    return F


def make_force_integand(
    constit: Callable[[ndarray], ndarray],
    velocity: Callable[[ndarray], ndarray],
    indentation: Callable[[ndarray], ndarray],
    tip: TipGeometry,
) -> Callable[[ndarray, float], ndarray]:
    a = tip.alpha
    b = tip.beta

    def _force_integrand(t_: ndarray, t: float) -> ndarray:
        return a * b * constit(t - t_) * velocity(t_) * indentation(t_) ** (b - 1)

    return _force_integrand


def calculate_t1(
    t: float,
    t_max: float,
    constit: Callable[[ndarray], ndarray],
    vel_app: Callable[[ndarray], ndarray],
    vel_ret: Callable[[ndarray], ndarray],
    **quad_kwargs,
) -> float:
    def _t1_objective(t1: float) -> float:
        integrand_app = quad(
            lambda t_: constit(t - t_) * vel_app(t_), t1, t_max, **quad_kwargs
        )
        integrand_ret = quad(
            lambda t_: constit(t - t_) * vel_ret(t_), t_max, t, **quad_kwargs
        )
        return integrand_app[0] + integrand_ret[0]

    try:
        sol = root_scalar(_t1_objective, method="bisect", bracket=(0, t_max))
        return sol.root
    except ValueError:
        return 0.0
