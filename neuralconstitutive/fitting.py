from functools import partial
import dataclasses

import numpy as np
from numpy import ndarray
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from .preprocessing import estimate_derivative
from .ting import force_approach, force_retract


def fit_approach(constit, time, indent, force, tip, **fixed_constit_params):
    t_app, _, indent_app, _, force_app, _ = split_approach_retract(time, indent, force)
    indent_app_ = interp1d(t_app, indent_app)
    vel_app_ = interp1d(t_app, estimate_derivative(t_app, indent_app))
    constit_factory = partial(type(constit), **fixed_constit_params)

    def objective(t_data, *constit_params):
        constit = constit_factory(*constit_params)
        f_app = force_approach(
            t_data, constit, indent_app_, vel_app_, tip, epsabs=1e-4, epsrel=1e-4
        )
        return f_app

    p0 = make_initial_guess(constit, **fixed_constit_params)
    popt, pcov = curve_fit(objective, t_app, force_app, p0=p0)

    return constit_factory(*popt), pcov


def fit_retract(constit, time, indent, force, tip, **fixed_constit_params):
    t_app, t_ret, indent_app, indent_ret, _, force_ret = split_approach_retract(
        time, indent, force
    )
    indent_app_ = interp1d(t_app, indent_app)
    vel_app_ = interp1d(t_app, estimate_derivative(t_app, indent_app))
    vel_ret_ = interp1d(t_ret, estimate_derivative(t_ret, indent_ret))
    constit_factory = partial(type(constit), **fixed_constit_params)

    def objective(t_data, *constit_params):
        constit = constit_factory(*constit_params)
        f_ret = force_retract(
            t_data,
            constit,
            indent_app_,
            vel_app_,
            vel_ret_,
            tip,
            epsabs=1e-4,
            epsrel=1e-4,
        )
        return f_ret

    p0 = make_initial_guess(constit, **fixed_constit_params)
    popt, pcov = curve_fit(objective, t_ret, force_ret, p0=p0)

    return constit_factory(*popt), pcov


def fit_total(constit, time, indent, force, tip, **fixed_constit_params):
    t_app, t_ret, indent_app, indent_ret, _, _ = split_approach_retract(
        time, indent, force
    )
    indent_app_ = interp1d(t_app, indent_app)
    vel_app_ = interp1d(t_app, estimate_derivative(t_app, indent_app))
    vel_ret_ = interp1d(t_ret, estimate_derivative(t_ret, indent_ret))
    constit_factory = partial(type(constit), **fixed_constit_params)

    def objective(t_data, *constit_params):
        constit = constit_factory(*constit_params)
        t_app, t_ret, _, _, _, _ = split_approach_retract(t_data, indent, force)
        f_app = force_approach(
            t_app, constit, indent_app_, vel_app_, tip, epsabs=1e-4, epsrel=1e-4
        )
        f_ret = force_retract(
            t_ret,
            constit,
            indent_app_,
            vel_app_,
            vel_ret_,
            tip,
            epsabs=1e-4,
            epsrel=1e-4,
        )
        return np.concatenate((f_app[:-1], f_ret), axis=0)

    p0 = make_initial_guess(constit, **fixed_constit_params)
    popt, pcov = curve_fit(objective, time, force, p0=p0)

    return constit_factory(*popt), pcov


def split_approach_retract(
    time: ndarray, indentation: ndarray, force: ndarray
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    split_idx = np.argmax(indentation)
    t_app, t_ret = time[: split_idx + 1], time[split_idx:]
    indent_app, indent_ret = indentation[: split_idx + 1], indentation[split_idx:]
    force_app, force_ret = force[: split_idx + 1], force[split_idx:]
    return t_app, t_ret, indent_app, indent_ret, force_app, force_ret


def make_initial_guess(constit, **fixed_constit_params) -> ndarray:
    num_total_params = len(dataclasses.asdict(constit))
    return np.ones(num_total_params - len(fixed_constit_params))
