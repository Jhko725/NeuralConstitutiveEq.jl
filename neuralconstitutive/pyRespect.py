import logging
from typing import Literal, TypeAlias

import numpy as np
from scipy.interpolate import interp1d

from ._pyRespect.common import get_kernel_matrix
from ._pyRespect.continuous import build_L_curve, get_H, initialize_H

SpectrumWindowMode: TypeAlias = Literal["lenient", "normal", "strict"]


def estimate_continous_spectrum(
    t: np.ndarray,
    G: np.ndarray,
    weights: np.ndarray | None = None,
    n_interpolation: int | None = 100,
    n_spectrum_points: int = 100,
    spectrum_window_mode: SpectrumWindowMode = "lenient",
    fit_plateau: bool = False,
    range_lambda: tuple[float, float] = (1e-10, 1e3),
    lambdas_per_decade: int = 2,
    L_curve_smoothness: float = 0.0,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    ## Initialize logger
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    ## Validate / Sanitize input data
    # TODO: Implement data sanitization step in the original pyRespect.common.GetExpData function
    # TODO: Also throw a warning if the data does need to be sanitized
    if n_interpolation is not None:
        print(f"Interpolating t, G to {n_interpolation} points")
        f = interp1d(t, G, fill_value="extrapolate")
        t = np.geomspace(np.min(t), np.max(t), n_interpolation)
        G = f(t)

    if weights is None:
        weights = np.ones_like(t)

    ## Preparation for the main algorithm
    logger.info("(*) Initial Setup...")

    s_minmax = get_spectrum_window(t, spectrum_window_mode)
    s = np.geomspace(*s_minmax, n_spectrum_points)
    kernel = get_kernel_matrix(s, t)

    ## Get initial guesses for the spectrum H and G0
    Hgs, G0 = initialize_H(G, weights, s, kernel, fit_plateau=fit_plateau)

    ## Find Optimum Lambda with 'L-curve'
    lamC, lam, rho, eta, logP, Hlam = build_L_curve(
        G,
        weights,
        Hgs,
        kernel,
        G0,
        range_lambda=range_lambda,
        lambdas_per_decade=lambdas_per_decade,
        smoothness=L_curve_smoothness,
    )

    ## Get the best spectrum
    H, G0 = get_H(lamC, G, weights, Hgs, kernel, G0)

    return s, np.exp(H), lamC


def get_spectrum_window(
    t: np.ndarray, spectrum_window_mode: SpectrumWindowMode = "lenient"
) -> tuple[float, float]:
    """
    Modes lenient, normal, strict correspoind to values 1, 2, 3 for the parameter
    FreqEnd in the original pyRespect code
    """
    t_min, t_max = t[0], t[-1]
    C = np.exp(np.pi / 2)

    match spectrum_window_mode:
        case "lenient":
            window = (t_min / C, t_max * C)
        case "normal":
            window = (t_min, t_max)
        case "strict":
            window = (t_min * C, t_max / C)

    return window
