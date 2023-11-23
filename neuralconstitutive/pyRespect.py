from typing import Literal, TypeAlias
import logging

import numpy as np

from ._pyRespect.common import get_kernel_matrix

SpectrumWindowMode: TypeAlias = Literal["lenient", "normal", "strict"]


def estimate_continous_spectrum(
    t: np.ndarray,
    G: np.ndarray,
    weights: np.ndarray | None,
    n_spectrum_points: int = 100,
    spectrum_window_mode: SpectrumWindowMode = "lenient",
    fit_plateau: bool = False,
    verbose: bool = True,
):
    ## Initialize logger
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    ## Validate / Sanitize input data
    # TODO: Implement data sanitization step in the original pyRespect.common.GetExpData function
    # TODO: Also throw a warning if the data does need to be sanitized
    if weights is None:
        weights = np.ones_like(t)

    ## Preparation for the main algorithm
    logger.info("(*) Initial Setup...")

    s_minmax = get_spectrum_window(t, spectrum_window_mode)
    s = np.geomspace(*s_minmax, len(t))
    kernel = get_kernel_matrix(s, t)


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
