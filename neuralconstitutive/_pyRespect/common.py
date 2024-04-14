import numpy as np


def get_kernel_matrix(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Corresponds to the getKernMat() function in pyRespect.common
    furnish kernel matrix for faster kernel evaluation

    given s, t generates hs * exp(-T/S) [n * ns matrix], where hs = wi = weights
    for trapezoidal rule integration.

    This matrix (K) times h = exp(H), Kh, is comparable with Gexp"""
    ns = len(s)
    hsv = np.zeros(ns)
    hsv[0] = 0.5 * np.log(s[1] / s[0])
    hsv[ns - 1] = 0.5 * np.log(s[ns - 1] / s[ns - 2])
    hsv[1 : ns - 1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0 : ns - 2]))
    S, T = np.meshgrid(s, t)

    return np.exp(-T / S) * hsv


def kernel_prestore(H: np.ndarray, kernel: np.ndarray, G0: float | None):
    """
    turbocharging kernel function evaluation by prestoring kernel matrix
        Function: kernel_prestore(input) returns K*h, where h = exp(H)

        Same as kernel, except prestoring hs, S, and T to improve speed 3x.

        outputs the n*1 dimensional vector K(H)(t) which is comparable to Gexp = Gt

        3/11/2019: returning Kh + G0

        Input: H = substituted CRS,
               kernMat = n*ns matrix [w * exp(-T/S)]

    """

    if G0 is None:
        G0 = 0.0

    return np.dot(kernel, np.exp(H)) + G0
