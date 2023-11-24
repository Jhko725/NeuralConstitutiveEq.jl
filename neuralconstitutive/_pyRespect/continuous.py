#
# 7/2023: allowing an optional weight column in the input data file
#         improving encapsulation of functions

# Help to find continuous spectrum
# March 2019 major update:
# (i)   added plateau modulus G0 (also in pyReSpect-time) calculation
# (ii)  following Hansen Bayesian interpretation of Tikhonov to extract p(lambda)
# (iii) simplifying lcurve (starting from high lambda to low)
# (iv)  changing definition of rho2 and eta2 (no longer dividing by 1/n and 1/nl)

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from .common import kernel_prestore

# HELPER FUNCTIONS


def initialize_H(
    G, weights, s, kernel, fit_plateau: bool = False
) -> tuple[np.ndarray, float | None]:
    """
    Function: InitializeH(input)

    Input:     G       = n*1 vector [Gt],
               weights = n*1 weight vector,
               s       = relaxation modes,
               kernMat = matrix for faster kernel evaluation
               G0      = optional; if plateau is nonzero

     Output:   H = guessed H
              G0 = optional guess if *argv is nonempty
    """
    #
    # To guess spectrum, pick a negative Hgs and a large value of lambda to get a
    # solution that is most determined by the regularization
    # March 2019; a single guess is good enough now, because going from large lambda to small
    #             lambda in lcurve.

    H = -5.0 * np.ones_like(s) + np.sin(np.pi * s)
    lambda_ = 1.0

    G0 = np.amin(G) if fit_plateau else None
    H_lam, G0 = get_H(lambda_, G, weights, H, kernel, G0)
    return H_lam, G0


def get_H(
    lambda_: float,
    G: np.ndarray,
    weights: np.ndarray,
    H: np.ndarray,
    kernel: np.ndarray,
    G0: float | None = None,
) -> tuple[np.ndarray, float | None]:
    """Purpose: Given a lambda, this function finds the H_lambda(s) that minimizes V(lambda)

             V(lambda) := ||(Gexp - kernel(H)) * (wexp/Gexp)||^2 +  lambda * ||L H||^2

    Input  : lambda  = regularization parameter,
             G    = experimental data,
             weights    = weighting factors,
             H       = guessed H,
             kernel = matrix for faster kernel evaluation
             G0      = optional plateau modulus value

    Output : H_lam, G0
             Default uses Trust-Region Method with Jacobian supplied by jacobianLM
    """

    # send Hplus = [H, G0], on return unpack H and G0
    H_ = H if G0 is None else np.append(H, G0)
    res_lsq = least_squares(
        residualLM, H_, jac=jacobianLM, args=(lambda_, G, weights, kernel)
    )
    result_ = res_lsq.x
    return (result_, None) if G0 is None else (result_[:-1], result_[-1])


def make_A_matrix(N: int) -> np.ndarray:
    """Generate symmetric matrix A = L' * L required for error analysis:
    helper function for lcurve in error determination"""
    # L is a N*N tridiagonal matrix with 1 -2 and 1 on its diagonal;
    nl = N - 2
    L = (
        np.diag(np.ones(N - 1), 1)
        + np.diag(np.ones(N - 1), -1)
        + np.diag(-2.0 * np.ones(N))
    )
    L = L[1 : nl + 1, :]

    return np.dot(L.T, L)


def make_B_matrix(
    H: np.ndarray,
    kernel: np.ndarray,
    G: np.ndarray,
    weights: np.ndarray,
    G0: float | None,
) -> np.ndarray:
    """Get the Bmatrix required for error analysis; helper for lcurve()
    not explicitly accounting for G0 in Jr because otherwise I get underflow problems"""
    N, Ns = kernel.shape
    K = np.dot((weights / G).reshape(N, 1), np.ones((1, Ns)))
    Jr = -kernelD(H, kernel) * K
    r = weights * (1 - kernel_prestore(H, kernel, G0) / G)
    B = np.dot(Jr.T, Jr) + np.diag(np.dot(r.T, Jr))
    return B


def build_L_curve(
    G: np.ndarray,
    weights: np.ndarray,
    H_guess: np.ndarray,
    kernel: np.ndarray,
    G0: float | None = None,
    range_lambda: tuple[float, float] = (1e-10, 1e3),
    lambdas_per_decade: int = 2,
    smoothness: float = 0.0,
):
    """
     Function: lcurve(input)

     Input: Gexp    = n*1 vector [Gt],
             wexp    = weights associated with datapoints
            Hgs     = guessed H,
            kernMat = matrix for faster kernel evaluation
            range_lambda = tuple corresponding to the min and max of the lambda parameter
                           originally corresponds to lam_min, lam_max in the inp file
            G0      = optionally

     Output: lamC and 3 vectors of size n_lambda*1 contains a range of lambda, rho
             and eta. "Elbow"  = lamC is estimated using a *NEW* heuristic AND by Hansen method


    March 2019: starting from large lambda to small cuts calculation time by a lot
                also gives an error estimate

    """
    l_min, l_max = range_lambda
    n_lambda = int(lambdas_per_decade * (np.log10(l_max) - np.log10(l_min)))
    lambdas = np.geomspace(l_min, l_max, n_lambda)

    eta, rho, logP = (
        np.zeros_like(lambdas),
        np.zeros_like(lambdas),
        np.zeros_like(lambdas),
    )

    H = H_guess.copy()
    ns = len(H)

    logPmax = -np.inf  # so nothing surprises me!
    Hlambda = np.zeros((ns, n_lambda))

    # Error Analysis: Furnish A_matrix
    A = make_A_matrix(ns)
    _, LogDetN = np.linalg.slogdet(A)

    #
    # This is the costliest step
    #
    for i, lambda_ in reversed(enumerate(lambdas)):
        H, G0 = get_H(lambda_, G, weights, H, kernel, G0)
        rho[i] = np.linalg.norm(weights * (1 - kernel_prestore(H, kernel, G0) / G))
        B = make_B_matrix(H, kernel, G, weights, G0)

        eta[i] = np.linalg.norm(np.diff(H, n=2))
        Hlambda[:, i] = H

        _, LogDetC = np.linalg.slogdet(lambda_ * A + B)
        V = rho[i] ** 2 + lambda_ * eta[i] ** 2

        # this assumes a prior exp(-lam)
        logP[i] = -V + 0.5 * (LogDetN + ns * np.log(lambda_) - LogDetC) - lambda_

        # Store needed parameters
        if logP[i] > logPmax:
            logPmax = logP[i]
        elif logP[i] < logPmax - 18:
            break

    # truncate all to significant lambda
    lambdas = lambdas[i:]
    logP = logP[i:]
    eta = eta[i:]
    rho = rho[i:]
    logP = logP - max(logP)

    Hlambda = Hlambda[:, i:]

    #
    # currently using both schemes to get optimal lamC
    # new lamM works better with actual experimental data
    #
    # lamC = oldLamC(par, lam, rho, eta)
    plam = np.exp(logP)
    plam = plam / np.sum(plam)
    lamM = np.exp(np.sum(plam * np.log(lambda_)))

    #
    # Dialling in the Smoothness Factor
    #
    if smoothness == 0:
        lamM = lamM
    elif smoothness > 0:
        lamM = np.exp(np.log(lamM) + smoothness * (max(np.log(lambda_)) - np.log(lamM)))
    else:
        lamM = np.exp(np.log(lamM) + smoothness * (np.log(lamM) - min(np.log(lambda_))))

    return lamM, lambda_, rho, eta, logP, Hlambda


def residualLM(H, lam, Gexp, wexp, kernMat):
    """
    %
    % HELPER FUNCTION: Gets Residuals r
     Input  : H       = guessed H,
              lambda  = regularization parameter ,
              Gexp    = experimental data,
              wexp    = weighting factors,
              kernMat = matrix for faster kernel evaluation
                G0      = plateau

     Output : a set of n+nl residuals,
              the first n correspond to the kernel
              the last  nl correspond to the smoothness criterion
    %"""

    n = kernMat.shape[0]
    ns = kernMat.shape[1]
    nl = ns - 2

    r = np.zeros(n + nl)

    # if plateau then unfurl G0
    if len(H) > ns:
        G0 = H[-1]
        H = H[:-1]
        # r[0:n] = (1. - kernel_prestore(H, kernMat, G0)/Gexp)  # the Gt and
        r[0:n] = wexp * (1.0 - kernel_prestore(H, kernMat, G0) / Gexp)  # the Gt and
    else:
        # r[0:n] = (1. - kernel_prestore(H, kernMat)/Gexp)
        r[0:n] = wexp * (1.0 - kernel_prestore(H, kernMat) / Gexp)

    # the curvature constraint is not affected by G0
    r[n : n + nl] = np.sqrt(lam) * np.diff(H, n=2)  # second derivative

    return r


def jacobianLM(H, lam, Gexp, wexp, kernMat):
    """
    HELPER FUNCTION for optimization: Get Jacobian J

    returns a (n+nl * ns) matrix Jr; (ns + 1) if G0 is also supplied.

    Jr_(i, j) = dr_i/dH_j

    It uses kernelD, which approximates dK_i/dH_j, where K is the kernel

    """
    n = kernMat.shape[0]
    ns = kernMat.shape[1]
    nl = ns - 2

    # L is a ns*ns tridiagonal matrix with 1 -2 and 1 on its diagonal;
    L = (
        np.diag(np.ones(ns - 1), 1)
        + np.diag(np.ones(ns - 1), -1)
        + np.diag(-2.0 * np.ones(ns))
    )
    L = L[1 : nl + 1, :]

    # Furnish the Jacobian Jr (n+ns)*ns matrix
    # Kmatrix         = np.dot((1./Gexp).reshape(n,1), np.ones((1,ns)));
    Kmatrix = np.dot((wexp / Gexp).reshape(n, 1), np.ones((1, ns)))

    if len(H) > ns:
        G0 = H[-1]
        H = H[:-1]

        Jr = np.zeros((n + nl, ns + 1))

        Jr[0:n, 0:ns] = -kernelD(H, kernMat) * Kmatrix
        # Jr[0:n, ns]     = -1./Gexp							# column for dr_i/dG0
        Jr[0:n, ns] = -wexp / Gexp  # column for dr_i/dG0

        Jr[n : n + nl, 0:ns] = np.sqrt(lam) * L
        Jr[n : n + nl, ns] = np.zeros(nl)  # column for dr_i/dG0 = 0

    else:
        Jr = np.zeros((n + nl, ns))

        Jr[0:n, 0:ns] = -kernelD(H, kernMat) * Kmatrix
        Jr[n : n + nl, 0:ns] = np.sqrt(lam) * L

    return Jr


def kernelD(H: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Function: kernelD(input)

    outputs the (n*ns) dimensional matrix DK(H)(t)
    It approximates dK_i/dH_j = K * e(H_j):

    Input: H       = substituted CRS,
           kernMat = matrix for faster kernel evaluation

    Output: DK = Jacobian of H
    """

    N, Ns = kernel.shape

    # A n*ns matrix with all the rows = H'
    Hsuper = np.dot(np.ones((N, 1)), np.exp(H).reshape(1, Ns))
    DK = kernel * Hsuper

    return DK
