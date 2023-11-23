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
    result = (result_, None) if G0 is None else (result_[:-1], result_[-1])

    return result


def getAmatrix(ns):
    """Generate symmetric matrix A = L' * L required for error analysis:
    helper function for lcurve in error determination"""
    # L is a ns*ns tridiagonal matrix with 1 -2 and 1 on its diagonal;
    nl = ns - 2
    L = (
        np.diag(np.ones(ns - 1), 1)
        + np.diag(np.ones(ns - 1), -1)
        + np.diag(-2.0 * np.ones(ns))
    )
    L = L[1 : nl + 1, :]

    return np.dot(L.T, L)


def getBmatrix(H, kernMat, Gexp, wexp, *argv):
    """get the Bmatrix required for error analysis; helper for lcurve()
    not explicitly accounting for G0 in Jr because otherwise I get underflow problems"""
    n = kernMat.shape[0]
    ns = kernMat.shape[1]
    nl = ns - 2
    r = np.zeros(n)
    # vector of size (n);

    # furnish relevant portion of Jacobian and residual

    # Kmatrix = np.dot((1./Gexp).reshape(n,1), np.ones((1,ns)));
    Kmatrix = np.dot((wexp / Gexp).reshape(n, 1), np.ones((1, ns)))
    Jr = -kernelD(H, kernMat) * Kmatrix

    # if plateau then unfurl G0
    if len(argv) > 0:
        G0 = argv[0]
        # r  = (1. - kernel_prestore(H, kernMat, G0)/Gexp)
        r = wexp * (1.0 - kernel_prestore(H, kernMat, G0) / Gexp)

    else:
        # r = (1. - kernel_prestore(H, kernMat)/Gexp)
        r = wexp * (1.0 - kernel_prestore(H, kernMat) / Gexp)

    B = np.dot(Jr.T, Jr) + np.diag(np.dot(r.T, Jr))

    return B


def lcurve(Gexp, wexp, Hgs, kernMat, par, *argv):
    """
     Function: lcurve(input)

     Input: Gexp    = n*1 vector [Gt],
             wexp    = weights associated with datapoints
            Hgs     = guessed H,
            kernMat = matrix for faster kernel evaluation
            par     = parameter dictionary
            G0      = optionally

     Output: lamC and 3 vectors of size npoints*1 contains a range of lambda, rho
             and eta. "Elbow"  = lamC is estimated using a *NEW* heuristic AND by Hansen method


    March 2019: starting from large lambda to small cuts calculation time by a lot
                also gives an error estimate

    """
    if par["plateau"]:
        G0 = argv[0]

    npoints = int(
        par["lamDensity"] * (np.log10(par["lam_max"]) - np.log10(par["lam_min"]))
    )
    hlam = (par["lam_max"] / par["lam_min"]) ** (1.0 / (npoints - 1.0))
    lam = par["lam_min"] * hlam ** np.arange(npoints)
    eta = np.zeros(npoints)
    rho = np.zeros(npoints)
    logP = np.zeros(npoints)
    H = Hgs.copy()
    n = len(Gexp)
    ns = len(H)
    nl = ns - 2
    logPmax = -np.inf  # so nothing surprises me!
    Hlambda = np.zeros((ns, npoints))

    # Error Analysis: Furnish A_matrix
    Amat = getAmatrix(len(H))
    _, LogDetN = np.linalg.slogdet(Amat)

    #
    # This is the costliest step
    #
    for i in reversed(range(len(lam))):
        lamb = lam[i]

        if par["plateau"]:
            H, G0 = get_H(lamb, Gexp, wexp, H, kernMat, G0)
            # rho[i]  = np.linalg.norm((1. - kernel_prestore(H, kernMat, G0)/Gexp))
            rho[i] = np.linalg.norm(
                wexp * (1.0 - kernel_prestore(H, kernMat, G0) / Gexp)
            )
            Bmat = getBmatrix(H, kernMat, Gexp, wexp, G0)
        else:
            H = get_H(lamb, Gexp, wexp, H, kernMat)
            # rho[i]  = np.linalg.norm((1. - kernel_prestore(H,kernMat)/Gexp))
            rho[i] = np.linalg.norm(wexp * (1.0 - kernel_prestore(H, kernMat) / Gexp))
            Bmat = getBmatrix(H, kernMat, Gexp, wexp)

        eta[i] = np.linalg.norm(np.diff(H, n=2))
        Hlambda[:, i] = H

        _, LogDetC = np.linalg.slogdet(lamb * Amat + Bmat)
        V = rho[i] ** 2 + lamb * eta[i] ** 2

        # this assumes a prior exp(-lam)
        logP[i] = -V + 0.5 * (LogDetN + ns * np.log(lamb) - LogDetC) - lamb

        if logP[i] > logPmax:
            logPmax = logP[i]
        elif logP[i] < logPmax - 18:
            break

    # truncate all to significant lambda
    lam = lam[i:]
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
    lamM = np.exp(np.sum(plam * np.log(lam)))

    #
    # Dialling in the Smoothness Factor
    #
    if par["SmFacLam"] > 0:
        lamM = np.exp(
            np.log(lamM) + par["SmFacLam"] * (max(np.log(lam)) - np.log(lamM))
        )
    elif par["SmFacLam"] < 0:
        lamM = np.exp(
            np.log(lamM) + par["SmFacLam"] * (np.log(lamM) - min(np.log(lam)))
        )

    #
    # printing this here for now because storing lamC for sometime only
    #
    if par["plotting"]:
        plt.clf()
        # plt.axvline(x=lamC, c='k', label=r'$\lambda_c$')
        plt.axvline(x=lamM, c="gray", label=r"$\lambda_m$")
        plt.ylim(-20, 1)
        plt.plot(lam, logP, "o-")
        plt.xscale("log")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$\log\,p(\lambda)$")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig("output/logP.pdf")

    return lamM, lam, rho, eta, logP, Hlambda


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


def kernelD(H, kernMat):
    """
    Function: kernelD(input)

    outputs the (n*ns) dimensional matrix DK(H)(t)
    It approximates dK_i/dH_j = K * e(H_j):

    Input: H       = substituted CRS,
           kernMat = matrix for faster kernel evaluation

    Output: DK = Jacobian of H
    """

    n = kernMat.shape[0]
    ns = kernMat.shape[1]

    # A n*ns matrix with all the rows = H'
    Hsuper = np.dot(np.ones((n, 1)), np.exp(H).reshape(1, ns))
    DK = kernMat * Hsuper

    return DK
