"""
Model diagnostics for GWR, MGWR, and STWR models.

Provides AIC, AICc, BIC, and CV (cross-validation) criteria for model
selection and bandwidth optimization.
"""

__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
from scipy import linalg
from spglm.family import Gaussian, Poisson, Binomial


def get_AICc(gwr):
    """
    Return the corrected Akaike Information Criterion (AICc).

    Gaussian: Fotheringham et al. (2002), eq. (2.33), p. 61:
        AICc = -2*LL + 2*n*(k+1) / (n - k - 2)

    GLM families: Nakaya et al. (2005), eq. (36), p. 2704:
        AICc = AIC + 2*k*(k+1) / (n - k - 1)

    Parameters
    ----------
    gwr : GWRResults, MGWRResults, or STWRResults
        Fitted model result object with attributes: n, tr_S, llf, family.

    Returns
    -------
    float : AICc value.
    """
    n = gwr.n
    k = gwr.tr_S
    if isinstance(gwr.family, Gaussian):
        aicc = -2.0 * gwr.llf + 2.0 * n * (k + 1.0) / (n - k - 2.0)
    elif isinstance(gwr.family, (Poisson, Binomial)):
        aicc = get_AIC(gwr) + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    return aicc


def get_AIC(gwr):
    """
    Return the Akaike Information Criterion (AIC).

    Gaussian: Fotheringham et al. (2002), eq. (4.22), p. 96:
        AIC = -2*LL + 2*(k+1)

    GLM families: AIC = deviance + 2*tr(S)

    Parameters
    ----------
    gwr : GWRResults, MGWRResults, or STWRResults
        Fitted model result object.

    Returns
    -------
    float : AIC value.
    """
    k = gwr.tr_S
    y = gwr.y
    mu = gwr.mu
    if isinstance(gwr.family, Gaussian):
        aic = -2.0 * gwr.llf + 2.0 * (k + 1)
    elif isinstance(gwr.family, (Poisson, Binomial)):
        aic = np.sum(gwr.family.resid_dev(y, mu)**2) + 2.0 * k
    return aic


def get_BIC(gwr):
    """
    Return the Bayesian Information Criterion (BIC).

    Gaussian: Fotheringham et al. (2002), eq. (2.34), p. 61:
        BIC = -2*LL + (k+1)*log(n)

    GLM families: BIC = deviance + tr(S)*log(n)

    Parameters
    ----------
    gwr : GWRResults, MGWRResults, or STWRResults
        Fitted model result object.

    Returns
    -------
    float : BIC value.
    """
    n = gwr.n
    k = gwr.tr_S
    y = gwr.y
    mu = gwr.mu
    if isinstance(gwr.family, Gaussian):
        bic = -2.0 * gwr.llf + (k + 1) * np.log(n)
    elif isinstance(gwr.family, (Poisson, Binomial)):
        bic = np.sum(gwr.family.resid_dev(y, mu)**2) + k * np.log(n)
    return bic


def get_CV(gwr):
    """
    Return the cross-validation (CV) score (Gaussian only).

    Fotheringham et al. (2002), eq. (2.31), p. 60 and eq. (9.4), p. 212:
        CV = sum( (resid_i / (1 - influ_i))^2 ) / n

    This is a leave-one-out cross-validation score used for bandwidth
    selection in GWR.

    Parameters
    ----------
    gwr : GWRResults or GWRResultsLite
        Fitted model result object.

    Returns
    -------
    float : CV score.
    """
    aa = gwr.resid_response.reshape((-1, 1)) / (1.0 - gwr.influ)
    cv = np.sum(aa**2) / gwr.n
    return cv


def corr(cov):
    """
    Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : array (p, p)
        Covariance matrix.

    Returns
    -------
    array (p, p) : Correlation matrix.
    """
    invsd = np.diag(1 / np.sqrt(np.diag(cov)))
    cors = np.dot(np.dot(invsd, cov), invsd)
    return cors