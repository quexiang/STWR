# Main GWR, MGWR, and STWR model classes
# STWR is developed by Xiang Que (xiangq@uidaho.edu, quexiang@fafu.edu.cn)
# GWR and MGWR are based on the original MGWR 2.0.2 implementation by Taylor Oshan (tayoshan@gmail.com)

import copy
from typing import Optional
import multiprocessing as mp

import numpy as np
import numpy.linalg as la
from scipy.stats import t
from scipy.special import factorial
from itertools import combinations as combo
from spglm.family import Gaussian, Binomial, Poisson
from spglm.glm import GLM, GLMResults
from spglm.iwls import iwls, _compute_betas_gwr
from spglm.utils import cache_readonly

from .diagnostics import get_AIC, get_AICc, get_BIC, corr
from .kernels import *
from .summary import *


class GWR(GLM):
    """
    Geographically Weighted Regression (GWR).

    Estimates Gaussian, Poisson, and logistic models built on a GLM framework.
    The GWR object prepares model input; call ``fit()`` to perform estimation
    and obtain a GWRResults object.

    Parameters
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x, y) coordinates of
                    observations; also used as calibration locations when
                    ``points`` is set to None.

    y             : array
                    n*1, dependent variable.

    X             : array
                    n*k, independent variables, excluding the constant.

    bw            : scalar
                    Bandwidth value: either a fixed distance or the number of
                    nearest neighbors (adaptive). User-specified or obtained
                    via ``Sel_BW``.

    family        : family object
                    Underlying probability model from ``spglm.family``
                    (Gaussian, Poisson, Binomial).

    offset        : array, optional
                    n*1, offset variable at each location. For Poisson models
                    this is often the population at risk or expected outcome
                    size in spatial epidemiology. Default is None (all 1.0).

    sigma2_v1     : boolean, optional
                    Form of the corrected denominator for sigma^2 in model
                    diagnostics:

                    - True:  n - tr(S) (default)
                    - False: n - 2(tr(S) + tr(S'S))

    kernel        : string, optional
                    Kernel function used to weight observations:
                    'gaussian', 'bisquare', 'exponential'. Default 'bisquare'.

    fixed         : boolean, optional
                    True for distance-based (fixed) kernel,
                    False for adaptive (nearest-neighbor) kernel. Default False.

    constant      : boolean, optional
                    True to include intercept (default), False to exclude it.

    spherical     : boolean, optional
                    True for spherical coordinates (longitude-latitude, Haversine distance),
                    False for projected/Euclidean coordinates. Default False.

    hat_matrix    : boolean, optional
                    True to store the full n*n hat matrix,
                    False to save memory (default).

    name_x        : list of strings, optional
                    Names of independent variables for output display.

    Attributes
    ----------
    coords        : array-like
                    n*2, calibration coordinate pairs.

    y             : array
                    n*1, dependent variable.

    X             : array
                    n*k, independent variables, excluding the constant.

    bw            : scalar
                    Bandwidth value (distance or neighbor count).

    family        : family object
                    Underlying probability model.

    offset        : array
                    n*1, offset variable.

    sigma2_v1     : boolean
                    Sigma^2 denominator form.

    kernel        : string
                    Kernel function name.

    fixed         : boolean
                    Whether bandwidth is fixed (True) or adaptive (False).

    constant      : boolean
                    Whether intercept is included.

    spherical     : boolean
                    Whether spherical coordinates are used.

    hat_matrix    : boolean
                    Whether full hat matrix is stored.

    n             : integer
                    Number of observations.

    k             : integer
                    Number of independent variables (including constant if enabled).

    mean_y        : float
                    Mean of y.

    std_y         : float
                    Standard deviation of y.

    fit_params    : dict
                    Parameters passed to the fit method.

    points        : array-like or None
                    n*2, prediction locations used instead of full observations;
                    defaults to None unless set via ``predict()``.

    P             : array or None
                    n*k, independent variables for prediction (excluding constant);
                    default None unless set via ``predict()``.

    exog_scale    : scalar or None
                    Estimated scale from sampled locations; default None.

    exog_resid    : array-like or None
                    Estimated residuals from sampled locations; default None.

    Examples
    --------
    Basic model calibration:

    >>> import libpysal as ps
    >>> from mgwr.gwr import GWR
    >>> data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
    >>> coords = list(zip(data.by_col('X'), data.by_col('Y')))
    >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
    >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
    >>> pov = np.array(data.by_col('PctPov')).reshape((-1,1))
    >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
    >>> X = np.hstack([rural, pov, african_amer])
    >>> model = GWR(coords, y, X, bw=90.000, fixed=False, kernel='bisquare')
    >>> results = model.fit()
    >>> print(results.params.shape)
    (159, 4)

    Prediction at unsampled locations:

    >>> index = np.arange(len(y))
    >>> test = index[-10:]
    >>> X_test = X[test]
    >>> coords_test = np.array(coords)[test]
    >>> model = GWR(coords, y, X, bw=94, fixed=False, kernel='bisquare')
    >>> results = model.predict(coords_test, X_test)
    >>> print(results.params.shape)
    (10, 4)

    """

    def __init__(self, coords, y, X, bw, family=Gaussian(), offset=None,
                 sigma2_v1=True, kernel='bisquare', fixed=False, constant=True,
                 spherical=False, hat_matrix=False, name_x=None):
        """
        Initialize class
        """
        GLM.__init__(self, y, X, family, constant=constant)
        self.constant = constant
        self.sigma2_v1 = sigma2_v1
        self.coords = np.array(coords)
        self.bw = bw
        self.kernel = kernel
        self.fixed = fixed
        if offset is None:
            self.offset = np.ones((self.n, 1))
        else:
            self.offset = offset * 1.0
        self.fit_params = {}

        self.points = None
        self.exog_scale = None
        self.exog_resid = None
        self.P = None
        self.spherical = spherical
        self.hat_matrix = hat_matrix
        self.name_x = name_x

    def _build_wi(self, i, bw):

        if bw == np.inf:
            wi = np.ones((self.n))
            return wi

        try:
            wi = Kernel(i, self.coords, bw, fixed=self.fixed,
                        function=self.kernel, points=self.points,
                        spherical=self.spherical).kernel
        except BaseException:
            raise  # TypeError('Unsupported kernel function  ', kernel)

        return wi

    def _local_fit(self, i):
        """
        Perform local model fitting at observation i.

        Computes the spatially weighted least-squares (or IWLS for
        GLM families) estimates using the kernel weights centered at
        location i.

        Returns
        -------
        tuple
            lite=True:  (influ, resid, predy, betas)
            lite=False: (influ, resid, predy, betas, w, Si, tr_STS_i, CCT)
        """
        wi = self._build_wi(i, self.bw).reshape(-1, 1)  # Local spatial weights

        if isinstance(self.family, Gaussian):
            betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
            predy = np.dot(self.X[i], betas)[0]
            resid = self.y[i] - predy
            influ = np.dot(self.X[i], inv_xtx_xt[:, i])
            w = 1

        elif isinstance(self.family, (Poisson, Binomial)):
            rslt = iwls(self.y, self.X, self.family, self.offset, None,
                        self.fit_params['ini_params'], self.fit_params['tol'],
                        self.fit_params['max_iter'], wi=wi)
            inv_xtx_xt = rslt[5]
            w = rslt[3][i][0]
            influ = np.dot(self.X[i], inv_xtx_xt[:, i]) * w
            predy = rslt[1][i]
            resid = self.y[i] - predy
            betas = rslt[0]

        if self.fit_params['lite']:
            return influ, resid, predy, betas.reshape(-1)
        else:
            Si = np.dot(self.X[i], inv_xtx_xt).reshape(-1)
            tr_STS_i = np.sum(Si * Si * w * w)
            CCT = np.diag(np.dot(inv_xtx_xt, inv_xtx_xt.T)).reshape(-1)
            if not self.hat_matrix:
                Si = None
            return influ, resid, predy, betas.reshape(-1), w, Si, tr_STS_i, CCT

    def fit(self, ini_params=None, tol=1.0e-5, max_iter=20, solve='iwls',
            lite=False, pool=None):
        """
        Fit the GWR model using the specified estimation routine.

        Parameters
        ----------
        ini_params     : array, optional
                        k*1, initial coefficient values including constant.
                        Default is None, which auto-calculates starting values.
        tol            : float, optional
                        Tolerance for estimation convergence. Default 1.0e-5.
        max_iter       : integer, optional
                        Maximum iterations if convergence not achieved. Default 20.
        solve          : string, optional
                        Estimation technique. Only 'iwls' (iteratively
                        reweighted least squares) is currently supported.
        lite           : bool, optional
                        If True, compute minimal diagnostics only (faster,
                        suitable for bandwidth selection). If False, compute
                        full GWR diagnostics. Default False.
        pool           : multiprocessing.Pool or None, optional
                        Pool object for parallel fitting. Default None (sequential).

        Returns
        -------
        result         : GWRResults or GWRResultsLite
                        Full results if lite=False, lightweight results if lite=True.
        """
        self.fit_params['ini_params'] = ini_params
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve'] = solve
        self.fit_params['lite'] = lite

        if solve.lower() == 'iwls':

            if self.points is None:
                m = self.y.shape[0]
            else:
                m = self.points.shape[0]

            if pool:
                rslt = pool.map(self._local_fit,
                                range(m))  #parallel using mp.Pool
            else:
                rslt = map(self._local_fit, range(m))  #sequential

            rslt_list = list(zip(*rslt))
            influ = np.array(rslt_list[0]).reshape(-1, 1)
            resid = np.array(rslt_list[1]).reshape(-1, 1)
            params = np.array(rslt_list[3])

            if lite:
                return GWRResultsLite(self, resid, influ, params)
            else:
                predy = np.array(rslt_list[2]).reshape(-1, 1)
                w = np.array(rslt_list[-4]).reshape(-1, 1)
                if self.hat_matrix:
                    S = np.array(rslt_list[-3])
                else:
                    S = None
                tr_STS = np.sum(np.array(rslt_list[-2]))
                CCT = np.array(rslt_list[-1])
                return GWRResults(self, params, predy, S, CCT, influ, tr_STS,
                                  w, self.name_x)


    def predict(self, points, P, exog_scale=None, exog_resid=None,
                fit_params={}):
        """
        Predict the dependent variable at un-sampled locations.

        Parameters
        ----------
        points        : array-like
                        n*2, (x, y) coordinates of prediction locations.
        P             : array
                        n*k, independent variables at prediction locations,
                        excluding the constant.
        exog_scale    : scalar, optional
                        Estimated scale from sampled locations. Default None
                        fits a training model from ``coords``.
        exog_resid    : array-like, optional
                        Estimated residuals from sampled locations (length
                        must match coords). Default None fits a training model.
        fit_params    : dict, optional
                        Key-value pairs passed to ``fit()`` to define the
                        estimation routine.
        """
        if (exog_scale is None) & (exog_resid is None):
            train_gwr = self.fit(**fit_params)
            self.exog_scale = train_gwr.scale
            self.exog_resid = train_gwr.resid_response
        elif (exog_scale is not None) & (exog_resid is not None):
            self.exog_scale = exog_scale
            self.exog_resid = exog_resid
        else:
            raise InputError('exog_scale and exog_resid must both either be'
                             'None or specified')
        self.points = points
        if self.constant:
            P = np.hstack([np.ones((len(P), 1)), P])
            self.P = P
        else:
            self.P = P
        gwr = self.fit(**fit_params)

        return gwr

    @cache_readonly
    def df_model(self):
        return None

    @cache_readonly
    def df_resid(self):
        return None


class GWRResults(GLMResults):
    """
    Results container for a fitted GWR model.

    Provides access to parameter estimates, diagnostics, and inference
    statistics. Most attributes are computed lazily via ``@cache_readonly``.

    Parameters
    ----------
    model               : GWR
                          Pointer to the GWR model object.

    params              : array
                          n*k, estimated coefficients.

    predy               : array
                          n*1, predicted y values.

    S                   : array
                          n*n, hat matrix (or None if hat_matrix=False).

    CCT                 : array
                          n*k, unscaled variance-covariance matrix.

    w                   : array
                          n*1, final IWLS weights. Default None.

    name_x              : list of strings, optional
                          Names of independent variables for output display.

    Attributes
    ----------
    model               : GWR
                          Pointer to the fitted GWR model.

    params              : array
                          n*k, parameter estimates.

    predy               : array
                          n*1, predicted y values.

    y                   : array
                          n*1, dependent variable.

    X                   : array
                          n*k, independent variables (including constant).

    family              : family object
                          Probability model from spglm.

    n                   : integer
                          Number of observations.

    k                   : integer
                          Number of independent variables.

    df_model            : integer
                          Model degrees of freedom.

    df_resid            : integer
                          Residual degrees of freedom.

    offset              : array
                          n*1, offset variable.

    scale               : float
                          Sigma^2 used for subsequent computations.

    w                   : array
                          n*1, final IWLS weights.

    resid_response      : array
                          n*1, response residuals.

    resid_ss            : scalar
                          Residual sum of squares.

    W                   : array
                          n*n, spatial weights for each observation from each
                          calibration point.

    S                   : array
                          n*n, hat matrix.

    CCT                 : array
                          n*k, scaled variance-covariance matrix.

    ENP                 : scalar
                          Effective number of parameters (depends on sigma2 form).

    tr_S                : float
                          Trace of S (hat) matrix.

    tr_STS              : float
                          Trace of S'S (computed when sigma2_v1 is False).

    y_bar               : array
                          n*1, weighted mean of y at each location.

    TSS                 : array
                          n*1, geographically weighted total sum of squares.

    RSS                 : array
                          n*1, geographically weighted residual sum of squares.

    R2                  : float
                          Global R-squared (Gaussian only).

    adj_R2              : float
                          Adjusted global R-squared (Gaussian only).

    aic                 : float
                          Akaike Information Criterion.

    aicc                : float
                          Corrected AIC (penalizes small bandwidths).

    bic                 : float
                          Bayesian Information Criterion.

    localR2             : array
                          n*1, local R-squared values (Gaussian only).

    sigma2              : float
                          Residual variance corrected for ENP.

    std_res             : array
                          n*1, standardized residuals.

    bse                 : array
                          n*k, standard errors of parameter estimates.

    influ               : array
                          n*1, diagonal of the hat matrix (leverage values).

    CooksD              : array
                          n*1, Cook's distance.

    tvalues             : array
                          n*k, local t-statistics.

    adj_alpha           : array
                          3*1, corrected alpha values for 90%, 95%, and 99%
                          confidence levels (multiple testing correction).

    deviance            : array
                          n*1, local model deviance (GLM families).

    resid_deviance      : array
                          n*1, local residual deviance (GLM families).

    llf                 : scalar
                          Log-likelihood of the full model.

    pDev                : float
                          Local percent of deviance explained (GLM analogue of R^2).

    D2                  : float
                          Percent deviance explained (GLM analogue of R^2).

    adj_D2              : float
                          Adjusted percent deviance explained.

    mu                  : array
                          n*1, predicted mean response.

    fit_params          : dict
                          Parameters passed to fit().

    predictions         : array
                          p*1, predicted values from predict().

    name_x              : list of strings
                          Names of independent variables for output display.
    """

    def __init__(self, model, params, predy, S, CCT, influ, tr_STS=None,
                 w=None, name_x=None):
        GLMResults.__init__(self, model, params, predy, w)
        self.offset = model.offset
        if w is not None:
            self.w = w
        self.predy = predy
        self.S = S
        self.tr_STS = tr_STS
        self.influ = influ
        self.CCT = self.cov_params(CCT, model.exog_scale)
        self._cache = {}
        self.name_x=name_x

    @cache_readonly
    def W(self):
        W = np.array(
            [self.model._build_wi(i, self.model.bw) for i in range(self.n)])
        return W

    @cache_readonly
    def resid_ss(self):
        if self.model.points is not None:
            raise NotImplementedError('Not available for GWR prediction')
        else:
            u = self.resid_response.flatten()
        return np.dot(u, u.T)

    @cache_readonly
    def scale(self, scale=None):
        if isinstance(self.family, Gaussian):
            scale = self.sigma2
        else:
            scale = 1.0
        return scale

    def cov_params(self, cov, exog_scale=None):
        """
        Return scaled covariance parameters.

        Parameters
        ----------
        cov         : array
                      Unscaled covariance.

        Returns
        -------
        array : Scaled covariance.
        """
        if exog_scale is not None:
            return cov * exog_scale
        else:
            return cov * self.scale

    @cache_readonly
    def tr_S(self):
        """
        Return the trace of the hat matrix S.
        """
        return np.sum(self.influ)

    @cache_readonly
    def ENP(self):
        """
        Return the effective number of parameters.

        When sigma2_v1=True: ENP = tr(S), as defined in Yu et al. (2019).
        When sigma2_v1=False: ENP = 2*tr(S) - tr(S'S).
        """
        if self.model.sigma2_v1:
            return self.tr_S
        else:
            return 2 * self.tr_S - self.tr_STS

    @cache_readonly
    def y_bar(self):
        """
        Return the spatially weighted mean of y at each location.

        Reference: Fotheringham et al. (2002), eq. (9.9).
        """
        if self.model.points is not None:
            n = len(self.model.points)
        else:
            n = self.n
        off = self.offset.reshape((-1, 1))
        arr_ybar = np.zeros(shape=(self.n, 1))
        for i in range(n):
            w_i = np.reshape(self.model._build_wi(i, self.model.bw), (-1, 1))
            sum_yw = np.sum(self.y.reshape((-1, 1)) * w_i)
            arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i * off)
        return arr_ybar

    @cache_readonly
    def TSS(self):
        """
        Return the geographically weighted total sum of squares.

        Reference: Fotheringham et al. (2002), eq. (9.9), p. 215.
        """
        if self.model.points is not None:
            n = len(self.model.points)
        else:
            n = self.n
        TSS = np.zeros(shape=(n, 1))
        for i in range(n):
            TSS[i] = np.sum(
                np.reshape(self.model._build_wi(i, self.model.bw),
                           (-1, 1)) * (self.y.reshape(
                               (-1, 1)) - self.y_bar[i])**2)
        return TSS

    @cache_readonly
    def RSS(self):
        """
        Return the geographically weighted residual sum of squares.

        Reference: Fotheringham et al. (2002), eq. (9.10), p. 215.
        """
        if self.model.points is not None:
            n = len(self.model.points)
            resid = self.model.exog_resid.reshape((-1, 1))
        else:
            n = self.n
            resid = self.resid_response.reshape((-1, 1))
        RSS = np.zeros(shape=(n, 1))
        for i in range(n):
            RSS[i] = np.sum(
                np.reshape(self.model._build_wi(i, self.model.bw),
                           (-1, 1)) * resid**2)
        return RSS

    @cache_readonly
    def localR2(self):
        """
        Return local R-squared values at each location (Gaussian only).

        Reference: Fotheringham et al. (2002), eq. (9.8), p. 215.
        """
        if isinstance(self.family, Gaussian):
            return (self.TSS - self.RSS) / self.TSS
        else:
            raise NotImplementedError('Only applicable to Gaussian')

    @cache_readonly
    def sigma2(self):
        """
        Return the residual variance, corrected for ENP.

        If sigma2_v1 is True: uses n - tr(S) in the denominator
        (Fotheringham et al. 2002, eq. 9.6, p. 214).

        If sigma2_v1 is False: uses n - 2*tr(S) + tr(S'S)
        (Fotheringham et al. 2002, eqs. 2.16-2.18, p. 55).
        """
        if self.model.sigma2_v1:
            return (self.resid_ss / (self.n - self.tr_S))
        else:
            return self.resid_ss / (self.n - 2.0 * self.tr_S + self.tr_STS)

    @cache_readonly
    def std_res(self):
        """
        Return the standardized residuals.

        Reference: Fotheringham et al. (2002), eq. (9.7), p. 215.
        """
        return self.resid_response.reshape(
            (-1, 1)) / (np.sqrt(self.scale * (1.0 - self.influ)))

    @cache_readonly
    def bse(self):
        """
        Return the standard errors of the parameter estimates.

        Reference: Fotheringham et al. (2002), eqs. (2.15) and (2.21), p. 215.
        """
        return np.sqrt(self.CCT)

    @cache_readonly
    def cooksD(self):
        """
        Return Cook's distance for each observation.

        Reference: Fotheringham et al. (2002), eq. (9.11), p. 216.
        Note: p in (9.11) is tr(S), the effective number of parameters.
        """
        return self.std_res**2 * self.influ / (self.tr_S * (1.0 - self.influ))

    @cache_readonly
    def deviance(self):
        off = self.offset.reshape((-1, 1)).T
        y = self.y
        ybar = self.y_bar
        if isinstance(self.family, Gaussian):
            raise NotImplementedError(
                'deviance not currently used for Gaussian')
        elif isinstance(self.family, Poisson):
            dev = np.sum(
                2.0 * self.W * (y * np.log(y / (ybar * off)) -
                                (y - ybar * off)), axis=1)
        elif isinstance(self.family, Binomial):
            dev = self.family.deviance(self.y, self.y_bar, self.W, axis=1)
        return dev.reshape((-1, 1))

    @cache_readonly
    def resid_deviance(self):
        if isinstance(self.family, Gaussian):
            raise NotImplementedError(
                'deviance not currently used for Gaussian')
        else:
            off = self.offset.reshape((-1, 1)).T
            y = self.y
            ybar = self.y_bar
            global_dev_res = ((self.family.resid_dev(self.y, self.mu))**2)
            dev_res = np.repeat(global_dev_res.flatten(), self.n)
            dev_res = dev_res.reshape((self.n, self.n))
            dev_res = np.sum(dev_res * self.W.T, axis=0)
            return dev_res.reshape((-1, 1))

    @cache_readonly
    def pDev(self):
        """
        Local percentage of deviance accounted for. Described in the GWR4
        manual. Equivalent to 1 - (deviance/null deviance)
        """
        if isinstance(self.family, Gaussian):
            raise NotImplementedError('Not implemented for Gaussian')
        else:
            return 1.0 - (self.resid_deviance / self.deviance)

    @cache_readonly
    def adj_alpha(self):
        """
        Return corrected critical alpha values for multiple hypothesis testing.

        Returns corrected values for 90% (.1), 95% (.05), and 99% (.01)
        confidence levels based on da Silva & Fotheringham (2015).
        """
        alpha = np.array([.1, .05, .001])
        pe = self.ENP
        p = self.k
        return (alpha * p) / pe

    def critical_tval(self, alpha=None):
        """
        Return the critical t-value for the given alpha or the 95% adjusted alpha.

        Parameters
        ----------
        alpha           : scalar, optional
                          Significance level. Default None uses adjusted alpha at 95% CI.

        Returns
        -------
        critical        : scalar
                          Two-tailed critical t-value.
        """
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha[1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):
        """
        Return t-values with non-significant entries set to zero.

        If ``critical_t`` is provided, it is used directly. If ``alpha`` is
        provided, the critical t-value is derived from it. If neither is
        given, the 95% adjusted alpha is used. If both are supplied,
        ``critical_t`` takes precedence.

        Parameters
        ----------
        critical_t      : scalar, optional
                          Direct critical t-value threshold.

        alpha           : scalar, optional
                          Significance level used to derive the critical t-value.

        Returns
        -------
        filtered        : array
                          n*k t-values; entries below the critical threshold
                          are set to zero.
        """
        n = self.n
        if critical_t is not None:
            critical = critical_t
        else:
            critical = self.critical_tval(alpha=alpha)

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    @cache_readonly
    def df_model(self):
        return self.n - self.tr_S

    @cache_readonly
    def df_resid(self):
        return self.n - 2.0 * self.tr_S + self.tr_STS

    @cache_readonly
    def normalized_cov_params(self):
        return None

    @cache_readonly
    def resid_pearson(self):
        return None

    @cache_readonly
    def resid_working(self):
        return None

    @cache_readonly
    def resid_anscombe(self):
        return None

    @cache_readonly
    def pearson_chi2(self):
        return None

    @cache_readonly
    def llnull(self):
        return None

    @cache_readonly
    def null_deviance(self):
        return self.family.deviance(self.y, self.null)

    @cache_readonly
    def global_deviance(self):
        deviance = np.sum(self.family.resid_dev(self.y, self.mu)**2)
        return deviance

    @cache_readonly
    def D2(self):
        """
        Return the percent of deviance explained (1 - deviance/null_deviance).
        """
        D2 = 1.0 - (self.global_deviance / self.null_deviance)
        return D2

    @cache_readonly
    def R2(self):
        """
        Return the global R-squared value (Gaussian models only).
        """
        if isinstance(self.family, Gaussian):
            return self.D2
        else:
            raise NotImplementedError('R2 only for Gaussian')

    @cache_readonly
    def adj_D2(self):
        """
        Return the adjusted percent of deviance explained.
        """
        adj_D2 = 1 - (1 - self.D2) * (self.n - 1) / (self.n - self.ENP - 1)
        return adj_D2

    @cache_readonly
    def adj_R2(self):
        """
        Return the adjusted global R-squared (Gaussian models only).
        """
        if isinstance(self.family, Gaussian):
            return self.adj_D2
        else:
            raise NotImplementedError('adjusted R2 only for Gaussian')

    @cache_readonly
    def aic(self):
        return get_AIC(self)

    @cache_readonly
    def aicc(self):
        return get_AICc(self)

    @cache_readonly
    def bic(self):
        return get_BIC(self)

    @cache_readonly
    def pseudoR2(self):
        return None

    @cache_readonly
    def adj_pseudoR2(self):
        return None

    @cache_readonly
    def pvalues(self):
        return None

    @cache_readonly
    def conf_int(self):
        return None

    @cache_readonly
    def use_t(self):
        return None

    def get_bws_intervals(self, selector, level=0.95):
        """
        Compute the bandwidth confidence interval for a GWR model.

        Based on Akaike weights from the bandwidth search history.
        See Li et al. (2020), Annals of the AAG.

        Parameters
        ----------
        selector : Sel_BW
            The bandwidth selector used to fit the GWR model.
        level : float
            Confidence level (default 0.95).

        Returns
        -------
        tuple : (lower_bound, upper_bound)
        """

        try:
            import pandas as pd
        except ImportError:
            return

        #Get AICcs and associated bw from the last iteration of back-fitting and make a DataFrame
        aiccs = pd.DataFrame(list(zip(*selector.sel_hist))[1],columns=["aicc"])
        aiccs['bw'] = list(zip(*selector.sel_hist))[0]
        #Sort DataFrame by the AICc values
        aiccs = aiccs.sort_values(by=['aicc'])
        #Calculate delta AICc
        d_aic_ak = aiccs.aicc - aiccs.aicc.min()
        #Calculate AICc weights
        w_aic_ak = np.exp(-0.5*d_aic_ak) / np.sum(np.exp(-0.5*d_aic_ak))
        aiccs['w_aic_ak'] = w_aic_ak/np.sum(w_aic_ak)
        #Calculate cum. AICc weights
        aiccs['cum_w_ak'] = aiccs.w_aic_ak.cumsum()
        #Find index where the cum weights above p-val
        index = len(aiccs[aiccs.cum_w_ak < level]) + 1
        #Get bw boundaries
        interval = (aiccs.iloc[:index,:].bw.min(),aiccs.iloc[:index,:].bw.max())
        return interval


    def local_collinearity(self):
        """
        Compute local multicollinearity diagnostics for the GWR design matrix.

        Returns four arrays:
        - Local correlation coefficients      (n, ((p^2) + p) / 2)
        - Local variance inflation factors    (n, p - 1)  (excludes constant)
        - Local condition numbers             (n, 1)
        - Local variance-decomposition proportions (n, p)

        Notes
        -----
        n = number of calibration locations,
        p = number of explanatory variables.
        Correlation and VIF are not computed for the constant term.
        """
        x = self.X
        w = self.W
        nvar = x.shape[1]
        nrow = len(w)
        if self.model.constant:
            ncor = (((nvar - 1)**2 + (nvar - 1)) / 2) - (nvar - 1)
            jk = list(combo(range(1, nvar), 2))
        else:
            ncor = (((nvar)**2 + (nvar)) / 2) - nvar
            jk = list(combo(range(nvar), 2))
        corr_mat = np.ndarray((nrow, int(ncor)))
        if self.model.constant:
            vifs_mat = np.ndarray((nrow, nvar - 1))
        else:
            vifs_mat = np.ndarray((nrow, nvar))
        vdp_idx = np.ndarray((nrow, nvar))
        vdp_pi = np.ndarray((nrow, nvar, nvar))

        for i in range(nrow):
            wi = self.model._build_wi(i, self.model.bw)
            sw = np.sum(wi)
            wi = wi / sw
            tag = 0

            for j, k in jk:
                corr_mat[i, tag] = corr(np.cov(x[:, j], x[:, k],
                                               aweights=wi))[0][1]
                tag = tag + 1

            if self.model.constant:
                corr_mati = corr(np.cov(x[:, 1:].T, aweights=wi))
                vifs_mat[i, ] = np.diag(
                    np.linalg.solve(corr_mati, np.identity((nvar - 1))))

            else:
                corr_mati = corr(np.cov(x.T, aweights=wi))
                vifs_mat[i, ] = np.diag(
                    np.linalg.solve(corr_mati, np.identity((nvar))))

            xw = x * wi.reshape((nrow, 1))
            sxw = np.sqrt(np.sum(xw**2, axis=0))
            sxw = np.transpose(xw.T / sxw.reshape((nvar, 1)))
            svdx = np.linalg.svd(sxw)
            vdp_idx[i, ] = svdx[1][0] / svdx[1]
            phi = np.dot(svdx[2].T, np.diag(1 / svdx[1]))
            phi = np.transpose(phi**2)
            pi_ij = phi / np.sum(phi, axis=0)
            vdp_pi[i, :, :] = pi_ij

        local_CN = vdp_idx[:, nvar - 1].reshape((-1, 1))
        VDP = vdp_pi[:, nvar - 1, :]

        return corr_mat, vifs_mat, local_CN, VDP

    def spatial_variability(self, selector, n_iters=1000, seed=None):
        """
        Monte Carlo test for spatial variability of each coefficient surface.

        Randomly permutes spatial coordinates to generate the null
        distribution of parameter surface standard deviations, then
        computes a pseudo p-value for each covariate.

        WARNING: This test is computationally very demanding.

        Parameters
        ----------
        selector        : Sel_BW
                          The bandwidth selector used for the GWR model.
        n_iters         : int, optional
                          Number of Monte Carlo permutations (default 1000).
        seed            : int, optional
                          Random seed for reproducibility (default 5536).

        Returns
        -------
        p_vals          : list of float
                          Pseudo p-values for each parameter surface.
        """
        temp_sel = copy.deepcopy(selector)
        temp_gwr = copy.deepcopy(self.model)

        if seed is None:
            np.random.seed(5536)
        else:
            np.random.seed(seed)

        fit_params = temp_gwr.fit_params
        search_params = temp_sel.search_params
        kernel = temp_gwr.kernel
        fixed = temp_gwr.fixed

        if self.model.constant:
            X = self.X[:, 1:]
        else:
            X = self.X

        init_sd = np.std(self.params, axis=0)
        SDs = []

        try:
            from tqdm.auto import tqdm  # if they have it, let users have a progress bar
        except ImportError:
            def tqdm(x, desc=''):  # otherwise, just passthrough the range
                return x

        for x in tqdm(range(n_iters), desc='Testing'):
            temp_coords = np.random.permutation(self.model.coords)
            temp_sel.coords = temp_coords
            temp_bw = temp_sel.search(**search_params)
            temp_gwr.bw = temp_bw
            temp_gwr.coords = temp_coords
            temp_params = temp_gwr.fit(**fit_params).params
            temp_sd = np.std(temp_params, axis=0)
            SDs.append(temp_sd)

        p_vals = (np.sum(np.array(SDs) > init_sd, axis=0) / float(n_iters))
        return p_vals

    @cache_readonly
    def predictions(self):
        P = self.model.P
        if P is None:
            raise TypeError('predictions only avaialble if predict'
                            'method is previously called on GWR model')
        else:
            predictions = np.sum(P * self.params, axis=1).reshape((-1, 1))
        return predictions

    def summary(self, as_str: bool = False) -> Optional[str]:
        """
        Print or return a formatted GWR model summary.

        The summary includes: model type and dimensions, global regression
        results for comparison, GWR diagnostic information, and summary
        statistics for parameter estimates.

        Parameters
        ----------
        as_str        : bool, optional
                        If True, return the summary as a string instead of
                        printing to stdout.

        Returns
        -------
        summary        : Optional[str]
                        Summary string when ``as_str=True``, otherwise None.
        """
        summary = summaryModel(self) + summaryGLM(self) + summaryGWR(self)

        if as_str:
            return summary

        print(summary)
        return None


class GWRResultsLite(object):
    """
    Lightweight GWR results for bandwidth selection.

    Computes only the minimal diagnostics (influence, residuals, log-likelihood)
    needed for bandwidth optimization. This is much faster than full estimation.
    See FastGWR (Li et al., 2019, IJGIS).

    Parameters
    ----------
    model               : GWR
                          Pointer to the GWR model object.

    resid               : array
                          n*1, response residuals.

    influ               : array
                          n*1, diagonal of the hat matrix.

    Attributes
    ----------
    tr_S                : float
                          Trace of the hat matrix.

    llf                 : scalar
                          Log-likelihood of the model.

    mu                  : array
                          n*1, predicted mean response.

    resid_ss            : scalar
                          Residual sum of squares.
    """

    def __init__(self, model, resid, influ, params):
        self.y = model.y
        self.family = model.family
        self.n = model.n
        self.influ = influ
        self.resid_response = resid
        self.model = model
        self.params = params

    @cache_readonly
    def tr_S(self):
        return np.sum(self.influ)

    @cache_readonly
    def llf(self):
        return self.family.loglike(self.y, self.mu)

    @cache_readonly
    def mu(self):
        return self.y - self.resid_response

    @cache_readonly
    def predy(self):
        return self.y - self.resid_response

    @cache_readonly
    def resid_ss(self):
        u = self.resid_response.flatten()
        return np.dot(u, u.T)


class MGWR(GWR):
    """
    Multiscale Geographically Weighted Regression (MGWR).

    Each covariate is assigned its own bandwidth, allowing for
    relationships that operate at different spatial scales. Model
    estimation uses iterative GAM backfitting or an exact closed-form
    solution. Currently supports Gaussian family only.

    References: Fotheringham et al. (2017), Yu et al. (2019).

    Parameters
    ----------
    coords        : array-like
                    n*2, (x, y) coordinates of observations.

    y             : array
                    n*1, dependent variable.

    X             : array
                    n*k, independent variables, excluding the constant.

    selector      : Sel_BW
                    A bandwidth selector object that has already run
                    ``search(multi=True)``. Provides the optimal bandwidths
                    and backfitting history.

    family        : family object, optional
                    Probability model. Gaussian only for MGWR.

    sigma2_v1     : boolean, optional
                    Sigma^2 denominator form:
                    True = n - tr(S) (default),
                    False = n - 2*tr(S) + tr(S'S).

    kernel        : string, optional
                    Kernel function: 'gaussian', 'bisquare', 'exponential'.
                    Default 'bisquare'.

    fixed         : boolean, optional
                    True for fixed (distance) bandwidth, False for adaptive
                    (nearest-neighbor). Default False.

    constant      : boolean, optional
                    True to include intercept (default), False to exclude it.

    spherical     : boolean, optional
                    True for spherical (lon/lat) coordinates with Haversine
                    distance; False for Euclidean. Default False.

    hat_matrix    : boolean, optional
                    True to compute and store covariate-specific hat matrices
                    R (n,n,k) and model hat matrix S (n,n). False (default)
                    to compute inference on the fly (less memory).

    name_x        : list of strings, optional
                    Names of independent variables for output display.

    Attributes
    ----------
    coords        : array-like
                    n*2, calibration coordinates.

    y             : array
                    n*1, dependent variable.

    X             : array
                    n*k, independent variables excluding the constant.

    selector      : Sel_BW
                    Bandwidth selector with search results.

    bw            : array-like
                    k bandwidth values (one per covariate). Order matches
                    the columns of X.

    family        : family object
                    Probability model (Gaussian).

    sigma2_v1     : boolean
                    Sigma^2 denominator form.

    kernel        : string
                    Kernel function name.

    fixed         : boolean
                    Whether bandwidths are fixed or adaptive.

    constant      : boolean
                    Whether intercept is included.

    spherical     : boolean
                    Whether spherical coordinates are used.

    n             : integer
                    Number of observations.

    k             : integer
                    Number of independent variables.

    mean_y        : float
                    Mean of y.

    std_y         : float
                    Standard deviation of y.

    fit_params    : dict
                    Parameters passed to fit().

    W             : list of array
                    k n*n spatial weight matrices, one per covariate.

    name_x        : list of strings
                    Names of independent variables for output display.

    Examples
    --------
    Basic MGWR calibration:

    >>> import libpysal as ps
    >>> from mgwr.gwr import MGWR
    >>> from mgwr.sel_bw import Sel_BW
    >>> data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
    >>> coords = list(zip(data.by_col('X'), data.by_col('Y')))
    >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
    >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
    >>> fb = np.array(data.by_col('PctFB')).reshape((-1,1))
    >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
    >>> X = np.hstack([fb, african_amer, rural])
    >>> X = (X - X.mean(axis=0)) / X.std(axis=0)
    >>> y = (y - y.mean(axis=0)) / y.std(axis=0)
    >>> selector = Sel_BW(coords, y, X, multi=True)
    >>> selector.search(multi_bw_min=[2])
    [92.0, 101.0, 136.0, 158.0]
    >>> model = MGWR(coords, y, X, selector, fixed=False, kernel='bisquare', sigma2_v1=True)
    >>> results = model.fit()
    >>> print(results.params.shape)
    (159, 4)

    """

    def __init__(self, coords, y, X, selector, sigma2_v1=True,
                 kernel='bisquare', fixed=False, constant=True,
                 spherical=False, hat_matrix=False, name_x=None):
        """
        Initialize class
        """
        self.selector = selector
        self.bws = self.selector.bw[0]  #final set of bandwidth
        self.bws_history = selector.bw[1]  #bws history in backfitting
        self.bw_init = self.selector.bw_init  #initialization bandiwdth
        self.family = Gaussian(
        )  # manually set since we only support Gassian MGWR for now
        GWR.__init__(self, coords, y, X, self.bw_init, family=self.family,
                     sigma2_v1=sigma2_v1, kernel=kernel, fixed=fixed,
                     constant=constant, spherical=spherical,
                     hat_matrix=hat_matrix)
        self.selector = selector
        self.sigma2_v1 = sigma2_v1
        self.points = None
        self.P = None
        self.offset = None
        self.exog_resid = None
        self.exog_scale = None
        self_fit_params = None
        self.name_x = name_x

    def _chunk_compute_R(self, chunk_id=0):
        """
        Compute one chunk of MGWR inference matrices R, CCT, and ENP_j.

        Processes a subset of observations to reduce peak memory usage.
        See Li & Fotheringham (2020), IJGIS.
        """
        n = self.n
        k = self.k
        n_chunks = self.n_chunks
        chunk_size = int(np.ceil(float(n / n_chunks)))
        ENP_j = np.zeros(self.k)
        CCT = np.zeros((self.n, self.k))

        chunk_index = np.arange(n)[chunk_id * chunk_size:(chunk_id + 1) *
                                   chunk_size]
        init_pR = np.zeros((n, len(chunk_index)))
        init_pR[chunk_index, :] = np.eye(len(chunk_index))
        pR = np.zeros((n, len(chunk_index),
                       k))  #partial R: n by chunk_size by k

        for i in range(n):
            wi = self._build_wi(i, self.bw_init).reshape(-1, 1)
            xT = (self.X * wi).T
            P = np.linalg.solve(xT.dot(self.X), xT).dot(init_pR).T
            pR[i, :, :] = P * self.X[i]

        err = init_pR - np.sum(pR, axis=2)  #n by chunk_size

        for iter_i in range(self.bws_history.shape[0]):
            for j in range(k):
                pRj_old = pR[:, :, j] + err
                Xj = self.X[:, j]
                n_chunks_Aj = n_chunks
                chunk_size_Aj = int(np.ceil(float(n / n_chunks_Aj)))
                for chunk_Aj in range(n_chunks_Aj):
                    chunk_index_Aj = np.arange(n)[chunk_Aj * chunk_size_Aj:(
                        chunk_Aj + 1) * chunk_size_Aj]
                    pAj = np.empty((len(chunk_index_Aj), n))
                    for i in range(len(chunk_index_Aj)):
                        index = chunk_index_Aj[i]
                        wi = self._build_wi(index, self.bws_history[iter_i, j])
                        xw = Xj * wi
                        pAj[i, :] = Xj[index] / np.sum(xw * Xj) * xw
                    pR[chunk_index_Aj, :, j] = pAj.dot(pRj_old)
                err = pRj_old - pR[:, :, j]

        for j in range(k):
            CCT[:, j] += ((pR[:, :, j] / self.X[:, j].reshape(-1, 1))**2).sum(
                axis=1)
        for i in range(len(chunk_index)):
            ENP_j += pR[chunk_index[i], i, :]

        if self.hat_matrix:
            return ENP_j, CCT, pR
        return ENP_j, CCT

    def fit(self, n_chunks=1, pool=None):
        """
        Fit the MGWR model and compute inference.

        Uses chunked computation to reduce memory footprint.
        See Li & Fotheringham (2020), IJGIS.

        Parameters
        ----------
        n_chunks      : integer, optional
                        Number of chunks for memory reduction (default 1).
                        With pool, total chunks = n_chunks * n_processes.
        pool          : multiprocessing.Pool or None, optional
                        Pool for parallel fitting (default None).

        Returns
        -------
        result        : MGWRResults
        """
        params = self.selector.params
        predy = np.sum(self.X * params, axis=1).reshape(-1, 1)

        try:
            from tqdm.autonotebook import tqdm  #progress bar
        except ImportError:

            def tqdm(x, total=0,
                     desc=''):  #otherwise, just passthrough the range
                return x

        if pool:
            self.n_chunks = pool._processes * n_chunks
            rslt = tqdm(
                pool.imap(self._chunk_compute_R, range(self.n_chunks)),
                total=self.n_chunks, desc='Inference')
        else:
            self.n_chunks = n_chunks
            rslt = map(self._chunk_compute_R,
                       tqdm(range(self.n_chunks), desc='Inference'))

        rslt_list = list(zip(*rslt))
        ENP_j = np.sum(np.array(rslt_list[0]), axis=0)
        CCT = np.sum(np.array(rslt_list[1]), axis=0)

        w = np.ones(self.n)
        if self.hat_matrix:
            R = np.hstack(rslt_list[2])
        else:
            R = None
        return MGWRResults(self, params, predy, CCT, ENP_j, w, R, self.name_x)


    def exact_fit(self):
        """
        Compute MGWR estimates using the closed-form exact solution.

        The backfitting algorithm in ``fit()`` converges to this result.
        Uses the block matrix formulation from Li & Fotheringham (2020),
        IJGIS, p. 4.

        Warning: Requires O(k * n^2) memory; not recommended for n > 5,000.

        Returns
        -------
        result        : MGWRResults
        """

        P = []
        Q = []
        I = np.eye(self.n)
        for j1 in range(self.k):
            Aj = GWR(self.coords,self.y,self.X[:,j1].reshape(-1,1),bw=self.bws[j1],hat_matrix=True,constant=False).fit().S
            Pj = []
            for j2 in range(self.k):
                if j1 == j2:
                    Pj.append(I)
                else:
                    Pj.append(Aj)
            P.append(Pj)
            Q.append([Aj])

        P = np.block(P)
        Q = np.block(Q)
        R = np.linalg.solve(P, Q)
        f = R.dot(self.y)

        params =  f/self.X.T.reshape(-1,1)
        params = params.reshape(-1,self.n).T

        R = np.stack(np.split(R,self.k),axis=2)
        ENP_j = np.trace(R, axis1=0, axis2=1)
        predy = np.sum(self.X * params, axis=1).reshape(-1, 1)
        w = np.ones(self.n)

        CCT = np.zeros((self.n,self.k))
        for j in range(self.k):
            CCT[:, j] = ((R[:, :, j] / self.X[:, j].reshape(-1, 1))**2).sum(axis=1)

        return MGWRResults(self, params, predy, CCT, ENP_j, w, R)


    def predict(self):
        '''
        Not implemented.
        '''
        raise NotImplementedError('N/A')


class MGWRResults(GWRResults):
    """
    Results container for a fitted MGWR model.

    Extends GWRResults with covariate-specific diagnostics (ENP_j,
    adj_alpha_j, bandwidths) that allow inference at multiple spatial scales.

    Parameters
    ----------
    model               : MGWR
                          Pointer to the fitted MGWR model.

    params              : array
                          n*k, estimated coefficients.

    predy               : array
                          n*1, predicted y values.

    CCT                 : array
                          n*k, scaled variance-covariance matrix.

    ENP_j               : array
                          k, effective number of parameters per covariate.

    w                   : array
                          n*1, IWLS weights (default None).

    R                   : array or None
                          n*n*k, covariate-specific hat matrices
                          (if hat_matrix=True).

    name_x              : list of strings, optional
                          Names of independent variables for output display.

    Attributes
    ----------
    (Inherits all attributes from GWRResults, plus the following:)

    ENP_j               : array
                          k, effective number of parameters per covariate.

    adj_alpha_j         : array
                          k*3, corrected alpha values per covariate at
                          90%, 95%, 99% confidence levels.

    W                   : list of array
                          k n*n spatial weight matrices, one per covariate.

    S                   : array or None
                          n*n model hat matrix (sum over R, if hat_matrix=True).

    R                   : array or None
                          n*n*k, covariate-specific hat matrices.
    """

    def __init__(self, model, params, predy, CCT, ENP_j, w, R, x_name=None):
        """
        Initialize class
        """
        self.ENP_j = ENP_j
        self.R = R
        GWRResults.__init__(self, model, params, predy, None, CCT, None, w)
        if model.hat_matrix:
            self.S = np.sum(self.R, axis=2)
        self.predy = predy
        self.x_name = x_name

    @cache_readonly
    def tr_S(self):
        return np.sum(self.ENP_j)

    @cache_readonly
    def W(self):
        Ws = []
        for bw_j in self.model.bws:
            W = np.array(
                [self.model._build_wi(i, bw_j) for i in range(self.n)])
            Ws.append(W)
        return Ws

    @cache_readonly
    def adj_alpha_j(self):
        """
        Return covariate-specific corrected alpha values for multiple testing.

        Returns a k*3 array with corrected values at 90%, 95%, and 99%
        confidence levels per covariate. Based on da Silva & Fotheringham (2015).
        """
        alpha = np.array([.1, .05, .001])
        pe = np.array(self.ENP_j).reshape((-1, 1))
        p = 1.
        return (alpha * p) / pe

    def critical_tval(self, alpha=None):
        """
        Return covariate-specific critical t-values for the given alpha.

        Parameters
        ----------
        alpha           : scalar, optional
                          Significance level. Default None uses
                          the 95% adjusted alpha per covariate.

        Returns
        -------
        critical        : array
                          k critical t-values, one per covariate.
        """
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha_j[:, 1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):
        """
        Return t-values with non-significant entries set to zero (covariate-specific).

        Uses the adjusted alpha at 95% CI per covariate by default.

        Parameters
        ----------
        critical_t      : scalar or array, optional
                          Direct critical t-value threshold(s).
        alpha           : scalar, optional
                          Significance level.

        Returns
        -------
        filtered        : array
                          n*k t-values; entries below threshold are zeroed.
        """
        n = self.n
        if critical_t is not None:
            critical = np.array(critical_t)
        elif alpha is not None and critical_t is None:
            critical = self.critical_tval(alpha=alpha)
        elif alpha is None and critical_t is None:
            critical = self.critical_tval()

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    @cache_readonly
    def RSS(self):
        raise NotImplementedError(
            'Not yet implemented for multiple bandwidths')

    @cache_readonly
    def TSS(self):
        raise NotImplementedError(
            'Not yet implemented for multiple bandwidths')

    @cache_readonly
    def localR2(self):
        raise NotImplementedError(
            'Not yet implemented for multiple bandwidths')

    @cache_readonly
    def y_bar(self):
        raise NotImplementedError(
            'Not yet implemented for multiple bandwidths')

    @cache_readonly
    def predictions(self):
        raise NotImplementedError('Not yet implemented for MGWR')

    def get_bws_intervals(self, selector, level=0.95):
        """
        Compute covariate-specific bandwidth confidence intervals for MGWR.

        Based on Akaike weights from the backfitting search history.
        See Li et al. (2020), Annals of the AAG.

        Parameters
        ----------
        selector : Sel_BW
            The bandwidth selector used to fit the MGWR model.
        level : float
            Confidence level (default 0.95).

        Returns
        -------
        list of tuple : [(lower, upper), ...], one per covariate.
        """
        intervals = []
        try:
            import pandas as pd
        except ImportError:
            return

        for j in range(self.k):
            #Get AICcs and associated bw from the last iteration of back-fitting and make a DataFrame
            aiccs = pd.DataFrame(list(zip(*selector.sel_hist[-self.k+j]))[1],columns=["aicc"])
            aiccs['bw'] = list(zip(*selector.sel_hist[-self.k+j]))[0]
            #Sort DataFrame by the AICc values
            aiccs = aiccs.sort_values(by=['aicc'])
            #Calculate delta AICc
            d_aic_ak = aiccs.aicc - aiccs.aicc.min()
            #Calculate AICc weights
            w_aic_ak = np.exp(-0.5*d_aic_ak) / np.sum(np.exp(-0.5*d_aic_ak))
            aiccs['w_aic_ak'] = w_aic_ak/np.sum(w_aic_ak)
            #Calculate cum. AICc weights
            aiccs['cum_w_ak'] = aiccs.w_aic_ak.cumsum()
            #Find index where the cum weights above p-val
            index = len(aiccs[aiccs.cum_w_ak < level]) + 1
            #Get bw boundaries
            interval = (aiccs.iloc[:index,:].bw.min(),aiccs.iloc[:index,:].bw.max())
            intervals += [interval]
        return intervals


    def local_collinearity(self):
        """
        Compute local multicollinearity diagnostics for the MGWR design matrix.

        Uses covariate-specific bandwidths when constructing the weighted
        design matrix at each location.

        Returns
        -------
        local_CN : array (n, 1)
            Local condition numbers.
        VDP      : array (n, p)
            Local variance-decomposition proportions.
        """
        x = self.X
        w = self.W
        nvar = x.shape[1]
        nrow = self.n
        vdp_idx = np.ndarray((nrow, nvar))
        vdp_pi = np.ndarray((nrow, nvar, nvar))

        for i in range(nrow):
            xw = np.zeros((x.shape))
            for j in range(nvar):
                wi = w[j][i]
                sw = np.sum(wi)
                wi = wi / sw
                xw[:, j] = x[:, j] * wi

            sxw = np.sqrt(np.sum(xw**2, axis=0))
            sxw = np.transpose(xw.T / sxw.reshape((nvar, 1)))
            svdx = np.linalg.svd(sxw)
            vdp_idx[i, ] = svdx[1][0] / svdx[1]

            phi = np.dot(svdx[2].T, np.diag(1 / svdx[1]))
            phi = np.transpose(phi**2)
            pi_ij = phi / np.sum(phi, axis=0)
            vdp_pi[i, :, :] = pi_ij

        local_CN = vdp_idx[:, nvar - 1].reshape((-1, 1))
        VDP = vdp_pi[:, nvar - 1, :]

        return local_CN, VDP

    def spatial_variability(self, selector, n_iters=1000, seed=None):
        """
        Monte Carlo test for spatial variability of each coefficient surface.

        Randomly permutes spatial coordinates to generate the null
        distribution of parameter surface standard deviations, then
        computes a pseudo p-value for each covariate.

        WARNING: This test is computationally very demanding.

        Parameters
        ----------
        selector        : Sel_BW
                          The bandwidth selector used for the GWR model.
        n_iters         : int, optional
                          Number of Monte Carlo permutations (default 1000).
        seed            : int, optional
                          Random seed for reproducibility (default 5536).

        Returns
        -------
        p_vals          : list of float
                          Pseudo p-values for each parameter surface.
        """
        temp_sel = copy.deepcopy(selector)

        if seed is None:
            np.random.seed(5536)
        else:
            np.random.seed(seed)

        search_params = temp_sel.search_params

        if self.model.constant:
            X = self.X[:, 1:]
        else:
            X = self.X

        init_sd = np.std(self.params, axis=0)
        SDs = []

        try:
            from tqdm.auto import tqdm  # if they have it, let users have a progress bar
        except ImportError:

            def tqdm(x, desc=''):  # otherwise, just passthrough the range
                return x

        for x in tqdm(range(n_iters), desc='Testing'):
            temp_coords = np.random.permutation(self.model.coords)
            temp_sel.coords = temp_coords
            temp_sel.search(**search_params)
            temp_params = temp_sel.params
            temp_sd = np.std(temp_params, axis=0)
            SDs.append(temp_sd)

        p_vals = (np.sum(np.array(SDs) > init_sd, axis=0) / float(n_iters))
        return p_vals

    def summary(self, as_str: bool=False) -> Optional[str]:
        """
        Print or return a formatted MGWR model summary.

        Includes global regression results, covariate-specific bandwidths
        and ENP_j, diagnostic information, and parameter estimate statistics.

        Parameters
        ----------
        as_str        : bool, optional
                        If True, return the summary as a string.

        Returns
        -------
        summary        : Optional[str]
                        Summary string when ``as_str=True``, otherwise None.
        """
        summary = summaryModel(self) + summaryGLM(self) + summaryMGWR(self)

        if as_str:
            return summary

        print(summary)
        return



class STWR(GLM):
    """
    Spatiotemporal Weighted Regression (STWR).

    Extends GWR into the temporal dimension by borrowing observations from
    past time stages to improve model fit for the latest (current) stage.
    Uses a novel spatiotemporal kernel that combines a spatial kernel
    (Gaussian or bi-square) with a sigmoid-form temporal kernel. The
    temporal weight is based on the rate of value change of nearby
    observations across time intervals.

    STWR is built on the GLM framework. Call ``fit()`` to calibrate the
    model and obtain an STWRResults object.

    Parameters
    ----------
    coords_list         : list of array
                          Coordinates from each time stage. Each array is
                          n*2 (x, y). Ordered from oldest to newest.

    y_list              : list of array
                          Dependent variable from each time stage.
                          Each array is n*1. Ordered oldest to newest.

    X_list              : list of array
                          Independent variables from each time stage.
                          Each array is n*k, excluding the constant.
                          Ordered oldest to newest.

    tick_times_intervel : list of float
                          Time intervals between consecutive stages,
                          ordered from first to last. The first element
                          is defined as 0.0 (no interval before the first
                          stage).

    sita                : float, optional
                          Slope of the linear relationship between spatial
                          bandwidth change and time. Default 0 means the
                          spatial bandwidth stays constant over time.
                          A positive value shrinks the bandwidth as time
                          intervals increase.

    gwr_bw0             : int, optional
                          Initial spatial bandwidth (number of nearest
                          neighbors). The actual spatial bandwidth distance
                          for each regression point is the maximum distance
                          to its ``gwr_bw0``-th nearest neighbor.

    tick_nums           : int, optional
                          Number of past time stages to use for calibration
                          (default 1, i.e., only the latest stage).

    dspal_mat_list      : list of array, optional
                          Pre-computed spatial distance matrices.
                          Each matrix in the list corresponds to one time
                          stage. Default None (computed automatically).

    sorted_dspal_list   : list of array, optional
                          Row-wise sorted spatial distance matrices, used
                          for bandwidth truncation. Default None.

    d_tmp_list          : list of array, optional
                          Pre-computed temporal distance matrices. Default None.

    dspmat              : array, optional
                          Horizontally concatenated spatial distance matrix
                          over all time stages. Default None.

    dtmat               : array, optional
                          Horizontally concatenated temporal distance matrix
                          over all time stages. Default None.

    alpha               : float, optional
                          Weight balancing spatial vs. temporal effect.
                          Range [0, 1]. 0 = spatial only, 1 = temporal only.
                          Default 0.3.

    family              : family object, optional
                          Probability model from ``spglm.family``.
                          Default Gaussian().

    offset              : array, optional
                          n*1, offset variable (e.g., population at risk
                          for Poisson models). Default None (all 1.0).

    sigma2_v1           : boolean, optional
                          Sigma^2 denominator form:
                          True = n - tr(S) (default),
                          False = n - 2*tr(S) + tr(S'S).

    kernel              : string, optional
                          Spatiotemporal kernel function:
                          'spt_gaussian', 'spt_bisquare', 'spt_exponential'.
                          Default 'spt_bisquare'.

    fixed               : boolean, optional
                          True for fixed bandwidth, False for adaptive
                          (default False).

    constant            : boolean, optional
                          True to include intercept (default).

    spherical           : boolean, optional
                          True for spherical (lon/lat) coordinates,
                          False for projected coordinates (default).

    compress            : boolean, optional
                          Whether to use memory compression for the
                          weight matrix during model fitting (default True).

    recorded            : int, optional
                          If 1, store the computed temporal distance
                          matrices for later use in prediction.
                          Default 0 (don't store).

    Examples
    --------
    Basic STWR calibration (conceptual outline):

    >>> import pandas as pd
    >>> import numpy as np
    >>> from stwr import STWR
    >>> from stwr.sel_bw import Sel_Spt_BW

    >>> # Prepare data lists for each time stage
    >>> cal_coords_list = []   # list of coordinate arrays per stage
    >>> cal_y_list = []        # list of y arrays per stage
    >>> cal_X_list = []        # list of X arrays per stage
    >>> delt_stwr_intervel = [0.0]  # time intervals between stages

    >>> # ... load and organize panel data by time stage ...

    >>> # Bandwidth selection
    >>> stwr_selector = Sel_Spt_BW(cal_coords_list, cal_y_list, cal_X_list,
    ...                             delt_stwr_intervel, spherical=True)
    >>> optalpha, optsita, opt_btticks, opt_gwr_bw0 = stwr_selector.search()

    >>> # Model fitting
    >>> stwr_model = STWR(cal_coords_list, cal_y_list, cal_X_list,
    ...                   delt_stwr_intervel, optsita, opt_gwr_bw0,
    ...                   tick_nums=opt_btticks, alpha=optalpha,
    ...                   spherical=True, recorded=1)
    >>> stwr_results = stwr_model.fit()
    >>> print(stwr_results.summary())
    """
    def __init__(self, coords_list,y_list, X_list,
                 tick_times_intervel =None,sita = None,gwr_bw0 = None,
                 tick_nums = 1,dspal_mat_list = None,sorted_dspal_list = None,
                 d_tmp_list = None,dspmat = None,dtmat=None,alpha =0.3,
                 family=Gaussian(), offset=None,
                 sigma2_v1=True, kernel='spt_bisquare',
                 fixed=False, constant=True,spherical=False,compress = True,recorded = 0):
        self.tick_nums = tick_nums
        self.y_list = y_list
        self.X_list = X_list
        self.coords_list = coords_list
        self.compress = compress #self.mbpred
        self.recorded = recorded        
        y_arr = np.asarray(self.y_list[-1]).reshape((-1,1))
        X_arr = self.X_list[-1]
        for i in range(self.tick_nums-1):
             X_arr = np.vstack((X_arr,self.X_list[-(2+i)]))
             y_arr_tick =np.asarray(self.y_list[-(2+i)]).reshape((-1,1))
             y_arr = np.vstack((y_arr,y_arr_tick))                   
        GLM.__init__(self, y_arr, X_arr, family, constant=constant)        
        self.y_arr = y_arr
        self.X_arr = X_arr
        self.constant = constant
        self.sigma2_v1 = sigma2_v1
        self.gwr_bw0 = gwr_bw0
        self.kernel = kernel
        self.fixed = fixed
        self.tick_times_intervel = tick_times_intervel  
        self.sita = sita  
        if offset is None:
            self.offset = np.ones((self.n, 1))
        else:
            self.offset = offset * 1.0        
        self.fit_params = {}
        self.points_list = None
        self.exog_scale = None
        self.exog_resid = None
        self.P = None
        self.alpha = alpha
        
        self.dspal_mat_list = dspal_mat_list
        self.sorted_dspal_list = sorted_dspal_list
        self.d_tmp_list = d_tmp_list
        self.dspmat = dspmat
        self.dtmat = dtmat
        self.spherical = spherical        
        self.W = self._build_spt_W(fixed, kernel,coords_list,self.y_list,tick_times_intervel,sita,tick_nums,self.points_list,self.alpha,gwr_bw0 = self.gwr_bw0)    
    
    def _build_spt_W(self, fixed, kernel, coords_list,y_list,tick_times_intervel,sita,tick_nums,points_list=None,alpha=0.3,gwr_bw0=None,pred = False):
        
        if self.recorded == 1:
             W,self.dtmat,self.d_tmp_list = ak[kernel](coords_list,y_list,tick_times_intervel,sita,tick_nums,
                                           gwr_bw0,self.dspal_mat_list,
                                           self.sorted_dspal_list,
                                           self.d_tmp_list,
                                           self.dspmat,self.dtmat,points_list=points_list,alpha=alpha,
                                           spherical=self.spherical,prediction = pred,rcdtype=self.recorded)
             return W
        else:  
            if self.compress:
                if fixed:
                    try:
                        W = fk[kernel](coords_list,y_list,tick_times_intervel,sita,tick_nums,
                                       gwr_bw0,self.dspal_mat_list,
                                       self.sorted_dspal_list,
                                       self.d_tmp_list,
                                       self.dspmat,self.dtmat,points_list =points_list ,alpha=alpha,
                                       spherical=self.spherical,prediction = pred)
                    except BaseException:
                        raise  # TypeError('Unsupported kernel function  ', kernel)
                else:
                    try:
                        W = ak[kernel](coords_list,y_list,tick_times_intervel,sita,tick_nums,
                                       gwr_bw0,self.dspal_mat_list,
                                       self.sorted_dspal_list,
                                       self.d_tmp_list,
                                       self.dspmat,self.dtmat,points_list =points_list,alpha=alpha,
                                       spherical=self.spherical,prediction = pred)
                    except BaseException:
                        raise  # TypeError('Unsupported kernel function  ', kernel)
            else:
                if fixed:
                    try:
                        W = fk[kernel](coords_list,y_list,tick_times_intervel,sita,tick_nums,
                                       gwr_bw0,self.dspal_mat_list,
                                       self.sorted_dspal_list,
                                       self.d_tmp_list,
                                       self.dspmat,self.dtmat,points_list =points_list ,alpha=alpha,mbpred=True,
                                       spherical=self.spherical,prediction = pred)
                    except BaseException:
                        raise  # TypeError('Unsupported kernel function  ', kernel)
                else:
                    try:
                        W = ak[kernel](coords_list,y_list,tick_times_intervel,sita,tick_nums,
                                       gwr_bw0,self.dspal_mat_list,
                                       self.sorted_dspal_list,
                                       self.d_tmp_list,
                                       self.dspmat,self.dtmat,points_list =points_list,alpha=alpha,mbpred=True,
                                       spherical=self.spherical,prediction = pred)
                    except BaseException:
                        raise  # TypeError('Unsupported kernel function  ', kernel)
            return W
    def fit(self, ini_params=None, tol=1.0e-5, max_iter=20,
            solve='iwls', searching=False):
        """
        Fit the STWR model.

        Can operate in two modes: a lightweight ``searching`` mode for
        bandwidth selection (returns STWRResultsLite), or full mode
        (returns STWRResults).

        Parameters
        ----------
        ini_params     : array, optional
                        Initial coefficient values. Default None (auto-computed).
        tol            : float, optional
                        Convergence tolerance (default 1.0e-5).
        max_iter       : int, optional
                        Maximum IWLS iterations (default 20).
        solve          : string, optional
                        Estimation method; only 'iwls' is supported.
        searching      : bool, optional
                        If True, compute minimal diagnostics for bandwidth
                        selection. If False, compute full results.

        Returns
        -------
        result         : STWRResults or STWRResultsLite
        """      
        self.fit_params['ini_params'] = ini_params
        self.fit_params['tol'] = tol
        self.fit_params['max_iter'] = max_iter
        self.fit_params['solve'] = solve
        if solve.lower() == 'iwls':
            m = self.W.shape[0]
            if searching:
                resid = np.zeros((m, 1))
                influ = np.zeros((m, 1))
                for i in range(m):
                    wi = self.W[i].reshape((-1, 1))
                    if isinstance(self.family, Gaussian): 
                        betas, inv_xtx_xt = _compute_betas_gwr(
                                 self.y, self.X, wi)
                        influ[i] = np.dot(self.X[i], inv_xtx_xt[:, i])
                        predy = np.dot(self.X[i], betas)[0]
                        resid[i] = self.y[i] - predy
                    elif isinstance(self.family, (Poisson, Binomial)):
                        rslt = iwls(self.y, self.X, self.family,
                                    self.offset, None, ini_params, tol,
                                    max_iter, wi=wi)
                        inv_xtx_xt = rslt[5]
                        influ[i] = np.dot(self.X[i], inv_xtx_xt[:, i]) * \
                                   rslt[3][i][0]
                        predy = rslt[1][i]
                        resid[i] = self.y[i] - predy        
                return STWRResultsLite(self, resid, influ,m)
            else:
                params = np.zeros((m, self.k))
                predy = np.zeros((m, 1))
                w = np.zeros((m, 1))
                S = np.zeros((m, self.n))
                CCT = np.zeros((m, self.k))
                for i in range(m):
                    wi = self.W[i].reshape((-1, 1))#[:m,]
                    rslt = iwls(self.y, self.X, self.family,
                                self.offset, None, ini_params, tol,
                                max_iter, wi=wi)
                    params[i, :] = rslt[0].T
                    predy[i] = rslt[1][i]
                    w[i] = rslt[3][i]
                    S[i] = np.dot(self.X[i], rslt[5])
                    CCT[i] = np.diag(np.dot(rslt[5], rslt[5].T))
                return STWRResults(self, params, predy, S, CCT,w,compress = self.compress)

    def predict(self, points_list, P, exog_scale=None, exog_resid=None,
                fit_params={}):
        """
        Predict the dependent variable at un-sampled locations.

        Currently supports prediction only at points from the same time
        stage as the latest observed stage. Temporal weights for prediction
        points are estimated via IDW interpolation from nearby observed
        points' temporal distance values.

        Parameters
        ----------
        points_list     : list of array
                          Coordinates of prediction locations (must be a
                          single-element list for the latest time stage).
        P               : list of array
                          Independent variables at prediction locations
                          (excluding constant). Must match points_list length.
        exog_scale      : scalar, optional
                          Estimated scale from training data.
        exog_resid      : array, optional
                          Estimated residuals from training data.
        fit_params      : dict, optional
                          Additional parameters passed to ``fit()``.

        Returns
        -------
        result          : STWRResults
        """
        if (exog_scale is None) & (exog_resid is None):
            train_gwr = self.fit(**fit_params)
            self.exog_scale = train_gwr.scale
            self.exog_resid = train_gwr.resid_response
        elif (exog_scale is not None) & (exog_resid is not None):
            self.exog_scale = exog_scale
            self.exog_resid = exog_resid
        else:
            raise InputError('exog_scale and exog_resid must both either be'
                             'None or specified')
        self.points_list = points_list
        pointslist_nums = len(self.points_list)
        if pointslist_nums != 1:
            raise InputError('Currently unsupported to predict at multiple time datasets')

        if self.constant:
            P[0] = np.hstack([np.ones((len(P[0]), 1)), P[0]])
            self.P = P[0]
        else:
            self.P = P[0]

        # Disable recorded mode during prediction (temporal matrices
        # must be re-computed for prediction locations).
        self.recorded = 0

        # Build spatiotemporal weights for prediction locations.
        # Currently only supports prediction at the latest time stage.
        # Future work: incorporate the rate of variation to predict
        # temporal weights for future time stages.
        self.W = self._build_spt_W(
            self.fixed,
            self.kernel,
            self.coords_list,
            self.y_list,
            tick_times_intervel=self.tick_times_intervel,
            sita=self.sita,
            tick_nums=self.tick_nums,
            gwr_bw0=self.gwr_bw0,
            points_list=self.points_list,
            alpha=self.alpha,
            pred=True)
        stwr = self.fit(**fit_params)
        return stwr          
    @cache_readonly
    def df_model(self):
        return None

    @cache_readonly
    def df_resid(self):
        return None

class STWRResultsLite(object):
    """
    Lightweight STWR results for bandwidth selection.

    Computes only the minimal diagnostics (influence, residuals,
    log-likelihood) needed for bandwidth optimization.

    Parameters
    ----------
    model   : STWR
              Pointer to the STWR model.
    resid   : array
              n*1, response residuals.
    influ   : array
              n*1, diagonal of the hat matrix.
    m       : int
              Number of observations in the latest time stage.
    """

    def __init__(self, model, resid, influ, m):
        self.m = m
        self.y = model.y[:m]
        self.family = model.family
        self.n = m
        self.influ = influ
        self.resid_response = resid
        self.model = model

    @cache_readonly
    def tr_S(self):
        """Trace of the hat matrix."""
        return np.sum(self.influ)

    @cache_readonly
    def llf(self):
        """Log-likelihood of the model."""
        return self.family.loglike(self.y, self.mu)

    @cache_readonly
    def mu(self):
        """Predicted mean response."""
        return self.y - self.resid_response

    @cache_readonly
    def resid_ss(self):
        """Residual sum of squares."""
        u = self.resid_response.flatten()
        return np.dot(u, u.T)


class STWRResults(GLMResults):
    """
    Results container for a fitted STWR model.

    Provides access to parameter estimates, diagnostics, and inference
    statistics for spatiotemporal weighted regression. Most attributes
    are computed lazily via ``@cache_readonly``.

    Parameters
    ----------
    model       : STWR
                  Pointer to the fitted STWR model.
    params      : array
                  n*k, estimated coefficients.
    predy       : array
                  n*1, predicted y values.
    S           : array
                  n*n, hat matrix.
    CCT         : array
                  n*k, unscaled variance-covariance matrix.
    w           : array, optional
                  n*1, IWLS weights.
    compress    : bool, optional
                  Whether to compress the model copy to memory.
    """
    def __init__(self, model, params, predy, S, CCT,w=None,compress = False):        
        self.model_copy = None  
        if(compress):
            self.model_copy =copy.deepcopy(model) 
            m = self.model_copy.W.shape[0]
            self.y = self.model_copy.y[:m]
            self.n = m
            self.y_ful = self.model_copy.y
            self.X_ful = self.model_copy.X
            self.model_copy.y = self.model_copy.y[:m]
            self.model_copy.X = self.model_copy.X[:m] 
        else:
           self.model_copy = model 
        GLMResults.__init__(self, self.model_copy, params, predy, w)
        self.W = self.model_copy.W       
        self.offset = self.model_copy.offset
        if w is not None:
            self.w = w
        self.predy = predy
        self.S = S
        self.CCT = self.cov_params(CCT, self.model_copy.exog_scale)
        self._cache = {}

    @cache_readonly
    def resid_ss(self):
        if self.model_copy.points_list is not None:
            raise NotImplementedError('Not available for STWR prediction')
        else:
            u = self.resid_response.flatten()
        return np.dot(u, u.T)

    @cache_readonly
    def scale(self, scale=None):
        if isinstance(self.family, Gaussian):
            scale = self.sigma2
        else:
            scale = 1.0
        return scale

    def cov_params(self, cov, exog_scale=None):
        """
        Returns scaled covariance parameters

        Parameters
        ----------
        cov         : array
                      estimated covariance parameters

        Returns
        -------
        Scaled covariance parameters

        """
        if exog_scale is not None:
            return cov * exog_scale
        else:
            return cov * self.scale

    @cache_readonly
    def tr_S(self):
        """
        Return the trace of the hat matrix S (weighted).
        """
        return np.trace(self.S * self.w)

    @cache_readonly
    def tr_STS(self):
        """
        Return the trace of S'S (weighted).
        """
        return np.trace(np.dot(self.S.T * self.w, self.S * self.w))

    @cache_readonly
    def ENP(self):
        """
        Return the effective number of parameters.

        When sigma2_v1=True: ENP = tr(S).
        When sigma2_v1=False: ENP = 2*tr(S) - tr(S'S).
        See Yu et al. (2019), Inference in Multiscale GWR.
        """
        if self.model_copy.sigma2_v1:
            return self.tr_S
        else:
            return 2 * self.tr_S - self.tr_STS

    @cache_readonly
    def y_bar(self):
        """
        Return the spatially weighted mean of y at each location.
        """
#        if self.model_copy.points is not None:
#            n = len(self.model_copy.points)
#        else:
#            n = self.n
#        off = self.offset.reshape((-1, 1))
#        arr_ybar = np.zeros(shape=(self.n, 1))
#        for i in range(n):
#            w_i = np.reshape(np.array(self.W[i]), (-1, 1))
#            sum_yw = np.sum(self.y.reshape((-1, 1)) * w_i)
#            arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i * off)
#        return arr_ybar
        if self.model_copy.points_list is not None:
            n = len(self.model_copy.points_list[-1])
        else:
            n =self.W.shape[0] 
        off = self.offset.reshape((-1, 1))
        arr_ybar = np.zeros(shape=(self.n, 1))
        for i in range(n):         
            w_i = self.W[i].reshape((-1, 1))
            sum_yw = np.sum(self.y_ful.reshape((-1, 1)) * w_i)
            arr_ybar[i] = 1.0 * sum_yw / np.sum(w_i * off)
        return arr_ybar[:n]

    @cache_readonly
    def TSS(self):
        """
        Return the geographically weighted total sum of squares.

        Reference: Fotheringham et al. (2002), eq. (9.9), p. 215.
        """
        #        if self.model_copy.points is not None:
#            n = len(self.model_copy.points)  
#        else:
#                    n = self.n         
        if  self.model_copy.points_list is not None:
            n = len(self.model_copy.points_list[-1])
        else:
            n = self.W.shape[0]  
            
        TSS = np.zeros(shape=(n, 1))
        for i in range(n):
            TSS[i] = np.sum(np.reshape(np.array(self.W[i]), (-1, 1)) *
                            (self.y_ful.reshape((-1, 1)) - self.y_bar[i])**2)
        return TSS

    @cache_readonly
    def RSS(self):
        """
        Return the geographically weighted residual sum of squares.

        Reference: Fotheringham et al. (2002), eq. (9.10), p. 215.
        """
        #        if self.model_copy.points is not None:
#            n = len(self.model_copy.points)
#            resid = self.model_copy.exog_resid.reshape((-1, 1))
#        else:
#            n = self.n
#            resid = self.resid_response.reshape((-1, 1))
        if self.model_copy.points_list is not None:
            n = len(self.model_copy.points_list[-1])
            resid = self.model_copy.exog_resid.reshape((-1, 1))
        else:
            n = self.W.shape[0]
            resid = self.resid_response.reshape((-1, 1))
        
        RSS = np.zeros(shape=(n, 1))      
        for i in range(n):      
            RSS[i] = np.sum(self.W[i].reshape((-1, 1))[:n]
                            * resid**2)
        return RSS

    @cache_readonly
    def localR2(self):
        """
        Return local R-squared values at each location (Gaussian only).

        Reference: Fotheringham et al. (2002), eq. (9.8), p. 215.
        """
        if isinstance(self.family, Gaussian):
            return (self.TSS - self.RSS) / self.TSS
        else:
            raise NotImplementedError('Only applicable to Gaussian')

    @cache_readonly
    def sigma2(self):
        """
        Return the residual variance, corrected for ENP.

        If sigma2_v1 is True: denominator = n - tr(S).
        If sigma2_v1 is False: denominator = n - 2*tr(S) + tr(S'S).
        """
        if self.model_copy.sigma2_v1:
            return (self.resid_ss / (self.n - self.tr_S))
        else:
            return self.resid_ss / (self.n - 2.0 * self.tr_S + self.tr_STS)

    @cache_readonly
    def std_res(self):
        """
        Return the standardized residuals.

        Reference: Fotheringham et al. (2002), eq. (9.7), p. 215.
        """
        return self.resid_response.reshape(
            (-1, 1)) / (np.sqrt(self.scale * (1.0 - self.influ)))

    @cache_readonly
    def bse(self):
        """
        Return the standard errors of the parameter estimates.

        Reference: Fotheringham et al. (2002), eqs. (2.15) and (2.21), p. 215.
        """
        return np.sqrt(self.CCT)

    @cache_readonly
    def influ(self):
        """
        Return the diagonal of the hat matrix (leverage values).
        """
        return np.reshape(np.diag(self.S), (-1, 1))

    @cache_readonly
    def cooksD(self):
        """
        Return Cook's distance for each observation.

        Reference: Fotheringham et al. (2002), eq. (9.11), p. 216.
        Note: p in (9.11) is tr(S), the effective number of parameters.
        """
        return self.std_res**2 * self.influ / (self.tr_S * (1.0 - self.influ))

    @cache_readonly
    def deviance(self):
        off = self.offset.reshape((-1, 1)).T
        y = self.y
        ybar = self.y_bar
        if isinstance(self.family, Gaussian):
            raise NotImplementedError(
                'deviance not currently used for Gaussian')
        elif isinstance(self.family, Poisson):
            dev = np.sum(
                2.0 * self.W * (y * np.log(y / (ybar * off)) - (y - ybar * off)), axis=1)
        elif isinstance(self.family, Binomial):
            dev = self.family.deviance(self.y, self.y_bar, self.W, axis=1)
        return dev.reshape((-1, 1))

    @cache_readonly
    def resid_deviance(self):
        if isinstance(self.family, Gaussian):
            raise NotImplementedError(
                'deviance not currently used for Gaussian')
        else:
            off = self.offset.reshape((-1, 1)).T
            y = self.y
            ybar = self.y_bar
            global_dev_res = ((self.family.resid_dev(self.y, self.mu))**2)
            dev_res = np.repeat(global_dev_res.flatten(), self.n)
            dev_res = dev_res.reshape((self.n, self.n))
            dev_res = np.sum(dev_res * self.W.T, axis=0)
            return dev_res.reshape((-1, 1))

    @cache_readonly
    def pDev(self):
        """
        Local percentage of deviance accounted for. Described in the GWR4
        manual. Equivalent to 1 - (deviance/null deviance)
        """
        if isinstance(self.family, Gaussian):
            raise NotImplementedError('Not implemented for Gaussian')
        else:
            return 1.0 - (self.resid_deviance / self.deviance)

    @cache_readonly
    def adj_alpha(self):
        """
        Return corrected critical alpha values for multiple hypothesis testing.

        Corrected for 90%, 95%, and 99% confidence levels based on
        da Silva & Fotheringham (2015).
        """
        alpha = np.array([.1, .05, .001])
        pe = self.ENP
        p = self.k
        return (alpha * p) / pe

    def critical_tval(self, alpha=None):
        """
        Return the critical t-value for the given alpha or the 95% adjusted alpha.

        Parameters
        ----------
        alpha           : scalar, optional
                          Significance level. Default None uses the 95% CI adjusted alpha.

        Returns
        -------
        critical        : scalar
                          Two-tailed critical t-value.
        """
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha[1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical

    def filter_tvals(self, critical_t=None, alpha=None):
        """
        Return t-values with non-significant entries set to zero.

        Parameters
        ----------
        critical_t      : scalar, optional
                          Direct critical t-value threshold.
        alpha           : scalar, optional
                          Significance level used to derive the critical value.

        Returns
        -------
        filtered        : array
                          n*k t-values; entries below the critical threshold
                          are set to zero.
        """
        n = self.n
        if critical_t is not None:
            critical = critical_t
        else:
            critical = self.critical_tval(alpha=alpha)

        subset = (self.tvalues < critical) & (self.tvalues > -1.0 * critical)
        tvalues = self.tvalues.copy()
        tvalues[subset] = 0
        return tvalues

    @cache_readonly
    def df_model(self):
        return self.n - self.tr_S

    @cache_readonly
    def df_resid(self):
        return self.n - 2.0 * self.tr_S + self.tr_STS

    @cache_readonly
    def normalized_cov_params(self):
        return None

    @cache_readonly
    def resid_pearson(self):
        return None

    @cache_readonly
    def resid_working(self):
        return None

    @cache_readonly
    def resid_anscombe(self):
        return None

    @cache_readonly
    def pearson_chi2(self):
        return None

    @cache_readonly
    def null(self):
        return None

    @cache_readonly
    def llnull(self):
        return None

    @cache_readonly
    def null_deviance(self):
        return None

    @cache_readonly
    def R2(self):
        """
        Return the global R-squared value (Gaussian models only).
        """
        if isinstance(self.family, Gaussian):
            TSS = np.sum((self.y.reshape((-1, 1)) -
                          np.mean(self.y.reshape((-1, 1))))**2)
            RSS = np.sum((self.y.reshape((-1, 1)) -
                          self.predy.reshape((-1, 1)))**2)
            return 1 - (RSS / TSS)
        else:
            raise NotImplementedError('Only available for Gaussian GWR')

    @cache_readonly
    def aic(self):
        return get_AIC(self)

    @cache_readonly
    def aicc(self):
        return get_AICc(self)

    @cache_readonly
    def bic(self):
        return get_BIC(self)

    @cache_readonly
    def D2(self):
        return None

    @cache_readonly
    def adj_D2(self):
        return None

    @cache_readonly
    def pseudoR2(self):
        return None

    @cache_readonly
    def adj_pseudoR2(self):
        return None

    @cache_readonly
    def pvalues(self):
        return None

    @cache_readonly
    def conf_int(self):
        return None

    @cache_readonly
    def use_t(self):
        return None

    def local_collinearity(self):
        """
        Compute local multicollinearity diagnostics for the STWR design matrix.

        Uses the spatiotemporal weight matrix at each location to construct
        the weighted design matrix.

        Returns
        -------
        corr_mat    : array (n, n_corr)
            Local correlation coefficients.
        vifs_mat    : array (n, p - 1)
            Local variance inflation factors (excludes constant).
        local_CN    : array (n, 1)
            Local condition numbers.
        VDP         : array (n, p)
            Local variance-decomposition proportions.

        Notes
        -----
        n = number of calibration locations,
        p = number of explanatory variables.
        """
        x = self.X
        w = self.W
        nvar = x.shape[1]
        nrow = len(w)
        if self.model_copy.constant:
            ncor = (((nvar - 1)**2 + (nvar - 1)) / 2) - (nvar - 1)
            jk = list(combo(range(1, nvar), 2))
        else:
            ncor = (((nvar)**2 + (nvar)) / 2) - nvar
            jk = list(combo(range(nvar), 2))
        corr_mat = np.ndarray((nrow, int(ncor)))
        if self.model_copy.constant:
            vifs_mat = np.ndarray((nrow, nvar - 1))
        else:
            vifs_mat = np.ndarray((nrow, nvar))
        vdp_idx = np.ndarray((nrow, nvar))
        vdp_pi = np.ndarray((nrow, nvar, nvar))

        for i in range(nrow):
            wi = w[i]
            sw = np.sum(wi)
            wi = wi / sw
            tag = 0

            for j, k in jk:
                corr_mat[i, tag] = corr(
                    np.cov(x[:, j], x[:, k], aweights=wi))[0][1]
                tag = tag + 1

            if self.model_copy.constant:
                corr_mati = corr(np.cov(x[:, 1:].T, aweights=wi))
                vifs_mat[i, ] = np.diag(np.linalg.solve(
                    corr_mati, np.identity((nvar - 1))))

            else:
                corr_mati = corr(np.cov(x.T, aweights=wi))
                vifs_mat[i, ] = np.diag(np.linalg.solve(
                    corr_mati, np.identity((nvar))))

            xw = x * wi.reshape((nrow, 1))
            sxw = np.sqrt(np.sum(xw**2, axis=0))
            sxw = np.transpose(xw.T / sxw.reshape((nvar, 1)))
            svdx = np.linalg.svd(sxw)
            vdp_idx[i, ] = svdx[1][0] / svdx[1]
            phi = np.dot(svdx[2].T, np.diag(1 / svdx[1]))
            phi = np.transpose(phi**2)
            pi_ij = phi / np.sum(phi, axis=0)
            vdp_pi[i, :, :] = pi_ij

        local_CN = vdp_idx[:, nvar - 1].reshape((-1, 1))
        VDP = vdp_pi[:, nvar - 1, :]

        return corr_mat, vifs_mat, local_CN, VDP

    def spatial_variability(self, selector, n_iters=1000, seed=None):
        """
        Monte Carlo test for spatial variability of each coefficient surface.

        Randomly permutes spatial coordinates to generate the null
        distribution of parameter surface standard deviations, then
        computes a pseudo p-value for each covariate.

        WARNING: This test is computationally very demanding.

        Parameters
        ----------
        selector        : Sel_BW
                          The bandwidth selector used for the GWR model.
        n_iters         : int, optional
                          Number of Monte Carlo permutations (default 1000).
        seed            : int, optional
                          Random seed for reproducibility (default 5536).

        Returns
        -------
        p_vals          : list of float
                          Pseudo p-values for each parameter surface.
        """
        temp_sel = copy.deepcopy(selector)
        temp_gwr = copy.deepcopy(self.model_copy)

        if seed is None:
            np.random.seed(5536)
        else:
            np.random.seed(seed)

        fit_params = temp_gwr.fit_params
        search_params = temp_sel.search_params
        kernel = temp_gwr.kernel
        fixed = temp_gwr.fixed

        if self.model_copy.constant:
            X = self.X[:, 1:]
        else:
            X = self.X

        init_sd = np.std(self.params, axis=0)
        SDs = []

        for x in range(n_iters):
            temp_coords = np.random.permutation(self.model_copy.coords)
            temp_sel.coords = temp_coords
            temp_sel._build_dMat()
            temp_bw = temp_sel.search(**search_params)

            temp_gwr.W = temp_gwr._build_W(fixed, kernel, temp_coords, temp_bw)
            temp_params = temp_gwr.fit(**fit_params).params

            temp_sd = np.std(temp_params, axis=0)
            SDs.append(temp_sd)

        p_vals = (np.sum(np.array(SDs) > init_sd, axis=0) / float(n_iters))
        return p_vals

    @cache_readonly
    def predictions(self):
        P = self.model_copy.P
        if P is None:
            raise TypeError('predictions only avaialble if predict'
                            'method is previously called on GWR model')
        else:
            predictions = np.sum(P * self.params, axis=1).reshape((-1, 1))
        return predictions

    def summary(self):
        """
        Print and return a formatted STWR model summary.

        Includes global regression results, spatiotemporal kernel
        configuration, diagnostic information, and parameter estimate
        statistics.
        """
        summary = summaryModel(self) + summaryGLM(self) + summarySTWR(self)
        print(summary)
        return summary