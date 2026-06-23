# STWR, GWR, and MGWR bandwidth selection classes
# STWR bandwidth selection developed by Xiang Que (xiangq@uidaho.edu, quexiang@fafu.edu.cn)
# GWR/MGWR bandwidth selection based on the MGWR 2.0.2 implementation by Taylor Oshan (tayoshan@gmail.com)

import numpy as np
import multiprocessing as mp
from scipy.spatial.distance import pdist
from scipy.optimize import minimize_scalar
import spreg.user_output as USER
from spglm.family import Gaussian, Poisson, Binomial

from .kernels import Kernel, local_cdist
from .gwr import GWR, STWR
from .search import golden_section, equal_interval, multi_bw
from .diagnostics import get_AICc, get_AIC, get_BIC, get_CV

# Mapping from criterion name to diagnostic function
getDiag = {'AICc': get_AICc, 'AIC': get_AIC, 'BIC': get_BIC, 'CV': get_CV}

class Sel_BW(object):
    """
    Bandwidth selector for GWR and MGWR models.

    Supports golden section search, equal interval search, and scipy
    optimization. Criterion options: AICc, AIC, BIC, CV. For MGWR
    (``multi=True``), uses iterative GAM backfitting to find
    covariate-specific bandwidths.

    Reference: Fotheringham et al. (2002), pp. 211-213.

    Parameters
    ----------
    coords         : array-like
                     n*2, (x, y) coordinates of observations.
    y              : array
                     n*1, dependent variable.
    X_loc          : array
                     n*k, locally-varying independent variables.
    X_glob         : array, optional
                     n*k1, globally fixed independent variables. Default None.
    family         : family object, optional
                     Probability model: Gaussian(), Poisson(), Binomial().
                     Default Gaussian().
    offset         : array, optional
                     n*1, offset variable (e.g., population at risk for
                     Poisson models). Default None (all 1.0).
    kernel         : string, optional
                     Kernel function: 'gaussian', 'bisquare', 'exponential'.
                     Default 'bisquare'.
    fixed          : boolean, optional
                     True for fixed (distance) bandwidth, False for adaptive
                     (nearest-neighbor). Default False.
    multi          : boolean, optional
                     True for covariate-specific (MGWR) bandwidths,
                     False for single bandwidth (GWR). Default False.
    constant       : boolean, optional
                     True to include intercept (default), False to exclude it.
    spherical      : boolean, optional
                     True for spherical (lon/lat) coordinates,
                     False for projected coordinates. Default False.

    Attributes
    ----------
    coords         : array-like
                     n*2, calibration coordinates.
    y              : array
                     n*1, dependent variable.
    X_loc          : array
                     n*k, local independent variables.
    X_glob         : array
                     n*k1, global independent variables.
    family         : family object
                     Probability model.
    kernel         : string
                     Kernel function name.
    fixed          : boolean
                     Whether bandwidth is fixed or adaptive.
    criterion      : string
                     Selection criterion: 'AICc', 'AIC', 'BIC', 'CV'.
    search_method  : string
                     Optimization method: 'golden_section', 'interval', 'scipy'.
    bw_min         : float
                     Minimum bandwidth value in search range.
    bw_max         : float
                     Maximum bandwidth value in search range.
    interval       : float
                     Step size for equal interval search.
    tol            : float
                     Convergence tolerance.
    max_iter       : int
                     Maximum iterations if convergence not achieved.
    multi          : boolean
                     Whether MGWR (multi=True) or GWR (multi=False) mode.
    int_score      : boolean
                     True for adaptive bandwidth (discrete search),
                     False for fixed bandwidth (continuous search).
    bw             : scalar or list
                     Optimal bandwidth value(s). Scalar for GWR, list of
                     scalars (one per covariate) for MGWR.
    params         : array
                     n*k, MGWR parameter estimates from backfitting.
    search_params  : dict
                     Arguments passed to the search routine.
    sel_hist       : list
                     Bandwidth search history [(bw, score), ...].
    bw_init        : scalar
                     Initial GWR bandwidth used to start MGWR backfitting.

    Examples
    --------
    Golden section search with AICc (adaptive bi-square):

    >>> import libpysal as ps
    >>> from mgwr.sel_bw import Sel_BW
    >>> data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
    >>> coords = list(zip(data.by_col('X'), data.by_col('Y')))
    >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
    >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
    >>> pov = np.array(data.by_col('PctPov')).reshape((-1,1))
    >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
    >>> X = np.hstack([rural, pov, african_amer])
    >>> bw = Sel_BW(coords, y, X).search(criterion='AICc')
    >>> print(bw)
    93.0

    Interval search with AICc (fixed bi-square):

    >>> sel = Sel_BW(coords, y, X, fixed=True)
    >>> bw = sel.search(search_method='interval', bw_min=211001.0, bw_max=211035.0, interval=2)
    >>> print(bw)
    211025.0

    MGWR covariate-specific bandwidths:

    >>> selector = Sel_BW(coords, y, X, multi=True)
    >>> bws = selector.search(multi_bw_min=[2])
    >>> print(bws)
    [92.0, 101.0, 136.0, 158.0]

    """

    def __init__(self, coords, y, X_loc, X_glob=None, family=Gaussian(),
                 offset=None, kernel='bisquare', fixed=False, multi=False,
                 constant=True, spherical=False):
        self.coords = np.array(coords)
        self.y = y
        self.X_loc = X_loc
        if X_glob is not None:
            self.X_glob = X_glob
        else:
            self.X_glob = []
        self.family = family
        self.fixed = fixed
        self.kernel = kernel
        if offset is None:
            self.offset = np.ones((len(y), 1))
        else:
            self.offset = offset * 1.0
        self.multi = multi
        self._functions = []
        self.constant = constant
        self.spherical = spherical
        self.search_params = {}

    def search(self, search_method='golden_section', criterion='AICc',
               bw_min=None, bw_max=None, interval=0.0, tol=1.0e-6,
               max_iter=200, init_multi=None, tol_multi=1.0e-5,
               rss_score=False, max_iter_multi=200, multi_bw_min=[None],
               multi_bw_max=[None
                             ], bws_same_times=5, pool=None, verbose=False):
        """
        Run bandwidth selection.

        Parameters
        ----------
        criterion      : string, optional
                         Selection criterion: 'AICc', 'AIC', 'BIC', 'CV'.
                         Default 'AICc'.
        search_method  : string, optional
                         Optimization method: 'golden_section', 'interval',
                         or 'scipy'. Default 'golden_section'.
        bw_min         : float, optional
                         Minimum bandwidth value. Default auto-determined.
        bw_max         : float, optional
                         Maximum bandwidth value. Default auto-determined.
        multi_bw_min   : list, optional
                         Per-covariate minimum bandwidths for MGWR. Can be a
                         single value or one per covariate (including intercept).
        multi_bw_max   : list, optional
                         Per-covariate maximum bandwidths for MGWR.
        interval       : float, optional
                         Interval step for equal interval search. Default 0.
        tol            : float, optional
                         Convergence tolerance for golden section search.
                         Default 1.0e-6.
        max_iter       : int, optional
                         Maximum iterations for golden section search.
                         Default 200.
        init_multi     : float, optional
                         Initial bandwidth for MGWR backfitting. Default None
                         uses a GWR-derived bandwidth.
        tol_multi      : float, optional
                         Convergence tolerance for MGWR backfitting.
                         Default 1.0e-5.
        max_iter_multi : int, optional
                         Maximum MGWR backfitting iterations. Default 200.
        rss_score      : bool, optional
                         If True, use RSS for backfitting convergence score;
                         if False, use a smooth function. Default False.
        bws_same_times : int, optional
                         Early-stopping: if all bandwidths stay the same for
                         this many consecutive backfitting iterations, stop.
                         Default 5.
        pool           : multiprocessing.Pool or None, optional
                         Pool for parallel fitting. Default None.
        verbose        : bool, optional
                         If True, print search progress. Default False.

        Returns
        -------
        bw             : scalar or list
                         Optimal bandwidth. Scalar for GWR (multi=False),
                         list of k scalars for MGWR (multi=True), matching
                         the column order of the design matrix.
        """
        k = self.X_loc.shape[1]
        if self.constant:  #k is the number of covariates
            k += 1
        self.search_method = search_method
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.bws_same_times = bws_same_times
        self.pool = pool
        self.verbose = verbose

        if len(multi_bw_min) == k:
            self.multi_bw_min = multi_bw_min
        elif len(multi_bw_min) == 1:
            self.multi_bw_min = multi_bw_min * k
        else:
            raise AttributeError(
                "multi_bw_min must be either a list containing"
                " a single entry or a list containing an entry for each of k"
                " covariates including the intercept")

        if len(multi_bw_max) == k:
            self.multi_bw_max = multi_bw_max
        elif len(multi_bw_max) == 1:
            self.multi_bw_max = multi_bw_max * k
        else:
            raise AttributeError(
                "multi_bw_max must be either a list containing"
                " a single entry or a list containing an entry for each of k"
                " covariates including the intercept")

        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter
        self.init_multi = init_multi
        self.tol_multi = tol_multi
        self.rss_score = rss_score
        self.max_iter_multi = max_iter_multi
        self.search_params['search_method'] = search_method
        self.search_params['criterion'] = criterion
        self.search_params['bw_min'] = bw_min
        self.search_params['bw_max'] = bw_max
        self.search_params['interval'] = interval
        self.search_params['tol'] = tol
        self.search_params['max_iter'] = max_iter
        #self._check_min_max()

        self.int_score = not self.fixed

        if self.multi:
            self._mbw()
            self.params = self.bw[3]  #params n by k
            self.sel_hist = self.bw[-2] #bw searching history
            self.bw_init = self.bw[
                -1]  #scalar, optimal bw from initial gwr model
        else:
            self._bw()
            self.sel_hist = self.bw[-1]

        self.pool = None
        return self.bw[0]

    def _bw(self):
        gwr_func = lambda bw: getDiag[self.criterion](GWR(
            self.coords, self.y, self.X_loc, bw, family=self.family, kernel=
            self.kernel, fixed=self.fixed, constant=self.constant, offset=self.
            offset, spherical=self.spherical).fit(lite=True, pool=self.pool))

        self._optimized_function = gwr_func

        if self.search_method == 'golden_section':
            a, c = self._init_section(self.X_glob, self.X_loc, self.coords,
                                      self.constant)
            delta = 0.38197  #1 - (np.sqrt(5.0)-1.0)/2.0
            self.bw = golden_section(a, c, delta, gwr_func, self.tol,
                                     self.max_iter, self.bw_max, self.int_score,
                                     self.verbose)
        elif self.search_method == 'interval':
            self.bw = equal_interval(self.bw_min, self.bw_max, self.interval,
                                     gwr_func, self.int_score, self.verbose)
        elif self.search_method == 'scipy':
            self.bw_min, self.bw_max = self._init_section(
                self.X_glob, self.X_loc, self.coords, self.constant)
            if self.bw_min == self.bw_max:
                raise Exception(
                    'Maximum bandwidth and minimum bandwidth must be distinct for scipy optimizer.'
                )
            self._optimize_result = minimize_scalar(
                gwr_func, bounds=(self.bw_min, self.bw_max), method='bounded')
            self.bw = [self._optimize_result.x, self._optimize_result.fun, []]
        else:
            raise TypeError('Unsupported computational search method ',
                            self.search_method)

    def _mbw(self):
        y = self.y
        if self.constant:
            X,keep_x,warn = USER.check_constant(self.X_loc)
        else:
            X = self.X_loc
        n, k = X.shape
        family = self.family
        offset = self.offset
        kernel = self.kernel
        fixed = self.fixed
        spherical = self.spherical
        coords = self.coords
        search_method = self.search_method
        criterion = self.criterion
        bw_min = self.bw_min
        bw_max = self.bw_max
        multi_bw_min = self.multi_bw_min
        multi_bw_max = self.multi_bw_max
        interval = self.interval
        tol = self.tol
        max_iter = self.max_iter
        bws_same_times = self.bws_same_times

        def gwr_func(y, X, bw):
            return GWR(coords, y, X, bw, family=family, kernel=kernel,
                       fixed=fixed, offset=offset, constant=False,
                       spherical=self.spherical, hat_matrix=False).fit(
                           lite=True, pool=self.pool)

        def bw_func(y, X):
            selector = Sel_BW(coords, y, X, X_glob=[], family=family,
                              kernel=kernel, fixed=fixed, offset=offset,
                              constant=False, spherical=self.spherical)
            return selector

        def sel_func(bw_func, bw_min=None, bw_max=None):
            return bw_func.search(
                search_method=search_method, criterion=criterion,
                bw_min=bw_min, bw_max=bw_max, interval=interval, tol=tol,
                max_iter=max_iter, pool=self.pool, verbose=False)

        self.bw = multi_bw(self.init_multi, y, X, n, k, family, self.tol_multi,
                           self.max_iter_multi, self.rss_score, gwr_func,
                           bw_func, sel_func, multi_bw_min, multi_bw_max,
                           bws_same_times, verbose=self.verbose)

    def _init_section(self, X_glob, X_loc, coords, constant):
        """
        Determine the initial search interval [a, c] for bandwidth selection.

        For adaptive bandwidth: a = 40 + 2*k, c = n.
        For fixed bandwidth: a = min_pairwise_distance / 2,
                             c = max_pairwise_distance * 2.
        User-specified bw_min/bw_max take precedence.
        """
        if len(X_glob) > 0:
            n_glob = X_glob.shape[1]
        else:
            n_glob = 0
        if len(X_loc) > 0:
            n_loc = X_loc.shape[1]
        else:
            n_loc = 0
        n_vars = n_glob + n_loc + (1 if constant else 0)
        n = np.array(coords).shape[0]

        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            # Compute pairwise distance bounds
            min_dist = np.min(np.array([np.min(np.delete(
                local_cdist(coords[i], coords, spherical=self.spherical), i))
                    for i in range(n)]))
            max_dist = np.max(np.array([np.max(
                local_cdist(coords[i], coords, spherical=self.spherical))
                    for i in range(n)]))

            a = min_dist / 2.0
            c = max_dist * 2.0

        if self.bw_min is not None:
            a = self.bw_min
        if self.bw_max is not None and self.bw_max is not np.inf:
            c = self.bw_max

        return a, c


# Legacy kernel dispatch table (referenced by number in some older code)
kernels = {1: fix_gauss, 2: adapt_gauss, 3: fix_bisquare, 4:
        adapt_bisquare, 5: fix_exp, 6: adapt_exp,
        7: fix_spt_bisquare, 8: adapt_spt_bisquare}


class Sel_Spt_BW(object):
    """
    Bandwidth and parameter selector for STWR models.

    Searches for the optimal combination of four parameters:
    - ``alpha``: balance between spatial and temporal effects [0, 1]
    - ``sita``:  slope of spatial bandwidth change over time
    - ``tick_nums``: number of past time stages to use
    - ``gwr_bw0``: initial spatial bandwidth (nearest neighbors)

    The search is a brute-force grid over all parameter combinations,
    evaluating each via cross-validation (CV) by default.

    Parameters
    ----------
    coordslist : list of array
        Coordinates per time stage, ordered oldest to newest.
    y_list : list of array
        Dependent variable per time stage.
    X_list : list of array
        Independent variables per time stage.
    tick_times_intervel : list of float
        Time intervals between consecutive stages.
    dspal_mat_list : list of array, optional
        Pre-computed spatial distance matrices.
    sorted_dspal_list : list of array, optional
        Pre-computed sorted spatial distances.
    d_tmp_list : list of array, optional
        Pre-computed temporal distance matrices.
    dspmat : array, optional
        Combined spatial distance matrix.
    dtmat : array, optional
        Combined temporal distance matrix.
    family : family object, optional
        Probability model. Default Gaussian().
    offset : array, optional
        Offset variable for Poisson models.
    kernel : string, optional
        Spatiotemporal kernel function. Default 'spt_bisquare'.
    fixed : bool, optional
        True for fixed bandwidth, False for adaptive. Default False.
    multi : bool, optional
        Reserved for MGWR compatibility.
    eps : float, optional
        Small tolerance for bandwidth computation. Default 1.0000001.
    constant : bool, optional
        True to include intercept. Default True.
    spherical : bool, optional
        True for spherical (lon/lat) coordinates. Default False.
    """
    def __init__(self, coordslist,y_list, X_list,
                 tick_times_intervel, 
                 dspal_mat_list=None,sorted_dspal_list=None,d_tmp_list=None,dspmat = None,dtmat=None,
                 family=Gaussian(),
                 offset=None, kernel='spt_bisquare', fixed=False, multi=False,eps=1.0000001,
                 constant=True, spherical=False):
        self.n_tick_nums = len(coordslist)
        self.coordslist = coordslist
        self.tick_timesIntls = tick_times_intervel
        self.y_list =y_list
        self.X_list = X_list
        self.dspal_mat_list =dspal_mat_list
        self.sorted_dspal_list = sorted_dspal_list
        self.d_tmp_list = d_tmp_list
        self.dspmat = dspmat
        self.dtmat = dtmat
        self.eps = eps

        self.family=family
        self.fixed = fixed
        self.kernel = kernel
        self._functions = []
        self.constant = constant
        self.spherical = spherical
        self._build_dMatlist()
        self.search_params = {}
    def search(self, search_method='golden_section',
               criterion='CV', sita_min=-np.pi/2, sita_max=np.pi/2,
               interval=np.pi/200, Intls=1, tol=1.0e-6, max_iter=200):
        """
        Search for optimal STWR parameters.

        Parameters
        ----------
        search_method : string, optional
            Search method (only 'golden_section' and 'interval' supported).
        criterion : string, optional
            Selection criterion (default 'CV').
        sita_min, sita_max : float, optional
            Search range for sita (radians). Default [-pi/2, pi/2].
        interval : float, optional
            Step size for alpha/interval search. Default pi/200.
        tol : float, optional
            Convergence tolerance. Default 1.0e-6.
        max_iter : int, optional
            Maximum iterations. Default 200.

        Returns
        -------
        opt_alpha : float
            Optimal alpha (spatial/temporal balance).
        opt_sita : float
            Optimal sita (bandwidth slope over time).
        opt_tick : int
            Optimal number of time stages.
        opt_gwr_bw0 : int
            Optimal initial spatial bandwidth (nearest neighbors).
        """
        self.search_method = search_method
        self.criterion = criterion
        self.sita_min = sita_min
        self.sita_max = sita_max  
        self.interval = interval
        self.tol = tol
        self.max_iter = max_iter        
        self.search_params['search_method'] = search_method
        self.search_params['criterion'] = criterion
        self.search_params['sita_min'] = sita_min
        self.search_params['sita_max'] = sita_max
        self.search_params['interval'] = interval
        self.search_params['tol'] = tol
        self.search_params['max_iter'] = max_iter
        self._opt_alpha,self._opt_bsita,self._opt_btticks,self._opt_bw0 = self._spt_bw()
        return self._opt_alpha,self._opt_bsita, self._opt_btticks,self._opt_bw0
    def _build_dMatlist(self):
        if self.fixed:  
            self.dspal_mat_list,self.d_tmp_list,self.dspmat,self.dtmat=cspatiltemporaldist(
                                                self.coordslist,self.coordslist,
                                                self.y_list,self.n_tick_nums,
                                                self.tick_timesIntls,self.spherical)
            self.sorted_dspal_list = None
        else:
            self.dspal_mat_list,self.d_tmp_list,self.dspmat,self.dtmat=cspatiltemporaldist(
                                        self.coordslist,self.coordslist,
                                        self.y_list,self.n_tick_nums,
                                        self.tick_timesIntls,self.spherical)
            if self.sorted_dspal_list is None:
                    self.sorted_dspal_list = []
            for i in range(self.n_tick_nums):
                self.sorted_dspal_list.append(np.sort(self.dspal_mat_list[i]))
    def _spt_bw(self):
        """
        Brute-force grid search for optimal STWR parameters.

        Iterates over all combinations of:
        - Number of time stages (1 to n_tick_nums)
        - Initial GWR bandwidth (min_bw0 to max_bw0)
        - Alpha (0 to 1, only when tick_nums > 1)

        For each combination, runs golden section or interval search to
        find the optimal alpha. Returns the combination with the lowest
        criterion score.
        """
        max_bw0 = self.coordslist[-1].shape[0]
        min_bw0 = (self.X_list[-1].shape[1] + 1)
        d_search_gw0 = max_bw0 - min_bw0
        opt_score_list = []
        opt_bsita_list = []
        opt_tick_list = []
        opt_alpha_list = []
        opt_gwr_bw0_list = []

        for i in range(self.n_tick_nums):
            for q in range(d_search_gw0):
                gwr_bw0_tick = q + min_bw0
                _opt_bsita_tick = 0
                stwr_func = lambda alpha: getDiag[self.criterion](
                    STWR(self.coordslist,
                         self.y_list, self.X_list,
                         self.tick_timesIntls,
                         _opt_bsita_tick, gwr_bw0_tick,
                         tick_nums=i + 1,
                         dspal_mat_list=self.dspal_mat_list,
                         sorted_dspal_list=self.sorted_dspal_list,
                         d_tmp_list=self.d_tmp_list,
                         dspmat=self.dspmat, dtmat=self.dtmat,
                         alpha=alpha,
                         family=self.family, kernel=self.kernel,
                         fixed=self.fixed,
                         constant=self.constant).fit(searching=True))
                self._optimized_function = stwr_func

                if self.search_method == 'golden_section':
                    mb_search, alpha_min, alpha_max, alpha_interval = \
                        self._init_alpha_search(i + 1, 25)
                    if mb_search:
                        opt_alpha_tick, opt_score, output_del = golden_section(
                            alpha_min, alpha_max, alpha_interval,
                            stwr_func, self.tol, self.max_iter)
                    else:
                        output_del = []
                        opt_score = stwr_func(0)
                        opt_bsita_tick = 0
                        opt_alpha_tick = 0
                        output_del.append((opt_bsita_tick, opt_score))
                    opt_bsita_list.append(opt_bsita_tick)
                    opt_score_list.append(opt_score)
                    opt_tick_list.append(i)
                    opt_alpha_list.append(opt_alpha_tick)
                    opt_gwr_bw0_list.append(gwr_bw0_tick)
                elif self.search_method == 'interval':
                    mb_search, alpha_min, alpha_max, alpha_interval = \
                        self._init_alpha_search(i + 1, 25)
                    if mb_search:
                        opt_alpha_tick, opt_score, output_del = equal_interval(
                            alpha_min, alpha_max, alpha_interval, stwr_func)
                    else:
                        output_del = []
                        opt_score = stwr_func(0)
                        opt_alpha_tick = 0
                        opt_bsita_tick = 0
                        output_del.append((opt_bsita_tick, opt_score))
                    opt_bsita_list.append(opt_bsita_tick)
                    opt_alpha_list.append(opt_alpha_tick)
                    opt_score_list.append(opt_score)
                    opt_tick_list.append(i)
                    opt_gwr_bw0_list.append(gwr_bw0_tick)
                else:
                    raise TypeError('Unsupported computational search method ',
                                    self.search_method)

        if len(opt_score_list) == 0:
            raise TypeError(
                'opt_score_list is empty. Please check your data!')
        index = opt_score_list.index(min(opt_score_list))

        return (opt_alpha_list[index], opt_bsita_list[index],
                opt_tick_list[index], opt_gwr_bw0_list[index])

    def _init_alpha_search(self, n_tick, brk_ticks):
        """
        Determine alpha search range.

        If only 1 time stage: alpha search is disabled (alpha=0,
        spatial kernel only). Otherwise: alpha in [0, 1].
        """
        if n_tick == 1:
            mb_search = False
            minalpha = 0
            maxalpha = 0
            intervel = 0
        else:
            mb_search = True
            minalpha = 0
            maxalpha = 1
            intervel = 1.0 / brk_ticks
        return mb_search, minalpha, maxalpha, intervel

    def _init_sita_section(self, n_tick, gwr_bw0, brk_ticks):
        """
        Determine sita (bandwidth slope) search range.

        Sita controls how spatial bandwidth shrinks over time stages.
        When n_tick == 1, sita is fixed at 0.
        """
        if n_tick == 1:
            sita_min = 0
            sita_max = 0
            sita_delta = 0
            mb_search = False
        else:
            mb_search = True
            tick_tims = self.tick_timesIntls[-n_tick:]
            lencd = self.coordslist[-1].shape[0]
            lenwd = n_tick - 1
            sita_arr = np.zeros((lencd, lenwd))
            gwr_bw0 = int(gwr_bw0) + 1
            gwr_bwmin = self.coordslist[-1].shape[1] + 1
            dspmat0 = self.sorted_dspal_list[0][:, :gwr_bw0]
            bw_0_lis = dspmat0[:, -1:].reshape((-1, 1))
            for i in range(n_tick - 1):
                sorted_dspmat_min = self.sorted_dspal_list[
                    -(i + 1)][:, :gwr_bwmin]
                bw_tick_lis = sorted_dspmat_min[:, -1:].reshape((-1, 1))
                delt_jundge = bw_0_lis <= bw_tick_lis
                if np.any(delt_jundge):
                    sita_min = 0
                    sita_max = 0
                    mb_search = False
                else:
                    tick_ticks = tick_tims[-(i + 1):]
                    delt_tic = np.sum(tick_ticks)
                    delt_sita_tick = (bw_0_lis - bw_tick_lis) / delt_tic
                    sita_arr[:, i:i + 1] = np.arctan(delt_sita_tick)
            sita_max = sita_arr.min()
            sita_min = 0
            sita_delta = (sita_max - sita_min) / brk_ticks
        return mb_search, sita_min, sita_max, sita_delta      