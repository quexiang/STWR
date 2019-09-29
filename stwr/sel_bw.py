# STWR and GWR Bandwidth selection class
__author__ = "STWR is XiangQue xiangq@uidaho.edu and GWR,MGWR is Taylor Oshan Tayoshan@gmail.com"

import spreg.user_output as USER
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.optimize import minimize_scalar
from spglm.family import Gaussian, Poisson, Binomial
from spglm.iwls import iwls,_compute_betas_gwr
from .kernels import *
from .gwr import GWR,STWR
from .search import golden_section, equal_interval, multi_bw
from .diagnostics import get_AICc, get_AIC, get_BIC, get_CV
from functools import partial
from math import atan,tan

kernels = {1: fix_gauss, 2: adapt_gauss, 3: fix_bisquare, 4:
        adapt_bisquare, 5: fix_exp, 6:adapt_exp,7:fix_spt_bisquare,8:adapt_spt_bisquare}
getDiag = {'AICc': get_AICc,'AIC':get_AIC, 'BIC': get_BIC, 'CV': get_CV}

class Sel_BW(object):
    """
    Select bandwidth for kernel

    Methods: p211 - p213, bandwidth selection
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.

    Parameters
    ----------
    y              : array
                     n*1, dependent variable.
    X_glob         : array
                     n*k1, fixed independent variable.
    X_loc          : array
                     n*k2, local independent variable, including constant.
    coords         : list of tuples
                     (x,y) of points used in bandwidth selection
    family         : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson''
    offset        : array
                    n*1, the offset variable at the ith location. For Poisson model
                    this term is often the size of the population at risk or
                    the expected size of the outcome in spatial epidemiology
                    Default is None where Ni becomes 1.0 for all locations
    kernel         : string
                     kernel function: 'gaussian', 'bisquare', 'exponetial'
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)
    multi          : True for multiple (covaraite-specific) bandwidths
                     False for a traditional (same for  all covariates)
                     bandwdith; defualt is False.
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.
    spherical     : boolean
                    True for shperical coordinates (long-lat),
                    False for projected coordinates (defalut).

    Attributes
    ----------
    y              : array
                     n*1, dependent variable.
    X_glob         : array
                     n*k1, fixed independent variable.
    X_loc          : array
                     n*k2, local independent variable, including constant.
    coords         : list of tuples
                     (x,y) of points used in bandwidth selection
    family         : string
                     GWR model type: 'Gaussian', 'logistic, 'Poisson''
    kernel         : string
                     type of kernel used and wether fixed or adaptive
    fixed          : boolean
                     True for fixed bandwidth and False for adaptive (NN)
    criterion      : string
                     bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
    search_method  : string
                     bw search method: 'golden', 'interval'
    bw_min         : float
                     min value used in bandwidth search
    bw_max         : float
                     max value used in bandwidth search
    interval       : float
                     interval increment used in interval search
    tol            : float
                     tolerance used to determine convergence
    max_iter       : integer
                     max interations if no convergence to tol
    multi          : True for multiple (covaraite-specific) bandwidths
                     False for a traditional (same for  all covariates)
                     bandwdith; defualt is False.
    constant       : boolean
                     True to include intercept (default) in model and False to exclude
                     intercept.
    offset        : array
                    n*1, the offset variable at the ith location. For Poisson model
                    this term is often the size of the population at risk or
                    the expected size of the outcome in spatial epidemiology
                    Default is None where Ni becomes 1.0 for all locations
    dmat          : array
                    n*n, distance matrix between calibration locations used
                    to compute weight matrix
                        
    sorted_dmat   : array
                    n*n, sorted distance matrix between calibration locations used
                    to compute weight matrix. Will be None for fixed bandwidths
        
    spherical     : boolean
                    True for shperical coordinates (long-lat),
                    False for projected coordinates (defalut).
    search_params : dict
                    stores search arguments
    int_score     : boolan
                    True if adaptive bandwidth is being used and bandwdith
                    selection should be discrete. False
                    if fixed bandwidth is being used and bandwidth does not have
                    to be discrete.
    bw            : scalar or array-like
                    Derived optimal bandwidth(s). Will be a scalar for GWR
                    (multi=False) and a list of scalars for MGWR (multi=True)
                    with one bandwidth for each covariate.
    S             : array
                    n*n, hat matrix derived from the iterative backfitting
                    algorthim for MGWR during bandwidth selection
    R             : array
                    n*n*k, partial hat matrices derived from the iterative
                    backfitting algoruthm for MGWR during bandwidth selection.
                    There is one n*n matrix for each of the k covariates.
    params        : array
                    n*k, calibrated parameter estimates for MGWR based on the
                    iterative backfitting algorithm - computed and saved here to
                    avoid having to do it again in the MGWR object.

    Examples
    --------

    >>> import libpysal as ps
    >>> from mgwr.sel_bw import Sel_BW
    >>> data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
    >>> coords = list(zip(data.by_col('X'), data.by_col('Y')))
    >>> y = np.array(data.by_col('PctBach')).reshape((-1,1))
    >>> rural = np.array(data.by_col('PctRural')).reshape((-1,1))
    >>> pov = np.array(data.by_col('PctPov')).reshape((-1,1))
    >>> african_amer = np.array(data.by_col('PctBlack')).reshape((-1,1))
    >>> X = np.hstack([rural, pov, african_amer])
    
    Golden section search AICc - adaptive bisquare

    >>> bw = Sel_BW(coords, y, X).search(criterion='AICc')
    >>> print(bw)
    93.0

    Golden section search AIC - adaptive Gaussian

    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='AIC')
    >>> print(bw)
    50.0

    Golden section search BIC - adaptive Gaussian

    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='BIC')
    >>> print(bw)
    62.0

    Golden section search CV - adaptive Gaussian

    >>> bw = Sel_BW(coords, y, X, kernel='gaussian').search(criterion='CV')
    >>> print(bw)
    68.0

    Interval AICc - fixed bisquare

    >>> sel = Sel_BW(coords, y, X, fixed=True)
    >>> bw = sel.search(search_method='interval', bw_min=211001.0, bw_max=211035.0, interval=2)
    >>> print(bw)
    211025.0

    """
    def __init__(self, coords, y, X_loc, X_glob=None, family=Gaussian(),
            offset=None, kernel='bisquare', fixed=False, multi=False,
            constant=True, spherical=False):
        self.coords = coords
        self.y = y
        self.X_loc = X_loc
        if X_glob is not None:
            self.X_glob = X_glob
        else:
            self.X_glob = []
        self.family=family
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
        self._build_dMat()
        self.search_params = {}

    def search(self, search_method='golden_section', criterion='AICc',
            bw_min=None, bw_max=None, interval=0.0, tol=1.0e-6, max_iter=200,
            init_multi=None, tol_multi=1.0e-5, rss_score=False,
            max_iter_multi=200, multi_bw_min=[None], multi_bw_max=[None]):
        """
        Method to select one unique bandwidth for a gwr model or a
        bandwidth vector for a mgwr model.

        Parameters
        ----------
        criterion      : string
                         bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
        search_method  : string
                         bw search method: 'golden', 'interval'
        bw_min         : float
                         min value used in bandwidth search
        bw_max         : float
                         max value used in bandwidth search
        multi_bw_min   : list 
                         min values used for each covariate in mgwr bandwidth search.
                         Must be either a single value or have one value for
                         each covariate including the intercept
        multi_bw_max   : list
                         max values used for each covariate in mgwr bandwidth
                         search. Must be either a single value or have one value
                         for each covariate including the intercept
        interval       : float
                         interval increment used in interval search
        tol            : float
                         tolerance used to determine convergence
        max_iter       : integer
                         max iterations if no convergence to tol
        init_multi     : float
                         None (default) to initialize MGWR with a bandwidth
                         derived from GWR. Otherwise this option will choose the
                         bandwidth to initialize MGWR with.
        tol_multi      : convergence tolerence for the multiple bandwidth
                         backfitting algorithm; a larger tolerance may stop the
                         algorith faster though it may result in a less optimal
                         model
        max_iter_multi : max iterations if no convergence to tol for multiple
                         bandwidth backfittign algorithm
        rss_score      : True to use the residual sum of sqaures to evaluate
                         each iteration of the multiple bandwidth backfitting
                         routine and False to use a smooth function; default is
                         False

        Returns
        -------
        bw             : scalar or array
                         optimal bandwidth value or values; returns scalar for
                         multi=False and array for multi=True; ordering of bandwidths
                         matches the ordering of the covariates (columns) of the
                         designs matrix, X
        """
        k = self.X_loc.shape[1]
        if self.constant: #k is the number of covariates
            k +=1
        self.search_method = search_method
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        
        if len(multi_bw_min) == k:
            self.multi_bw_min = multi_bw_min
        elif len(multi_bw_min) == 1:
            self.multi_bw_min = multi_bw_min*k
        else:
            raise AttributeError("multi_bw_min must be either a list containing"
            " a single entry or a list containing an entry for each of k"
            " covariates including the intercept")
        
        if len(multi_bw_max) == k:
            self.multi_bw_max = multi_bw_max
        elif len(multi_bw_max) == 1:
            self.multi_bw_max = multi_bw_max*k
        else:
            raise AttributeError("multi_bw_max must be either a list containing"
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

        if self.fixed:
            if self.kernel == 'gaussian':
                ktype = 1
            elif self.kernel == 'bisquare':
                ktype = 3
            elif self.kernel == 'exponential':
                ktype = 5
            else:
                raise TypeError('Unsupported kernel function ', self.kernel)
        else:
            if self.kernel == 'gaussian':
              ktype = 2
            elif self.kernel == 'bisquare':
                ktype = 4
            elif self.kernel == 'exponential':
                ktype = 6
            else:
                raise TypeError('Unsupported kernel function ', self.kernel)

        if ktype % 2 == 0:
            int_score = True
        else:
            int_score = False
        self.int_score = int_score #isn't this just self.fixed?

        if self.multi:
            self._mbw()
            self.params = self.bw[3] #params
            self.S = self.bw[-2] #(n,n)
            self.R = self.bw[-1] #(n,n,k)
        else:
            self._bw()

        return self.bw[0]
    
    def _build_dMat(self):
        if self.fixed:
            self.dmat = cdist(self.coords,self.coords,self.spherical)
            self.sorted_dmat = None
        else:
            self.dmat = cdist(self.coords,self.coords,self.spherical)
            self.sorted_dmat = np.sort(self.dmat)


    def _bw(self):

        gwr_func = lambda bw: getDiag[self.criterion](GWR(self.coords, self.y, 
            self.X_loc, bw, family=self.family, kernel=self.kernel,
            fixed=self.fixed, constant=self.constant,
            dmat=self.dmat,sorted_dmat=self.sorted_dmat).fit(searching = True))
        
        self._optimized_function = gwr_func

        if self.search_method == 'golden_section':
            a,c = self._init_section(self.X_glob, self.X_loc, self.coords,
                    self.constant)
            delta = 0.38197 #1 - (np.sqrt(5.0)-1.0)/2.0
            self.bw = golden_section(a, c, delta, gwr_func, self.tol,
                    self.max_iter, self.int_score)
        elif self.search_method == 'interval':
            self.bw = equal_interval(self.bw_min, self.bw_max, self.interval,
                    gwr_func, self.int_score)
        elif self.search_method == 'scipy':
            self.bw_min, self.bw_max = self._init_section(self.X_glob, self.X_loc,
                    self.coords, self.constant)
            if self.bw_min == self.bw_max:
                raise Exception('Maximum bandwidth and minimum bandwidth must be distinct for scipy optimizer.')
            self._optimize_result = minimize_scalar(gwr_func, bounds=(self.bw_min,
                self.bw_max), method='bounded')
            self.bw = [self._optimize_result.x, self._optimize_result.fun, []]
        else:
            raise TypeError('Unsupported computational search method ',
                    self.search_method)

    def _mbw(self):
        y = self.y
        if self.constant:
            X = USER.check_constant(self.X_loc)
        else:
            X = self.X_loc
        n, k = X.shape
        family = self.family
        offset = self.offset
        kernel = self.kernel
        fixed = self.fixed
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
        def gwr_func(y,X,bw):
            return GWR(coords, y,X,bw,family=family, kernel=kernel, fixed=fixed,
                    offset=offset, constant=False).fit()
        def bw_func(y,X):
            return Sel_BW(coords, y,X,X_glob=[], family=family, kernel=kernel,
                    fixed=fixed, offset=offset, constant=False)
        def sel_func(bw_func, bw_min=None, bw_max=None):
            return bw_func.search(search_method=search_method, criterion=criterion,
                    bw_min=bw_min, bw_max=bw_max, interval=interval, tol=tol, max_iter=max_iter)
        self.bw = multi_bw(self.init_multi, y, X, n, k, family,
                self.tol_multi, self.max_iter_multi, self.rss_score, gwr_func,
                bw_func, sel_func, multi_bw_min, multi_bw_max)

    def _init_section(self, X_glob, X_loc, coords, constant):
        if len(X_glob) > 0:
            n_glob = X_glob.shape[1]
        else:
            n_glob = 0
        if len(X_loc) > 0:
            n_loc = X_loc.shape[1]
        else:
            n_loc = 0
        if constant:
            n_vars = n_glob + n_loc + 1
        else:
            n_vars = n_glob + n_loc
        n = np.array(coords).shape[0]

        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            sq_dists = pdist(coords)
            a = np.min(sq_dists)/2.0
            c = np.max(sq_dists)*2.0

        if self.bw_min is not None:
            a = self.bw_min
        if self.bw_max is not None:
            c = self.bw_max
        
        return a, c
        
class Sel_Spt_BW(object):
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
               criterion='CV',sita_min=-np.pi/2, sita_max=np.pi/2, interval=np.pi/200,Intls =1, tol=1.0e-6, max_iter=200):
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
            max_bw0 = self.coordslist[-1].shape[0]
            min_bw0 = (self.X_list[-1].shape[1]+1)
            d_search_gw0 = max_bw0-min_bw0
            opt_score_list = []
            opt_bsita_list = []
            opt_tick_list = []
            opt_alpha_list =[]
            opt_gwr_bw0_list=[]                   
            for i in range (self.n_tick_nums):
                for q in range(d_search_gw0):
                    gwr_bw0_tick = q+min_bw0
                    _opt_bsita_tick = 0
                    stwr_func = lambda alpha: getDiag[self.criterion](STWR(self.coordslist,
                                                               self.y_list,self.X_list, 
                                                               self.tick_timesIntls,
                                                              _opt_bsita_tick,gwr_bw0_tick,
                                                              tick_nums =i+1,dspal_mat_list=self.dspal_mat_list,
                                                              sorted_dspal_list=self.sorted_dspal_list,
                                                              d_tmp_list=self.d_tmp_list,
                                                              dspmat=self.dspmat,dtmat=self.dtmat,alpha =alpha,
                                                              family=self.family, kernel=self.kernel,fixed=self.fixed,
                                                              constant=self.constant).fit(searching = True))
                    self._optimized_function =stwr_func 
                    if self.search_method == 'golden_section':
                            mb_search,alpha_min,alpha_max,alpha_interval=self._init_alpha_search(i+1,25)
                            if mb_search:
                                opt_alpha_tick, opt_score,output_del = golden_section(alpha_min, alpha_max,alpha_interval,stwr_func, self.tol,self.max_iter)
                            else: 
                                output_del = []
                                opt_score = stwr_func(0)
                                opt_bsita_tick = 0
                                opt_alpha_tick =0
                                output_del.append((opt_bsita_tick,opt_score))
                            opt_bsita_list.append(opt_bsita_tick)
                            opt_score_list.append(opt_score)
                            opt_tick_list.append(i)
                            opt_alpha_list.append(opt_alpha_tick)
                            opt_gwr_bw0_list.append(gwr_bw0_tick)
                    elif self.search_method == 'interval':
                            mb_search,alpha_min,alpha_max,alpha_interval=self._init_alpha_search(i+1,25)
                            if (mb_search):
                                opt_alpha_tick,opt_score,output_del = equal_interval(alpha_min, alpha_max,alpha_interval,stwr_func)
                            else:
                                output_del = []
                                opt_score = stwr_func(0)
                                opt_alpha_tick =0
                                opt_bsita_tick = 0
                                output_del.append((opt_bsita_tick,opt_score))
                            opt_bsita_list.append(opt_bsita_tick)
                            opt_alpha_list.append(opt_alpha_tick)
                            opt_score_list.append(opt_score)
                            opt_tick_list.append(i)
                            opt_gwr_bw0_list.append(gwr_bw0_tick)
                    else:
                        raise TypeError('Unsupported computational search method ',
                                self.search_method)
            if len(opt_score_list) == 0:
                 raise TypeError('opt_score_list is empty,Please check your data!' )            
            index = opt_score_list.index(min(opt_score_list))          
            return  opt_alpha_list[index],opt_bsita_list[index],opt_tick_list[index],opt_gwr_bw0_list[index]
        
    def _init_alpha_search(self,n_tick,brk_ticks): 
            if(n_tick ==1):
                mb_search = False
                minalpha = 0
                maxalpha = 0
                intervel = 0
            else: 
                mb_search = True
                minalpha = 0
                maxalpha = 1
                intervel = 1.0/brk_ticks
            return mb_search,minalpha,maxalpha,intervel
                
                
    def _init_sita_section(self,n_tick,gwr_bw0,brk_ticks):           
            if(n_tick == 1):
                sita_min = 0
                sita_max = 0
                sita_delta=0
                mb_search = False
            else: 
                mb_search = True
                tick_tims = self.tick_timesIntls[-n_tick:]
                lencd = self.coordslist[-1].shape[0]
                lenwd = n_tick-1 
                sita_arr  = np.zeros((lencd,lenwd))
                gwr_bw0 =int(gwr_bw0) + 1
                gwr_bwmin = self.coordslist[-1].shape[1]+1
                dspmat0 =self.sorted_dspal_list[0][:,:gwr_bw0]
                bw_0_lis =dspmat0[:,-1:].reshape((-1,1))
                for i in range(n_tick-1):
                    sorted_dspmat_min = self.sorted_dspal_list[-(i+1)][:,:gwr_bwmin]
                    bw_tick_lis = sorted_dspmat_min[:,-1:].reshape((-1,1))
                    delt_jundge = bw_0_lis <= bw_tick_lis
                    if np.any(delt_jundge):
                            sita_min = 0
                            sita_max = 0 
                            mb_search = False
                    else:
                        tick_ticks = tick_tims[-(i+1):]
                        delt_tic = np.sum(tick_ticks)
                        delt_sita_tick =(bw_0_lis - bw_tick_lis)/delt_tic 
                        sita_arr[:,i:i+1] = np.arctan(delt_sita_tick)
                sita_max = sita_arr.min()
                sita_min = 0
                sita_delta = (sita_max - sita_min)/brk_ticks
            return mb_search,sita_min, sita_max,sita_delta      