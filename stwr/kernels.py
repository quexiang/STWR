# GWR and STWR kernel function specifications
# STWR kernels developed by Xiang Que (xiangq@uidaho.edu, quexiang@fafu.edu.cn)
# GWR kernels based on the MGWR 2.0.2 implementation by Taylor Oshan (tayoshan@gmail.com)

import copy

import numpy as np

# ---------------------------------------------------------------------------
# Optional numba JIT compilation for distance computations.
# When numba is available, local_cdist is JIT-compiled for speed;
# otherwise it falls back to pure Python.
# ---------------------------------------------------------------------------
try:
    from numba import njit
except ImportError:

    def njit(func):
        return func


@njit
def local_cdist(coords_i, coords, spherical):
    """
    Compute distances from a single point to a set of points.

    Parameters
    ----------
    coords_i : array (2,)
        Coordinates of the reference point (x, y) or (lon, lat).
    coords : array (n, 2)
        Coordinates of target points.
    spherical : bool
        If True, use Haversine distance (km). If False, use Euclidean.

    Returns
    -------
    array (n,)
        Distances from coords_i to each point in coords.
    """
    if spherical:
        dLat = np.radians(coords[:, 1] - coords_i[1])
        dLon = np.radians(coords[:, 0] - coords_i[0])
        lat1 = np.radians(coords[:, 1])
        lat2 = np.radians(coords_i[1])
        a = np.sin(
            dLat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0  # Earth's mean radius in km
        return R * c
    else:
        return np.sqrt(np.sum((coords_i - coords)**2, axis=1))


# ==============================================================================
# Spatial-only kernel (used by GWR and MGWR)
# ==============================================================================


class Kernel(object):
    """
    Spatial kernel for GWR/MGWR models.

    Computes kernel weights for a single calibration point i based on
    distances to all data points, using either a fixed bandwidth
    (distance) or an adaptive bandwidth (k-nearest-neighbor distance).

    Parameters
    ----------
    i : int
        Index of the calibration point.
    data : array (n, 2)
        Coordinates of all observations.
    bw : float or int
        Bandwidth value: fixed distance if ``fixed=True``, or number
        of nearest neighbors if ``fixed=False``.
    fixed : bool
        True for fixed (distance-based) bandwidth, False for adaptive.
    function : str
        Kernel function name: 'gaussian', 'bisquare', 'exponential',
        'triangular', 'uniform', 'quadratic', 'quartic'.
    eps : float
        Small tolerance to prevent division by zero in adaptive bandwidth.
    points : array, optional
        If provided, use as calibration locations instead of ``data``.
    spherical : bool
        If True, compute Haversine distances for lon/lat coordinates.
    """

    def __init__(self, i, data, bw=None, fixed=True, function='triangular',
                 eps=1.0000001, ids=None, points=None, spherical=False):

        if points is None:
            self.dvec = local_cdist(data[i], data, spherical).reshape(-1)
        else:
            self.dvec = local_cdist(points[i], data, spherical).reshape(-1)

        self.function = function.lower()

        if fixed:
            self.bandwidth = float(bw)
        else:
            # Adaptive bandwidth: use the distance to the bw-th nearest neighbor.
            # ``np.partition`` gives O(n) partial sort.
            self.bandwidth = np.partition(
                self.dvec,
                int(bw) - 1)[int(bw) - 1] * eps

        self.kernel = self._kernel_funcs(self.dvec / self.bandwidth)

        if self.function == "bisquare":  # Truncate weights beyond bandwidth
            self.kernel[(self.dvec >= self.bandwidth)] = 0

    def _kernel_funcs(self, zs):
        """
        Evaluate the kernel function on standardized distances zs = d / bw.

        Functions follow Anselin and Rey (2010), Table 5.4.
        """
        if self.function == 'triangular':
            return 1 - zs
        elif self.function == 'uniform':
            return np.ones(zs.shape) * 0.5
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - zs**2)
        elif self.function == 'quartic':
            return (15. / 16) * (1 - zs**2)**2
        elif self.function == 'gaussian':
            return np.exp(-0.5 * (zs)**2)
        elif self.function == 'bisquare':
            return (1 - (zs)**2)**2
        elif self.function == 'exponential':
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)





# ==============================================================================
# Spatiotemporal kernel dispatch functions
#
# These functions are the public API for building STWR weight matrices.
# They are registered in the ``ak`` and ``fk`` dispatch dictionaries in
# sel_bw.py and called by STWR._build_spt_W().
#
# Each function returns either:
#   - The kernel weight matrix W alone (rcdtype=0, default), or
#   - A tuple (W, dtmat_tol, dst_dtamplist) when rcdtype=1 (for prediction).
# ==============================================================================


def fix_spt_bisquare(coords_list, y_list, tick_times_intervel, sita,
                     tick_nums, gwr_bw0, points_list=None, alpha=0.3,
                     dspmat=None, dtmat=None, sorted_dmat=None,
                     mbpred=False, spherical=False, prediction=False,
                     rcdtype=0):
    """
    Fixed-bandwidth spatiotemporal bi-square kernel.

    Uses a fixed spatial bandwidth (distance) at each time stage,
    combined with a sigmoid temporal kernel via ``_SpatiotemporalKernel``.
    """
    gwr_bw_list = np.repeat(gwr_bw0, tick_nums).tolist()
    w = _SpatiotemporalKernel(
        coords_list, y_list, tick_times_intervel, sita,
        gwr_bw_list, function='spt_bisquare',
        points_list=points_list, dspmat_tol=dspmat, dtmat_tol=dtmat,
        sorted_dmat=sorted_dmat, alpha=alpha, mbpred=mbpred,
        spherical=spherical, pred=prediction)

    if rcdtype == 1:
        return w.kernel, w.dtmat_tol, w.dst_dtamplist
    else:
        return w.kernel


def adapt_spt_bisquare(coords_list, y_list, tick_times_intervel,
                       sita, tick_nums, gwr_bw0,
                       dspal_m_list=None, dsorteds_m_list=None,
                       d_t_list=None, dspmat=None, dtmat=None,
                       points_list=None, alpha=0.3, mbpred=False,
                       spherical=False, prediction=False, rcdtype=0):
    """
    Adaptive-bandwidth spatiotemporal bi-square kernel.

    Uses the k-nearest-neighbor distance as the spatial bandwidth at
    each time stage, combined with the sigmoid temporal kernel.
    """
    gwr_bw_list = np.repeat(gwr_bw0, tick_nums).tolist()
    w = _SpatiotemporalKernel(
        coords_list, y_list, tick_times_intervel, sita,
        bk_list=gwr_bw_list, fixed=False, function='spt_bisquare',
        dspal_mat_list=dspal_m_list,
        sorted_dspal_list=dsorteds_m_list,
        d_tmp_list=d_t_list,
        dspmat_tol=dspmat, dtmat_tol=dtmat,
        points_list=points_list,
        alpha=alpha, mbpred=mbpred, spherical=spherical, pred=prediction)

    if rcdtype == 1:
        return w.kernel, w.dtmat_tol, w.dst_dtamplist
    else:
        return w.kernel


def fix_spt_gwr_gaussian(coords_list, y_list, tick_times_intervel, sita,
                         tick_nums, gwr_bw0, points_list=None, alpha=0.3,
                         dspmat=None, dtmat=None, sorted_dmat=None,
                         mbpred=False, spherical=False, prediction=False,
                         rcdtype=0):
    """
    Fixed-bandwidth spatiotemporal Gaussian kernel.

    Uses a fixed spatial bandwidth (distance) at each time stage.
    """
    gwr_bw_list = np.repeat(gwr_bw0, tick_nums).tolist()
    w = _SpatiotemporalKernel(
        coords_list, y_list, tick_times_intervel, sita,
        gwr_bw_list, function='spt_gwr_gaussian',
        truncate=False, points_list=points_list,
        dspmat_tol=dspmat, dtmat_tol=dtmat,
        sorted_dmat=sorted_dmat, alpha=alpha, mbpred=mbpred,
        spherical=spherical, pred=prediction)

    if rcdtype == 1:
        return w.kernel, w.dtmat_tol, w.dst_dtamplist
    else:
        return w.kernel


def spt_gwr_gaussian(coords_list, y_list, tick_times_intervel,
                     sita, tick_nums, gwr_bw0,
                     dspal_m_list=None, dsorteds_m_list=None,
                     d_t_list=None, dspmat=None, dtmat=None,
                     points_list=None, alpha=0.3, mbpred=False,
                     spherical=False, prediction=False, rcdtype=0):
    """
    Adaptive-bandwidth spatiotemporal Gaussian kernel.

    Uses the k-nearest-neighbor distance as the spatial bandwidth at
    each time stage.
    """
    gwr_bw_list = np.repeat(gwr_bw0, tick_nums).tolist()
    w = _SpatiotemporalKernel(
        coords_list, y_list, tick_times_intervel, sita,
        bk_list=gwr_bw_list, fixed=False, function='spt_gwr_gaussian',
        truncate=False,
        dspal_mat_list=dspal_m_list,
        sorted_dspal_list=dsorteds_m_list,
        d_tmp_list=d_t_list,
        dspmat_tol=dspmat, dtmat_tol=dtmat,
        points_list=points_list,
        alpha=alpha, mbpred=mbpred, spherical=spherical, pred=prediction)

    if rcdtype == 1:
        return w.kernel, w.dtmat_tol, w.dst_dtamplist
    else:
        return w.kernel


# ==============================================================================
# Spatiotemporal distance computation
# ==============================================================================


def cspatiltemporaldist(cal_data_list1, cal_data_list2, y_valuelist,
                        bt_size, deta_t_list, spherical, pred=False):
    """
    Compute spatial and temporal distance matrices for STWR.

    Builds the block-diagonal spatial and temporal distance matrices
    that combine observations from the latest time stage with those
    from past stages.

    Parameters
    ----------
    cal_data_list1 : list of array
        Calibration coordinates per time stage (destination).
    cal_data_list2 : list of array
        Data coordinates per time stage (source).
    y_valuelist : list of array
        Dependent variable values per time stage. Used to compute
        temporal distance as the rate of value change.
    bt_size : int
        Number of time stages to use.
    deta_t_list : list of float
        Time intervals between stages.
    spherical : bool
        If True, use Haversine distances.
    pred : bool
        If True, only compute spatial distances (temporal distances
        are set to zero, to be estimated separately during prediction).

    Returns
    -------
    dspatialmat_list : list of array
        Spatial distance matrices, one per time stage.
    dtemporalmat_list : list of array
        Temporal distance matrices, one per time stage.
    dspatialmat_tol : array
        Horizontally concatenated spatial distance matrix.
    detemporalmat_tol : array
        Horizontally concatenated temporal distance matrix.
    """
    # Determine sizes
    nsize = len(cal_data_list1[-1])          # Number of points in latest stage
    msize_0 = len(cal_data_list2[-1])        # Number of points in latest stage (source)
    mlist = [msize_0]
    msize = msize_0
    for p in range(bt_size - 1):
        tick_size = len(cal_data_list2[-(p + 2)])
        msize += tick_size
        mlist.append(tick_size)

    # Allocate combined matrices
    dspatialmat_tol = np.zeros((nsize, msize))
    detemporalmat_tol = np.zeros((nsize, msize))
    dspatialmat = local_cdist(cal_data_list1[-1], cal_data_list2[-1], spherical)
    dtemporalmat = np.zeros((nsize, msize_0))
    dspatialmat_list = []
    dtemporalmat_list = []

    # Latest time stage: no temporal distance (it is the reference stage)
    dspatialmat_tol[0:nsize, 0:msize_0] = dspatialmat
    dspatialmat_list.append(dspatialmat)

    mb_caltmp = (bt_size != 1)  # Only compute temporal distances if > 1 stage

    if not mb_caltmp:
        detemporalmat_tol[0:nsize, 0:msize_0] = dtemporalmat
        dtemporalmat_list.append(dtemporalmat)
    elif not pred:
        # --- Calibration mode: compute temporal distances ---
        delta_t_total = np.sum(deta_t_list) * 1.0
        m_size_tick = msize_0

        # First stage: temporal distance is zero (reference)
        dtempfirst = np.zeros((nsize, mlist[0]))
        detemporalmat_tol[:nsize, :mlist[0]] = dtempfirst
        dtemporalmat_list.append(dtempfirst)

        y_value_0 = y_valuelist[-1]  # Latest stage y values (reference)

        for i in range(bt_size - 1):
            dspatialmat_tick = local_cdist(
                cal_data_list1[-1], cal_data_list2[-(2 + i)], spherical)
            dtemporalmat_tick = np.zeros((nsize, mlist[i + 1]))

            delt_tick_tol = np.sum(deta_t_list[-(2 + i):])
            y_value_tick = y_valuelist[-(i + 2)].flatten()

            # Temporal distance = normalized rate of y-value change:
            #   d_tmp = delta_t_total * |(y_tick - y_0) / y_tick| / delt_tick_tol
            # This captures how much each past observation's value has changed
            # relative to the latest observation at the same location.
            y_value_tick = np.where(y_value_tick == 0, 1e-6, y_value_tick)
            for j in range(nsize):
                ydelt_j = np.repeat(y_value_0[j], mlist[i + 1], axis=0)
                dtemporalmat_tick[j] = (
                    delta_t_total *
                    np.abs((y_value_tick - ydelt_j) / y_value_tick) /
                    delt_tick_tol)

            dspatialmat_tol[:nsize,
                            m_size_tick:m_size_tick + mlist[i + 1]] = dspatialmat_tick
            detemporalmat_tol[:nsize,
                              m_size_tick:m_size_tick + mlist[i + 1]] = dtemporalmat_tick
            dspatialmat_list.append(dspatialmat_tick)
            dtemporalmat_list.append(dtemporalmat_tick)
            m_size_tick += mlist[i + 1]
    else:
        # --- Prediction mode: spatial distances only ---
        m_size_tick = msize_0
        detemporalmat_tol = dtemporalmat  # Temporal distances filled later
        for i in range(bt_size - 1):
            dspatialmat_tick = local_cdist(
                cal_data_list1[-1], cal_data_list2[-(2 + i)], spherical)
            dspatialmat_tol[:nsize,
                            m_size_tick:m_size_tick + mlist[i + 1]] = dspatialmat_tick
            dspatialmat_list.append(dspatialmat_tick)
            m_size_tick += mlist[i + 1]

    return (dspatialmat_list, dtemporalmat_list,
            dspatialmat_tol, detemporalmat_tol)


def spatialtemporalkernel_funcs(fname, d_spa, d_tmp, m_dtm0, alpha=0.3):
    """
    Evaluate the spatiotemporal kernel function.

    Combines a spatial kernel with a sigmoid temporal kernel using the
    ``alpha`` weighting parameter.

    Parameters
    ----------
    fname : str
        Kernel function name (e.g., 'spt_bisquare', 'spt_gwr_gaussian').
    d_spa : array
        Standardized spatial distances (d / bandwidth).
    d_tmp : array
        Temporal distances (rate-of-change based).
    m_dtm0 : bool
        If True, use spatial-only kernel (single time stage).
    alpha : float
        Balance parameter in [0, 1]. 0 = spatial only, 1 = temporal only.

    Returns
    -------
    array : Kernel weights.

    Notes
    -----
    The combined kernel is:

        W = (1 - alpha) * K_spatial(d_spa) + alpha * K_temporal(d_tmp)

    where K_temporal(d) = 2 / (1 + exp(-d)) - 1  (shifted sigmoid).
    For 'spt_gwr_gaussian', a multiplicative combination is used instead.
    """
    if m_dtm0:
        # Spatial-only kernel (single time stage)
        if fname == 'spt_triangular':
            return (1 - d_spa)
        elif fname == 'spt_uniform':
            return np.ones(d_spa.shape) * 0.5
        elif fname == 'spt_quadratic':
            return (3. / 4) * (1 - d_spa ** 2)
        elif fname == 'spt_quartic':
            return (15. / 16) * (1 - d_spa ** 2) ** 2
        elif fname == 'spt_gaussian':
            c = np.pi * 2
            c = c ** (-0.5)
            return c * np.exp(-(d_spa ** 2) / 2.)
        elif fname == 'spt_gwr_gaussian':
            return np.exp(-0.5 * (d_spa)**2)
        elif fname == 'spt_bisquare':
            return (1 - (d_spa)**2)**2
        elif fname == 'spt_exponential':
            return np.exp(-d_spa)
        else:
            print('Unsupported kernel function', fname)
    else:
        # Combined spatiotemporal kernel
        # Temporal component: shifted sigmoid = 2/(1+exp(-d)) - 1, range [0,1]
        if fname == 'spt_triangular':
            return ((1 - alpha) * (1 - d_spa)
                    + alpha * (2 / (1 + np.exp(-d_tmp)) - 1))
        elif fname == 'spt_uniform':
            return ((1 - alpha) * np.ones(d_spa.shape) * 0.5
                    + alpha * (2 / (1 + np.exp(-d_tmp)) - 1))
        elif fname == 'spt_quadratic':
            return ((1 - alpha) * (3. / 4) * (1 - d_spa ** 2)
                    + alpha * (2 / (1 + np.exp(-d_tmp)) - 1))
        elif fname == 'spt_quartic':
            return ((1 - alpha) * (15. / 16)
                    * (1 - d_spa ** 2) ** 2
                    + alpha * (2 / (1 + np.exp(-d_tmp)) - 1))
        elif fname == 'spt_gaussian':
            c = np.pi * 2
            c = c ** (-0.5)
            return ((1 - alpha) * (c * np.exp(-(d_spa ** 2) / 2.))
                    + alpha * (2 / (1 + np.exp(-d_tmp)) - 1))
        elif fname == 'spt_gwr_gaussian':
            # Multiplicative combination for Gaussian STWR
            return (np.exp(-0.5 * (d_spa)**2)
                    * (1 / (1 + np.exp(-d_tmp)) - 0.5))
        elif fname == 'spt_bisquare':
            return ((1 - alpha) * ((1 - (d_spa)**2)**2)
                    + alpha * (2 / (1 + np.exp(-d_tmp)) - 1))
        elif fname == 'spt_exponential':
            return ((1 - alpha) * np.exp(-d_spa)
                    + alpha * (2 / (1 + np.exp(-d_tmp)) - 1))
        else:
            print('Unsupported kernel function', fname)

class _SpatiotemporalKernel(object):
    """
    Spatiotemporal kernel for STWR models.

    Computes a combined spatial-temporal weight matrix that weights
    observations from multiple time stages. The spatial component uses
    standard GWR kernel functions (Gaussian, bi-square, exponential),
    while the temporal component uses a sigmoid function driven by the
    rate of value change between time stages.

    The combined weight at location i for past observation j is:

        W_ij = (1 - alpha) * K_spatial(d_ij / bw) + alpha * K_temporal(dt_ij)

    where K_temporal(dt) = 2/(1 + exp(-dt)) - 1, a shifted sigmoid.

    The spatial bandwidth can shrink linearly over time stages via the
    ``sita`` parameter: bw_t = bw_0 - tan(sita) * delta_t.

    Parameters
    ----------
    data_list : list of array
        Coordinates for each time stage, oldest to newest.
    y_list : list of array
        Dependent variable for each time stage.
    tick_times_intervel : list of float
        Time intervals between consecutive stages.
    sita : float
        Slope of bandwidth change over time (radians). Spatial bandwidth
        shrinks as bw_t = bw_0 - tan(sita) * delta_t.
    gwr_bw_list : list of float, optional
        Fixed bandwidth values per time stage.
    bk_list : list of int, optional
        Adaptive bandwidth (k nearest neighbors) per time stage.
    fixed : bool
        True for fixed (distance) bandwidth, False for adaptive.
    function : str
        Kernel function name (e.g., 'spt_bisquare').
    eps : float
        Small value added to weights to avoid zero-variance issues.
    truncate : bool
        If True, set weights to zero beyond the bandwidth.
    points_list : list of array, optional
        Prediction point coordinates per time stage.
    dspal_mat_list : list of array, optional
        Pre-computed spatial distance matrices.
    sorted_dspal_list : list of array, optional
        Pre-computed sorted spatial distances.
    d_tmp_list : list of array, optional
        Pre-computed temporal distance matrices.
    dspmat_tol : array, optional
        Horizontally concatenated spatial distance matrix.
    dtmat_tol : array, optional
        Horizontally concatenated temporal distance matrix.
    alpha : float
        Balance parameter in [0, 1]. 0 = spatial only, 1 = temporal only.
    mbpred : bool
        If True, use memory-efficient prediction mode.
    spherical : bool
        If True, compute Haversine distances.
    pred : bool
        If True, prediction mode (spatial distances only for prediction
        points with temporal distances estimated via IDW).
    """

    def __init__(self, data_list, y_list, tick_times_intervel,
                 sita=None, gwr_bw_list=None,
                 bk_list =None, fixed=True,function='spt_bisquare',eps=1.0000001,
                 truncate=True,points_list=None, dspal_mat_list=None,sorted_dspal_list=None,
                 d_tmp_list=None,dspmat_tol=None,dtmat_tol=None,alpha =0.3,mbpred=False,spherical=False,pred = False,seed = None):       
        #datalens = len(data_list) 
        # if issubclass(type(data_list[0]), scipy.spatial.KDTree):
        #     for i in range(datalens):
        #         self.data_list[i] = data_list[i].data
        #         data_list[i] = self.data_list[i]
        # else:
        #     self.data_list = data_list
        if bk_list is not None:
            self.bk_list = bk_list
            self.nbt_len = len(bk_list)
            for i in range(self.nbt_len):
                self.bk_list[i] = int(bk_list[i]) + 1     
        else:
            self.bk_list = bk_list          
        if gwr_bw_list is not None:
            self.gwr_bw_list = gwr_bw_list
            self.nbt_len = len(gwr_bw_list)
        else:
            self.gwr_bw_list = gwr_bw_list          
        self.y_val_list = y_list[-self.nbt_len:]
        self.tick_times_intls = tick_times_intervel[-self.nbt_len:]
        self.sita = sita             
        self.spherical = spherical
        self.searching = True         
        self.fname = function.lower()
        self.eps = eps      
        self.m_minbw = data_list[-1].shape[1]+1#最小带宽
        m_eps = self.eps-1
        m_dtm0 = False
        self.alpha = alpha
        self.mbpred = mbpred
        self.pred = pred        
        self.dst_dtamplist = None  
        self.pre_masktol = None
        
        nsizes =len(data_list[-1])
        mlist = [nsizes] 
        msizetol = len(data_list[-1])
        for p in range(self.nbt_len-1):
            tick_size = len(data_list[-(p+2)])
            msizetol += tick_size 
            mlist.append(tick_size)
        
        self.fixed = fixed
        self.trunc = truncate
        if seed is None:
            np.random.seed(5536)
        else:
            np.random.seed(seed)
        if(self.nbt_len == 1):
            m_dtm0 = True
        if dspmat_tol is None:
            self.searching = False
        if self.searching:
            self.d_spa_list = dspal_mat_list[:self.nbt_len]
            self.dsorted_spa =sorted_dspal_list[:self.nbt_len] 
            self.d_tmp_list = d_tmp_list[:self.nbt_len]
 
            m_curml = mlist[:self.nbt_len]
            m_curtol = sum(m_curml)
            self.dspmat_tol = dspmat_tol[:nsizes,:m_curtol]
            m_tmp = np.zeros(self.dspmat_tol.shape)       
            if m_dtm0:
                self.d_tmp_list[0] = m_tmp

            else: 
                m_tmp[:nsizes,:mlist[0]] =m_eps*np.random.rand(nsizes,mlist[0])

                m_size_tick = mlist[0]
                for i in range(self.nbt_len-1):
                    m_tmp_tick = dtmat_tol[:nsizes,m_size_tick:(m_size_tick+mlist[i+1])]
                    m_tmp[:nsizes,m_size_tick:(m_size_tick+mlist[i+1])] =m_tmp_tick
                    m_size_tick += mlist[i+1]
                    self.d_tmp_list[i+1] = m_tmp_tick
            self.dtmat_tol = m_tmp        
        else:

            if self.sita is None:
                raise TypeError('Please Enter a sita ',
                                self.sita)

            if points_list is None:
                self.data_list = self.data_list[-self.nbt_len:]
                self.d_spa_list, self.d_tmp_list,self.dspmat_tol,self.dtmat_tol = cspatiltemporaldist(self.data_list, self.data_list,
                                                    self.y_val_list ,self.nbt_len,
                                                    self.tick_times_intls,
                                                    self.spherical)
                self.dst_dtamplist = copy.deepcopy(self.d_tmp_list)
            else:
                self.points_list = points_list[-self.nbt_len:]
                d_spa_Pre_list,d_Pre_tmplist,d_Pre_spamat_tol,d_Pre_tmat_tol =cspatiltemporaldist(self.points_list, self.data_list, 
                                                             self.y_val_list,self.nbt_len,
                                                             self.tick_times_intls,
                                                             self.spherical,self.pred) 
                msizetol = np.sum(mlist)
                if m_dtm0 :
                    self.d_spa_list = d_spa_Pre_list
                    self.dspmat_tol = d_Pre_spamat_tol
                    self.d_tmp_list = d_Pre_tmplist
                    self.dtmat_tol  = d_Pre_tmat_tol
                else:

                    len_pred = len(self.points_list[0])
                    self.my_bw_list = np.zeros((self.nbt_len,1)).tolist()
                    mask_find_neighbors = d_spa_Pre_list[0].copy()
                    if self.fixed == False:
                        d_sort_spa_tick = d_spa_Pre_list[0].copy()
                        d_sort_spa_tick = np.sort(d_sort_spa_tick)
                        dspmat0 = d_sort_spa_tick[:,:self.bk_list[-1]]
                        dspmat_tick = d_sort_spa_tick[:,1:2] 
                        dspmat_last = d_sort_spa_tick[:,-1:]
                        delt_jundge = dspmat_tick == dspmat_last
                        add_neighbor=1
                        while np.any(delt_jundge):
                            dspmat0 = d_sort_spa_tick[:,:self.bk_list[-1]+add_neighbor]
                            dspmat_tick = dspmat0[:,1:2] 
                            dspmat_last = dspmat0[:,-1:]
                            delt_jundge = dspmat_tick == dspmat_last
                            add_neighbor = add_neighbor+1
                        self.my_bw_list[-1] = dspmat0.max(axis=1) * self.eps
                        self.my_bw_list[-1].shape = (self.my_bw_list[-1].size, 1)
                    if self.trunc:
                            self.pre_masktol = np.zeros((len_pred,msizetol))
                            if gwr_bw_list is not None:
                                  self.my_bw_list =self.gwr_bw_list
                            mask_pre_tick = self.my_bw_list[-1].copy()
                            mask_pre_tick = np.repeat(mask_pre_tick,mlist[0], axis=1)                     
                            mask_find_neighbors[(mask_find_neighbors>mask_pre_tick)]=0
                            self.pre_masktol[:,:mlist[0]] =mask_pre_tick

                    tol_pre_col = mask_find_neighbors.shape[1]
                    tmp_cal_matrix = d_tmp_list[1].copy() 
                    for cal_timetick in range(self.nbt_len-2):
                        tmp_cal_tick = d_tmp_list[cal_timetick+2].copy()
                        tmp_cal_matrix = np.hstack((tmp_cal_matrix,tmp_cal_tick))
                    d_pre_tmpweight = np.zeros((len_pred,msizetol-mlist[0]))
                    for Pre_row in  range(len_pred):
                        compress_matrix = []
                        prev_distances = []
                        for Pre_col in range(tol_pre_col):
                            if(mask_find_neighbors[Pre_row,Pre_col]>0):
                                no_zero_val = d_Pre_spamat_tol[Pre_row,Pre_col+mlist[0]] 
                                if no_zero_val == 0:
                                      no_zero_val = d_Pre_spamat_tol[Pre_row,Pre_col+mlist[0]]+m_eps              
                                pv_dist = 1.0/no_zero_val
                                prev_distances.append(pv_dist)
                                compress_matrix.append(pv_dist*tmp_cal_matrix[Pre_col])
                        compress_matrix = np.asarray(compress_matrix)
                        tol_distances = np.sum(prev_distances)
                        compress_matrix= compress_matrix/tol_distances
                        d_pre_tmpweight[Pre_row] = np.sum(compress_matrix, axis=0)
                        #simple Mean
                               #d_pre_tmpweight[Pre_row] = np.mean(compress_matrix, axis=0)
                        #IDW     
                    self.d_spa_list = d_spa_Pre_list
                    self.dspmat_tol = d_Pre_spamat_tol
                    
                    self.d_tmp_list = d_Pre_tmplist
                    d_Pre_tmat_tol = np.hstack((d_Pre_tmat_tol,d_pre_tmpweight))
                    self.dtmat_tol = d_Pre_tmat_tol

        if self.pred:
                 self.my_bw_list = np.zeros((self.nbt_len,1)).tolist()
                 self._set_spt_sita()
                 self.kernel = np.zeros(self.dspmat_tol.shape) 
                 len_pred = len(self.points_list[0])            
                 cal_sptol = self.d_spa_list[0]
                 if m_dtm0:
                    cal_sptol=cal_sptol/self.my_bw_list[-1]
                    cal_my_ttol = self.d_tmp_list[0]
                    self.kernel[:,:mlist[0]] = spatialtemporalkernel_funcs(self.fname,cal_sptol,cal_my_ttol,m_dtm0,self.alpha)
                 else:
                    m_bwcop =  self.my_bw_list[-1].copy()
                    cal_sptol = cal_sptol/m_bwcop
                    cal_my_ttol = self.dtmat_tol[:,:mlist[0]]
                    if self.trunc:
                        cal_sptol[cal_sptol>=1]=1 
                        cal_my_ttol[cal_sptol>=1] =0
                    self.kernel[:,:mlist[0]] = spatialtemporalkernel_funcs(self.fname,cal_sptol,cal_my_ttol,True,self.alpha) 
                 
                 m_size_tick = mlist[0]
                 for i in range(self.nbt_len-1):
                     self.my_bw_list[-(2+i)] =  self.my_bw_list[-(i+1)] - np.tan(sita)*self.tick_times_intls[-(i+1)]
                     bw_expand =self.my_bw_list[-(2+i)].copy()
                     d_sort_spa_tick = self.d_spa_list[i+1].copy()
                     d_sort_spa_tick = np.sort(d_sort_spa_tick)
                     sorted_dspmat_min = d_sort_spa_tick[:,:self.m_minbw]
                     dspmat_tick = sorted_dspmat_min[:,1:2]
                     dspmat_last = sorted_dspmat_min[:,-1:]
                     delt_jundge_same = dspmat_tick == dspmat_last
                     add_neighbor=1
                     while np.any(delt_jundge_same):
                          sorted_dspmat_min = d_sort_spa_tick[:,:self.m_minbw+add_neighbor]
                          dspmat_tick = sorted_dspmat_min[:,1:2] 
                          dspmat_last = sorted_dspmat_min[:,-1:]
                          delt_jundge_same = dspmat_tick == dspmat_last
                          add_neighbor = add_neighbor+1 
                     cal_bw_min = sorted_dspmat_min[:,-1:].reshape((-1,1)) 
                     delt_jundge = bw_expand < cal_bw_min
                     if np.any(delt_jundge):
                         np.copyto(bw_expand,cal_bw_min*self.eps,where=delt_jundge) 
                     cal_sptol_tick = self.d_spa_list[i+1].copy()
                     cal_tmp_tick = self.dtmat_tol[:,m_size_tick:(m_size_tick+mlist[i+1])]
                     if self.trunc:
                         mask_tick = np.repeat(bw_expand,mlist[i+1], axis=1)
                         cal_sptol_tick = cal_sptol_tick/bw_expand
                         cal_sptol_tick[cal_sptol_tick>=1]=1
                         cal_tmp_tick[cal_sptol_tick>=1] =0 
                         self.pre_masktol[:,m_size_tick:(m_size_tick+mlist[i+1])]=mask_tick
                     else:
                         cal_sptol_tick = cal_sptol_tick/bw_expand  
                        
                     self.kernel[:,m_size_tick:(m_size_tick+mlist[i+1])]=spatialtemporalkernel_funcs(self.fname,
                             cal_sptol_tick,cal_tmp_tick,m_dtm0,self.alpha)
                     m_size_tick +=mlist[i+1]
                 if self.pre_masktol is not None:
                     self.kernel[(self.dspmat_tol > self.pre_masktol)] = 0
                 self.kernel += np.random.normal(0,m_eps,self.kernel.shape)
        else:
            if gwr_bw_list is not None:
                try:
                    self.gwr_bw_list[-1] = np.array(self.gwr_bw_list[-1])
                    self.gwr_bw_list[-1].shape = (len(self.gwr_bw_list[-1]),1)
                except:
                    self.gwr_bw_list[-1] = np.ones((len(data_list[-1]),1),'float')*self.gwr_bw_list[-1] 
                for i in range(self.nbt_len-1):
                    self.gwr_bw_list[-(2+i)] =  self.gwr_bw_list[-(i+1)] - np.tan(sita)*self.tick_times_intls[-(i+1)]
                    try:
                        self.gwr_bw_list[-(2+i)] = np.array(self.gwr_bw_list[-(2+i)])
                        self.gwr_bw_list[-(2+i)].shape = (len(self.gwr_bw_list[-(2+i)]), 1)
                    except:
                        self.gwr_bw_list[-(2+i)] = np.ones((len(data_list[-(2+i)]), 1), 'float') * self.gwr_bw_list[-(2+i)]               
            else:
                     self.my_bw_list = np.zeros((self.nbt_len,1)).tolist()
                     self._set_spt_sita()
            spa_mat = self.d_spa_list[0]
            nsizes = spa_mat.shape[0]           
            cal_sptol = spa_mat.copy()          
            self.dspmat_tol[:nsizes,:mlist[0]]=spa_mat
            if self.searching and m_dtm0:          
                self.kernel = m_eps*np.random.rand(self.dspmat_tol.shape[0],self.dspmat_tol.shape[1])
            else:
                self.kernel = np.zeros(self.dspmat_tol.shape)       
            mask_tol = np.zeros((nsizes,msizetol)) 
            if self.trunc:
                    if gwr_bw_list is not None:
                          self.my_bw_list =self.gwr_bw_list
                    mask_tick = self.my_bw_list[-1].copy()
                    mask_tick = np.repeat(mask_tick,mlist[0], axis=1)
                    mask_tol[:nsizes,:mlist[0]] = mask_tick
                    cal_sptol[(cal_sptol>mask_tick)]=0
            if m_dtm0:
                cal_sptol=cal_sptol/self.my_bw_list[-1]
                cal_my_ttol = self.d_tmp_list[0]
                self.kernel[:nsizes,:mlist[0]] = spatialtemporalkernel_funcs(self.fname,cal_sptol,cal_my_ttol,m_dtm0,self.alpha)
            else:
                m_bwcop =  self.my_bw_list[-1].copy()
                cal_sptol = cal_sptol/m_bwcop
                cal_my_ttol = self.d_tmp_list[0]
                self.kernel[:nsizes,:mlist[0]] = spatialtemporalkernel_funcs(self.fname,cal_sptol,cal_my_ttol,m_dtm0,self.alpha) 
            m_size_tick = mlist[0]   
            for i in range(self.nbt_len-1):
                     if gwr_bw_list is not None:
                         self.my_bw_list =self.gwr_bw_list  
                     self.my_bw_list[-(2+i)] =  self.my_bw_list[-(i+1)] - np.tan(sita)*self.tick_times_intls[-(i+1)]
                     bw_expand =self.my_bw_list[-(2+i)].copy()
                     if self.searching:
                         sorted_dspmat_min = self.dsorted_spa[i+1][:,:self.m_minbw]
                         cal_bw_min = sorted_dspmat_min[:,-1:].reshape((-1,1)) 
                         delt_jundge = bw_expand < cal_bw_min
                         if np.any(delt_jundge):
                             np.copyto(bw_expand,cal_bw_min*self.eps,where=delt_jundge)
                     spa_mat_tick= self.d_spa_list[i+1]
                     cal_sptol_tick = spa_mat_tick.copy()
                     self.dspmat_tol[:nsizes,m_size_tick:(m_size_tick+mlist[i+1])]=spa_mat_tick
                     if self.trunc:
                         mask_tick = np.repeat(bw_expand,mlist[i+1], axis=1)
                         cal_sptol_tick[(cal_sptol_tick>mask_tick)]=0
                         mask_tol[:nsizes,m_size_tick:(m_size_tick+mlist[i+1])]=mask_tick
                   
                     cal_sptol_tick= cal_sptol_tick/bw_expand
                     self.kernel[:nsizes,m_size_tick:(m_size_tick+mlist[i+1])]=spatialtemporalkernel_funcs(self.fname,
                             cal_sptol_tick,self.d_tmp_list[i+1],m_dtm0,self.alpha)
                     m_size_tick +=mlist[i+1]
            self.kernel[(self.dspmat_tol > mask_tol)] = 0
            self.kernel += np.random.normal(0,m_eps,self.kernel.shape)
        
    def _set_spt_sita(self):
        """
        Determine the initial spatial bandwidth for the latest time stage.

        For fixed bandwidth: uses the maximum distance across all neighbors.
        For adaptive bandwidth: uses the local k-nearest-neighbor distance.
        """
        if self.searching:
            if self.bk_list is not None:
                dspmat0 = self.dsorted_spa[0][:, :self.bk_list[-1]]
            else:
                dspmat0 = self.d_spa_list[0]
        else:
            if self.bk_list is not None:
                dspmat0 = np.sort(self.d_spa_list[0])[:, :self.bk_list[-1]]
            else:
                dspmat0 = self.d_spa_list[0]
        if self.fixed:
            bandwidth0 = dspmat0.max() * self.eps
            n = len(self.data_list[-1])
            self.my_bw_list[-1] = np.ones((n, 1), 'float') * bandwidth0
        else:
            self.my_bw_list[-1] = dspmat0.max(axis=1) * self.eps
            self.my_bw_list[-1].shape = (self.my_bw_list[-1].size, 1)
