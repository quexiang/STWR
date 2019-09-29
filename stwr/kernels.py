# GWR kernel function specifications

__author__ = "STWR is XiangQue xiangq@uidaho.edu and GWR,MGWR is Taylor Oshan Tayoshan@gmail.com"

import scipy
from scipy.spatial.kdtree import KDTree
import numpy as np
from scipy.spatial.distance import cdist as cdist_scipy
from math import radians, sin, cos, sqrt, asin,exp,atan,tan
import copy

#adaptive specifications should be parameterized with nn-1 to match original gwr
#implementation. That is, pysal counts self neighbors with knn automatically.

def fix_gauss(coords, bw, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Fixed Gaussian kernel.
    """
    w = _Kernel(coords, function='gwr_gaussian', bandwidth=bw,
            truncate=False, points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel

def adapt_gauss(coords, nn, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Spatially adaptive Gaussian kernel.
    """
    w = _Kernel(coords, fixed=False, k=nn-1, function='gwr_gaussian',
            truncate=False, points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel

def fix_bisquare(coords, bw, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Fixed bisquare kernel.
    """
    w = _Kernel(coords, function='bisquare', bandwidth=bw, points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel

def adapt_bisquare(coords, nn, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Spatially adaptive bisquare kernel.
    """
    w = _Kernel(coords, fixed=False, k=nn-1, function='bisquare', points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel

def fix_exp(coords, bw, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Fixed exponential kernel.
    """
    w = _Kernel(coords, function='exponential', bandwidth=bw,
            truncate=False, points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel

def adapt_exp(coords, nn, points=None, dmat=None,sorted_dmat=None,spherical=False):
    """
    Spatially adaptive exponential kernel.
    """
    w = _Kernel(coords, fixed=False, k=nn-1, function='exponential',
            truncate=False, points=points, dmat=dmat,sorted_dmat=sorted_dmat,spherical=spherical)
    return w.kernel
def fix_spt_bisquare(coords_list,y_list,tick_times_intervel,sita, tick_nums,gwr_bw0,
                     points_list=None, alpha =0.3,dspmat=None,dtmat=None,sorted_dmat=None,mbpred = False,spherical=False,prediction =False,rcdtype = 0):
    """
    Fixed spatiotemporal kernel.
    """
    gwr_bw_list = np.repeat(gwr_bw0,tick_nums).tolist()
    w = _SpatiotemporalKernel(coords_list, y_list,tick_times_intervel,sita, gwr_bw_list,function='spt_bisquare',
             points_list=points_list, dspmat_tol=dspmat,dtmat_tol=dtmat,sorted_dmat=sorted_dmat,alpha =alpha,mbpred=mbpred,spherical=spherical,pred =prediction)
    
    if rcdtype ==1:
        return w.kernel,w.dtmat_tol,w.dst_dtamplist
    else:
        return w.kernel

def adapt_spt_bisquare(coords_list,y_list,tick_times_intervel,
                       sita, tick_nums,gwr_bw0,
                       dspal_m_list = None,dsorteds_m_list = None,
                       d_t_list = None,dspmat = None,dtmat=None,
                       points_list=None,alpha =0.3,mbpred = False,spherical=False,prediction =False,rcdtype = 0):
    """
    Spatially adaptive spatiotemporal kernel.
    """
    gwr_bw_list = np.repeat(gwr_bw0,tick_nums).tolist()
    w = _SpatiotemporalKernel(coords_list, y_list,tick_times_intervel,sita, 
                              bk_list=gwr_bw_list, fixed=False,function='spt_bisquare',
                              dspal_mat_list = dspal_m_list,
                              sorted_dspal_list=dsorteds_m_list,
                              d_tmp_list=d_t_list,
                              dspmat_tol=dspmat,dtmat_tol=dtmat,points_list=points_list,
                              alpha =alpha,mbpred = mbpred,spherical=spherical,pred =prediction)#,truncate=False
    
    if rcdtype ==1:
        return w.kernel,w.dtmat_tol,w.dst_dtamplist
    else:
        return w.kernel
def fix_spt_gwr_gaussian(coords_list,y_list,tick_times_intervel,sita, tick_nums,gwr_bw0,points_list=None,alpha =0.3,
                         dspmat=None,dtmat=None,sorted_dmat=None,mbpred = False,spherical=False,prediction =False,rcdtype = 0):
    """
    Fixed spatiotemporal kernel.
    """
    gwr_bw_list = np.repeat(gwr_bw0,tick_nums).tolist()
    w = _SpatiotemporalKernel(coords_list, y_list,tick_times_intervel,sita, gwr_bw_list,function='spt_gwr_gaussian',
             truncate=False,points_list=points_list, dspmat_tol=dspmat,dtmat_tol=dtmat,
             sorted_dmat=sorted_dmat,alpha =alpha,mbpred=mbpred,spherical=spherical,pred =prediction)
    
    if rcdtype ==1:
        return w.kernel,w.dtmat_tol,w.dst_dtamplist
    else:
        return w.kernel

def spt_gwr_gaussian(coords_list,y_list,tick_times_intervel,
                       sita, tick_nums,gwr_bw0,
                       dspal_m_list = None,dsorteds_m_list = None,
                       d_t_list = None,dspmat = None,dtmat=None,points_list=None,alpha =0.3,mbpred = False,
                       spherical=False,prediction =False,rcdtype = 0):
    """
    Spatially adaptive spatiotemporal kernel.
    """
    gwr_bw_list = np.repeat(gwr_bw0,tick_nums).tolist()
    w = _SpatiotemporalKernel(coords_list, y_list,tick_times_intervel,sita, 
                              bk_list=gwr_bw_list, fixed=False,function='spt_gwr_gaussian',
                              truncate=False,
                              dspal_mat_list = dspal_m_list,
                              sorted_dspal_list=dsorteds_m_list,
                              d_tmp_list=d_t_list,
                              dspmat_tol=dspmat,dtmat_tol=dtmat,points_list=points_list,
                              alpha =alpha,mbpred = mbpred,spherical=spherical,pred =prediction)#,truncate=False
    
    if rcdtype ==1:
        return w.kernel,w.dtmat_tol,w.dst_dtamplist
    else:
        return w.kernel

from scipy.spatial.distance import cdist

#Customized Kernel class user for GWR because the default PySAL kernel class
#favors memory optimization over speed optimizations and GWR often needs the 
#speed optimization since it is not always assume points far awary
# are truncated #to zero

def cdist(coords1,coords2,spherical):
    def _haversine(lon1, lat1, lon2, lat2):
        R = 6371.0 # Earth radius in kilometers
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
        c = 2*asin(sqrt(a))
        return R * c
    n = len(coords1)
    m = len(coords2)
#    dmat = np.zeros((n,n))
    dmat = np.zeros((n,m))

    if spherical:
        for i in range(n) :
            for j in range(m):
                dmat[i,j] = _haversine(coords1[i][0], coords1[i][1], coords2[j][0], coords2[j][1])
    else:
        dmat = cdist_scipy(coords1,coords2)

    return dmat

class _Kernel(object):
    """
    GWR kernel function specifications.

    """
    def __init__(self, data, bandwidth=None, fixed=True, k=None,
                 function='triangular', eps=1.0000001, ids=None, truncate=True,
                 points=None, dmat=None,sorted_dmat=None, spherical=False): #Added truncate flag
        

        if issubclass(type(data), scipy.spatial.KDTree):
            self.data = data.data
            data = self.data
        else:
            self.data = data
        if k is not None:
            self.k = int(k) + 1
        else:
            self.k = k    
        self.spherical = spherical
        self.searching = True
        
        if dmat is None:
            self.searching = False
        
        if self.searching:
            self.dmat = dmat
            self.sorted_dmat = sorted_dmat
        else:
            if points is None:
                self.dmat = cdist(self.data, self.data, self.spherical)
            else:
                self.points = points
                self.dmat = cdist(self.points, self.data, self.spherical)

        self.function = function.lower()
        self.fixed = fixed
        self.eps = eps
        self.trunc = truncate
        if bandwidth:
            try:
                bandwidth = np.array(bandwidth)
                bandwidth.shape = (len(bandwidth), 1)
            except:
                bandwidth = np.ones((len(data), 1), 'float') * bandwidth
            self.bandwidth = bandwidth
        else:
            self._set_bw()
        self.kernel = self._kernel_funcs(self.dmat/self.bandwidth)
        if self.trunc:
            mask = np.repeat(self.bandwidth, len(self.data), axis=1)
            self.kernel[(self.dmat >= mask)] = 0
                
    def _set_bw(self):
        if self.searching:
            if self.k is not None:
                dmat = self.sorted_dmat[:,:self.k]
            else:
                dmat = self.dmat
        else:
            if self.k is not None:
                dmat = np.sort(self.dmat)[:,:self.k]
            else:
                dmat = self.dmat
        
        if self.fixed:
            # use max knn distance as bandwidth
            bandwidth = dmat.max() * self.eps
            n = len(self.data)
            self.bandwidth = np.ones((n, 1), 'float') * bandwidth
        else:
            # use local max knn distance
            self.bandwidth = dmat.max(axis=1) * self.eps
            self.bandwidth.shape = (self.bandwidth.size, 1)

    def _kernel_funcs(self, zs):
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            return 1 - zs
        elif self.function == 'uniform':
            return np.ones(zs.shape) * 0.5
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - zs ** 2)
        elif self.function == 'quartic':
            return (15. / 16) * (1 - zs ** 2) ** 2
        elif self.function == 'gaussian':
            c = np.pi * 2
            c = c ** (-0.5)
            return c * np.exp(-(zs ** 2) / 2.)
        elif self.function == 'gwr_gaussian':
            return np.exp(-0.5*(zs)**2)
        elif self.function == 'bisquare':
            return (1-(zs)**2)**2
        elif self.function =='exponential':
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)

          
def cspatiltemporaldist(cal_data_list1,cal_data_list2,y_valuelist,bt_size,deta_t_list,spherical,pred = False):#sita,fname
    def _haversine(lon1, lat1, lon2, lat2):
        R = 6371.0 # Earth radius in kilometers
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
        c = 2*asin(sqrt(a))
        return R * c
    nsize =len(cal_data_list1[-1])
    msize_0 =len(cal_data_list2[-1]) 
    mlist = [msize_0] 
    msize = len(cal_data_list2[-1])
    for p in range(bt_size-1):
        tick_size = len(cal_data_list2[-(p+2)])
        msize += tick_size 
        mlist.append(tick_size)
    dspatialmat_tol = np.zeros((nsize,msize)) 
    detemporalmat_tol = np.zeros((nsize,msize)) 
    dspatialmat  = np.zeros((nsize,msize_0))
    dtemporalmat = np.zeros((nsize,msize_0)) 
    dspatialmat_list = []
    dtemporalmat_list = []
    mb_caltmp = True;
    if bt_size == 1:
        mb_caltmp = False
    if spherical:
        for i in range(nsize) :
            for j in range(msize_0):
                dspatialmat[i,j]  = _haversine(cal_data_list1[-1][i][0], cal_data_list1[-1][i][1], cal_data_list2[-1][j][0], cal_data_list2[-1][j][1])               
    else:
        dspatialmat = cdist_scipy(cal_data_list1[-1],cal_data_list2[-1])#/gwr_bw_list[-1]  

    dspatialmat_tol[0:nsize,0:msize_0] = dspatialmat
    dspatialmat_list.append(dspatialmat)
    if(mb_caltmp == False):
         detemporalmat_tol[0:nsize,0:msize_0] =dtemporalmat 
         dtemporalmat_list.append(dtemporalmat)  
    elif(pred == False):
        delta_t_total =np.sum(deta_t_list)*1.0
      
        m_size_tick = msize_0       
        dtempfirst = np.zeros((nsize,mlist[0]))
        detemporalmat_tol[:nsize,:mlist[0]]= dtempfirst
        dtemporalmat_list.append(dtempfirst)

        y_value_0 = y_valuelist[-1]
             
        for i in range(bt_size-1): 
            dspatialmat_tick  = np.zeros((nsize,mlist[i+1]))
            dtemporalmat_tick = np.zeros((nsize,mlist[i+1]))

            delt_tick_tol = np.sum(deta_t_list[-(2+i):])
            y_value_tick = y_valuelist[-(i+2)]                  
            if spherical:
                for j in range(nsize) :
                    for q in range(mlist[i+1]):
                        dspatialmat_tick[j,q]  = _haversine(cal_data_list1[-1][j][0], cal_data_list1[-1][j][1], cal_data_list2[-(2+i)][q][0], cal_data_list2[-(2+i)][q][1])
                        dtemporalmat_tick[j,q] =  delta_t_total*abs((y_value_tick[q]-y_value_0[j])/y_value_tick[q])/delt_tick_tol
            else:
                dspatialmat_tick = cdist_scipy(cal_data_list1[-1],cal_data_list2[-(2+i)])     
                y_value_tick = y_value_tick.flatten()
                for j in range(nsize) :
                    ydelt_j = np.repeat(y_value_0[j],mlist[i+1],axis=0)
                    dtemporalmat_tick[j] =  delta_t_total*(np.absolute(( y_value_tick- ydelt_j)/y_value_tick))/delt_tick_tol
                                           
            dspatialmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] = dspatialmat_tick    
            detemporalmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] =dtemporalmat_tick
            dspatialmat_list.append(dspatialmat_tick)
            dtemporalmat_list.append(dtemporalmat_tick)
            m_size_tick +=  mlist[i+1] 
    else:
         m_size_tick = msize_0 
         detemporalmat_tol = dtemporalmat
         for i in range(bt_size-1):
             dspatialmat_tick  = np.zeros((nsize,mlist[i+1]))
             if spherical:
                for j in range(nsize) :
                    for q in range(mlist[i+1]):
                        dspatialmat_tick[j,q]  = _haversine(cal_data_list1[-1][j][0], cal_data_list1[-1][j][1], cal_data_list2[-(2+i)][q][0], cal_data_list2[-(2+i)][q][1])
             else:
                dspatialmat_tick = cdist_scipy(cal_data_list1[-1],cal_data_list2[-(2+i)])#/gwr_bw_list[-(2+i)] 
             dspatialmat_tol[:nsize,m_size_tick:m_size_tick+mlist[i+1]] = dspatialmat_tick
             dspatialmat_list.append(dspatialmat_tick)
             m_size_tick = m_size_tick + mlist[i+1] 
    return dspatialmat_list,dtemporalmat_list,dspatialmat_tol,detemporalmat_tol


def spatialtemporalkernel_funcs(fname,d_spa,d_tmp,m_dtm0,alpha=0.3):
        # functions follow Anselin and Rey (2010) table 5.4
        if m_dtm0:
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
                return np.exp(-0.5*(d_spa)**2)
            elif fname == 'spt_bisquare':
                return (1-(d_spa)**2)**2
            elif fname =='spt_exponential':
                return np.exp(-d_spa)
            else:
                print('Unsupported kernel function',fname)
        else:
            if fname == 'spt_triangular':
                return ((1-alpha)*(1 - d_spa)+ alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname == 'spt_uniform':
                return ((1-alpha)*np.ones(d_spa.shape) * 0.5+ alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname == 'spt_quadratic':
                return ((1-alpha)*(3. / 4) * (1 - d_spa ** 2)+ alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname == 'spt_quartic':
                return ((1-alpha)*(15. / 16) * (1 - d_spa ** 2) ** 2 + alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname == 'spt_gaussian':
                c = np.pi * 2
                c = c ** (-0.5)
                return ((1-alpha)*(c * np.exp(-(d_spa ** 2) / 2.))+alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname == 'spt_gwr_gaussian':
                return np.exp(-0.5*(d_spa)**2)*(1/(1+np.exp(-d_tmp))-0.5)
            elif fname == 'spt_bisquare':
                return ((1-alpha)*((1-(d_spa)**2)**2)+alpha*(2/(1+np.exp(-d_tmp))-1))
            elif fname =='spt_exponential':
                return ((1-alpha)*np.exp(-d_spa)+alpha*(2/(1+np.exp(-d_tmp))-1))
            else:
                print('Unsupported kernel function',fname) 

class _SpatiotemporalKernel(object):
    """
    STWR Spatiotemporal kernel function specifications.

    """ 
    def __init__(self, data_list,y_list,tick_times_intervel,sita = None, gwr_bw_list = None,
                 bk_list =None, fixed=True,function='spt_bisquare',eps=1.0000001,
                 truncate=True,points_list=None, dspal_mat_list=None,sorted_dspal_list=None,
                 d_tmp_list=None,dspmat_tol=None,dtmat_tol=None,alpha =0.3,mbpred=False,spherical=False,pred = False):       
        datalens = len(data_list) 
        if issubclass(type(data_list[0]), scipy.spatial.KDTree):
            for i in range(datalens):
                self.data_list[i] = data_list[i].data
                data_list[i] = self.data_list[i]
        else:
            self.data_list = data_list
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
                     self.my_bw_list[-(2+i)] =  self.my_bw_list[-(i+1)] - tan(sita)*self.tick_times_intls[-(i+1)]
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
                    self.gwr_bw_list[-(2+i)] =  self.gwr_bw_list[-(i+1)] - tan(sita)*self.tick_times_intls[-(i+1)]
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
                     self.my_bw_list[-(2+i)] =  self.my_bw_list[-(i+1)] - tan(sita)*self.tick_times_intls[-(i+1)]
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
            if self.searching:
                if self.bk_list is not None:
                    dspmat0 =self.dsorted_spa[0][:,:self.bk_list[-1]]
                else:    
                    dspmat0 = self.d_spa_list[0] 
            else:
                if self.bk_list is not None:
                    dspmat0  = np.sort(self.d_spa_list[0])[:,:self.bk_list[-1]]
                else:
                    dspmat0 =  self.d_spa_list[0] 
            if self.fixed:
                bandwidth0 = dspmat0.max() * self.eps
                n = len(self.data_list[-1])
                self.my_bw_list[-1] = np.ones((n, 1), 'float') * bandwidth0
            else:
                self.my_bw_list[-1] = dspmat0.max(axis=1) * self.eps
                self.my_bw_list[-1].shape = (self.my_bw_list[-1].size, 1)
