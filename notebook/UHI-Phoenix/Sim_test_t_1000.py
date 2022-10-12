import  numpy  as  np
import libpysal as ps
from stwr.gwr import GWR, MGWR,STWR
from stwr.sel_bw import *
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
from matplotlib.gridspec import GridSpec
import multiprocessing as mp
import time
import csv 
import copy 
import rasterio
import rasterio.plot
import rasterio.features
import rasterio.warp
import pyproj
import os
from spglm.family import Gaussian, Binomial, Poisson
from spglm.glm import GLM, GLMResults

#read data read The image is 354 * 355 in size #95
r95gndvi = rasterio.open('D:/STWR/Data-Run/FanChao/1995/X2_GNDVI.tif')
b95gndvi = r95gndvi.read(1)
b95gndvi_f = b95gndvi[1:-1,1:]

r95gpnbi  = rasterio.open('D:/STWR/Data-Run/FanChao/1995/X1_GNDBI.tif')
b95gpnbi = r95gpnbi.read(1)
b95gpnbi_f = b95gpnbi[1:-1,1:]

r95moran =  rasterio.open('D:/STWR/Data-Run/FanChao/1995/X3_LNDVI.tif')
b95moran = r95moran.read(1)
b95moran_f = b95moran[1:-1,1:]

#00
r00gndvi = rasterio.open('D:/STWR/Data-Run/FanChao/2000/X2_GNDVI.tif')
b00gndvi = r00gndvi.read(1)
b00gndvi_f = b00gndvi[1:-1,1:]

r00gpnbi  = rasterio.open('D:/STWR/Data-Run/FanChao/2000/X1_GNDBI.tif')
b00gpnbi = r00gpnbi.read(1)
b00gpnbi_f = b00gpnbi[1:-1,1:]

r00moran =  rasterio.open('D:/STWR/Data-Run/FanChao/2000/X3_LNDVI.tif')
b00moran = r00moran.read(1)
b00moran_f = b00moran[1:-1,1:]

#05
r05gndvi = rasterio.open('D:/STWR/Data-Run/FanChao/2005/X2_GNDVI.tif')
b05gndvi = r05gndvi.read(1)
b05gndvi_f = b05gndvi[1:-1,1:]

r05gpnbi  = rasterio.open('D:/STWR/Data-Run/FanChao/2005/X1_GNDBI.tif')
b05gpnbi = r05gpnbi.read(1)
b05gpnbi_f = b05gpnbi[1:-1,1:]

r05moran =  rasterio.open('D:/STWR/Data-Run/FanChao/2005/X3_LNDVI.tif')
b05moran = r05moran.read(1)
b05moran_f = b05moran[1:-1,1:]

#10
r10gndvi = rasterio.open('D:/STWR/Data-Run/FanChao/2010/X2_GNDVI.tif')
b10gndvi = r10gndvi.read(1)
b10gndvi_f = b10gndvi[1:-1,1:]

r10gpnbi  = rasterio.open('D:/STWR/Data-Run/FanChao/2010/X1_GNDBI.tif')
b10gpnbi = r10gpnbi.read(1)
b10gpnbi_f = b10gpnbi[1:-1,1:]

r10moran =  rasterio.open('D:/STWR/Data-Run/FanChao/2010/X3_LNDVI.tif')
b10moran = r10moran.read(1)
b10moran_f = b10moran[1:-1,1:]

#15
r15gndvi = rasterio.open('D:/STWR/Data-Run/FanChao/2015/X2_GNDVI.tif')
b15gndvi = r15gndvi.read(1)
b15gndvi_f = b15gndvi[1:-1,1:]

r15gpnbi  = rasterio.open('D:/STWR/Data-Run/FanChao/2015/X1_GNDBI.tif')
b15gpnbi = r15gpnbi.read(1)
b15gpnbi_f = b15gpnbi[1:-1,1:]

r15moran =  rasterio.open('D:/STWR/Data-Run/FanChao/2015/X3_LNDVI.tif')
b15moran = r15moran.read(1)
b15moran_f = b15moran[1:-1,1:]

#2020
r20gndvi = rasterio.open('D:/STWR/Data-Run/FanChao/2020/X2_GNDVI.tif')
b20gndvi = r20gndvi.read(1)
b20gndvi_f = b20gndvi[1:-1,1:]

r20gpnbi  = rasterio.open('D:/STWR/Data-Run/FanChao/2020/X1_GNDBI.tif')
b20gpnbi = r20gpnbi.read(1)
b20gpnbi_f = b10gpnbi[1:-1,1:]

r20moran =  rasterio.open('D:/STWR/Data-Run/FanChao/2020/X3_LNDVI.tif')
b20moran = r20moran.read(1)
b20moran_f = b20moran[1:-1,1:]

#############################
pf = r95gndvi.profile
transform =r95gndvi.profile['transform']
nodata = pf['nodata']

time_dif = [0.0,5.0,5.0,5.0,5.0,5.0]
#simulation case 1 
mu_0  = 1
beta0= mu_0* np.ones((353,353))
mu_1 = 3
beta1 = mu_1* np.ones((353,353)) 
mu_2 = -1
beta2 = mu_2* np.ones((353,353)) 
mu_3 = -6
beta3 = mu_3* np.ones((353,353)) 

beta_min_0 = np.amin(beta0)  
beta_max_0 = np.amax(beta0)
beta_min = beta_min_0 
beta_max = beta_max_0
beta_min_1 = np.amin(beta1)  
beta_max_1 = np.amax(beta1)

if beta_min > beta_min_1:
    beta_min = beta_min_1
if beta_max < beta_max_1:
    beta_max = beta_max_1
    
beta_min_2 = np.amin(beta2)  
beta_max_2 = np.amax(beta2)
if beta_min > beta_min_2:
    beta_min = beta_min_2
if beta_max < beta_max_2:
    beta_max = beta_max_2
beta_min_3 = np.amin(beta3)  
beta_max_3 = np.amax(beta3)
if beta_min > beta_min_3:
    beta_min = beta_min_3
if beta_max < beta_max_3:
    beta_max = beta_max_3
beta_min_list =[beta_min_0,beta_min_1,beta_min_2,beta_min_3]
beta_max_list =[beta_max_0,beta_max_1,beta_max_2,beta_max_3]

beta_min_1 = np.amin(beta1) 
beta_max_1 = np.amax(beta1)
beta_min_list.append(beta_min_1)
beta_max_list.append(beta_max_1)
if beta_min_1<beta_min:
    beta_min =  beta_min_1
if beta_max_1> beta_max:
    beta_max =  beta_max_1
    
beta_min_2 = np.amin(beta2) 
beta_max_2 = np.amax(beta2)
beta_min_list.append(beta_min_2)
beta_max_list.append(beta_max_2)
if beta_min_2<beta_min:
    beta_min =  beta_min_2
if beta_max_2> beta_max:
    beta_max = beta_max_2
    
beta_min_3 = np.amin(beta3) 
beta_max_3 = np.amax(beta3)
beta_min_list.append(beta_min_3)
beta_max_list.append(beta_max_3)
if np.amin(beta3)<beta_min:
    beta_min =  beta_min_3
if np.amax(beta3)> beta_max:
    beta_max =  beta_max_3

#read coordinates from csv
csvFile = open("D:/STWR/Data-Run/FanChao/1995/Sample_point_100_1995.csv", "r")
df = pd.read_csv(csvFile,header = 0,names=['POINT_X','POINT_Y','X1_GNDBI','X2_GNDVI','X3_LNDVI','cal_y_LST','timestamp'],
                 dtype = {"POINT_X" : "float64","POINT_Y":"float64","X1_GNDBI":"float64",
                          "X2_GNDVI":"float64","X3_LNDVI":"float64","cal_y_LST":"float64","timestamp":"float64"},
                 skip_blank_lines = True,
                 keep_default_na = False)
all_data = df.values
# cal_coord_tick = []
row_col = []
lensdata = len(all_data)
samples_nums = []  

for row in range(lensdata): 
    row_tick,col_tick = r95gndvi.index(all_data[row,0], all_data[row,1])
    #cutted，c and r should -1
    row_col_ticks = np.array((row_tick-1, col_tick-1))
    row_col.append(row_col_ticks) 
    sam_tic = (row_tick-1)*353+(col_tick-1)
    samples_nums.append(sam_tic)
cal_cords = np.asarray(row_col)
samples_nums = np.asarray(samples_nums)

samples_len = samples_nums.shape[0]


ols_predbeta_results = []
ols_resid_results= []
ols_predy_results = []

gwr_prebeta0_results =[]
gwr_prebeta1_results =[]
gwr_prebeta2_results =[]
gwr_prebeta3_results =[]
gwr_resid_results = []
gwr_predy_results = []


stwr_prebeta0_results =[]
stwr_prebeta1_results =[]
stwr_prebeta2_results =[]
stwr_prebeta3_results =[]
stwr_resid_results = []
stwr_predy_results = []

random_times = 1000
for rand_times in range(random_times):
    #add error
    mu_err, sigma_err = 0, 2
    err = np.random.normal(mu_err, sigma_err, 6*353*353*1)
    err = err.reshape((6,353*353,1))
    
    lst95 = beta0 + beta1 * b95gpnbi_f + beta2 * b95gndvi_f + beta3 * b95moran_f + np.random.choice((-1,1))*err[0].reshape((353,353))
    lst00 = beta0 + beta1 * b00gpnbi_f + beta2 * b00gndvi_f + beta3 * b00moran_f + np.random.choice((-1,1))*err[1].reshape((353,353))
    lst05 = beta0 + beta1 * b05gpnbi_f + beta2 * b05gndvi_f + beta3 * b05moran_f + np.random.choice((-1,1))*err[2].reshape((353,353))
    lst10 = beta0 + beta1 * b10gpnbi_f + beta2 * b10gndvi_f + beta3 * b10moran_f + np.random.choice((-1,1))*err[3].reshape((353,353))
    lst15 = beta0 + beta1 * b15gpnbi_f + beta2 * b15gndvi_f + beta3 * b15moran_f + np.random.choice((-1,1))*err[4].reshape((353,353))
    lst20 = beta0 + beta1 * b20gpnbi_f + beta2 * b20gndvi_f + beta3 * b20moran_f + np.random.choice((-1,1))*err[5].reshape((353,353))
    #get all_coord from tif
    all_coords_list = []
    mask_r95gndvi = r95gndvi.dataset_mask()
    mask_r95gndvi = mask_r95gndvi[1:-1,1:]
    for row in range(mask_r95gndvi.shape[0]):
        for col in range (mask_r95gndvi.shape[1]):
            if(mask_r95gndvi[row,col]>0):
                all_coords_list.append(r95gndvi.xy(row+1,col+1)) 
    cal_coords_list = []
    cal_y_list = []
    cal_X_list = []
    
    pre_coords_list = []
    Pre_y_list = []
    Pre_X_list = []
    
    #95
    mask_ticks_95 = np.ones(353*353,dtype=bool).flatten()  
    mask_ticks_95[samples_nums] = False

    cal_y_tick = lst95.flatten()[~mask_ticks_95]
    cal_y_list.append(cal_y_tick.reshape((-1,1)))
    b95gndvi_f_x2 = b95gndvi_f.flatten()[~mask_ticks_95].reshape((-1,1))
    b95gpnbi_f_x1 = b95gpnbi_f.flatten()[~mask_ticks_95].reshape((-1,1))
    b95moran_f_x3 = b95moran_f.flatten()[~mask_ticks_95].reshape((-1,1))
    cal_X_tick = np.concatenate((b95gpnbi_f_x1,b95gndvi_f_x2,b95moran_f_x3),axis=1)
    cal_X_list.append(cal_X_tick)
    all_coords_arr = np.asarray(all_coords_list) 
    cal_coords_list.append(cal_cords)
 
    pred_coords_tick = all_coords_arr[mask_ticks_95]
    pre_coords_list.append(pred_coords_tick)
    pred_y_tick = lst95.flatten()[mask_ticks_95]
    Pre_y_list.append(pred_y_tick.reshape((-1,1)))
    b95gndvi_p_x2 = b95gndvi_f.flatten()[mask_ticks_95].reshape((-1,1))
    b95gpnbi_p_x1 = b95gpnbi_f.flatten()[mask_ticks_95].reshape((-1,1))
    b95moran_p_x3 = b95moran_f.flatten()[mask_ticks_95].reshape((-1,1))
    pre_X_tick = np.concatenate((b95gpnbi_p_x1,b95gndvi_p_x2,b95moran_p_x3),axis=1)
    Pre_X_list.append(pre_X_tick)
   
    #00
    mask_ticks_00 = np.ones(353*353,dtype=bool).flatten() 
    mask_ticks_00[samples_nums] = False
    cal_y_tick = lst00.flatten()[~mask_ticks_00]
    cal_y_list.append(cal_y_tick.reshape((-1,1)))
    b00gndvi_f_x2 = b00gndvi_f.flatten()[~mask_ticks_00].reshape((-1,1))
    b00gpnbi_f_x1 = b00gpnbi_f.flatten()[~mask_ticks_00].reshape((-1,1))
    b00moran_f_x3 = b00moran_f.flatten()[~mask_ticks_00].reshape((-1,1))
    cal_X_tick = np.concatenate((b00gpnbi_f_x1,b00gndvi_f_x2,b00moran_f_x3),axis=1)
    cal_X_list.append(cal_X_tick)
    all_coords_arr = np.asarray(all_coords_list)
    cal_coords_list.append(cal_cords)
    
    pred_coords_tick = all_coords_arr[mask_ticks_00]
    pre_coords_list.append(pred_coords_tick)
    pred_y_tick = lst00.flatten()[mask_ticks_00]
    Pre_y_list.append(pred_y_tick.reshape((-1,1)))
    b00gndvi_p_x2 = b00gndvi_f.flatten()[mask_ticks_00].reshape((-1,1))
    b00gpnbi_p_x1 = b00gpnbi_f.flatten()[mask_ticks_00].reshape((-1,1))
    b00moran_p_x3 = b00moran_f.flatten()[mask_ticks_00].reshape((-1,1))
    pre_X_tick = np.concatenate((b00gpnbi_p_x1,b00gndvi_p_x2,b00moran_p_x3),axis=1)
    Pre_X_list.append(pre_X_tick)
    
    #05
    mask_ticks_05 = np.ones(353*353,dtype=bool).flatten() 
    mask_ticks_05[samples_nums] = False
    cal_y_tick = lst05.flatten()[~mask_ticks_05]
    cal_y_list.append(cal_y_tick.reshape((-1,1)))
    b05gndvi_f_x2 = b05gndvi_f.flatten()[~mask_ticks_05].reshape((-1,1))
    b05gpnbi_f_x1 = b05gpnbi_f.flatten()[~mask_ticks_05].reshape((-1,1))
    b05moran_f_x3 = b05moran_f.flatten()[~mask_ticks_05].reshape((-1,1))
    cal_X_tick = np.concatenate((b05gpnbi_f_x1 ,b05gndvi_f_x2,b05moran_f_x3),axis=1)
    cal_X_list.append(cal_X_tick)
    all_coords_arr = np.asarray(all_coords_list)
    cal_coords_list.append(cal_cords)
    
    pred_coords_tick = all_coords_arr[mask_ticks_05]
    pre_coords_list.append(pred_coords_tick)
    pred_y_tick = lst05.flatten()[mask_ticks_05]
    Pre_y_list.append(pred_y_tick.reshape((-1,1)))
    b05gndvi_p_x2 = b05gndvi_f.flatten()[mask_ticks_05].reshape((-1,1))
    b05gpnbi_p_x1 = b05gpnbi_f.flatten()[mask_ticks_05].reshape((-1,1))
    b05moran_p_x3 = b05moran_f.flatten()[mask_ticks_05].reshape((-1,1))
    pre_X_tick = np.concatenate((b05gpnbi_p_x1, b05gndvi_p_x2,b05moran_p_x3),axis=1)
    Pre_X_list.append(pre_X_tick)
    #10
    mask_ticks_10 = np.ones(353*353,dtype=bool).flatten() 
    mask_ticks_10[samples_nums] = False
    cal_y_tick = lst10.flatten()[~mask_ticks_10]
    cal_y_list.append(cal_y_tick.reshape((-1,1)))
    b10gndvi_f_x2 = b10gndvi_f.flatten()[~mask_ticks_10].reshape((-1,1))
    b10gpnbi_f_x1 = b10gpnbi_f.flatten()[~mask_ticks_10].reshape((-1,1))
    b10moran_f_x3 = b10moran_f.flatten()[~mask_ticks_10].reshape((-1,1))
    cal_X_tick = np.concatenate((b10gpnbi_f_x1,b10gndvi_f_x2,b10moran_f_x3),axis=1)
    cal_X_list.append(cal_X_tick)
    all_coords_arr = np.asarray(all_coords_list)
    cal_coords_list.append(cal_cords)
    
    pred_coords_tick = all_coords_arr[mask_ticks_10]
    pre_coords_list.append(pred_coords_tick)
    pred_y_tick = lst10.flatten()[mask_ticks_10]
    Pre_y_list.append(pred_y_tick.reshape((-1,1)))
    b10gndvi_p_x2 = b10gndvi_f.flatten()[mask_ticks_10].reshape((-1,1))
    b10gpnbi_p_x1 = b10gpnbi_f.flatten()[mask_ticks_10].reshape((-1,1))
    b10moran_p_x3 = b10moran_f.flatten()[mask_ticks_10].reshape((-1,1))
    pre_X_tick = np.concatenate((b10gpnbi_p_x1,b10gndvi_p_x2,b10moran_p_x3),axis=1)
    Pre_X_list.append(pre_X_tick)
    
    #15
    mask_ticks_15 = np.ones(353*353,dtype=bool).flatten() 
    mask_ticks_15[samples_nums] = False
    cal_y_tick = lst15.flatten()[~mask_ticks_15]
    cal_y_list.append(cal_y_tick.reshape((-1,1)))
    b15gndvi_f_x2 = b15gndvi_f.flatten()[~mask_ticks_15].reshape((-1,1))
    b15gpnbi_f_x1 = b15gpnbi_f.flatten()[~mask_ticks_15].reshape((-1,1))
    b15moran_f_x3 = b15moran_f.flatten()[~mask_ticks_15].reshape((-1,1))
    cal_X_tick = np.concatenate((b15gpnbi_f_x1, b15gndvi_f_x2,b15moran_f_x3),axis=1)
    cal_X_list.append(cal_X_tick)
    all_coords_arr = np.asarray(all_coords_list)
    cal_coords_list.append(cal_cords)
    
    pred_coords_tick = all_coords_arr[mask_ticks_15]
    pre_coords_list.append(pred_coords_tick)
    pred_y_tick =  lst15.flatten()[mask_ticks_15]
    Pre_y_list.append(pred_y_tick.reshape((-1,1)))
    b15gndvi_p_x2 = b15gndvi_f.flatten()[mask_ticks_15].reshape((-1,1))
    b15gpnbi_p_x1 = b15gpnbi_f.flatten()[mask_ticks_15].reshape((-1,1))
    b15moran_p_x3 = b15moran_f.flatten()[mask_ticks_15].reshape((-1,1))
    pre_X_tick = np.concatenate((b15gpnbi_p_x1,b15gndvi_p_x2,b15moran_p_x3),axis=1)
    Pre_X_list.append(pre_X_tick)
    
    #20
    mask_ticks_20 = np.ones(353*353,dtype=bool).flatten() 
    mask_ticks_20[samples_nums] = False
    cal_y_tick = lst20.flatten()[~mask_ticks_20]
    cal_y_list.append(cal_y_tick.reshape((-1,1)))
    b20gndvi_f_x2 = b20gndvi_f.flatten()[~mask_ticks_20].reshape((-1,1))
    b20gpnbi_f_x1 = b20gpnbi_f.flatten()[~mask_ticks_20].reshape((-1,1))
    b20moran_f_x3 = b20moran_f.flatten()[~mask_ticks_20].reshape((-1,1))
    cal_X_tick = np.concatenate((b20gpnbi_f_x1, b20gndvi_f_x2,b20moran_f_x3),axis=1)
    cal_X_list.append(cal_X_tick)
    all_coords_arr = np.asarray(all_coords_list)
    cal_coords_list.append(cal_cords)
    
    pred_coords_tick = all_coords_arr[mask_ticks_20]
    pre_coords_list.append(pred_coords_tick)
    pred_y_tick =  lst20.flatten()[mask_ticks_20]
    Pre_y_list.append(pred_y_tick.reshape((-1,1)))
    b20gndvi_p_x2 = b20gndvi_f.flatten()[mask_ticks_20].reshape((-1,1))
    b20gpnbi_p_x1 = b20gpnbi_f.flatten()[mask_ticks_20].reshape((-1,1))
    b20moran_p_x3 = b20moran_f.flatten()[mask_ticks_20].reshape((-1,1))
    pre_X_tick = np.concatenate((b20gpnbi_p_x1,b20gndvi_p_x2,b20moran_p_x3),axis=1)
    Pre_X_list.append(pre_X_tick)
    #############################
    #4. test the prediction of beta surfaces
    #OLS
    ols_X = cal_X_list[-1]
    ols_X = np.concatenate((np.ones((ols_X.shape[0], 1)), ols_X),axis=1)
    ols_result = GLM(cal_y_list[-1],ols_X,constant=False,family=Gaussian()).fit()
    ols_predbeta_results.append(ols_result.params) 
    ols_resid_results.append(ols_result.resid_response)
    #OLS
    #gwr  
    gwr_selector = Sel_BW(cal_coords_list[-1], cal_y_list[-1], cal_X_list[-1], spherical = False)
    gwr_bw= gwr_selector.search(bw_min=2)
    gwr_model = GWR(cal_coords_list[-1], cal_y_list[-1], cal_X_list[-1], gwr_bw,spherical = False)
    gwr_results = gwr_model.fit()
    print(gwr_results.summary())
    gwr_scale = gwr_results.scale 
    gwr_residuals = gwr_results.resid_response
    gwr_resid_results.append(gwr_residuals)
    
    #stwr
    stwr_selector_ = Sel_Spt_BW(cal_coords_list, cal_y_list, cal_X_list,time_dif ,spherical = False)
    #F-STWR
    optalpha,optsita,opt_btticks,opt_gwr_bw0 = stwr_selector_.search(nproc = 20) 
    stwr_model = STWR(cal_coords_list,cal_y_list,cal_X_list,time_dif,optsita,opt_gwr_bw0,tick_nums=opt_btticks,alpha =optalpha,spherical = False,recorded=1)#recorded = True)
    #F-STWR
    stwr_results = stwr_model.fit()
    print(stwr_results.summary())
    stwr_scale = stwr_results.scale 
    stwr_residuals = stwr_results.resid_response
    stwr_resid_results.append(stwr_residuals)
    
    allklen_stwr = cal_X_list[-1].shape[1]+1
    stwr_cal_parmas = np.reshape(stwr_results.params.flatten(),(-1,allklen_stwr)) 
    gwr_cal_parmas = np.reshape(gwr_results.params.flatten(),(-1,allklen_stwr))
 
    gwr_prebeta0_results.append(gwr_cal_parmas[:,0])
    gwr_prebeta1_results.append(gwr_cal_parmas[:,1])
    gwr_prebeta2_results.append(gwr_cal_parmas[:,2])
    gwr_prebeta3_results.append(gwr_cal_parmas[:,3])
    stwr_prebeta0_results.append(stwr_cal_parmas[:,0])
    stwr_prebeta1_results.append(stwr_cal_parmas[:,1])
    stwr_prebeta2_results.append(stwr_cal_parmas[:,2])
    stwr_prebeta3_results.append(stwr_cal_parmas[:,3])
    

rpt_betas = np.zeros((random_times,4))
for i in range(random_times):
   rpt_betas[i] = np.array([1,3,-1,-6]) 
    
## OLS Results#
ols_betas = np.asarray(ols_predbeta_results)
ols_betas_means = np.mean(ols_betas, axis=0)
ols_betas_std = np.std(ols_betas, axis=0)

ols_net_errs = np.absolute(rpt_betas-ols_betas)
ols_abs_net_means = np.mean(ols_net_errs, axis=0)
ols_betas_mse = np.sqrt(np.mean((rpt_betas-ols_betas)*(rpt_betas-ols_betas),axis=0))

ols_resid_results_arr =np.asarray(ols_resid_results) 
ols_resid_results_arr_m = np.mean(ols_resid_results_arr,axis = 0)
ols_ols_resid_arr_mae = np.mean(np.absolute(ols_resid_results_arr))
ols_ols_resid_arr_rmae = np.mean((ols_resid_results_arr*ols_resid_results_arr))

##  GWR Results#
gwr_pbeta0_arr = np.asarray(gwr_prebeta0_results)
gwr_pbeta0_means = np.mean(gwr_pbeta0_arr, axis=0)
gwr_pbeta0_std = np.std(gwr_pbeta0_arr, axis=0)
gwr_pbeta0_mm = np.mean(gwr_pbeta0_means)

rpt_betas0 = np.zeros((random_times,lensdata))
for i in range(random_times):
   rpt_betas0[i] = np.repeat([1],lensdata) 
gwr_pbeta0_errs =  np.absolute(rpt_betas0-gwr_pbeta0_arr)
gwr_pbeta0_errs_mean = np.mean(gwr_pbeta0_errs)
gwr_beta0_mse = np.sqrt(np.mean((rpt_betas0-gwr_pbeta0_arr)*(rpt_betas0-gwr_pbeta0_arr)))


gwr_pbeta1_arr = np.asarray(gwr_prebeta1_results)
gwr_pbeta1_means = np.mean(gwr_pbeta1_arr, axis=0)
gwr_pbeta1_std = np.std(gwr_pbeta1_arr, axis=0)
gwr_pbeta1_mm = np.mean(gwr_pbeta1_means)

rpt_beta1 = np.zeros((random_times,lensdata))
for i in range(random_times):
   rpt_beta1[i] = np.repeat([3],lensdata) 
gwr_pbeta1_errs =  np.absolute(rpt_beta1-gwr_pbeta1_arr)
gwr_pbeta1_errs_mean = np.mean(gwr_pbeta1_errs)
gwr_beta1_mse = np.sqrt(np.mean((rpt_beta1-gwr_pbeta1_arr)*(rpt_beta1-gwr_pbeta1_arr)))


gwr_pbeta2_arr = np.asarray(gwr_prebeta2_results)
gwr_pbeta2_means = np.mean(gwr_pbeta2_arr, axis=0)
gwr_pbeta2_std = np.std(gwr_pbeta2_arr, axis=0)
gwr_pbeta2_mm = np.mean(gwr_pbeta2_means)

rpt_beta2 = np.zeros((random_times,lensdata))
for i in range(random_times):
   rpt_beta2[i] = np.repeat([-1],lensdata) 
gwr_pbeta2_errs =  np.absolute(rpt_beta2-gwr_pbeta2_arr)
gwr_pbeta2_errs_mean= np.mean(gwr_pbeta2_errs)
gwr_beta2_mse = np.sqrt(np.mean((rpt_beta2-gwr_pbeta2_arr)*(rpt_beta2-gwr_pbeta2_arr)))

gwr_pbeta3_arr = np.asarray(gwr_prebeta3_results)
gwr_pbeta3_means = np.mean(gwr_pbeta3_arr, axis=0)
gwr_pbeta3_std = np.std(gwr_pbeta3_arr, axis=0)
gwr_pbeta3_mm = np.mean(gwr_pbeta3_means)

rpt_beta3 = np.zeros((random_times,lensdata))
for i in range(random_times):
   rpt_beta3[i] = np.repeat([-6],lensdata) 
gwr_pbeta3_errs =  np.absolute(rpt_beta3-gwr_pbeta3_arr)
gwr_pbeta3_errs_mean = np.mean(gwr_pbeta3_errs)
gwr_beta3_mse = np.sqrt(np.mean((rpt_beta3-gwr_pbeta3_arr)*(rpt_beta3-gwr_pbeta3_arr)))



gwr_resid_results_arr =np.asarray(gwr_resid_results) 
gwr_resid_results_arr_m = np.mean(gwr_resid_results_arr,axis = 0)
gwr_resid_results_arr_mae = np.mean(np.absolute(gwr_resid_results_arr))
gwr_resid_results_arr_rmse = np.mean((gwr_resid_results_arr*gwr_resid_results_arr))



##  STWR Results#
stwr_pbeta0_arr = np.asarray(stwr_prebeta0_results)
stwr_pbeta0_means = np.mean(stwr_pbeta0_arr, axis=0)
stwr_pbeta0_std = np.std(stwr_pbeta0_arr, axis=0)
stwr_pbeta0_mm = np.mean(stwr_pbeta0_means)

rptstwr_beta0 = np.zeros((random_times,lensdata))
for i in range(random_times):
   rptstwr_beta0[i] = np.repeat([1],lensdata) 
stwr_pbeta0_errs =  np.absolute(rptstwr_beta0-stwr_pbeta0_arr)
stwr_pbeta0_errs_mean = np.mean(stwr_pbeta0_errs)
stwr_beta0_mse = np.sqrt(np.mean((rptstwr_beta0-stwr_pbeta0_arr)*(rptstwr_beta0-stwr_pbeta0_arr)))


stwr_pbeta1_arr = np.asarray(stwr_prebeta1_results)
stwr_pbeta1_means = np.mean(stwr_pbeta1_arr, axis=0)
stwr_pbeta1_std = np.std(stwr_pbeta1_arr, axis=0)
stwr_pbeta1_mm = np.mean(stwr_pbeta1_means)

rptstwr_beta1 = np.zeros((random_times,lensdata))
for i in range(random_times):
   rptstwr_beta1[i] = np.repeat([3],lensdata) 
stwr_pbeta1_errs =  np.absolute(rptstwr_beta1-stwr_pbeta1_arr)
stwr_pbeta1_errs_mean = np.mean(stwr_pbeta1_errs)
stwr_beta1_mse = np.sqrt(np.mean((rptstwr_beta1-stwr_pbeta1_arr)*(rptstwr_beta1-stwr_pbeta1_arr)))

stwr_pbeta2_arr = np.asarray(stwr_prebeta2_results)
stwr_pbeta2_means = np.mean(stwr_pbeta2_arr, axis=0)
stwr_pbeta2_std = np.std(stwr_pbeta2_arr, axis=0)
stwr_pbeta2_mm = np.mean(stwr_pbeta2_means)

rptstwr_beta2 = np.zeros((random_times,lensdata))
for i in range(random_times):
   rptstwr_beta2[i] = np.repeat([-1],lensdata) 
stwr_pbeta2_errs =  np.absolute(rptstwr_beta2-stwr_pbeta2_arr)
stwr_pbeta2_errs_mean = np.mean(stwr_pbeta2_errs)
stwr_beta2_mse = np.sqrt(np.mean((rptstwr_beta2-stwr_pbeta2_arr)*(rptstwr_beta2-stwr_pbeta2_arr)))

stwr_pbeta3_arr = np.asarray(stwr_prebeta3_results)
stwr_pbeta3_means = np.mean(stwr_pbeta3_arr, axis=0)
stwr_pbeta3_std = np.std(stwr_pbeta3_arr, axis=0)
stwr_pbeta3_mm = np.mean(stwr_pbeta3_means)

rptstwr_beta3 = np.zeros((random_times,lensdata))
for i in range(random_times):
   rptstwr_beta3[i] = np.repeat([-1],lensdata) 
stwr_pbeta3_errs =  np.absolute(rptstwr_beta3-stwr_pbeta3_arr)
stwr_pbeta3_errs_mean= np.mean(stwr_pbeta3_errs)
stwr_beta3_mse = np.sqrt(np.mean((rptstwr_beta3-stwr_pbeta3_arr)*(rptstwr_beta3-stwr_pbeta3_arr)))


stwr_resid_results_arr =np.asarray(stwr_resid_results) 
stwr_resid_results_arr_m = np.mean(stwr_resid_results_arr,axis = 0)
stwr_resid_results_arr_mae = np.mean(np.absolute(stwr_resid_results_arr))
stwr_resid_results_arr_rmse = np.mean((stwr_resid_results_arr*stwr_resid_results_arr))

######################################################
gwr_pbeta0_mm = np.mean(gwr_pbeta0_means)
gwr_pbeta1_mm = np.mean(gwr_pbeta1_means)
gwr_pbeta2_mm = np.mean(gwr_pbeta2_means)
gwr_pbeta3_mm = np.mean(gwr_pbeta3_means)

stwr_pbeta0_mm = np.mean(stwr_pbeta0_means)
stwr_pbeta1_mm = np.mean(stwr_pbeta1_means)
stwr_pbeta2_mm = np.mean(stwr_pbeta2_means)
stwr_pbeta3_mm = np.mean(stwr_pbeta3_means)

#######save results to txt files
with open('D:/STWR/Data-Run/results/100_test_olsbeta_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(ols_betas.shape))
    np.savetxt(outfile, ols_betas, fmt='%-7.7f')
 

with open('D:/STWR/Data-Run/results/100_test_olspredy_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(ols_predy_arr.shape))
    for data_slice in ols_predy_arr:
        np.savetxt(outfile, data_slice, fmt='%-7.7f')
        outfile.write('# New slice\n')
                      
with open('D:/STWR/Data-Run/results/100_test_gwr_predbeta0arr_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(gwr_predbeta0_arr.shape))
    for data_slice in gwr_predbeta0_arr:
        np.savetxt(outfile, data_slice, fmt='%-7.7f')
        outfile.write('# New slice\n')
with open('D:/STWR/Data-Run/results/100_test_gwr_predbeta1arr_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(gwr_predbeta1_arr.shape))
    for data_slice in gwr_predbeta1_arr:
        np.savetxt(outfile, data_slice, fmt='%-7.7f')
        outfile.write('# New slice\n')                     
with open('D:/STWR/Data-Run/results/100_test_gwr_predbeta2arr_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(gwr_predbeta2_arr.shape))
    for data_slice in gwr_predbeta2_arr:
        np.savetxt(outfile, data_slice, fmt='%-7.7f')
        outfile.write('# New slice\n')           
with open('D:/STWR/Data-Run/results/100_test_gwr_predbeta3arr_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(gwr_predbeta3_arr.shape))
    for data_slice in gwr_predbeta3_arr:
        np.savetxt(outfile, data_slice, fmt='%-7.7f')
        outfile.write('# New slice\n')
with open('D:/STWR/Data-Run/results/100_test_gwr_predy_arr_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(gwr_predy_arr.shape))
    for data_slice in gwr_predy_arr: 
        np.savetxt(outfile, data_slice, fmt='%-7.7f')
        outfile.write('# New slice\n')            
#STWR
with open('D:/STWR/Data-Run/results/100_test_stwr_predbeta0arr_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(stwr_predbeta0_arr.shape))
    for data_slice in stwr_predbeta0_arr:
        np.savetxt(outfile, data_slice, fmt='%-7.7f')
        outfile.write('# New slice\n')
with open('D:/STWR/Data-Run/results/100_test_stwr_predbeta1arr_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(stwr_predbeta1_arr.shape))
    for data_slice in stwr_predbeta1_arr:  
        np.savetxt(outfile, data_slice, fmt='%-7.7f')
        outfile.write('# New slice\n')                     
with open('D:/STWR/Data-Run/results/100_test_stwr_predbeta2arr_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(stwr_predbeta2_arr.shape))
    for data_slice in stwr_predbeta2_arr:
        np.savetxt(outfile, data_slice, fmt='%-7.7f')
        outfile.write('# New slice\n')           
with open('D:/STWR/Data-Run/results/100_test_stwr_predbeta3arr_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(stwr_predbeta3_arr.shape))
    for data_slice in stwr_predbeta3_arr:
        np.savetxt(outfile, data_slice, fmt='%-7.7f')
        outfile.write('# New slice\n')
with open('D:/STWR/Data-Run/results/100_test_stwr_predy_arr_1000_hc.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(stwr_predy_arr.shape))
    for data_slice in stwr_predy_arr: 
        np.savetxt(outfile, data_slice, fmt='%-7.7f')
        outfile.write('# New slice\n')          

#############draw the pics#########################
#Comparision surfaces of Y_true,prdictions mean surface by OLS,GWR and STWR.
print(" Comparision surfaces of Y_true,prdictions mean surface by  STWR , GWR and OLS with same color bars of Y_true: \n")           
fig_pred_cmp_stwr = plt.figure(figsize=(12,3),constrained_layout=True)
gs_stwr = GridSpec(1, 4, figure=fig_pred_cmp_stwr)
jet  = plt.get_cmap('jet',256)
#use the same color bar of true_y  for prediction of STWR and GWR
vmin_ytrue =np.amin(lst10)
vmax_ytrue =np.amax(lst10)

ax_true  = fig_pred_cmp_stwr.add_subplot(gs_stwr[0,0])
psm_true = ax_true.pcolormesh(lst10, cmap=jet, rasterized=True, vmin=vmin_ytrue, vmax=vmax_ytrue)
ax_stwr  = fig_pred_cmp_stwr.add_subplot(gs_stwr[0,1])
psm_stwr = ax_stwr.pcolormesh(stwr_predy_means, cmap=jet, rasterized=True, vmin=vmin_ytrue, vmax=vmax_ytrue)
ax_gwr  =  fig_pred_cmp_stwr.add_subplot(gs_stwr[0,2])
psm_gwr =  ax_gwr.pcolormesh(gwr_predy_means, cmap=jet, rasterized=True, vmin=vmin_ytrue, vmax=vmax_ytrue)
ax_ols =  fig_pred_cmp_stwr.add_subplot(gs_stwr[0,3])
psm_ols=  ax_ols.pcolormesh(ols_predy_means, cmap=jet, rasterized=True, vmin=vmin_ytrue, vmax=vmax_ytrue)

fig_pred_cmp_stwr.colorbar(psm_true, ax = fig_pred_cmp_stwr.axes[3])
plt.show()


print("Comparision surfaces of Y_true,prdictions mean surface by  STWR , GWR and OLS with seperated color bars: \n")
#use different colorbar of predicton y of STWR and GWR 

fig_pred_cmp_stwr = plt.figure(figsize=(16,3),constrained_layout=True)
gs_stwr = GridSpec(1, 4, figure=fig_pred_cmp_stwr)
jet  = plt.get_cmap('jet',256)
vmin_stwr= np.amin(stwr_predy_means)
vmax_stwr= np.amax(stwr_predy_means)
vmin_gwr = np.amin(gwr_predy_means)
vmax_gwr = np.amax(gwr_predy_means)

vmin_ols = np.amin(ols_predy_means)
vmax_ols = np.amax(ols_predy_means)

ax_true  = fig_pred_cmp_stwr.add_subplot(gs_stwr[0,0])
psm_true = ax_true.pcolormesh(lst10, cmap=jet, rasterized=True, vmin=vmin_ytrue, vmax=vmax_ytrue)
fig_pred_cmp_stwr.colorbar(psm_true, ax = ax_true)
ax_stwr  = fig_pred_cmp_stwr.add_subplot(gs_stwr[0,1])
psm_stwr = ax_stwr.pcolormesh(stwr_predy_means, cmap=jet, rasterized=True, vmin=vmin_stwr, vmax=vmax_stwr)
fig_pred_cmp_stwr.colorbar(psm_stwr, ax = ax_stwr)
ax_gwr  =  fig_pred_cmp_stwr.add_subplot(gs_stwr[0,2])
psm_gwr =  ax_gwr.pcolormesh(gwr_predy_means, cmap=jet, rasterized=True, vmin=vmin_gwr, vmax=vmax_gwr)
fig_pred_cmp_stwr.colorbar(psm_gwr, ax = ax_gwr)

ax_ols  =  fig_pred_cmp_stwr.add_subplot(gs_stwr[0,3])
psm_ols =  ax_ols.pcolormesh(ols_predy_means, cmap=jet, rasterized=True, vmin=vmin_ols, vmax=vmax_ols)
fig_pred_cmp_stwr.colorbar(psm_ols, ax = ax_ols)
plt.show()  

#predy error
y_self_err = np.abs(lst10-lst10)          
y_pre_stwr_err = np.abs(lst10 - stwr_predy_means)            
y_pre_gwr_err = np.abs(lst10 - gwr_predy_means)          
y_pre_ols_err = np.abs(lst10- ols_predy_means)  

y_pre_stwr_eee = stwr_predy_means -lst10
y_pre_gwr_eee  =  gwr_predy_means -lst10
y_pre_ols_eee  =  ols_predy_means -lst10         

print("Comparision Absolute  Error Surfaces of Y_true,prdictions mean surface by STWR , GWR and OLS with same color bars: \n")
fig_prederr_cmp_gtwr = plt.figure(figsize=(12,3),constrained_layout=True)
gs_stwr_err = GridSpec(1, 4, figure=fig_prederr_cmp_gtwr)
jet  = plt.get_cmap('jet',256)
vmin1  =np.amin(y_pre_stwr_err)
vmin2  =np.amin(y_pre_gwr_err)
if(vmin1 <vmin2):
    vmin_stwr_err=vmin1
else:
    vmin_stwr_err= vmin2
#####OLS####
vmin3 = np.amin(y_pre_ols_err)
if (vmin_stwr_err > vmin3 ):
        vmin_stwr_err = vmin3
#####OLS####           
vmax1 = np.amax(y_pre_stwr_err)
vmax2 = np.amax(y_pre_gwr_err)
if(vmax1 >vmax2):
    vmax_stwr_err=vmax1
else:
    vmax_stwr_err= vmax2

if (vmax_stwr_err>vmax_stwr):
        vmax_stwr_err = vmax_stwr
#####OLS####
vmax3 =  np.amax(y_pre_ols_err) 
if (vmax_stwr_err < vmax3):
        vmax_stwr_err = vmax3       
#####OLS####   
ax_true_err  = fig_prederr_cmp_gtwr.add_subplot(gs_stwr_err[0,0])
psm_true_err = ax_true_err.pcolormesh(y_self_err, cmap=jet, rasterized=True, vmin=vmin_stwr_err, vmax=vmax_stwr_err)
ax_stwr_err  = fig_prederr_cmp_gtwr.add_subplot(gs_stwr_err[0,1])
psm_stwr_err = ax_stwr_err.pcolormesh(y_pre_stwr_err, cmap=jet, rasterized=True, vmin=vmin_stwr_err, vmax=vmax_stwr_err)
ax_gwr_err  =  fig_prederr_cmp_gtwr.add_subplot(gs_stwr_err[0,2])
psm_gwr_err =  ax_gwr_err.pcolormesh(y_pre_gwr_err, cmap=jet, rasterized=True, vmin=vmin_stwr_err, vmax=vmax_stwr_err)

ax_ols_err  =  fig_prederr_cmp_gtwr.add_subplot(gs_stwr_err[0,3])
psm_ols_err =  ax_ols_err.pcolormesh(y_pre_ols_err, cmap=jet, rasterized=True, vmin=vmin_stwr_err, vmax=vmax_stwr_err)

fig_prederr_cmp_gtwr.colorbar(psm_true_err, ax = fig_prederr_cmp_gtwr.axes[3])
plt.show()

#####Generate a MAE for each Models######
stwr_mae = np.mean(y_pre_stwr_err)
gwr_mae = np.mean(y_pre_gwr_err)
ols_mae = np.mean(y_pre_ols_err)
print("MAE of STWR is:{}". format(stwr_mae))
print("MAE of GWR is:{}". format(gwr_mae))
print("MAE of OLS is:{}". format(ols_mae))
#####Generate a MAE for each Models######

  
print("Comparision Error Surfaces of Y_true,prdicted mean surface by STWR , GWR and OLS with seperated color bars: \n")
fig_prederr_cmp_gtwr = plt.figure(figsize=(15,3),constrained_layout=True)
gs_stwr_err = GridSpec(1, 4, figure=fig_prederr_cmp_gtwr)
jet  = plt.get_cmap('jet',256)
ax_true_err  = fig_prederr_cmp_gtwr.add_subplot(gs_stwr_err[0,0])
psm_true_err = ax_true_err.pcolormesh(y_self_err, cmap=jet, rasterized=True,  vmin=np.amin(y_pre_stwr_eee), vmax=np.amax(y_pre_stwr_eee))
fig_prederr_cmp_gtwr.colorbar(psm_true_err, ax = ax_true_err)

ax_stwr_err  = fig_prederr_cmp_gtwr.add_subplot(gs_stwr_err[0,1])
psm_stwr_err = ax_stwr_err.pcolormesh(y_pre_stwr_eee, cmap=jet, rasterized=True, vmin=np.amin(y_pre_stwr_eee), vmax=np.amax(y_pre_stwr_eee))
fig_prederr_cmp_gtwr.colorbar(psm_stwr_err, ax = ax_stwr_err)

ax_gwr_err  =  fig_prederr_cmp_gtwr.add_subplot(gs_stwr_err[0,2])
psm_gwr_err =  ax_gwr_err.pcolormesh(y_pre_gwr_eee, cmap=jet, rasterized=True, vmin=np.amin(y_pre_stwr_eee), vmax=np.amax(y_pre_stwr_eee))
fig_prederr_cmp_gtwr.colorbar(psm_gwr_err, ax = ax_gwr_err)

ax_ols_err  =  fig_prederr_cmp_gtwr.add_subplot(gs_stwr_err[0,3])
psm_ols_err =  ax_ols_err.pcolormesh(y_pre_ols_eee, cmap=jet, rasterized=True, vmin=np.amin(y_pre_stwr_eee), vmax=np.amax(y_pre_stwr_eee))
fig_pred_cmp_stwr.colorbar(psm_ols_err, ax = ax_ols_err)
plt.show()

##########################predicted beta surfaces#########################
###Draw figures of different beta surfaces 
print("Draw figures of different heterogeneity surfaces: \n")          
fig = plt.figure(figsize=(16,3),constrained_layout=True)
gs = GridSpec(1, 4, figure=fig)
jet  = plt.get_cmap('jet',256)
psm =0        

ax = fig.add_subplot(gs[0,0])
psm = ax.pcolormesh(beta0, cmap=jet, rasterized=True, vmin=beta_min_0, vmax=beta_max_0)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[0,1])
psm = ax.pcolormesh(beta1, cmap=jet, rasterized=True, vmin=beta_min_1, vmax=beta_max_1)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[0,2])
psm = ax.pcolormesh(beta2, cmap=jet, rasterized=True, vmin=beta_min_2, vmax=beta_max_2)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[0,3])
psm = ax.pcolormesh(beta3, cmap=jet, rasterized=True, vmin=beta_min_3, vmax=beta_max_3)
fig.colorbar(psm, ax = ax)
plt.show()

##Four coefficient surfaces predicted by STWR
print("Four mean coefficient surfaces predicted by STWR: \n")   
fig_parmas_stwr = plt.figure(figsize=(16,3),constrained_layout=True)
gs_stwr = GridSpec(1, 4, figure=fig_parmas_stwr)
jet  = plt.get_cmap('jet',256)
psm_pam_stwr =0        
ax_pam_stwr = fig_parmas_stwr.add_subplot(gs_stwr[0,0])
psm_pam_stwr = ax_pam_stwr.pcolormesh(stwr_predbeta0_means, cmap=jet, rasterized=True, vmin=np.amin(stwr_predbeta0_means), vmax=np.amax(stwr_predbeta0_means))
fig_parmas_stwr.colorbar(psm_pam_stwr, ax = ax_pam_stwr)
ax_pam_stwr = fig_parmas_stwr.add_subplot(gs_stwr[0,1])
psm_pam_stwr = ax_pam_stwr.pcolormesh(stwr_predbeta1_means, cmap=jet, rasterized=True, vmin=np.amin(stwr_predbeta1_means), vmax=np.amax(stwr_predbeta1_means))
fig_parmas_stwr.colorbar(psm_pam_stwr, ax = ax_pam_stwr)
ax_pam_stwr = fig_parmas_stwr.add_subplot(gs_stwr[0,2])
psm_pam_stwr = ax_pam_stwr.pcolormesh(stwr_predbeta2_means, cmap=jet, rasterized=True, vmin=np.amin(stwr_predbeta2_means), vmax=np.amax(stwr_predbeta2_means))
fig_parmas_stwr.colorbar(psm_pam_stwr, ax = ax_pam_stwr)
ax_pam_stwr = fig_parmas_stwr.add_subplot(gs_stwr[0,3])
psm_pam_stwr = ax_pam_stwr.pcolormesh(stwr_predbeta3_means, cmap=jet, rasterized=True, vmin=np.amin(stwr_predbeta3_means), vmax=np.amax(stwr_predbeta3_means))
fig_parmas_stwr.colorbar(psm_pam_stwr, ax = ax_pam_stwr)
plt.show()

print("Four mean coefficient surfaces predicted by STWR with same colorbar: \n")   
#use the same colorbar with true beta
fig_parmas_stwr = plt.figure(figsize=(16,3),constrained_layout=True)
gs_stwr = GridSpec(1, 4, figure=fig_parmas_stwr)
jet  = plt.get_cmap('jet',256)
psm_pam_stwr =0        
ax_pam_stwr = fig_parmas_stwr.add_subplot(gs_stwr[0,0])
psm_pam_stwr = ax_pam_stwr.pcolormesh(stwr_predbeta0_means, cmap=jet, rasterized=True, vmin=beta_min_list[0], vmax=beta_max_list[0])
fig_parmas_stwr.colorbar(psm_pam_stwr, ax = ax_pam_stwr)
ax_pam_stwr = fig_parmas_stwr.add_subplot(gs_stwr[0,1])
psm_pam_stwr = ax_pam_stwr.pcolormesh(stwr_predbeta1_means, cmap=jet, rasterized=True, vmin=beta_min_list[1], vmax=beta_max_list[1])
fig_parmas_stwr.colorbar(psm_pam_stwr, ax = ax_pam_stwr)
ax_pam_stwr = fig_parmas_stwr.add_subplot(gs_stwr[0,2])
psm_pam_stwr = ax_pam_stwr.pcolormesh(stwr_predbeta2_means, cmap=jet, rasterized=True,vmin=beta_min_list[2], vmax=beta_max_list[2])
fig_parmas_stwr.colorbar(psm_pam_stwr, ax = ax_pam_stwr)
ax_pam_stwr = fig_parmas_stwr.add_subplot(gs_stwr[0,3])
psm_pam_stwr = ax_pam_stwr.pcolormesh(stwr_predbeta3_means, cmap=jet, rasterized=True,vmin=beta_min_list[3], vmax=beta_max_list[3])
fig_parmas_stwr.colorbar(psm_pam_stwr, ax = ax_pam_stwr)
plt.show()


#Four coefficient surfaces predicted by GWR
print("Four coefficient surfaces predicted by GWR: \n")   
fig_parmas_gwr = plt.figure(figsize=(16,3),constrained_layout=True)
gs_gwr = GridSpec(1, 4, figure=fig_parmas_gwr)
jet  = plt.get_cmap('jet',256)
psm_pam_gwr =0        
ax_pam_gwr = fig_parmas_gwr.add_subplot(gs_gwr[0,0])
psm_pam_gwr = ax_pam_gwr.pcolormesh(gwr_predbeta0_means, cmap=jet, rasterized=True, vmin=np.amin(gwr_predbeta0_means), vmax=np.amax(gwr_predbeta0_means))
fig_parmas_gwr.colorbar(psm_pam_gwr, ax = ax_pam_gwr)
ax_pam_gwr = fig_parmas_gwr.add_subplot(gs_gwr[0,1])
psm_pam_gwr = ax_pam_gwr.pcolormesh(gwr_predbeta1_means, cmap=jet, rasterized=True, vmin=np.amin(gwr_predbeta1_means), vmax=np.amax(gwr_predbeta1_means))
fig_parmas_gwr.colorbar(psm_pam_gwr, ax = ax_pam_gwr)
ax_pam_gwr = fig_parmas_gwr.add_subplot(gs_gwr[0,2])
psm_pam_gwr = ax_pam_gwr.pcolormesh(gwr_predbeta2_means, cmap=jet, rasterized=True, vmin=np.amin(gwr_predbeta2_means), vmax=np.amax(gwr_predbeta2_means))
fig_parmas_gwr.colorbar(psm_pam_gwr, ax = ax_pam_gwr)
ax_pam_gwr = fig_parmas_gwr.add_subplot(gs_gwr[0,3])
psm_pam_gwr = ax_pam_gwr.pcolormesh(gwr_predbeta3_means, cmap=jet, rasterized=True, vmin=np.amin(gwr_predbeta3_means), vmax=np.amax(gwr_predbeta3_means))
fig_parmas_gwr.colorbar(psm_pam_gwr, ax = ax_pam_gwr)
plt.show()

print("Four mean coefficient surfaces predicted by GWR with same colorbar: \n") 
#use the same colorbar with true beta
fig_parmas_gwr = plt.figure(figsize=(16,3),constrained_layout=True)
gs_gwr = GridSpec(1, 4, figure=fig_parmas_gwr)
jet  = plt.get_cmap('jet',256)
psm_pam_gwr =0        
ax_pam_gwr = fig_parmas_gwr.add_subplot(gs_gwr[0,0])
psm_pam_gwr = ax_pam_gwr.pcolormesh(gwr_predbeta0_means, cmap=jet, rasterized=True, vmin=beta_min_list[0], vmax=beta_max_list[0])
fig_parmas_gwr.colorbar(psm_pam_gwr, ax = ax_pam_gwr)
ax_pam_gwr = fig_parmas_gwr.add_subplot(gs_gwr[0,1])
psm_pam_gwr = ax_pam_gwr.pcolormesh(gwr_predbeta1_means, cmap=jet, rasterized=True, vmin=beta_min_list[1], vmax=beta_max_list[1])
fig_parmas_gwr.colorbar(psm_pam_gwr, ax = ax_pam_gwr)
ax_pam_gwr = fig_parmas_gwr.add_subplot(gs_gwr[0,2])
psm_pam_gwr = ax_pam_gwr.pcolormesh(gwr_predbeta2_means, cmap=jet, rasterized=True,vmin=beta_min_list[2], vmax=beta_max_list[2])
fig_parmas_gwr.colorbar(psm_pam_gwr, ax = ax_pam_gwr)
ax_pam_gwr = fig_parmas_gwr.add_subplot(gs_gwr[0,3])
psm_pam_gwr = ax_pam_gwr.pcolormesh(gwr_predbeta3_means, cmap=jet, rasterized=True, vmin=beta_min_list[3], vmax=beta_max_list[3])
fig_parmas_gwr.colorbar(psm_pam_gwr, ax = ax_pam_gwr)
plt.show()


print("Four mean coefficient surfaces predicted by OLS with same colorbar: \n") 
#use the same colorbar with true beta
fig_parmas_ols = plt.figure(figsize=(16,3),constrained_layout=True)
gs_ols = GridSpec(1, 4, figure=fig_parmas_ols)
jet  = plt.get_cmap('jet',256)
psm_pam_ols =0        
ax_pam_ols = fig_parmas_ols.add_subplot(gs_ols[0,0])
psm_pam_ols = ax_pam_ols.pcolormesh(ols_betas_means[0]*np.ones_like(beta0), cmap=jet, rasterized=True, vmin=beta_min_list[0], vmax=beta_max_list[0])
fig_parmas_ols.colorbar(psm_pam_ols, ax = ax_pam_ols)
ax_pam_ols = fig_parmas_ols.add_subplot(gs_ols[0,1])
psm_pam_ols = ax_pam_ols.pcolormesh(ols_betas_means[1]*np.ones_like(beta1), cmap=jet, rasterized=True, vmin=beta_min_list[1], vmax=beta_max_list[1])
fig_parmas_ols.colorbar(psm_pam_ols, ax = ax_pam_ols)
ax_pam_ols = fig_parmas_ols.add_subplot(gs_ols[0,2])
psm_pam_ols = ax_pam_ols.pcolormesh(ols_betas_means[2]*np.ones_like(beta2), cmap=jet, rasterized=True,vmin=beta_min_list[2], vmax=beta_max_list[2])
fig_parmas_ols.colorbar(psm_pam_ols, ax = ax_pam_ols)
ax_pam_ols = fig_parmas_ols.add_subplot(gs_ols[0,3])
psm_pam_ols = ax_pam_ols.pcolormesh(ols_betas_means[3]*np.ones_like(beta3), cmap=jet, rasterized=True, vmin=beta_min_list[3], vmax=beta_max_list[3])
fig_parmas_ols.colorbar(psm_pam_ols, ax = ax_pam_ols)
plt.show()


#6.Draw different of prediction paramas surfaces
print("Four Absoulate Errors of mean prediction beta surfaces by STWR(row0),GWR(row1) and OLS(row2): \n") 
ppstwr_err_beta0 = np.abs(beta0 - stwr_predbeta0_means)
ppgwr_err_beta0 = np.abs(beta0 - gwr_predbeta0_means)
ppols_err_beta0 = np.abs(beta0 - ols_betas_means[0]*np.ones_like(beta0))


err_min_0 = np.amin(ppgwr_err_beta0)
err_max_0 = np.amax(ppgwr_err_beta0)
if(np.amin(ppstwr_err_beta0)<err_min_0):
    err_min_0 = np.amin(ppstwr_err_beta0)
if(np.amax(ppstwr_err_beta0)>err_max_0):
    err_max_0 = np.amax(ppstwr_err_beta0)

if err_min_0 >  np.min(ppols_err_beta0):
        err_min_0  = np.min(ppols_err_beta0)
if err_max_0 < np.max(ppols_err_beta0):
        err_max_0 =  np.max(ppols_err_beta0)


ppstwr_err_beta1 = np.abs(beta1 - stwr_predbeta1_means)
ppgwr_err_beta1 = np.abs(beta1 - gwr_predbeta1_means)
ppols_err_beta1 = np.abs(beta1 - ols_betas_means[1]*np.ones_like(beta1))

err_min_1 = np.amin(ppgwr_err_beta1)
err_max_1 = np.amax(ppgwr_err_beta1)
if(np.amin(ppstwr_err_beta1)<err_min_1):
    err_min_1 = np.amin(ppstwr_err_beta1)
if(np.amax(ppstwr_err_beta1)>err_max_1):
    err_max_1 = np.amax(ppstwr_err_beta1)

if err_min_1 >  np.min(ppols_err_beta1):
        err_min_1  = np.min(ppols_err_beta1)
if err_max_1 < np.max(ppols_err_beta1):
        err_max_1 =  np.max(ppols_err_beta1)  


ppstwr_err_beta2 = np.abs(beta2 - stwr_predbeta2_means)
ppgwr_err_beta2 = np.abs(beta2 -  gwr_predbeta2_means)
ppols_err_beta2 = np.abs(beta2 - ols_betas_means[2]*np.ones_like(beta2))

err_min_2 = np.amin(ppgwr_err_beta2)
err_max_2 = np.amax(ppgwr_err_beta2)
if(np.amin(ppstwr_err_beta2)<err_min_2):
    err_min_2 = np.amin(ppstwr_err_beta2)
if(np.amax(ppstwr_err_beta2)>err_max_2):
    err_max_2 = np.amax(ppstwr_err_beta2)
    
if err_min_2 >  np.min(ppols_err_beta2):
        err_min_2  = np.min(ppols_err_beta2)
if err_max_2 < np.max(ppols_err_beta2):
        err_max_2 =  np.max(ppols_err_beta2)  
                            
    
ppstwr_err_beta3 = np.abs(beta3 - stwr_predbeta3_means)
ppgwr_err_beta3 = np.abs(beta3 -  gwr_predbeta3_means)
ppols_err_beta3 = np.abs(beta3 - ols_betas_means[3]*np.ones_like(beta3))

err_min_3 = np.amin(ppgwr_err_beta3)
err_max_3 = np.amax(ppgwr_err_beta3)
if(np.amin(ppstwr_err_beta3)<err_min_3):
    err_min_3 = np.amin(ppstwr_err_beta3)
if(np.amax(ppstwr_err_beta3)>err_max_3):
    err_max_3 = np.amax(ppstwr_err_beta3)
    
if err_min_3 >  np.min(ppols_err_beta3):
        err_min_3  = np.min(ppols_err_beta3)
if err_max_3 < np.max(ppols_err_beta3):
        err_max_3 =  np.max(ppols_err_beta3)   


fig = plt.figure(figsize=(16,8),constrained_layout=True)
gs = GridSpec(3, 4, figure=fig)
jet  = plt.get_cmap('jet',256)
psm =0        
####STWR paramas####       

ax = fig.add_subplot(gs[0,0])
psm = ax.pcolormesh(ppstwr_err_beta0, cmap=jet, rasterized=True, vmin=err_min_0, vmax=err_max_0)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[0,1])
psm = ax.pcolormesh(ppstwr_err_beta1, cmap=jet, rasterized=True, vmin=err_min_1, vmax=err_max_1)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[0,2])
psm = ax.pcolormesh(ppstwr_err_beta2, cmap=jet, rasterized=True, vmin=err_min_2, vmax=err_max_2)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[0,3])
psm = ax.pcolormesh(ppstwr_err_beta3, cmap=jet, rasterized=True, vmin=err_min_3, vmax=err_max_3)
fig.colorbar(psm, ax = ax)

####GWR  paramas####     
ax = fig.add_subplot(gs[1,0])
psm = ax.pcolormesh(ppgwr_err_beta0, cmap=jet, rasterized=True, vmin=err_min_0, vmax=err_max_0)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[1,1])
psm = ax.pcolormesh(ppgwr_err_beta1, cmap=jet, rasterized=True, vmin=err_min_1, vmax=err_max_1)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[1,2])
psm = ax.pcolormesh(ppgwr_err_beta2, cmap=jet, rasterized=True, vmin=err_min_2, vmax=err_max_2)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[1,3])
psm = ax.pcolormesh(ppgwr_err_beta3, cmap=jet, rasterized=True, vmin=err_min_3, vmax=err_max_3)
fig.colorbar(psm, ax = ax)


####OLS  paramas####     
ax = fig.add_subplot(gs[2,0])
psm = ax.pcolormesh(np.abs(beta0 - ols_result.params[0]*np.ones_like(b10gndvi_f)), cmap=jet, rasterized=True, vmin=err_min_0, vmax=err_max_0)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[2,1])
psm = ax.pcolormesh(np.abs(beta1 - ols_result.params[1]*np.ones_like(b10gndvi_f)), cmap=jet, rasterized=True, vmin=err_min_1, vmax=err_max_1)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[2,2])
psm = ax.pcolormesh(np.abs(beta2 - ols_result.params[2]*np.ones_like(b10gndvi_f)), cmap=jet, rasterized=True, vmin=err_min_2, vmax=err_max_2)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[2,3])
psm = ax.pcolormesh(np.abs(beta3 - ols_result.params[3]*np.ones_like(b10gndvi_f)), cmap=jet, rasterized=True, vmin=err_min_3, vmax=err_max_3)
fig.colorbar(psm, ax = ax)

#fig.colorbar(psm, ax = fig.axes[3])
plt.show()


#####Generate a MAE for each Models######
stwr_mae_beta0 = np.mean(ppstwr_err_beta0)
gwr_mae_btea0 = np.mean(ppgwr_err_beta0)
ols_mae_beta0 = np.mean(ppols_err_beta0)

print("The MAE of beta0 predicted by STWR is:{}". format(stwr_mae_beta0))
print("The MAE of beta0 predicted by GWR is:{}". format(gwr_mae_btea0))
print("The MAE of beta0 predicted byOLS is:{}". format(ols_mae_beta0))

stwr_mae_beta1 = np.mean(ppstwr_err_beta1)
gwr_mae_btea1 = np.mean(ppgwr_err_beta1)
ols_mae_beta1 = np.mean(ppols_err_beta1)

print("The MAE of beta1 predicted by STWR is:{}". format(stwr_mae_beta1))
print("The MAE of beta1 predicted by GWR is:{}". format(gwr_mae_btea1))
print("The MAE of beta1 predicted byOLS is:{}". format(ols_mae_beta1))

stwr_mae_beta2 = np.mean(ppstwr_err_beta2)
gwr_mae_btea2 = np.mean(ppgwr_err_beta2)
ols_mae_beta2 = np.mean(ppols_err_beta2)

print("The MAE of beta2 predicted by STWR is:{}". format(stwr_mae_beta2))
print("The MAE of beta2 predicted by GWR is:{}". format(gwr_mae_btea2))
print("The MAE of beta2 predicted byOLS is:{}". format(ols_mae_beta2))

stwr_mae_beta3 = np.mean(ppstwr_err_beta3)
gwr_mae_btea3 = np.mean(ppgwr_err_beta3)
ols_mae_beta3 = np.mean(ppols_err_beta3)

print("The MAE of beta3 predicted by STWR is:{}". format(stwr_mae_beta3))
print("The MAE of beta3 predicted by GWR is:{}". format(gwr_mae_btea3))
print("The MAE of beta3 predicted byOLS is:{}". format(ols_mae_beta3))
#####Generate a MAE for each Models######
print("Four Errors of mean prediction beta surfaces by STWR(row0),GWR(row1) and OLS(row2): \n") 
ppstwr_eee_beta0 = stwr_predbeta0_means - beta0
ppgwr_eee_beta0 = gwr_predbeta0_means - beta0 
ppols_eee_beta0 = ols_betas_means[0]*np.ones_like(beta0) - beta0 


eee_min_0 = np.amin(ppgwr_eee_beta0)
eee_max_0 = np.amax(ppgwr_eee_beta0)
if(np.amin(ppstwr_eee_beta0)<eee_min_0):
    eee_min_0 = np.amin(ppstwr_eee_beta0)
if(np.amax(ppstwr_eee_beta0)>eee_max_0):
    eee_max_0 = np.amax(ppstwr_eee_beta0)

if eee_min_0 >  np.min(ppols_eee_beta0):
        eee_min_0  = np.min(ppols_eee_beta0)
if eee_max_0 < np.max(ppols_eee_beta0):
        eee_max_0 =  np.max(ppols_eee_beta0)


ppstwr_eee_beta1 = stwr_predbeta1_means - beta1 
ppgwr_eee_beta1 = gwr_predbeta1_means - beta1 
ppols_eee_beta1 = ols_betas_means[1]*np.ones_like(beta1) - beta1

eee_min_1 = np.amin(ppgwr_eee_beta1)
eee_max_1 = np.amax(ppgwr_eee_beta1)
if(np.amin(ppstwr_eee_beta1)<eee_min_1):
    eee_min_1 = np.amin(ppstwr_eee_beta1)
if(np.amax(ppstwr_eee_beta1)>eee_max_1):
    eee_max_1 = np.amax(ppstwr_eee_beta1)

if eee_min_1 >  np.min(ppols_eee_beta1):
        eee_min_1  = np.min(ppols_eee_beta1)
if eee_max_1 < np.max(ppols_eee_beta1):
        eee_max_1 =  np.max(ppols_eee_beta1)  


ppstwr_eee_beta2 = stwr_predbeta2_means - beta2 
ppgwr_eee_beta2 = gwr_predbeta2_means - beta2 
ppols_eee_beta2 = ols_betas_means[2]*np.ones_like(beta2) - beta2 

eee_min_2 = np.amin(ppgwr_eee_beta2)
eee_max_2 = np.amax(ppgwr_eee_beta2)
if(np.amin(ppstwr_eee_beta2)<eee_min_2):
    eee_min_2 = np.amin(ppstwr_eee_beta2)
if(np.amax(ppstwr_eee_beta2)>eee_max_2):
    eee_max_2 = np.amax(ppstwr_eee_beta2)
    
if eee_min_2 >  np.min(ppols_eee_beta2):
        eee_min_2  = np.min(ppols_eee_beta2)
if eee_max_2 < np.max(ppols_eee_beta2):
        eee_max_2 =  np.max(ppols_eee_beta2)  
                            
    
ppstwr_eee_beta3 = stwr_predbeta3_means - beta3 
ppgwr_eee_beta3 =  gwr_predbeta3_means - beta3 
ppols_eee_beta3 = ols_betas_means[3]*np.ones_like(beta3) - beta3

eee_min_3 = np.amin(ppgwr_eee_beta3)
eee_max_3 = np.amax(ppgwr_eee_beta3)
if(np.amin(ppstwr_eee_beta3)<eee_min_3):
    eee_min_3 = np.amin(ppstwr_eee_beta3)
if(np.amax(ppstwr_eee_beta3)>eee_max_3):
    eee_max_3 = np.amax(ppstwr_eee_beta3)
    
if eee_min_3 >  np.min(ppols_eee_beta3):
        eee_min_3  = np.min(ppols_eee_beta3)
if eee_max_3 < np.max(ppols_eee_beta3):
        eee_max_3 =  np.max(ppols_eee_beta3)   


fig = plt.figure(figsize=(16,8),constrained_layout=True)
gs = GridSpec(3, 4, figure=fig)
jet  = plt.get_cmap('jet',256)
psm =0        
####STWR paramas####       

ax = fig.add_subplot(gs[0,0])
psm = ax.pcolormesh(ppstwr_eee_beta0, cmap=jet, rasterized=True, vmin=eee_min_0, vmax=eee_max_0)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[0,1])
psm = ax.pcolormesh(ppstwr_eee_beta1, cmap=jet, rasterized=True, vmin=eee_min_1, vmax=eee_max_1)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[0,2])
psm = ax.pcolormesh(ppstwr_eee_beta2, cmap=jet, rasterized=True, vmin=eee_min_2, vmax=eee_max_2)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[0,3])
psm = ax.pcolormesh(ppstwr_eee_beta3, cmap=jet, rasterized=True, vmin=eee_min_3, vmax=eee_max_3)
fig.colorbar(psm, ax = ax)

####GWR  paramas####     
ax = fig.add_subplot(gs[1,0])
psm = ax.pcolormesh(ppgwr_eee_beta0, cmap=jet, rasterized=True, vmin=eee_min_0, vmax=eee_max_0)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[1,1])
psm = ax.pcolormesh(ppgwr_eee_beta1, cmap=jet, rasterized=True, vmin=eee_min_1, vmax=eee_max_1)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[1,2])
psm = ax.pcolormesh(ppgwr_eee_beta2, cmap=jet, rasterized=True, vmin=eee_min_2, vmax=eee_max_2)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[1,3])
psm = ax.pcolormesh(ppgwr_eee_beta3, cmap=jet, rasterized=True, vmin=eee_min_3, vmax=eee_max_3)
fig.colorbar(psm, ax = ax)


####OLS  paramas####     
ax = fig.add_subplot(gs[2,0])
psm = ax.pcolormesh( ols_result.params[0]*np.ones_like(b10gndvi_f) - beta0 , cmap=jet, rasterized=True, vmin=eee_min_0, vmax=eee_max_0)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[2,1])
psm = ax.pcolormesh(ols_result.params[1]*np.ones_like(b10gndvi_f) - beta1  , cmap=jet, rasterized=True, vmin=eee_min_1, vmax=eee_max_1)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[2,2])
psm = ax.pcolormesh(ols_result.params[2]*np.ones_like(b10gndvi_f) - beta2 , cmap=jet, rasterized=True, vmin=eee_min_2, vmax=eee_max_2)
fig.colorbar(psm, ax = ax)

ax = fig.add_subplot(gs[2,3])
psm = ax.pcolormesh(ols_result.params[3]*np.ones_like(b10gndvi_f) - beta3 , cmap=jet, rasterized=True, vmin=eee_min_3, vmax=eee_max_3)
fig.colorbar(psm, ax = ax)

#fig.colorbar(psm, ax = fig.axes[3])
plt.show()


#comparison of STWR,GWR,and OLS.
sum_err2_stwr = np.zeros_like(lst10)
for idx_py_stwr in stwr_predy_arr:
    sum_err2_stwr +=   (idx_py_stwr-lst10)*(idx_py_stwr-lst10)    
rmse_stwr_predy = np.sqrt(sum_err2_stwr/random_times)

sum_err2_gwr = np.zeros_like(lst10)
for idx_py_gwr in gwr_predy_arr:
    sum_err2_gwr +=   (idx_py_gwr-lst10)*(idx_py_gwr-lst10)    
rmse_gwr_predy = np.sqrt(sum_err2_gwr/random_times)

sum_err2_ols = np.zeros_like(lst10)
for idx_py_ols in ols_predy_arr :   
    sum_err2_ols +=   (idx_py_ols-lst10)*(idx_py_ols-lst10)    
rmse_ols_predy = np.sqrt(sum_err2_ols/random_times)
#Comparision surfaces of Y_true,prdictions mean surface by OLS,GWR and STWR.

print(" Comparision RMSE surfaces of pred_y  by  STWR , GWR and OLS: \n")   
vmin_predy_rmse = np.amin(rmse_stwr_predy)
vmax_predy_rmse =np.amax(rmse_stwr_predy)

if vmin_predy_rmse > np.amin(rmse_gwr_predy):
        vmin_predy_rmse = np.amin(rmse_gwr_predy)
if vmin_predy_rmse > np.amin(rmse_ols_predy):
        vmin_predy_rmse = np.amin(rmse_ols_predy)
        
if vmax_predy_rmse < np.amax(rmse_gwr_predy):
        vmax_predy_rmse = np.amax(rmse_gwr_predy)
if vmax_predy_rmse < np.amax(rmse_ols_predy):
        vmax_predy_rmse = np.amax(rmse_ols_predy)
        
fig_pred_rmse = plt.figure(figsize=(16,3),constrained_layout=True)
gs_stwr = GridSpec(1, 3, figure=fig_pred_rmse)
jet  = plt.get_cmap('jet',256)

ax_stwr  = fig_pred_rmse.add_subplot(gs_stwr[0,0])
psm_stwr = ax_stwr.pcolormesh(rmse_stwr_predy, cmap=jet, rasterized=True, vmin=vmin_predy_rmse, vmax=vmax_predy_rmse)
fig_pred_rmse.colorbar(psm_stwr, ax = ax_stwr)
ax_gwr  =  fig_pred_rmse.add_subplot(gs_stwr[0,1])
psm_gwr =  ax_gwr.pcolormesh(rmse_gwr_predy, cmap=jet, rasterized=True, vmin=vmin_predy_rmse, vmax=vmax_predy_rmse)
fig_pred_rmse.colorbar(psm_gwr, ax = ax_gwr)
ax_ols =  fig_pred_rmse.add_subplot(gs_stwr[0,2])
psm_ols=  ax_ols.pcolormesh(rmse_ols_predy, cmap=jet, rasterized=True, vmin=vmin_predy_rmse, vmax=vmax_predy_rmse)
fig_pred_rmse.colorbar(psm_ols, ax = ax_ols)
plt.show()

stwr_rmse_py_a = np.mean(rmse_stwr_predy)
gwr_rmse_py_a = np.mean(rmse_gwr_predy)
ols_rmse_py_a = np.mean(rmse_ols_predy)

print("The RMSE of pred_y predicted by STWR is:{}". format(stwr_rmse_py_a))
print("The RMSE of pred_y predicted by GWR is:{}". format(gwr_rmse_py_a))
print("The RMSE of pred_y predicted by OLS is:{}". format(ols_rmse_py_a))




print(" Comparision RMSE surfaces of beta0  by  STWR , GWR and OLS: \n") 
sum_err2_stwr_beta0 = np.zeros_like(beta0)
for idx_py_stwr in stwr_predbeta0_arr:
    sum_err2_stwr_beta0 +=   (idx_py_stwr-beta0)*(idx_py_stwr-beta0)    
rmse_stwr_beta0 = np.sqrt(sum_err2_stwr_beta0/random_times)

sum_err2_gwr_beta0 = np.zeros_like(beta0)
for idx_py_gwr in gwr_predbeta0_arr:
    sum_err2_gwr_beta0 +=   (idx_py_gwr-beta0)*(idx_py_gwr-beta0)    
rmse_gwr_beta0 = np.sqrt((sum_err2_gwr_beta0/random_times))

sum_err2_ols_beta0 = np.zeros_like(beta0)
for idx_py_ols in ols_betas[:,0] :
   sum_err2_ols_beta0 +=   (idx_py_ols*np.ones_like(beta0)-beta0)*(idx_py_ols*np.ones_like(beta0)-beta0)    
rmse_ols_beta0 = np.sqrt(sum_err2_ols_beta0/random_times)
          

vmin_beta0 = np.amin(rmse_stwr_beta0)
vmax_beta0 =np.amax(rmse_stwr_beta0)

if vmin_beta0 > np.amin(rmse_gwr_beta0):
        vmin_beta0 = np.amin(rmse_gwr_beta0)
if vmin_beta0 > np.amin(rmse_ols_beta0):
        vmin_beta0 = np.amin(rmse_ols_beta0)
        
if vmax_beta0 < np.amax(rmse_gwr_beta0):
        vmax_beta0 = np.amax(rmse_gwr_beta0)
if vmax_beta0 < np.amax(rmse_ols_beta0):
        vmax_beta0 = np.amax(rmse_ols_beta0)

fig_pred_rmse = plt.figure(figsize=(16,3),constrained_layout=True)
gs_stwr = GridSpec(1, 3, figure=fig_pred_rmse)
jet  = plt.get_cmap('jet',256)

ax_stwr  = fig_pred_rmse.add_subplot(gs_stwr[0,0])
psm_stwr = ax_stwr.pcolormesh(rmse_stwr_beta0, cmap=jet, rasterized=True, vmin=vmin_beta0, vmax=vmax_beta0)
fig_pred_rmse.colorbar(psm_stwr, ax = ax_stwr)
ax_gwr  =  fig_pred_rmse.add_subplot(gs_stwr[0,1])
psm_gwr =  ax_gwr.pcolormesh(rmse_gwr_beta0, cmap=jet, rasterized=True, vmin=vmin_beta0, vmax=vmax_beta0)
fig_pred_rmse.colorbar(psm_gwr, ax = ax_gwr)
ax_ols =  fig_pred_rmse.add_subplot(gs_stwr[0,2])
psm_ols=  ax_ols.pcolormesh(rmse_ols_beta0, cmap=jet, rasterized=True, vmin=vmin_beta0, vmax=vmax_beta0)
fig_pred_rmse.colorbar(psm_ols, ax = ax_ols)
plt.show()

stwr_rmse_b0_a = np.mean(rmse_stwr_beta0)
gwr_rmse_b0_a = np.mean(rmse_gwr_beta0)
ols_rmse_b0_a = np.mean(rmse_ols_beta0)

print("The RMSE of beta0 predicted by STWR is:{}". format(stwr_rmse_b0_a))
print("The RMSE of beta0 predicted by GWR is:{}". format(gwr_rmse_b0_a))
print("The RMSE of beta0 predicted by OLS is:{}". format(ols_rmse_b0_a))

print(" Comparision RMSE surfaces of beta1  by  STWR , GWR and OLS: \n") 
sum_err2_stwr_beta1 = np.zeros_like(beta1)
for idx_py_stwr in stwr_predbeta1_arr:
    sum_err2_stwr_beta1 +=   (idx_py_stwr-beta1)*(idx_py_stwr-beta1)    
rmse_stwr_beta1 = np.sqrt(sum_err2_stwr_beta1/sample_num)

sum_err2_gwr_beta1 = np.zeros_like(beta1)
for idx_py_gwr in gwr_predbeta1_arr:
    sum_err2_gwr_beta1 +=   (idx_py_gwr-beta1)*(idx_py_gwr-beta1)    
rmse_gwr_beta1 = np.sqrt((sum_err2_gwr_beta1/sample_num))

sum_err2_ols_beta1 = np.zeros_like(beta1)
for idx_py_ols in ols_betas[:,1] :   
    sum_err2_ols_beta1 +=   (idx_py_ols*np.ones_like(beta1)-beta1)*(idx_py_ols*np.ones_like(beta1)-beta1)    
rmse_ols_beta1 = np.sqrt(sum_err2_ols_beta1/sample_num)
          

vmin_beta1 = np.amin(rmse_stwr_beta1)
vmax_beta1 =np.amax(rmse_stwr_beta1)

if vmin_beta1 > np.amin(rmse_gwr_beta1):
        vmin_beta1 = np.amin(rmse_gwr_beta1)
if vmin_beta1 > np.amin(rmse_ols_beta1):
        vmin_beta1 = np.amin(rmse_ols_beta1)
        
if vmax_beta1 < np.amax(rmse_gwr_beta1):
        vmax_beta1 = np.amax(rmse_gwr_beta1)
if vmax_beta1 < np.amax(rmse_ols_beta1):
        vmax_beta1 = np.amax(rmse_ols_beta1)

fig_pred_rmse = plt.figure(figsize=(16,3),constrained_layout=True)
gs_stwr = GridSpec(1, 3, figure=fig_pred_rmse)
jet  = plt.get_cmap('jet',256)

ax_stwr  = fig_pred_rmse.add_subplot(gs_stwr[0,0])
psm_stwr = ax_stwr.pcolormesh(rmse_stwr_beta1, cmap=jet, rasterized=True, vmin=vmin_beta1, vmax=vmax_beta1)
fig_pred_rmse.colorbar(psm_stwr, ax = ax_stwr)
ax_gwr  =  fig_pred_rmse.add_subplot(gs_stwr[0,1])
psm_gwr =  ax_gwr.pcolormesh(rmse_gwr_beta1, cmap=jet, rasterized=True, vmin=vmin_beta1, vmax=vmax_beta1)
fig_pred_rmse.colorbar(psm_gwr, ax = ax_gwr)
ax_ols =  fig_pred_rmse.add_subplot(gs_stwr[0,2])
psm_ols=  ax_ols.pcolormesh(rmse_ols_beta1, cmap=jet, rasterized=True, vmin=vmin_beta1, vmax=vmax_beta1)
fig_pred_rmse.colorbar(psm_ols, ax = ax_ols)
plt.show()

stwr_rmse_b1_a = np.mean(rmse_stwr_beta1)
gwr_rmse_b1_a = np.mean(rmse_gwr_beta1)
ols_rmse_b1_a = np.mean(rmse_ols_beta1)

print("The RMSE of beta1 predicted by STWR is:{}". format(stwr_rmse_b1_a))
print("The RMSE of beta1 predicted by GWR is:{}". format(gwr_rmse_b1_a))
print("The RMSE of beta1 predicted by OLS is:{}". format(ols_rmse_b1_a))


print(" Comparision RMSE surfaces of beta2  by  STWR , GWR and OLS: \n") 
sum_err2_stwr_beta2 = np.zeros_like(beta2)
for idx_py_stwr in stwr_predbeta2_arr:
    sum_err2_stwr_beta2 +=   (idx_py_stwr-beta2)*(idx_py_stwr-beta2)    
rmse_stwr_beta2 = np.sqrt(sum_err2_stwr_beta2/sample_num)

sum_err2_gwr_beta2 = np.zeros_like(beta2)
for idx_py_gwr in gwr_predbeta2_arr:
    sum_err2_gwr_beta2 +=   (idx_py_gwr-beta2)*(idx_py_gwr-beta2)    
rmse_gwr_beta2 = np.sqrt((sum_err2_gwr_beta2/sample_num))

sum_err2_ols_beta2 = np.zeros_like(beta2)
for idx_py_ols in ols_betas[:,2] :   
    sum_err2_ols_beta2 +=   (idx_py_ols*np.ones_like(beta2)-beta2)*(idx_py_ols*np.ones_like(beta2)-beta2)    
rmse_ols_beta2 = np.sqrt(sum_err2_ols_beta2/sample_num)
          

vmin_beta2 = np.amin(rmse_stwr_beta2)
vmax_beta2 =np.amax(rmse_stwr_beta2)

if vmin_beta2 > np.amin(rmse_gwr_beta2):
        vmin_beta2 = np.amin(rmse_gwr_beta2)
if vmin_beta2 > np.amin(rmse_ols_beta2):
        vmin_beta2 = np.amin(rmse_ols_beta2)
        
if vmax_beta2 < np.amax(rmse_gwr_beta2):
        vmax_beta2 = np.amax(rmse_gwr_beta2)
if vmax_beta2 < np.amax(rmse_ols_beta2):
        vmax_beta2 = np.amax(rmse_ols_beta2)

fig_pred_rmse = plt.figure(figsize=(16,3),constrained_layout=True)
gs_stwr = GridSpec(1, 3, figure=fig_pred_rmse)
jet  = plt.get_cmap('jet',256)

ax_stwr  = fig_pred_rmse.add_subplot(gs_stwr[0,0])
psm_stwr = ax_stwr.pcolormesh(rmse_stwr_beta2, cmap=jet, rasterized=True, vmin=vmin_beta2, vmax=vmax_beta2)
fig_pred_rmse.colorbar(psm_stwr, ax = ax_stwr)
ax_gwr  =  fig_pred_rmse.add_subplot(gs_stwr[0,1])
psm_gwr =  ax_gwr.pcolormesh(rmse_gwr_beta2, cmap=jet, rasterized=True, vmin=vmin_beta2, vmax=vmax_beta2)
fig_pred_rmse.colorbar(psm_gwr, ax = ax_gwr)
ax_ols =  fig_pred_rmse.add_subplot(gs_stwr[0,2])
psm_ols=  ax_ols.pcolormesh(rmse_ols_beta2, cmap=jet, rasterized=True, vmin=vmin_beta2, vmax=vmax_beta2)
fig_pred_rmse.colorbar(psm_ols, ax = ax_ols)
plt.show()

stwr_rmse_b2_a = np.mean(rmse_stwr_beta2)
gwr_rmse_b2_a = np.mean(rmse_gwr_beta2)
ols_rmse_b2_a = np.mean(rmse_ols_beta2)

print("The RMSE of beta2 predicted by STWR is:{}". format(stwr_rmse_b2_a))
print("The RMSE of beta2 predicted by GWR is:{}". format(gwr_rmse_b2_a))
print("The RMSE of beta2 predicted by OLS is:{}". format(ols_rmse_b2_a))


print(" Comparision RMSE surfaces of beta3  by  STWR , GWR and OLS: \n") 
sum_err2_stwr_beta3 = np.zeros_like(beta3)
for idx_py_stwr in stwr_predbeta3_arr:
    sum_err2_stwr_beta3 +=   (idx_py_stwr-beta3)*(idx_py_stwr-beta3)    
rmse_stwr_beta3 = np.sqrt(sum_err2_stwr_beta3/random_times)

sum_err2_gwr_beta3 = np.zeros_like(beta3)
for idx_py_gwr in gwr_predbeta3_arr:
    sum_err2_gwr_beta3 +=   (idx_py_gwr-beta3)*(idx_py_gwr-beta3)    
rmse_gwr_beta3 = np.sqrt((sum_err2_gwr_beta3/random_times))

sum_err2_ols_beta3 = np.zeros_like(beta3)
for idx_py_ols in ols_betas[:,3] :   
    sum_err2_ols_beta3 +=   (idx_py_ols*np.ones_like(beta3)-beta3)*(idx_py_ols*np.ones_like(beta3)-beta3)    
rmse_ols_beta3 = np.sqrt(sum_err2_ols_beta3/random_times)
          

vmin_beta3 = np.amin(rmse_stwr_beta3)
vmax_beta3 =np.amax(rmse_stwr_beta3)

if vmin_beta3 > np.amin(rmse_gwr_beta3):
        vmin_beta3 = np.amin(rmse_gwr_beta3)
if vmin_beta3 > np.amin(rmse_ols_beta3):
        vmin_beta3 = np.amin(rmse_ols_beta3)
        
if vmax_beta3 < np.amax(rmse_gwr_beta3):
        vmax_beta2 = np.amax(rmse_gwr_beta3)
if vmax_beta3 < np.amax(rmse_ols_beta3):
        vmax_beta3 = np.amax(rmse_ols_beta3)

fig_pred_rmse = plt.figure(figsize=(16,3),constrained_layout=True)
gs_stwr = GridSpec(1, 3, figure=fig_pred_rmse)
jet  = plt.get_cmap('jet',256)

ax_stwr  = fig_pred_rmse.add_subplot(gs_stwr[0,0])
psm_stwr = ax_stwr.pcolormesh(rmse_stwr_beta3, cmap=jet, rasterized=True, vmin=vmin_beta3, vmax=vmax_beta3)
fig_pred_rmse.colorbar(psm_stwr, ax = ax_stwr)
ax_gwr  =  fig_pred_rmse.add_subplot(gs_stwr[0,1])
psm_gwr =  ax_gwr.pcolormesh(rmse_gwr_beta3, cmap=jet, rasterized=True, vmin=vmin_beta3, vmax=vmax_beta3)
fig_pred_rmse.colorbar(psm_gwr, ax = ax_gwr)
ax_ols =  fig_pred_rmse.add_subplot(gs_stwr[0,2])
psm_ols=  ax_ols.pcolormesh(rmse_ols_beta3, cmap=jet, rasterized=True, vmin=vmin_beta3, vmax=vmax_beta3)
fig_pred_rmse.colorbar(psm_ols, ax = ax_ols)
plt.show()

stwr_rmse_b3_a = np.mean(rmse_stwr_beta3)
gwr_rmse_b3_a = np.mean(rmse_gwr_beta3)
ols_rmse_b3_a = np.mean(rmse_ols_beta3)

print("The RMSE of beta3 predicted by STWR is:{}". format(stwr_rmse_b3_a))
print("The RMSE of beta3 predicted by GWR is:{}". format(gwr_rmse_b3_a))
print("The RMSE of beta3 predicted by OLS is:{}". format(ols_rmse_b3_a))





