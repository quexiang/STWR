import  numpy  as  np
import libpysal as ps
from stwr.gwr import GWR, MGWR,STWR
from stwr.sel_bw import *
from stwr.utils import shift_colormap, truncate_colormap
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
from matplotlib.gridspec import GridSpec
import time
import csv 
import copy 
from scipy import stats

import os
import rasterio
import rasterio.plot
import rasterio.features
import rasterio.warp

from rasterio.transform import Affine
###########################################DataPrepared###############################################
str_Xfilepath = 'D:/transactions/Postdoc/FanChao-wangzhe/0515X/'
str_yfilepath = 'D:/transactions/Postdoc/FanChao-wangzhe/LST/'
out_dataPrepared = 'D:/transactions/Postdoc/FanChao-wangzhe/LST_prepared/'
out_sampledata = 'D:/transactions/Postdoc/FanChao-wangzhe/Re_Samples_data/'

lst_fldname = ['2000','2010','2020']

rlst1995 = rasterio.open(str_yfilepath +'1995LST_Small.tif')
lst_1995_dataset = rlst1995.read(1)

rlst2000 = rasterio.open(str_yfilepath +'2000LST_Small.tif')
lst_2000_dataset = rlst2000.read(1)

rlst2005 = rasterio.open(str_yfilepath +'2005LST_Small.tif')
lst_2005_dataset = rlst2005.read(1)

rlst2010 = rasterio.open(str_yfilepath +'2010LST_Small.tif')
lst_2010_dataset = rlst2010.read(1)

rlst2015 = rasterio.open(str_yfilepath +'2015LST_Small.tif')
lst_2015_dataset = rlst2015.read(1)

rlst2020 = rasterio.open(str_yfilepath +'2020LST_Small.tif')
lst_2020_dataset = rlst2020.read(1)


pf_1995 = rlst1995.profile
transform_lst1995 =rlst1995.profile['transform']
nodata = pf_1995['nodata']
print (pf_1995)
data_crs = pf_1995['crs']

#for fname in lst_fldname:
fname = '1995'
str_Cur_Xfilepath = str_Xfilepath  + fname + '/'
rgndbi95 = rasterio.open(str_Cur_Xfilepath +'GNDBI_clip.tif')
bgndbi95 = rgndbi95.read(1)
bgndbi95 = bgndbi95[:-2,:-1]

rgndvi95 = rasterio.open(str_Cur_Xfilepath + 'GNDVI_clip.tif')
bgndvi95 = rgndvi95.read(1)
bgndvi95 = bgndvi95[:-2,:-1]

rlndvi95 = rasterio.open(str_Cur_Xfilepath + 'LNDVI_clip.tif')
blndvi95 = rlndvi95.read(1)  
blndvi95 = blndvi95[:-2,:-1]


pf_gndbi = rgndbi95.profile
transform_gndbi =rgndbi95.profile['transform']
nodata_gndbi = pf_gndbi['nodata']
print (pf_gndbi)
data_crs_gndbi = pf_gndbi['crs']

all_y_95_data = []
all_y_00_data = []
all_y_05_data = []
all_y_10_data = []
all_y_15_data = []
all_y_20_data = []

all_coords_list2 = []
mask_rblst2 = rgndbi95.dataset_mask()
mask_rblst2 = mask_rblst2[:-2,:-1]
for row in range(mask_rblst2.shape[0]):
    for col in range (mask_rblst2.shape[1]):
        if(mask_rblst2[row,col]>0):
            all_coords_list2.append(rgndbi95.xy(row,col))
            x_curidx,y_curidx = rgndbi95.xy(row,col)
            n_yrow,nycol = rlst1995.index(x_curidx,y_curidx)
            all_y_95_data.append(lst_1995_dataset[n_yrow,nycol])
            all_y_00_data.append(lst_2000_dataset[n_yrow,nycol])
            all_y_05_data.append(lst_2005_dataset[n_yrow,nycol])
            all_y_10_data.append(lst_2010_dataset[n_yrow,nycol])
            all_y_15_data.append(lst_2015_dataset[n_yrow,nycol])
            all_y_20_data.append(lst_2020_dataset[n_yrow,nycol])
    
#output the y_data to tifs
data_out_y95 = np.asarray(all_y_95_data,dtype = bgndbi95.dtype).reshape(mask_rblst2.shape)
data_out_y00 = np.asarray(all_y_00_data,dtype = bgndbi95.dtype).reshape(mask_rblst2.shape)
data_out_y05 = np.asarray(all_y_05_data,dtype = bgndbi95.dtype).reshape(mask_rblst2.shape)
data_out_y10 = np.asarray(all_y_10_data,dtype = bgndbi95.dtype).reshape(mask_rblst2.shape)
data_out_y15 = np.asarray(all_y_15_data,dtype = bgndbi95.dtype).reshape(mask_rblst2.shape)
data_out_y20 = np.asarray(all_y_20_data,dtype = bgndbi95.dtype).reshape(mask_rblst2.shape)

            
with rasterio.open(out_dataPrepared +'pre_1995.tif', 'w', driver='GTiff', 
                   height=data_out_y95.shape[0],
                   width=data_out_y95.shape[1], count=1, dtype= bgndbi95.dtype,
                   crs= data_crs_gndbi, transform=transform_gndbi,nodata = nodata_gndbi) as dsdata_out:
    dsdata_out.write(data_out_y95, 1)

with rasterio.open(out_dataPrepared +'pre_2000.tif', 'w', driver='GTiff', 
                   height=data_out_y00.shape[0],
                   width=data_out_y00.shape[1], count=1, dtype= bgndbi95.dtype,
                   crs= data_crs_gndbi, transform=transform_gndbi,nodata = nodata_gndbi) as dsdata_out:
    dsdata_out.write(data_out_y00, 1)   
    
with rasterio.open(out_dataPrepared +'pre_2005.tif', 'w', driver='GTiff', 
                   height=data_out_y05.shape[0],
                   width=data_out_y05.shape[1], count=1, dtype= bgndbi95.dtype,
                   crs= data_crs_gndbi, transform=transform_gndbi,nodata = nodata_gndbi) as dsdata_out:
    dsdata_out.write(data_out_y05, 1)
    
with rasterio.open(out_dataPrepared +'pre_2010.tif', 'w', driver='GTiff', 
                   height=data_out_y10.shape[0],
                   width=data_out_y10.shape[1], count=1, dtype= bgndbi95.dtype,
                   crs= data_crs_gndbi, transform=transform_gndbi,nodata = nodata_gndbi) as dsdata_out:
    dsdata_out.write(data_out_y10, 1)          
with rasterio.open(out_dataPrepared +'pre_2015.tif', 'w', driver='GTiff', 
                   height=data_out_y15.shape[0],
                   width=data_out_y15.shape[1], count=1, dtype= bgndbi95.dtype,
                   crs= data_crs_gndbi, transform=transform_gndbi,nodata = nodata_gndbi) as dsdata_out:
    dsdata_out.write(data_out_y15, 1) 

with rasterio.open(out_dataPrepared +'pre_2020.tif', 'w', driver='GTiff', 
                   height=data_out_y20.shape[0],
                   width=data_out_y20.shape[1], count=1, dtype= bgndbi95.dtype,
                   crs= data_crs_gndbi, transform=transform_gndbi,nodata = nodata_gndbi) as dsdata_out:
    dsdata_out.write(data_out_y20, 1)      
###################################predicted betas and LST ###########################################
#Sample form the generated surfaces for tests.
np.random.seed(1000)
sample_num = 3000
samples_ticks =  np.random.choice(range(bgndbi95.shape[0]*bgndbi95.shape[1]),sample_num,replace=False)
mask_ticks = np.ones_like(all_y_95_data,dtype=bool).flatten() 
mask_ticks[samples_ticks] = False
cal_coords_tick = np.asarray(all_coords_list2)[~mask_ticks]
cal_y_95_tick = np.asarray(all_y_95_data)[~mask_ticks]
x1_95 = bgndbi95.flatten().reshape((-1,1))
x2_95 = bgndvi95.flatten().reshape((-1,1))
x3_95 = blndvi95.flatten().reshape((-1,1))
cal_X_95 = np.concatenate((x1_95,x2_95), axis=1)
cal_X_95 = np.concatenate((cal_X_95,x3_95), axis=1)
cal_X_95_tick = cal_X_95[~mask_ticks]

year95 = np.ones_like(cal_y_95_tick)* 1995

#output 95
dataframe = pd.DataFrame({'LST':cal_y_95_tick.tolist(),'GNDBI':cal_X_95_tick[:,0].tolist(),'GNDVI':cal_X_95_tick[:,1].tolist(),'LNDVI':cal_X_95_tick[:,2].tolist(),'X':cal_coords_tick[:,0].tolist(),'Y':cal_coords_tick[:,1].tolist(),'Year':year95.tolist()})
dataframe.to_csv(out_sampledata+"samples_{:d}.csv".format(1995))
    
###############################################################################################################

fname = '2000'
str_Cur_Xfilepath = str_Xfilepath  + fname + '/'
rgndbi00 = rasterio.open(str_Cur_Xfilepath +'GNDBI_clip.tif')
bgndbi00 = rgndbi00.read(1)
bgndbi00 = bgndbi00[:-2,:-1]

rgndvi00 = rasterio.open(str_Cur_Xfilepath + 'GNDVI_clip.tif')
bgndvi00 = rgndvi00.read(1)
bgndvi00 = bgndvi00[:-2,:-1]

rlndvi00 = rasterio.open(str_Cur_Xfilepath + 'LNDVI_clip.tif')
blndvi00 = rlndvi00.read(1)  
blndvi00 = blndvi00[:-2,:-1]

cal_y_00_tick = np.asarray(all_y_00_data)[~mask_ticks]
x1_00 = bgndbi00.flatten().reshape((-1,1))
x2_00 = bgndvi00.flatten().reshape((-1,1))
x3_00 = blndvi00.flatten().reshape((-1,1))
cal_X_00 = np.concatenate((x1_00,x2_00), axis=1)
cal_X_00 = np.concatenate((cal_X_00,x3_00), axis=1)
cal_X_00_tick = cal_X_00[~mask_ticks]
year00 = np.ones_like(cal_y_00_tick)* 2000
#output 00
dataframe = pd.DataFrame({'LST':cal_y_00_tick.tolist(),'GNDBI':cal_X_00_tick[:,0].tolist(),'GNDVI':cal_X_00_tick[:,1].tolist(),'LNDVI':cal_X_00_tick[:,2].tolist(),'X':cal_coords_tick[:,0].tolist(),'Y':cal_coords_tick[:,1].tolist(),'Year':year00.tolist()})
dataframe.to_csv(out_sampledata+"samples_{:d}.csv".format(2000))
#######################################################################################################################
fname = '2005'
str_Cur_Xfilepath = str_Xfilepath  + fname + '/'
rgndbi05 = rasterio.open(str_Cur_Xfilepath +'GNDBI_clip.tif')
bgndbi05 = rgndbi05.read(1)
bgndbi05 = bgndbi05[:-2,:-1]

rgndvi05 = rasterio.open(str_Cur_Xfilepath + 'GNDVI_clip.tif')
bgndvi05 = rgndvi05.read(1)
bgndvi05 = bgndvi05[:-2,:-1]

rlndvi05 = rasterio.open(str_Cur_Xfilepath + 'LNDVI_clip.tif')
blndvi05 = rlndvi05.read(1)  
blndvi05 = blndvi05[:-2,:-1]

cal_y_05_tick = np.asarray(all_y_05_data)[~mask_ticks]
x1_05 = bgndbi05.flatten().reshape((-1,1))
x2_05 = bgndvi05.flatten().reshape((-1,1))
x3_05 = blndvi05.flatten().reshape((-1,1))
cal_X_05 = np.concatenate((x1_05,x2_05), axis=1)
cal_X_05 = np.concatenate((cal_X_05,x3_05), axis=1)
cal_X_05_tick = cal_X_05[~mask_ticks]
year05 = np.ones_like(cal_y_05_tick)* 2005
#output 05
dataframe = pd.DataFrame({'LST':cal_y_05_tick.tolist(),'GNDBI':cal_X_05_tick[:,0].tolist(),'GNDVI':cal_X_05_tick[:,1].tolist(),'LNDVI':cal_X_05_tick[:,2].tolist(),'X':cal_coords_tick[:,0].tolist(),'Y':cal_coords_tick[:,1].tolist(),'Year':year05.tolist()})
dataframe.to_csv(out_sampledata+"samples_{:d}.csv".format(2005))
###############################################################################################################
fname = '2010'
str_Cur_Xfilepath = str_Xfilepath  + fname + '/'
rgndbi10 = rasterio.open(str_Cur_Xfilepath +'GNDBI_clip.tif')
bgndbi10 = rgndbi10.read(1)
bgndbi10 = bgndbi10[:-2,:-1]

rgndvi10 = rasterio.open(str_Cur_Xfilepath + 'GNDVI_clip.tif')
bgndvi10 = rgndvi10.read(1)
bgndvi10 = bgndvi10[:-2,:-1]

rlndvi10 = rasterio.open(str_Cur_Xfilepath + 'LNDVI_clip.tif')
blndvi10 = rlndvi10.read(1)  
blndvi10 = blndvi10[:-2,:-1]

cal_y_10_tick = np.asarray(all_y_10_data)[~mask_ticks]
x1_10 = bgndbi10.flatten().reshape((-1,1))
x2_10 = bgndvi10.flatten().reshape((-1,1))
x3_10 = blndvi10.flatten().reshape((-1,1))
cal_X_10 = np.concatenate((x1_10,x2_10), axis=1)
cal_X_10 = np.concatenate((cal_X_10,x3_10), axis=1)
cal_X_10_tick = cal_X_10[~mask_ticks]
year10 = np.ones_like(cal_y_10_tick)* 2010
#output 10
dataframe = pd.DataFrame({'LST':cal_y_10_tick.tolist(),'GNDBI':cal_X_10_tick[:,0].tolist(),'GNDVI':cal_X_10_tick[:,1].tolist(),'LNDVI':cal_X_10_tick[:,2].tolist(),'X':cal_coords_tick[:,0].tolist(),'Y':cal_coords_tick[:,1].tolist(),'Year':year10.tolist()})
dataframe.to_csv(out_sampledata+"samples_{:d}.csv".format(2010))

###############################################################################################################
fname = '2015'
str_Cur_Xfilepath = str_Xfilepath  + fname + '/'
rgndbi15 = rasterio.open(str_Cur_Xfilepath +'GNDBI_clip.tif')
bgndbi15 = rgndbi15.read(1)
bgndbi15 = bgndbi15[:-2,:-1]

rgndvi15 = rasterio.open(str_Cur_Xfilepath + 'GNDVI_clip.tif')
bgndvi15 = rgndvi15.read(1)
bgndvi15 = bgndvi15[:-2,:-1]

rlndvi15 = rasterio.open(str_Cur_Xfilepath + 'LNDVI_clip.tif')
blndvi15 = rlndvi15.read(1)  
blndvi15 = blndvi15[:-2,:-1]

cal_y_15_tick = np.asarray(all_y_15_data)[~mask_ticks]
x1_15 = bgndbi15.flatten().reshape((-1,1))
x2_15 = bgndvi15.flatten().reshape((-1,1))
x3_15 = blndvi15.flatten().reshape((-1,1))
cal_X_15 = np.concatenate((x1_15,x2_15), axis=1)
cal_X_15 = np.concatenate((cal_X_15,x3_15), axis=1)
cal_X_15_tick = cal_X_15[~mask_ticks]
year15 = np.ones_like(cal_y_15_tick)* 2015
#output 15
dataframe = pd.DataFrame({'LST':cal_y_15_tick.tolist(),'GNDBI':cal_X_15_tick[:,0].tolist(),'GNDVI':cal_X_15_tick[:,1].tolist(),'LNDVI':cal_X_15_tick[:,2].tolist(),'X':cal_coords_tick[:,0].tolist(),'Y':cal_coords_tick[:,1].tolist(),'Year':year15.tolist()})
dataframe.to_csv(out_sampledata+"samples_{:d}.csv".format(2015))
###############################################################################################################
fname = '2020'
str_Cur_Xfilepath = str_Xfilepath  + fname + '/'
rgndbi20 = rasterio.open(str_Cur_Xfilepath +'GNDBI_clip.tif')
bgndbi20 = rgndbi20.read(1)
bgndbi20 = bgndbi20[:-2,:-1]

rgndvi20 = rasterio.open(str_Cur_Xfilepath + 'GNDVI_clip.tif')
bgndvi20 = rgndvi20.read(1)
bgndvi20 = bgndvi20[:-2,:-1]

rlndvi20 = rasterio.open(str_Cur_Xfilepath + 'LNDVI_clip.tif')
blndvi20 = rlndvi20.read(1)  
blndvi20 = blndvi20[:-2,:-1]

cal_y_20_tick = np.asarray(all_y_20_data)[~mask_ticks]
x1_20 = bgndbi20.flatten().reshape((-1,1))
x2_20 = bgndvi20.flatten().reshape((-1,1))
x3_20 = blndvi20.flatten().reshape((-1,1))
cal_X_20 = np.concatenate((x1_20,x2_20), axis=1)
cal_X_20 = np.concatenate((cal_X_20,x3_20), axis=1)
cal_X_20_tick = cal_X_20[~mask_ticks]
year20 = np.ones_like(cal_y_20_tick)* 2020
#output 20
dataframe = pd.DataFrame({'LST':cal_y_20_tick.tolist(),'GNDBI':cal_X_20_tick[:,0].tolist(),'GNDVI':cal_X_20_tick[:,1].tolist(),'LNDVI':cal_X_20_tick[:,2].tolist(),'X':cal_coords_tick[:,0].tolist(),'Y':cal_coords_tick[:,1].tolist(),'Year':year20.tolist()})
dataframe.to_csv(out_sampledata+"samples_{:d}.csv".format(2020))

def extract_str_filedate(filename):
    datetime_string = filename.replace('.csv', '')
    return datetime_string

datapath ="D:/transactions/Postdoc/FanChao-wangzhe/0515"
outpath = "D:/transactions/Postdoc/FanChao-wangzhe/0515-run/"
lst_fldname = ['2000','2010','2020']


cidx_xfile = 0

for filename in os.listdir(datapath):
        c_filepath = extract_str_filedate(filename)
        cur_outpath = outpath + c_filepath + '/'
        if os.path.exists(cur_outpath) == False:
                   os.makedirs(cur_outpath)                   
        FilesPath = os.path.join(datapath,filename)
        print (FilesPath)
        cal_coords_list =[]
        cal_y_list =[]
        cal_X_list =[]
        delt_stwr_intervel =[0.0]
        
        gwr_predbeta0_results=[]
        gwr_predbeta1_results=[]
        gwr_predbeta2_results=[]
        gwr_predbeta3_results=[]
        gwr_predy_results = []
        
        stwr_predbeta0_results=[]
        stwr_predbeta1_results=[]
        stwr_predbeta2_results=[]
        stwr_predbeta3_results=[]
        stwr_predy_results = []
    
        csvFile = open(FilesPath, "r")
        df =pd.read_csv(csvFile,header = 0,names=['LST','GNDBI','GNDVI','LNDVI','X','Y','Year'],
                 dtype = {"LST" : "float64","GNDBI":"float64","GNDVI":"float64",
                          "LNDVI":"float64","X":"float64","Y":"float64","Year":"float64"
                          },
                 skip_blank_lines = True,
                 keep_default_na = False)
        
        df.info()
        all_data = df.values
        all_data.shape
        #output for the calibration
        df_GTWR = pd.DataFrame({
                                'X':all_data[:,4],
                                'Y':all_data[:,5],
                                'LST':all_data[:,0],
                                'X1_GNDBI':all_data[:,1],
                                'X2_GNDVI':all_data[:,2],
                                'X3_LNDVI':all_data[:,3],
                                'Year':all_data[:,6]
                                })
        df_GTWR.to_csv(cur_outpath + "for_gtwr_compare.csv") 
        df_GTWR = df_GTWR.sort_values(by=['Year'])  
        #update all_data
        all_data  = df_GTWR.values
        tick_time = all_data[0,-1]
        cal_coord_tick = []
        cal_X_tick =[]
        cal_y_tick =[]
        time_tol = 1.0e-7
        lensdata = len(all_data)
        for row in range(lensdata):
            cur_time = all_data[row,-1]
            if(abs(cur_time-tick_time)>time_tol):
                cal_coords_list.append(np.asarray(cal_coord_tick))
                cal_X_list.append(np.asarray(cal_X_tick))
                cal_y_list.append(np.asarray(cal_y_tick))
                delt_t = cur_time - tick_time
                delt_stwr_intervel.append(delt_t) 
                tick_time =cur_time
                cal_coord_tick = []
                cal_X_tick =[]
                cal_y_tick =[]
            coords_tick = np.array([all_data[row,0],all_data[row,1]])
            cal_coord_tick.append(coords_tick)
            x_tick = np.array([all_data[row,3],all_data[row,4],all_data[row,5]])
            cal_X_tick.append(x_tick)
            y_tick = np.array([all_data[row,2]])
            cal_y_tick.append(y_tick)  
        #gwr 
        cal_cord_gwr = np.asarray(cal_coord_tick)
        cal_X_gwr  = np.asarray(cal_X_tick)
        cal_y_gwr = np.asarray(cal_y_tick)  
        cal_coords_list.append(np.asarray(cal_coord_tick))
        cal_X_list.append(np.asarray(cal_X_tick))
        cal_y_list.append(np.asarray(cal_y_tick))
        #stwr
        stwr_selector_ = Sel_Spt_BW(cal_coords_list, cal_y_list, cal_X_list,delt_stwr_intervel ,spherical = False)
        optalpha,optsita,opt_btticks,opt_gwr_bw0 = stwr_selector_.search(nproc = 12) 

        stwr_model = STWR(cal_coords_list,cal_y_list,cal_X_list,delt_stwr_intervel,optsita,opt_gwr_bw0,tick_nums=opt_btticks,alpha =optalpha,spherical = False,recorded=1)
        #F-STWR
        #stwr
        stwr_results = stwr_model.fit()
        stwr_scale = stwr_results.scale 
        stwr_residuals = stwr_results.resid_response
        stwr_beta_se = stwr_results.bse 
        stwr_r2 = stwr_results.localR2
        
        #gwr
        gwr_selector = Sel_BW(cal_cord_gwr, cal_y_gwr, cal_X_gwr,spherical = False)
        gwr_bw= gwr_selector.search(bw_min=2)
        gwr_model = GWR(cal_cord_gwr, cal_y_gwr, cal_X_gwr, gwr_bw,spherical = False)
        gwr_results = gwr_model.fit()
        gwr_scale = gwr_results.scale 
        gwr_residuals = gwr_results.resid_response
        gwr_beta_bse = gwr_results.bse
        gwr_r2 = gwr_results.localR2
        
        da_summary=open(cur_outpath+"/summary.txt",'w+') 
        print(stwr_results.summary(),file=da_summary)
        print(gwr_results.summary(),file=da_summary)
        da_summary.close()
        
        
        df_stwr_r2 = pd.DataFrame({'R2':stwr_r2[:,0]
                        })
        df_stwr_r2.to_csv( cur_outpath+"stwr_R2.csv") 
        df_gwr_r2 = pd.DataFrame({
                                'R2':gwr_r2[:,0]
                                })
        df_gwr_r2.to_csv( cur_outpath+"gwr_R2.csv") 
        
        df_stwr_resids = pd.DataFrame({
                                        'X':cal_cord_gwr[:,0],
                                        'Y': cal_cord_gwr[:,1],
                                        'resid':stwr_residuals
                                        })
        df_stwr_resids.to_csv(cur_outpath+"/stwr_residuals.csv") 
           
        df_gwr_resids = pd.DataFrame({
                                        'X':cal_cord_gwr[:,0],
                                        'Y':cal_cord_gwr[:,1],
                                        'resid':gwr_residuals
                                           })
        df_gwr_resids.to_csv(cur_outpath+"/gwr_residuals.csv") 

        allklen_stwr = cal_X_list[-1].shape[1]+1
        stwr_cal_parmas = np.reshape(stwr_results.params.flatten(),(-1,allklen_stwr))
        gwr_cal_parmas  = np.reshape(gwr_results.params.flatten(),(-1,allklen_stwr))
        
        df_stwr_parmas = pd.DataFrame({
                                'X':cal_cord_gwr[:,0],
                                'Y': cal_cord_gwr[:,1],
                                'constant':stwr_cal_parmas[:,0],
                                'X1_GNDBI':stwr_cal_parmas[:,1],
                                'X2_GNDVI':stwr_cal_parmas[:,2],
                                'X3_LNDVI':stwr_cal_parmas[:,3]
                                })
        df_stwr_parmas.to_csv(cur_outpath + "stwr_parmas.csv") 
        df_gwr_parmas = pd.DataFrame({
                                'X':cal_cord_gwr[:,0],
                                'Y': cal_cord_gwr[:,1],
                                'constant':gwr_cal_parmas[:,0],
                                'X1_GNDBI':gwr_cal_parmas[:,1],
                                'X2_GNDVI':gwr_cal_parmas[:,2],
                                'X3_LNDVI':gwr_cal_parmas[:,3]
                                })
        df_gwr_parmas.to_csv( cur_outpath+ "gwr_parmas.csv") 
        # stwr_cal_tvalues = stwr_cal_parmas/stwr_beta_se
        # gwr_cal_tvalues =  gwr_cal_parmas/gwr_beta_bse
        
        stwr_cal_tvalues = stwr_results.tvalues
        gwr_cal_tvalues = gwr_results.tvalues
        
        stwr_cal_pvalues = stats.t.sf(np.abs(stwr_cal_tvalues),stwr_results.df_model )*2 
        gwr_cal_pvalues = stats.t.sf(np.abs(gwr_cal_tvalues),gwr_results.df_model )*2 
        
        ############################################################################## 
        df_stwr_tvalue = pd.DataFrame({
                                'X':cal_cord_gwr[:,0],
                                'Y': cal_cord_gwr[:,1],
                                'constant':stwr_cal_tvalues[:,0],
                                'X1_GNDBI':stwr_cal_tvalues[:,1],
                                'X2_GNDVI':stwr_cal_tvalues[:,2],
                                'X3_LNDVI':stwr_cal_tvalues[:,3]
                                })
        df_stwr_tvalue.to_csv(cur_outpath+"stwr_t-Values.csv") 
        df_gwr_tvalue = pd.DataFrame({
                                'X':cal_cord_gwr[:,0],
                                'Y': cal_cord_gwr[:,1],
                                'constant':gwr_cal_tvalues[:,0],
                                'X1_GNDBI':gwr_cal_tvalues[:,1],
                                'X2_GNDVI':gwr_cal_tvalues[:,2],
                                'X3_LNDVI':gwr_cal_tvalues[:,3]
                                })
        df_gwr_tvalue.to_csv(cur_outpath+"gwr_t-Values.csv") 
        
        
        df_stwr_pvalue = pd.DataFrame({
                                'X':cal_cord_gwr[:,0],
                                'Y': cal_cord_gwr[:,1],
                                'constant':stwr_cal_pvalues[:,0],
                                'X1_GNDBI':stwr_cal_pvalues[:,1],
                                'X2_GNDVI':stwr_cal_pvalues[:,2],
                                'X3_LNDVI':stwr_cal_pvalues[:,3]
                                })
        df_stwr_pvalue.to_csv(cur_outpath+"stwr_p-Values.csv") 
        df_gwr_pvalue = pd.DataFrame({
                                'X':cal_cord_gwr[:,0],
                                'Y': cal_cord_gwr[:,1],
                                'constant':gwr_cal_pvalues[:,0],
                                'X1_GNDBI':gwr_cal_pvalues[:,1],
                                'X2_GNDVI':gwr_cal_pvalues[:,2],
                                'X3_LNDVI':gwr_cal_pvalues[:,3]
                                })
        df_gwr_pvalue.to_csv(cur_outpath+"gwr_p-Values.csv") 
        
        
        ##############################################################################
        stwr_cal_mean = np.mean(stwr_cal_parmas, axis=0)
        stwr_cal_mean
        
        gwr_cal_mean = np.mean(gwr_cal_parmas, axis=0)
        gwr_cal_mean
        
        strw_cal_std = np.std(stwr_cal_parmas, axis=0)
        strw_cal_std
        
        grw_cal_std = np.std(gwr_cal_parmas, axis=0)
        grw_cal_std
        
        stwr_cal_parmas_zscore = np.zeros_like(stwr_cal_parmas)
        tick_num = 0
        for stwr_tick in stwr_cal_parmas:
            stwr_cal_parmas_zscore[tick_num] = (stwr_tick - stwr_cal_mean)/strw_cal_std
            tick_num += 1
        stwr_cal_parmas_zscore
        
        gwr_cal_parmas_zscore = np.zeros_like(gwr_cal_parmas)
        tick_num = 0
        for gwr_tick in gwr_cal_parmas:
            gwr_cal_parmas_zscore[tick_num] = (gwr_tick - gwr_cal_mean)/grw_cal_std
            tick_num += 1
        gwr_cal_parmas_zscore
        df_stwr_pz = pd.DataFrame({
                                'X':cal_cord_gwr[:,0],
                                'Y': cal_cord_gwr[:,1],
                                'constant':stwr_cal_parmas_zscore[:,0],
                                'X1_GNDBI':stwr_cal_parmas_zscore[:,1],
                                'X2_GNDVI':stwr_cal_parmas_zscore[:,2],
                                'X3_LNDVI':stwr_cal_parmas_zscore[:,3]
                                })
        df_stwr_pz.to_csv(cur_outpath+"stwr_z-Scores.csv") 
        df_gwr_pz = pd.DataFrame({
                                'X':cal_cord_gwr[:,0],
                                'Y': cal_cord_gwr[:,1],
                                'constant':gwr_cal_parmas_zscore[:,0],
                                'X1_GNDBI':gwr_cal_parmas_zscore[:,1],
                                'X2_GNDVI':gwr_cal_parmas_zscore[:,2],
                                'X3_LNDVI':gwr_cal_parmas_zscore[:,3]
                                })
        df_gwr_pz.to_csv(cur_outpath+"gwr_z-Scores.csv") 
        
        ###################################predicted betas and LST ###########################################
        str_Xfilepath = 'D:/transactions/Postdoc/FanChao-wangzhe/0515X/'
        str_Xfilepath = str_Xfilepath  + lst_fldname[cidx_xfile] + '/'
        #read surfaces of different X
        rgndbi = rasterio.open(str_Xfilepath +'GNDBI_clip.tif')
        bgndbi = rgndbi.read(1)
        bgpnbi = bgndbi[1:-1,1:]
        
        rgndvi = rasterio.open(str_Xfilepath + 'GNDVI_clip.tif')
        bgndvi = rgndvi.read(1)
        bgndvi = bgndvi[1:-1,1:]
        
        rlndvi = rasterio.open(str_Xfilepath + 'LNDVI_clip.tif')
        blndvi = rlndvi.read(1)  
        blndvi = blndvi[1:-1,1:]
        
        
        #pf = lst2013.profile  #CRS
        pf = rgndbi.profile
        transform =rgndbi.profile['transform']
        nodata = pf['nodata']
        print (pf)
        
        data_crs = pf['crs']
        all_coords_list = []
        Pre_X_list = []
        Pre_y_list = []
        mask_rgndbi = rgndbi.dataset_mask()
        mask_rgndbi = mask_rgndbi[1:-1,1:]
        for row in range(mask_rgndbi.shape[0]):
            for col in range (mask_rgndbi.shape[1]):
                if(mask_rgndbi[row,col]>0):
                    all_coords_list.append(rgndbi.xy(row+1,col+1)) 
        all_coords_arr = np.asarray(all_coords_list)   
        lstpred = np.copy(bgndbi)
        lstpred_y_tick = lstpred.flatten()
        Pre_y_list.append(lstpred_y_tick.reshape((-1,1)))
                
        bgndbi_x1 = bgndbi.flatten()[mask_rgndbi].reshape((-1,1))
        bgndvi_x2 = bgndvi.flatten()[mask_rgndbi].reshape((-1,1))
        blndvi_x3 = blndvi.flatten()[mask_rgndbi].reshape((-1,1))
        pre_X_tick = np.concatenate((bgndbi_x1,bgndvi_x2,blndvi_x3),axis=1)
        Pre_X_list.append(pre_X_tick)
        
        #compare different methods  compare predicted Beta surfaces of latest time stages
        alllen_stwr = len(all_coords_arr)
        allklen_stwr = cal_X_list[-1].shape[1]+1
        rec_parmas_stwr = np.ones((alllen_stwr,allklen_stwr))
        calen_stwr =  len(cal_y_list[-1])
        prelen_stwr = len(Pre_X_list[-1])
        
        
        stwr_cal_parmas = np.reshape(stwr_results.params.flatten(),(-1,allklen_stwr))
        stwr_pre_parmas = np.ones((prelen_stwr,allklen_stwr))

        gwr_pre_parmas = np.ones((prelen_stwr,allklen_stwr))
        gwr_cal_parmas = np.reshape(gwr_results.params.flatten(),(-1,allklen_stwr))

        if (calen_stwr>=prelen_stwr):
               #predPointList = [pre_coords_list[-1]]
               predPointList = [all_coords_arr]
               PreX_list = [Pre_X_list[-1]]
               pred_stwr_dir_result = stwr_model.predict(predPointList,PreX_list,stwr_scale,stwr_residuals)
               pre_y_stwr = pred_stwr_dir_result.predictions
               pre_parmas_stwr=np.reshape(pred_stwr_dir_result.params.flatten(),(-1,allklen_stwr))
               #gwr
               pred_gwr_dir_result = gwr_model.predict(pre_coords_list[-1],PreX_list,gwr_scale,gwr_residuals)
               pre_y_gwr = pred_gwr_dir_result.predictions
               pre_parmas_gwr=np.reshape(pred_stwr_dir_result.params.flatten(),(-1,allklen_stwr))
               #gwr
        else:
                spl_parts_stwr = math.ceil(prelen_stwr*1.0/calen_stwr)
                spl_X_stwr = np.array_split(Pre_X_list[-1], spl_parts_stwr, axis = 0)
                spl_coords_stwr = np.array_split(all_coords_arr, spl_parts_stwr, axis = 0)
                
                pred_stwr_result = np.array_split(Pre_y_list[-1], spl_parts_stwr, axis = 0)
                
                pred_stwrparmas_result = np.array_split(stwr_pre_parmas, spl_parts_stwr, axis = 0)
                #gwr
                pred_gwr_result = np.array_split(Pre_y_list[-1], spl_parts_stwr, axis = 0)
                
                pred_gwrparmas_result = np.array_split(gwr_pre_parmas, spl_parts_stwr, axis = 0)
                
                #gwr
                for j in range(spl_parts_stwr):
                        predPointList_tick = [spl_coords_stwr[j]]
                        PreX_list_tick = [spl_X_stwr[j]]
                        pred_stwr_spl_result =  stwr_model.predict(predPointList_tick,PreX_list_tick,stwr_scale,stwr_residuals)
                        pred_stwr_result[j] =pred_stwr_spl_result.predictions
                        pred_stwrparmas_result[j] =np.reshape(pred_stwr_spl_result.params.flatten(),(-1,allklen_stwr))
                        #gwr
                        pred_gwr_spl_result =  gwr_model.predict(spl_coords_stwr[j],spl_X_stwr[j],gwr_scale,gwr_residuals)
                        pred_gwr_result[j] =pred_gwr_spl_result.predictions
                        pred_gwrparmas_result[j] =np.reshape(pred_gwr_spl_result.params.flatten(),(-1,allklen_stwr))
                        #gwr
                pre_y_stwr = pred_stwr_result[0]
                pre_parmas_stwr = pred_stwrparmas_result[0]
                #gwr
                pre_y_gwr = pred_gwr_result[0]
                pre_parmas_gwr = pred_gwrparmas_result[0]
                #gwr
                combnum = spl_parts_stwr-1
                for s in range(combnum):
                    pre_y_stwr = np.vstack((pre_y_stwr,pred_stwr_result[s+1]))
                    pre_parmas_stwr = np.vstack((pre_parmas_stwr,pred_stwrparmas_result[s+1]))
                    #gwr
                    pre_y_gwr = np.vstack((pre_y_gwr,pred_gwr_result[s+1]))
                    pre_parmas_gwr = np.vstack((pre_parmas_gwr,pred_gwrparmas_result[s+1]))
                    #gwr

        draw_vals_stwr = np.ones([353,353])
        draw_stwr_parms = np.ones([4,353,353])
        #gwr
        draw_vals_gwr = np.ones([353,353])
        draw_gwr_parms = np.ones([4,353,353])
        #gwr

        for j in range(prelen_stwr):
                row_c,col_c= rgndvi.index(all_coords_arr[j,0],all_coords_arr[j,1])
                row_c = row_c-1
                col_c = col_c -1
                draw_vals_stwr[row_c,col_c]=pre_y_stwr[j]
                draw_stwr_parms[0,row_c,col_c]=pre_parmas_stwr[j,0]
                draw_stwr_parms[1,row_c,col_c]=pre_parmas_stwr[j,1]
                draw_stwr_parms[2,row_c,col_c]=pre_parmas_stwr[j,2]
                draw_stwr_parms[3,row_c,col_c]=pre_parmas_stwr[j,3]
                #gwr
                draw_vals_gwr[row_c,col_c]=pre_y_gwr[j]
                draw_gwr_parms[0,row_c,col_c]=pre_parmas_gwr[j,0]
                draw_gwr_parms[1,row_c,col_c]=pre_parmas_gwr[j,1]
                draw_gwr_parms[2,row_c,col_c]=pre_parmas_gwr[j,2]
                draw_gwr_parms[3,row_c,col_c]=pre_parmas_gwr[j,3]
                #gwr
        for j in range(calen_stwr):
                row_c,col_c= rgndvi.index(all_coords_arr[j,0],all_coords_arr[j,1])
                row_c = row_c-1
                col_c = col_c -1
                draw_vals_stwr[row_c,col_c]=cal_y_list[-1][j]
                draw_stwr_parms[0,row_c,col_c]=stwr_cal_parmas[j,0]
                draw_stwr_parms[1,row_c,col_c]=stwr_cal_parmas[j,1]
                draw_stwr_parms[2,row_c,col_c]=stwr_cal_parmas[j,2]
                draw_stwr_parms[3,row_c,col_c]=stwr_cal_parmas[j,3]
                #gwr
                draw_vals_gwr[row_c,col_c]=cal_y_list[-1][j]
                draw_gwr_parms[0,row_c,col_c]=gwr_cal_parmas[j,0]
                draw_gwr_parms[1,row_c,col_c]=gwr_cal_parmas[j,1]
                draw_gwr_parms[2,row_c,col_c]=gwr_cal_parmas[j,2]
                draw_gwr_parms[3,row_c,col_c]=gwr_cal_parmas[j,3]
                #gwr

        gwr_predbeta0_results.append(draw_gwr_parms[0])
        gwr_predbeta1_results.append(draw_gwr_parms[1])
        gwr_predbeta2_results.append(draw_gwr_parms[2])
        gwr_predbeta3_results.append(draw_gwr_parms[3])
        gwr_predy_results.append(draw_vals_gwr)

        stwr_predbeta0_results.append(draw_stwr_parms[0])
        stwr_predbeta1_results.append(draw_stwr_parms[1])
        stwr_predbeta2_results.append(draw_stwr_parms[2])
        stwr_predbeta3_results.append(draw_stwr_parms[3])
        stwr_predy_results.append(draw_vals_stwr)
        
        
        #STWR results output
        pre_yb_outpath = outpath + lst_fldname[cidx_xfile]+ '/'
        if os.path.exists(pre_yb_outpath) == False:
                   os.makedirs(pre_yb_outpath)  
                   
        stwr_predbeta0_arr = np.asarray(stwr_predbeta0_results).reshape((353,353))
        stwr_predbeta1_arr = np.asarray(stwr_predbeta1_results).reshape((353,353))
        stwr_predbeta2_arr = np.asarray(stwr_predbeta2_results).reshape((353,353))
        stwr_predbeta3_arr = np.asarray(stwr_predbeta3_results).reshape((353,353))
        stwr_predy_arr = np.asarray(stwr_predy_results).reshape((353,353))
        
        gwr_predbeta0_arr = np.asarray(gwr_predbeta0_results).reshape((353,353))
        gwr_predbeta1_arr = np.asarray(gwr_predbeta1_results).reshape((353,353))
        gwr_predbeta2_arr = np.asarray(gwr_predbeta2_results).reshape((353,353))
        gwr_predbeta3_arr = np.asarray(gwr_predbeta3_results).reshape((353,353))
        gwr_predy_arr = np.asarray(gwr_predy_results).reshape((353,353))
        
        
        with open(pre_yb_outpath + 'stwr_predbeta0.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(stwr_predbeta0_arr.shape))
            for data_slice in stwr_predbeta0_arr:
                np.savetxt(outfile, data_slice, fmt='%-7.7f')
        
        with open(pre_yb_outpath + 'stwr_predbeta1.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(stwr_predbeta1_arr.shape))
            for data_slice in stwr_predbeta1_arr:
                np.savetxt(outfile, data_slice, fmt='%-7.7f')
        
        
        with open(pre_yb_outpath + 'stwr_predbeta2.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(stwr_predbeta2_arr.shape))
            for data_slice in stwr_predbeta2_arr:
                np.savetxt(outfile, data_slice, fmt='%-7.7f')

        with open(pre_yb_outpath + 'stwr_predbeta3.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(stwr_predbeta3_arr.shape))
            for data_slice in stwr_predbeta3_arr:
                np.savetxt(outfile, data_slice, fmt='%-7.7f')        

        with open(pre_yb_outpath + 'stwr_predy.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(stwr_predy_arr.shape))
            for data_slice in stwr_predy_arr:
                np.savetxt(outfile, data_slice, fmt='%-7.7f')       
        
        with open(pre_yb_outpath + 'gwr_predbeta0.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(gwr_predbeta0_arr.shape))
            for data_slice in gwr_predbeta0_arr:
                np.savetxt(outfile, data_slice, fmt='%-7.7f')
        
        with open(pre_yb_outpath + 'gwr_predbeta1.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(gwr_predbeta1_arr.shape))
            for data_slice in gwr_predbeta1_arr:
                np.savetxt(outfile, data_slice, fmt='%-7.7f')
        
        
        with open(pre_yb_outpath + 'gwr_predbeta2.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(gwr_predbeta2_arr.shape))
            for data_slice in gwr_predbeta2_arr:
                np.savetxt(outfile, data_slice, fmt='%-7.7f')

        with open(pre_yb_outpath + 'gwr_predbeta3.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(gwr_predbeta3_arr.shape))
            for data_slice in gwr_predbeta3_arr:
                np.savetxt(outfile, data_slice, fmt='%-7.7f')        

        with open(pre_yb_outpath + 'gwr_predy.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(gwr_predy_arr.shape))
            for data_slice in gwr_predy_arr:
                np.savetxt(outfile, data_slice, fmt='%-7.7f')       
        
        ################################output tifs ###################################################
        with rasterio.open(pre_yb_outpath + 'stwr_beta0.tif', 'w', driver='GTiff', 
                   height=stwr_predbeta0_arr.shape[0],
                   width=stwr_predbeta0_arr.shape[1], count=1, dtype= stwr_predbeta0_arr.dtype,
                   crs=data_crs, transform=transform,nodata = nodata) as dststwr:
            dststwr.write(stwr_predbeta0_arr, 1)
        
        with rasterio.open(pre_yb_outpath + 'stwr_beta1.tif', 'w', driver='GTiff', 
                   height=stwr_predbeta1_arr.shape[0],
                   width=stwr_predbeta1_arr.shape[1], count=1, dtype= stwr_predbeta1_arr.dtype,
                   crs=data_crs, transform=transform,nodata = nodata) as dststwr:
            dststwr.write(stwr_predbeta1_arr, 1)
        
        with rasterio.open(pre_yb_outpath + 'stwr_beta2.tif', 'w', driver='GTiff', 
                   height=stwr_predbeta2_arr.shape[0],
                   width=stwr_predbeta2_arr.shape[1], count=1, dtype= stwr_predbeta2_arr.dtype,
                   crs=data_crs, transform=transform,nodata = nodata) as dststwr:
            dststwr.write(stwr_predbeta2_arr, 1)
        
        with rasterio.open(pre_yb_outpath + 'stwr_beta3.tif', 'w', driver='GTiff', 
                   height=stwr_predbeta3_arr.shape[0],
                   width=stwr_predbeta3_arr.shape[1], count=1, dtype= stwr_predbeta3_arr.dtype,
                   crs=data_crs, transform=transform,nodata = nodata) as dststwr:
            dststwr.write(stwr_predbeta3_arr, 1)
        
        with rasterio.open(pre_yb_outpath + 'stwr_predy.tif', 'w', driver='GTiff', 
                   height=stwr_predy_arr.shape[0],
                   width=stwr_predy_arr.shape[1], count=1, dtype= stwr_predy_arr.dtype,
                   crs=data_crs, transform=transform,nodata = nodata) as dststwr:
            dststwr.write(stwr_predy_arr, 1)
        
        #GWR
        with rasterio.open(pre_yb_outpath + 'gwr_beta0.tif', 'w', driver='GTiff', 
                   height=gwr_predbeta0_arr.shape[0],
                   width=gwr_predbeta0_arr.shape[1], count=1, dtype= gwr_predbeta0_arr.dtype,
                   crs=data_crs, transform=transform,nodata = nodata) as dststwr:
            dststwr.write(gwr_predbeta0_arr, 1)
        
        with rasterio.open(pre_yb_outpath + 'gwr_beta1.tif', 'w', driver='GTiff', 
                   height=gwr_predbeta1_arr.shape[0],
                   width=gwr_predbeta1_arr.shape[1], count=1, dtype= gwr_predbeta1_arr.dtype,
                   crs=data_crs, transform=transform,nodata = nodata) as dststwr:
            dststwr.write(gwr_predbeta1_arr, 1)
        
        with rasterio.open(pre_yb_outpath + 'gwr_beta2.tif', 'w', driver='GTiff', 
                   height=gwr_predbeta2_arr.shape[0],
                   width=gwr_predbeta2_arr.shape[1], count=1, dtype= gwr_predbeta2_arr.dtype,
                   crs=data_crs, transform=transform,nodata = nodata) as dststwr:
            dststwr.write(gwr_predbeta2_arr, 1)
        
        with rasterio.open(pre_yb_outpath + 'gwr_beta3.tif', 'w', driver='GTiff', 
                   height=gwr_predbeta3_arr.shape[0],
                   width=gwr_predbeta3_arr.shape[1], count=1, dtype= gwr_predbeta3_arr.dtype,
                   crs=data_crs, transform=transform,nodata = nodata) as dststwr:
            dststwr.write(gwr_predbeta3_arr, 1)
        
        with rasterio.open(pre_yb_outpath + 'gwr_predy.tif', 'w', driver='GTiff', 
                   height=gwr_predy_arr.shape[0],
                   width=gwr_predy_arr.shape[1], count=1, dtype= gwr_predy_arr.dtype,
                   crs=data_crs, transform=transform,nodata = nodata) as dststwr:
            dststwr.write(gwr_predy_arr, 1) 
        ################################output tifs ###################################################
        cidx_xfile = cidx_xfile + 1

