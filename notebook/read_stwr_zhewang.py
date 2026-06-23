import  numpy  as  np
import libpysal as ps
from stwr.gwr import GWR, MGWR,STWR
from stwr.sel_bw import *
from stwr.utils import shift_colormap, truncate_colormap
#import matplotlib.pyplot as plt
#import matplotlib as mpl
import pandas as pd
import math
#from matplotlib.gridspec import GridSpec
import time
import csv 
import copy 

#import rasterio
#import rasterio.plot
#import rasterio.features
#import rasterio.warp
#import pyproj

#from sklearn import preprocessing
# import sklearn.preprocessing as preprocessing
import os
from scipy import stats

# mu, sigma = 0, 0.0000001 # mean and standard deviation
#ssss = np.random.normal(mu, sigma,100).reshape((-1,1))


#读入数据需要有这些
cal_coords_list =[]
cal_y_list =[]
cal_X_list =[]
cal_geoid_list = []
delt_stwr_intervel =[0.0]

#csvFile = open("../runningdata/zuojiani/E0823.csv", "r")#("../../Sanaz/FFCTWithoutNan.csv", "r") #../.
csvFile = open("../Data_STWR/Lyme_disease/CT_STWR_16.csv", "r")
df = pd.read_csv(csvFile,header = 0,
                 skip_blank_lines = True,
                 keep_default_na = False)
df.info()
#df = df[~df["FFCT"].isnull()]
#df = df[~df["CFCT"].isnull()]
#df = df[~df["UFCT"].isnull()]
								
#OBJECTID	longitude	latitude	y_cuc	x1_dem	x2_podu	x3_poxiang	x4_shui	x5_wendu	x6_ndvi	x7_yejian	timestamp								
df_stwr = df.loc[:, ["LID","LymeDisease","year","CONTIG_MN_cls_41_ln","GYRATE_RA_cls_95_ln","CONTIG_MD_cls_90_ln","GYRATE_MN_cls_41_squared","CIRCLE_SD_cls_42_squared","X","Y"]]
df_stwr = df_stwr.sort_values(by=['year'])
all_data = df_stwr.values


tick_time = all_data[0,2]
cal_coord_tick = []
cal_X_tick =[]
cal_y_tick =[]
cal_geoid_tick = []


time_list = [tick_time]
time_tol = 1.0e-7
lensdata = len(all_data)
for row in range(lensdata):
    cur_time = all_data[row,2]
    if(abs(cur_time-tick_time)>time_tol):  
        cal_coords_list.append(np.asarray(cal_coord_tick))
       # normanlized = np.asarray(cal_X_tick)
        #normanlized_X = preprocessing.normalize(normanlized,axis = 0)
        #cal_X_list.append(normanlized_X)#np.asarray(cal_X_tick) 
        cal_X_list.append(np.asarray(cal_X_tick))
        cal_y_list.append(np.asarray(cal_y_tick))
        cal_geoid_list.append(np.asarray(cal_geoid_tick))
        delt_t = cur_time - tick_time
        delt_stwr_intervel.append(delt_t) 
        tick_time =cur_time
        time_list.append(tick_time)
        cal_coord_tick = []
        cal_X_tick =[]
        cal_y_tick =[]
        cal_geoid_tick = []
 
    coords_tick = np.array([all_data[row,8],all_data[row,9]])
    cal_coord_tick.append(coords_tick)
    x_tick = np.array([all_data[row,3],all_data[row,4],all_data[row,5],all_data[row,6],all_data[row,7]])   
    #normalized_x_tick = preprocessing.normalize(x_tick,axis=1)
    cal_X_tick.append(x_tick) #normalized_x_tick)
    y_tick = np.array([all_data[row,1]])
    cal_y_tick.append(y_tick)
    geoid_tick = np.array([all_data[row,0]])
    cal_geoid_tick.append(geoid_tick)
    
    
#最后在放一次
#gwr解出最后一期 
cal_cord_gwr = np.asarray(cal_coord_tick)
cal_X_gwr  = np.asarray(cal_X_tick)
cal_y_gwr = np.asarray(cal_y_tick)  
cal_coords_list.append(np.asarray(cal_coord_tick))
cal_X_list.append(np.asarray(cal_X_tick))
cal_y_list.append(np.asarray(cal_y_tick))
cal_geoid_list.append(np.asarray(cal_geoid_tick))


#np.random.seed(1000)
#sample_num = 9000

#sampled_y_list = []
#sampled_X_list = []
#sampled_coords_list = []
#sampled_geoid_list = []

#for linumber in range (len(cal_y_list)):
    #samples_ticks =  np.random.choice(range(cal_y_list[linumber].shape[0]),sample_num,replace=False)
#    mask_ticks = np.ones_like(cal_y_list[linumber],dtype=bool).flatten() 
#    mask_ticks[samples_ticks] = False
#    sampled_cal_y = cal_y_list[linumber][~mask_ticks]
#    sampled_y_list.append(sampled_cal_y)
#    sampled_cal_x = cal_X_list[linumber][~mask_ticks]
#    sampled_X_list.append(sampled_cal_x)
#    sample_coords = cal_coords_list[linumber][~mask_ticks]
#    sampled_coords_list.append(sample_coords)
#    sample_geoid = cal_geoid_list[linumber][~mask_ticks]
#    sampled_geoid_list.append(sample_geoid)
    
    
 
cur_coords_list = []
cur_y_list =[]
cur_X_list = []
cur_interval_list = []
out_put_path = "../Data_STWR/Lyme_disease/results/"
                 
lenlist = len(delt_stwr_intervel)
for il in range(lenlist):
        #cur_time_str = 
        cur_outs = out_put_path + str(int(time_list[il]))+ '/'
        if os.path.exists(cur_outs) == False:
                 os.makedirs(cur_outs)      
        cur_coords_list = cal_coords_list[:il+1]
        cur_y_list = cal_y_list[:il+1]
        cur_X_list = cal_X_list[:il+1]
        cur_interval_list = delt_stwr_intervel[:il+1]
        cur_geoid_list = cal_geoid_list[:il+1]
        #stwr 
        stwr_selector_ = Sel_Spt_BW(cur_coords_list, cur_y_list, cur_X_list,#gwr_bw0,
                                    cur_interval_list,spherical = False)

        optalpha,optsita,opt_btticks,opt_gwr_bw0 = stwr_selector_.search() 
        stwr_model = STWR(cur_coords_list,cur_y_list,cur_X_list,cur_interval_list,
                          optsita,opt_gwr_bw0,tick_nums=opt_btticks+1,alpha =optalpha,spherical = False,recorded = 1)      
        stwr_results = stwr_model.fit()
        print(stwr_results.summary())
        fsum_stwr = cur_outs +'stwr_summary.txt'
        with open(fsum_stwr, 'w') as f:
                 f.write(stwr_results.summary())
#        stwr 
        stwr_scale = stwr_results.scale 
        stwr_residuals = stwr_results.resid_response
        stwr_beta_se = stwr_results.bse 
        stwr_r2 = stwr_results.localR2
        
        #gwr
        #normalized the X array
       # normalized_arr = preprocessing.normalize(cur_X_list[-1] ) 
        gwr_selector = Sel_BW(cur_coords_list[-1], cur_y_list[-1],cur_X_list[-1],spherical = False)
        gwr_bw= gwr_selector.search(bw_min=2)
        gwr_model = GWR(cur_coords_list[-1], cur_y_list[-1],cur_X_list[-1], gwr_bw,spherical = False)
        gwr_results = gwr_model.fit()
        #print(gwr_results.summary())
        fsum_gwr = cur_outs + 'gwr_summary.txt'
        with open(fsum_gwr, 'w') as f:
                 f.write(gwr_results.summary())
        
        gwr_scale = gwr_results.scale 
        gwr_residuals = gwr_results.resid_response
        gwr_beta_bse = gwr_results.bse
        gwr_r2 = gwr_results.localR2
        
        
        allklen_stwr = cal_X_list[-1].shape[1]+1
        stwr_cal_parmas = np.reshape(stwr_results.params.flatten(),(-1,allklen_stwr))
        gwr_cal_parmas  = np.reshape(gwr_results.params.flatten(),(-1,allklen_stwr))
        
        fresid_stwr = cur_outs + 'stwr_resid.csv'
        df_stwr_resids = pd.DataFrame({
                                'OBJECTID': cur_geoid_list[-1][:,0],
                                'resid':stwr_residuals
                                        
                                })
        df_stwr_resids.to_csv(fresid_stwr) 
        
        fresid_gwr = cur_outs + 'gwr_resid.csv'
        df_gwr_resids = pd.DataFrame({
                                  'OBJECTID': cur_geoid_list[-1][:,0],
                                  'resid':gwr_residuals
                                })
        df_gwr_resids.to_csv(fresid_gwr) 
        
        fr2_stwr = cur_outs + 'stwr_R2.csv'
        df_stwr_r2 = pd.DataFrame({
                                'OBJECTID': cur_geoid_list[-1][:,0],
                                'R2':stwr_r2[:,0]
                                })
        df_stwr_r2.to_csv(fr2_stwr) 
        
        fr2_gwr = cur_outs + 'gwr_R2.csv'
        df_gwr_r2 = pd.DataFrame({
                                'OBJECTID': cur_geoid_list[-1][:,0],
                                'R2':gwr_r2[:,0]
                                })
        
        df_gwr_r2.to_csv(fr2_gwr) 
         
        fparmas_stwr = cur_outs + 'stwr_parmas.csv'
        # 	U_m_s_	V_m_s_	TEMP_K_	RH___	PSFC_Pa_
        df_stwr_parmas = pd.DataFrame({
                                'OBJECTID': cur_geoid_list[-1][:,0],
                                'constant':stwr_cal_parmas[:,0],
                                "CONTIG_MN_cls_41_ln":stwr_cal_parmas[:,1],
                                "GYRATE_RA_cls_95_ln":stwr_cal_parmas[:,2],
                                "CONTIG_MD_cls_90_ln":stwr_cal_parmas[:,3],
                                "GYRATE_MN_cls_41_squared":stwr_cal_parmas[:,4],
                                "CIRCLE_SD_cls_42_squared":stwr_cal_parmas[:,5]
                                })
        df_stwr_parmas.to_csv(fparmas_stwr) 
        
        fparmas_gwr = cur_outs + 'gwr_parmas.csv'
        df_gwr_parmas = pd.DataFrame({
                                'OBJECTID': cur_geoid_list[-1][:,0],
                                'constant':gwr_cal_parmas[:,0],
                                "CONTIG_MN_cls_41_ln":gwr_cal_parmas[:,1],
                                "GYRATE_RA_cls_95_ln":gwr_cal_parmas[:,2],
                                "CONTIG_MD_cls_90_ln":gwr_cal_parmas[:,3],
                                "GYRATE_MN_cls_41_squared":gwr_cal_parmas[:,4],
                                "CIRCLE_SD_cls_42_squared":gwr_cal_parmas[:,5]
                                         
                                })
        df_gwr_parmas.to_csv(fparmas_gwr) 
        
        
        ftvalue_stwr = cur_outs + 'stwr_t-Values.csv'
#        
        stwr_cal_tvalues = stwr_cal_parmas/stwr_beta_se
        gwr_cal_tvalues =  gwr_cal_parmas/gwr_beta_bse
#        
       
        stwr_cal_pvalues = stats.t.sf(np.abs(stwr_cal_tvalues),stwr_results.df_model )*2 
        gwr_cal_pvalues = stats.t.sf(np.abs(gwr_cal_tvalues),gwr_results.df_model )*2 
#        ##############################################################################
#        
        df_stwr_tvalue = pd.DataFrame({
                                'OBJECTID': cur_geoid_list[-1][:,0],
                                'constant':stwr_cal_tvalues[:,0],                           
                                "CONTIG_MN_cls_41_ln":stwr_cal_tvalues[:,1],
                                "GYRATE_RA_cls_95_ln":stwr_cal_tvalues[:,2],
                                "CONTIG_MD_cls_90_ln":stwr_cal_tvalues[:,3],
                                "GYRATE_MN_cls_41_squared":stwr_cal_tvalues[:,4],
                                "CIRCLE_SD_cls_42_squared":stwr_cal_tvalues[:,5] 
                                })
        df_stwr_tvalue.to_csv(ftvalue_stwr) 
        
        ftvalue_gwr = cur_outs + 'gwr_t-Values.csv'
        
        df_gwr_tvalue = pd.DataFrame({
                                'OBJECTID': cur_geoid_list[-1][:,0],
                                'constant':gwr_cal_tvalues[:,0],                            
                                "CONTIG_MN_cls_41_ln":gwr_cal_tvalues[:,1],
                                "GYRATE_RA_cls_95_ln":gwr_cal_tvalues[:,2],
                                "CONTIG_MD_cls_90_ln":gwr_cal_tvalues[:,3],
                                "GYRATE_MN_cls_41_squared":gwr_cal_tvalues[:,4],
                                "CIRCLE_SD_cls_42_squared":gwr_cal_tvalues[:,5]  
                                })
        df_gwr_tvalue.to_csv(ftvalue_gwr) 
        
        
        fp_stwr = cur_outs + 'stwr_p-Values.csv'
       
        
        df_stwr_pvalue = pd.DataFrame({
                                'OBJECTID': cur_geoid_list[-1][:,0],
                                'constant':stwr_cal_pvalues[:,0],
                                "CONTIG_MN_cls_41_ln":stwr_cal_pvalues[:,1],
                                "GYRATE_RA_cls_95_ln":stwr_cal_pvalues[:,2],
                                "CONTIG_MD_cls_90_ln":stwr_cal_pvalues[:,3],
                                "GYRATE_MN_cls_41_squared":stwr_cal_pvalues[:,4],
                                "CIRCLE_SD_cls_42_squared":stwr_cal_pvalues[:,5]  
                                         
                                })
        df_stwr_pvalue.to_csv(fp_stwr) 
        
        fp_gwr = cur_outs + 'gwr_p-Values.csv'
        df_gwr_pvalue = pd.DataFrame({
                                'OBJECTID': cur_geoid_list[-1][:,0],
                                'constant':gwr_cal_pvalues[:,0], 
                                "CONTIG_MN_cls_41_ln":gwr_cal_pvalues[:,1],
                                "GYRATE_RA_cls_95_ln":gwr_cal_pvalues[:,2],
                                "CONTIG_MD_cls_90_ln":gwr_cal_pvalues[:,3],
                                "GYRATE_MN_cls_41_squared":gwr_cal_pvalues[:,4],
                                "CIRCLE_SD_cls_42_squared":gwr_cal_pvalues[:,5]  
                                })
        df_gwr_pvalue.to_csv(fp_gwr) 
        
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
#        stwr_cal_parmas_zscore
#        
        gwr_cal_parmas_zscore = np.zeros_like(gwr_cal_parmas)
        tick_num = 0
        for gwr_tick in gwr_cal_parmas:
            gwr_cal_parmas_zscore[tick_num] = (gwr_tick - gwr_cal_mean)/grw_cal_std
            tick_num += 1
#        gwr_cal_parmas_zscore
        fz_stwr = cur_outs + 'stwr_z-Scores.csv'
        
        df_stwr_pz = pd.DataFrame({
                                'OBJECTID': cur_geoid_list[-1][:,0],
                                'constant':stwr_cal_parmas_zscore[:,0], 
                                "CONTIG_MN_cls_41_ln":stwr_cal_parmas_zscore[:,1],
                                "GYRATE_RA_cls_95_ln":stwr_cal_parmas_zscore[:,2],
                                "CONTIG_MD_cls_90_ln":stwr_cal_parmas_zscore[:,3],
                                "GYRATE_MN_cls_41_squared":stwr_cal_parmas_zscore[:,4],
                                "CIRCLE_SD_cls_42_squared":stwr_cal_parmas_zscore[:,5]  
                                })
        df_stwr_pz.to_csv(fz_stwr) 
        
        fz_gwr = cur_outs + 'gwr_z-Scores.csv'
        df_gwr_pz = pd.DataFrame({
                                'OBJECTID': cur_geoid_list[-1][:,0],
                                'constant':gwr_cal_parmas_zscore[:,0],                            
                                "CONTIG_MN_cls_41_ln":gwr_cal_parmas_zscore[:,1],
                                "GYRATE_RA_cls_95_ln":gwr_cal_parmas_zscore[:,2],
                                "CONTIG_MD_cls_90_ln":gwr_cal_parmas_zscore[:,3],
                                "GYRATE_MN_cls_41_squared":gwr_cal_parmas_zscore[:,4],
                                "CIRCLE_SD_cls_42_squared":gwr_cal_parmas_zscore[:,5]   
                                })
        df_gwr_pz.to_csv(fz_gwr) 

