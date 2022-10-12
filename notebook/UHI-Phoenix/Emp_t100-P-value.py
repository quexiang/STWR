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

def extract_str_filedate(filename):
    datetime_string = filename.replace('.csv', '')
    return datetime_string

datapath ="D:/transactions/Postdoc/FanChao-wangzhe/0628"
outpath = "D:/transactions/Postdoc/FanChao-wangzhe/0628-run/"
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
        #print(stwr_results.summary())
        stwr_scale = stwr_results.scale 
        stwr_residuals = stwr_results.resid_response
        stwr_beta_se = stwr_results.bse 
        stwr_r2 = stwr_results.localR2
        #gwr
        gwr_selector = Sel_BW(cal_cord_gwr, cal_y_gwr, cal_X_gwr,spherical = False)
        gwr_bw= gwr_selector.search(bw_min=2)
        gwr_model = GWR(cal_cord_gwr, cal_y_gwr, cal_X_gwr, gwr_bw,spherical = False)
        gwr_results = gwr_model.fit()
        #print(gwr_results.summary())
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
        bgpnbi = bgndbi[:-2,:-1]
        
        rgndvi = rasterio.open(str_Xfilepath + 'GNDVI_clip.tif')
        bgndvi = rgndvi.read(1)
        bgndvi = bgndvi[:-2,:-1]
        
        rlndvi = rasterio.open(str_Xfilepath + 'LNDVI_clip.tif')
        blndvi = rlndvi.read(1)  
        blndvi = blndvi[:-2,:-1]

        #pf = profile  #CRS
        pf = rgndbi.profile
        transform =rgndbi.profile['transform']
        nodata = pf['nodata']
        print (pf)
        
        data_crs = pf['crs']
        
        all_coords_list = []
        Pre_X_list = []
        Pre_y_list = []
        mask_rgndbi = rgndbi.dataset_mask()
        mask_rgndbi = mask_rgndbi[:-2,:-1]
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

