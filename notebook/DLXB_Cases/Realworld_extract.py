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
import pyproj
import os


#read position_info
csvFile = open("D:/STWR/applicationofgwr/Application_CHXB/position_info.csv","r")
df_pos = pd.read_csv(csvFile,header = 0,
                 skip_blank_lines = True,
                 keep_default_na = False)       
df_pos.info()
df_pos = df_pos[['number', 'height','lon','lat']]
pos_arr =df_pos.values


csv_beijing = open("D:/STWR/applicationofgwr/Application_CHXB/chuliAQI/real_time_beijing.csv","r")
df_beijing = pd.read_csv(csv_beijing,header = 0,
                 skip_blank_lines = True,
                 keep_default_na = False)       
df_beijing .info()

beijing_arr =df_beijing.values


np_rcds = np.zeros((pos_arr.shape[0],29*24,10))
in_rcd = 0    
for i_pos in pos_arr[:,0]:
    trick = np.where(beijing_arr[:,0] == i_pos)
    beijing_trick = beijing_arr[trick]
    beijing_trick_trunk = beijing_trick[3:,:]
    beijing_trick_trunk = beijing_trick_trunk[:-21,:]
    time_h_tick= 0
    tt_rcd = 0
    for row_in in beijing_trick_trunk:
        np_rcds[in_rcd][tt_rcd,:2]  = pos_arr[in_rcd,-2:]
        np_rcds[in_rcd][tt_rcd,2:-2] = row_in[5:]
        np_rcds[in_rcd][tt_rcd,-2] = pos_arr[in_rcd,-3]
        np_rcds[in_rcd][tt_rcd,-1] = time_h_tick
        time_h_tick = time_h_tick+1
        tt_rcd += 1
    in_rcd += 1
    


arr = os.listdir("D:/STWR/applicationofgwr/Application_CHXB/20160401-0430/")
str_path = 'D:/STWR/applicationofgwr/Application_CHXB/20160401-0430/'
out_path ='D:/STWR/applicationofgwr/Application_CHXB/chuliAQI/' 

hour_idx = 0
day_idx = 0
for fi in arr:
        file_path = str_path +fi
#        csvFile = open("D:/STWR/applicationofgwr/Application_CHXB/20160401-0430/beijing_all_20160401.csv","r", encoding='utf-8')
        csvFile = open(file_path,"r", encoding='utf-8')
        df = pd.read_csv(csvFile,header = 0,
                         skip_blank_lines = True,
                         keep_default_na = False)
        
        df.info()
        df2 = df[['顺义', '万柳','延庆','密云','怀柔','密云水库','平谷','通州','农展馆','昌平','门头沟','天坛','古城','丰台花园','大兴','房山']]
        
        df2.dtypes
        
        df2['顺义' ]=pd.to_numeric(df2['顺义'])
        df2['万柳' ]=pd.to_numeric(df2['万柳'])
        df2['延庆' ]=pd.to_numeric(df2['延庆'])
        df2['密云' ]=pd.to_numeric(df2['密云'])
        df2['怀柔' ]=pd.to_numeric(df2['怀柔'])
        df2['密云水库' ]=pd.to_numeric(df2['密云水库'])#可能对不上
        df2['平谷' ]=pd.to_numeric(df2['平谷'])
        df2['通州' ]=pd.to_numeric(df2['通州'])
        df2['农展馆' ]=pd.to_numeric(df2['农展馆'])
        df2['昌平' ]=pd.to_numeric(df2['昌平'])
        df2['门头沟' ]=pd.to_numeric(df2['门头沟'])
        df2['天坛' ]=pd.to_numeric(df2['天坛'])
        df2['古城' ]=pd.to_numeric(df2['古城'])
        df2['丰台花园' ]=pd.to_numeric(df2['丰台花园'])
        df2['大兴' ]=pd.to_numeric(df2['大兴'])
        df2['房山' ]=pd.to_numeric(df2['房山'])
        
        all_data = df2.values
        trow = all_data.shape[0]
        tcol = all_data.shape[1]
        
        pm25 = []
        pm25_24 = []
        pm10=[]
        pm10_24=[]
        AQI=[]
        time_stamp=[]
        trans_arr = None
        
#        coord_info =np.array( 
#        [[116.6166667,40.133],
#        [116.2833,39.9833],
#        [115.917,40.4],
#        [116.87,40.38],
#        [116.633,40.367],
#        [117.117,40.65],#密云水库可能要拿掉
#        [117.117,40.167],
#        [116.75,39.85],
#        [116.5,39.95],
#        [116.217,40.217],
#        [116.1,39.93],
#        [116.47,39.8],
#        [116.2,39.95],
#        [116.25,39.87],
#        [116.35,39.717],
#        [116.2,39.77]])
    
        for hour in range(24): 
                cur_hour_idx = day_idx*24 + hour
                tick_pm25 = all_data[hour*5+ 0].reshape((16,1))
                pm25.append(tick_pm25)
                tick_pm25_24 = all_data[hour*5+ 1].reshape((16,1))
                pm25_24.append(tick_pm25_24) 
                tick_pm10 =all_data[hour*5+ 2].reshape((16,1))
                pm10.append(tick_pm10) 
                tick_pm10_24 = all_data[hour*5+ 3].reshape((16,1))
                pm10_24.append(tick_pm10_24) 
                tick_aqi = all_data[hour*5+ 4].reshape((16,1))
                AQI.append(tick_aqi) 
                tick_time = np.ones((16,1))*hour
                time_stamp.append(tick_time)
                
                npvxs = np.array((np_rcds[0][cur_hour_idx],
                np_rcds[1][cur_hour_idx],np_rcds[2][cur_hour_idx],np_rcds[3][cur_hour_idx],
                np_rcds[4][cur_hour_idx],np_rcds[5][cur_hour_idx],np_rcds[6][cur_hour_idx],
                np_rcds[7][cur_hour_idx],np_rcds[8][cur_hour_idx],np_rcds[9][cur_hour_idx],
                np_rcds[10][cur_hour_idx],np_rcds[11][cur_hour_idx],np_rcds[12][cur_hour_idx],
                np_rcds[13][cur_hour_idx],np_rcds[14][cur_hour_idx],np_rcds[15][cur_hour_idx]
                ))
                
                trans_tick = np.concatenate((tick_pm25,tick_pm25_24,tick_pm10,tick_pm10_24,tick_aqi,tick_time),axis=1)
                trans_tick = np.concatenate((npvxs,trans_tick),axis=1)    
#                trans_tick = np.concatenate((coord_info,trans_tick),axis=1)
                if trans_arr is None:     
                    trans_arr = trans_tick
                else:
                    trans_arr = np.concatenate((trans_arr,trans_tick),axis=0)
                    
        df_out = pd.DataFrame(trans_arr, index=[i for i in range(trans_arr.shape[0])],columns=['lon','lat','temp','hpa','wet','speed','dir','mm','height','t_time','pm25','pm25_24','pm10','pm10_24','aqi','time'])
        out_file = out_path + fi
        df_out.to_csv(out_file)
        day_idx +=1
#        df_out.to_csv("D:/STWR/applicationofgwr/Application_CHXB/chuliAQI/test_20160401.csv")     

    
#read data
#arr = os.listdir("D:/STWR/Application/chao216/Bioturbation/Data/To_Chao/To_Chao/independentVariable")
#str_path = 'D:/STWR/Application/chao216/Bioturbation/Data/To_Chao/To_Chao/independentVariable/'
#idx = 3
#min_t = -100000000
#max_t = 100000000
#max_trunc_val = 0.00000001
#min_trunc_val = -0.00000001
#m_eps = 0.00000001
#for fi in arr:
#    #get the X value and writ to csv file  
#    file_path = str_path +fi
#    ppt1= rasterio.open(file_path)
#    bppt1 = ppt1.read(1)
#    pf = ppt1.profile
#    transform =ppt1.profile['transform']
#    nodata = pf['nodata']
#    crs_all = pf['crs']
#    rows, cols = rasterio.transform.rowcol(transform, cal_data[:,2], cal_data[:,1])
#    vals = bppt1[rows,cols]
#    vals = np.float64(vals)
#    vals[vals == nodata] = np.nan
#    vals[(vals < max_trunc_val) & (vals >min_trunc_val)] = np.nan
#    vals[(vals<min_t) | (vals>max_t)] = np.nan
#    index_of_dot = fi.index('.')
#    fieldname = fi[:index_of_dot]
#    df2.insert(idx,fieldname,vals,True)
#    idx +=1
#
#
#
#all_data = df2.values
##remove nan in all_data
#all_data = all_data[~np.isnan(all_data).any(axis=1)]
##log y
#all_data[:,0] =  np.log(all_data[:,0])
##remove log y = inf
#all_data = all_data[~np.isinf(all_data).any(axis=1)]
#
#
#cal_y_gwr  = np.array(all_data[:,0]).reshape((-1,1))
#cal_cord_gwr = np.asarray(all_data[:,1:3])
#cal_cord_gwr[:,[0,1]] = cal_cord_gwr[:,[1,0]]
#cal_X_gwr  = np.asarray(all_data[:,3:])
#
#
##from sklearn.preprocessing import StandardScaler 
##sc = StandardScaler()
##cal_X_gwr = sc.fit_transform(cal_X_gwr)
#from sklearn.preprocessing import Normalizer 
#nr = Normalizer()
#cal_X_gwr = nr.fit_transform(cal_X_gwr)
#
#cal_X_gwr += np.random.normal(0,m_eps,cal_X_gwr.shape)
#gwr_selector = Sel_BW(cal_cord_gwr, cal_y_gwr, cal_X_gwr,spherical = True)
#gwr_bw= gwr_selector.search()
#gwr_model = GWR(cal_cord_gwr, cal_y_gwr, cal_X_gwr, gwr_bw,spherical = True)
#gwr_results = gwr_model.fit()
#print(gwr_results.summary())
#
#cal_log_y =np.log(cal_y_gwr)
#cal_log_y_gwr =  cal_log_y
#cal_log_gwr = cal_X_gwr
#cal_log_cord = cal_cord_gwr
#dix = 0
#for cinf in cal_log_y:    
#    if math.isinf(cinf):
#            cal_log_y_gwr = np.delete(cal_log_y_gwr,dix,0)
#            cal_log_gwr = np.delete(cal_log_gwr,dix,0)
#            cal_log_cord =  np.delete(cal_log_cord,dix,0)
#            dix -= 1
#    dix +=1
#
#
##cal_X_gwr += np.random.normal(0,0.00001,cal_X_gwr.shape)
#
#gwr_selector = Sel_BW(cal_log_cord, cal_log_y_gwr, cal_log_gwr,spherical = True)
#gwr_bw= gwr_selector.search()
#gwr_model = GWR(cal_log_cord, cal_log_y_gwr, cal_log_gwr, gwr_bw,spherical = True)
#gwr_results = gwr_model.fit()
#print(gwr_results.summary())






