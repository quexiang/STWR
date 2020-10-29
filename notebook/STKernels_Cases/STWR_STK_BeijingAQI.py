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

#from spglm.family import Gaussian, Binomial, Poisson
#read data
cal_coords_list =[]
cal_y_list =[]
cal_X_list =[]
delt_stwr_intervel =[0.0]
#20160427
#20160423
csvFile = open("D:/STWR/applicationofgwr/Application_CHXB/chuliAQI/beijing_all_20160423.csv", "r")
df = pd.read_csv(csvFile,header = 0,
                 skip_blank_lines = True,
                 keep_default_na = False)
df.info()

#drop others 
df = df.drop(columns=['pm25', 'pm25_24','pm10','pm10_24'])
#no drop 
#df['pm25' ]=pd.to_numeric(df['pm25'])
#df['pm25_24' ]=pd.to_numeric(df['pm25_24'])
#df['pm10' ]=pd.to_numeric(df['pm10'])
#df['pm10_24' ]=pd.to_numeric(df['pm10_24'])
df['aqi' ]=pd.to_numeric(df['aqi'])

df = df.sort_values(['time']) 
all_data = df.values
#remove nan in all_data
all_data = all_data[~np.isnan(all_data).any(axis=1)]
all_data = all_data[:,1:]

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

    x_tick = np.array([all_data[row,2],all_data[row,3],all_data[row,4],all_data[row,5],all_data[row,6],all_data[row,8]])
    cal_X_tick.append(x_tick)
    y_tick = np.array([all_data[row,-2]])
    cal_y_tick.append(y_tick)
#最后在放一次
#gwr解出最后一期 
cal_cord_gwr = np.asarray(cal_coord_tick)
cal_X_gwr  = np.asarray(cal_X_tick)
cal_y_gwr = np.asarray(cal_y_tick)  
cal_coords_list.append(np.asarray(cal_coord_tick))
cal_X_list.append(np.asarray(cal_X_tick))
cal_y_list.append(np.asarray(cal_y_tick))

#stwr 
stwr_selector_ = Sel_Spt_BW(cal_coords_list, cal_y_list, cal_X_list,delt_stwr_intervel,spherical = True)
optalpha,optsita,opt_btticks,opt_gwr_bw0 = stwr_selector_.search() 
stwr_model = STWR(cal_coords_list,cal_y_list,cal_X_list,delt_stwr_intervel,optsita,opt_gwr_bw0,tick_nums=opt_btticks+1,alpha =optalpha,spherical = True,recorded = 1)

#optalpha,optsita,opt_btticks,opt_gwr_bw0 = stwr_selector_.search() 
#stwr_model = STWR(cal_coords_list,cal_y_list,cal_X_list,delt_stwr_intervel,
#                  optsita,opt_gwr_bw0,tick_nums=opt_btticks+1,alpha =optalpha,spherical = True,recorded=1)#recorded = True)

stwr_results = stwr_model.fit()
print(stwr_results.summary())
stwr_localr2 = stwr_results.localR2
print(stwr_localr2)

#gwr  数据只有最后一期
gwr_selector = Sel_BW(cal_cord_gwr, cal_y_gwr, cal_X_gwr,spherical = True)
gwr_bw= gwr_selector.search(bw_min=2)
#search_method='golden_section', criterion='AICc',bw_min=None, bw_max=None, interval=0.0,
#gwr_bw= gwr_selector.search(search_method='interval', bw_min=2.0, bw_max=16.0, interval=1)
                            
gwr_model = GWR(cal_cord_gwr, cal_y_gwr, cal_X_gwr, gwr_bw,spherical = True)
gwr_results = gwr_model.fit()
print(gwr_results.summary())
gwr_localr2 = gwr_results.localR2
print(gwr_localr2)