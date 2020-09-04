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

#read all data 
#cal_coords_list =[]
#cal_y_list =[]
#cal_X_list =[]
#delt_stwr_intervel =[0.0]
#csvFile = open("D:/STWR/applicationofgwr/Application_CHXB/430stwr.csv", "r")
#df = pd.read_csv(csvFile,header = 0,names=['lon','lat','tmp','hpa','wet','speed','dir','height','AQI','timestamp'],
#                 dtype = {"lon" : "float64","lat":"float64",
#                          "tmp":"float64","hpa":"float64","wet":"float64","speed":"float64","dir":"float64","height":"float64","AQI":"float64",
#                          "timestamp":"float64"},
#                 skip_blank_lines = True,
#                 keep_default_na = False)
#
#df.info()
#df = df.sort_values(by=['timestamp'])  
#all_data = df.values

#read data
cal_coords_list =[]
cal_y_list =[]
cal_X_list =[]
delt_stwr_intervel =[0.0]

#beijing_all_20160403
#beijing_all_20160417
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
#read data



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




f = open('D:/STWR/applicationofgwr/Application_CHXB/testresults_20160423_LOOCV.txt','w')
error_stwr = []
error_gwr =[]
errsum_stwr = []
errsum_gwr=[]
errtol_stwr =[]
errtol_gwr =[]
spl_parts_cv = cal_X_list[-1].shape[0]

cal_coord_last_tick= np.array_split(cal_coord_tick,spl_parts_cv,axis =0)
cal_last_X_tick = np.array_split(cal_X_tick, spl_parts_cv, axis = 0)
cal_last_y_tick = np.array_split(cal_y_tick, spl_parts_cv, axis = 0)
for i in range(spl_parts_cv):
        cal_x_gwr_cur = []
        cal_y_gwr_cur = []
        cal_coord_gwr_cur=[]
        cal_stwr_coordlist = cal_coords_list.copy()
        cal_stwr_Xlist =cal_X_list.copy()
        cal_stwr_ylist = cal_y_list.copy() 
        for j in range(spl_parts_cv):
                if (i != j):
                    # 作为训练集
                     cal_coord_gwr_cur.extend(cal_coord_last_tick[j])
                     cal_x_gwr_cur.extend(cal_last_X_tick[j])
                     cal_y_gwr_cur.extend(cal_last_y_tick[j])
        cal_X_gwr =np.asarray(cal_x_gwr_cur)
        cal_y_gwr = np.asarray(cal_y_gwr_cur)
        cal_cord_gwr = np.asarray(cal_coord_gwr_cur)
        cal_stwr_coordlist.append(cal_cord_gwr)
        cal_stwr_Xlist.append(cal_X_gwr)
        cal_stwr_ylist.append(cal_y_gwr)     
        #stwr 
        stwr_selector_ = Sel_Spt_BW(cal_stwr_coordlist, cal_stwr_ylist, cal_stwr_Xlist,#gwr_bw0,
                                    delt_stwr_intervel,spherical = True)
        optalpha,optsita,opt_btticks,opt_gwr_bw0 = stwr_selector_.search() 
        stwr_model = STWR(cal_stwr_coordlist,cal_stwr_ylist,cal_stwr_Xlist,delt_stwr_intervel,
                          optsita,opt_gwr_bw0,tick_nums=opt_btticks+1,alpha =optalpha,spherical = True,recorded = True)       
        stwr_results = stwr_model.fit()  
        f.write(stwr_results.summary())            
        stwr_scale = stwr_results.scale 
        stwr_residuals = stwr_results.resid_response     
        #gwr  
        gwr_selector = Sel_BW(cal_cord_gwr, cal_y_gwr, cal_X_gwr,spherical = True)
        gwr_bw= gwr_selector.search(bw_min=2)
#        gwr_bw= gwr_selector.search(search_method='interval', bw_min=2.0, bw_max=16.0, interval=1)        
        gwr_model = GWR(cal_cord_gwr, cal_y_gwr, cal_X_gwr, gwr_bw,spherical = True)
        gwr_results = gwr_model.fit()
        f.write(gwr_results.summary())
        gw_rscale = gwr_results.scale 
        gwr_residuals = gwr_results.resid_response
        #predition results
        predPointList = cal_coord_last_tick[i]
        PreX_list = cal_last_X_tick[i]
        vd_y = cal_last_y_tick[i]
        #stwr
        pred_stwr_dir_result = stwr_model.predict([predPointList],[PreX_list],stwr_scale,stwr_residuals)
        pre_y_stwr = pred_stwr_dir_result.predictions 
        #pre_parmas_stwr=np.reshape(pred_stwr_dir_result.params.flatten(),(-1,allklen_stwr)) 
        #gwr
        pred_gwr_dir_result = gwr_model.predict(predPointList,PreX_list,gw_rscale,gwr_residuals)
        pre_y_gwr = pred_gwr_dir_result.predictions
        #gwr
        #pre_parmas_gwr=np.reshape(pred_gwr_dir_result.params.flatten(),(-1,allklen_stwr))
        #error append
        lenpre = vd_y.shape[0]
        errsum_stwr.append(math.sqrt(np.sum(np.square(vd_y-pre_y_stwr))/lenpre))
        errsum_gwr.append(math.sqrt(np.sum(np.square(vd_y-pre_y_gwr))/lenpre))
        error_stwr.append(vd_y-pre_y_stwr)
        error_gwr.append(vd_y-pre_y_gwr)
        errtol_stwr.append(np.sum(np.square(vd_y-pre_y_stwr)))
        errtol_gwr.append(np.sum(np.square(vd_y-pre_y_gwr)))
        f.write('The result of {:d}'.format(i))
f.close()                  
sstw_total = np.sum(errsum_stwr)
sgwr_total = np.sum(errsum_gwr)
sstw_total
sgwr_total

################################draw LOOCV ############################

from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt

font1 = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
font2 = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }


csvFile = open("D:/STWR/applicationofgwr/Application_CHXB/20160423LOOCV.csv", "r")
df = pd.read_csv(csvFile,header = 0,names=['NOP','GWR_err','STWR_err','GWR_err2','STWR_err2'],
                 dtype = {"NOP" : "float64","GWR":"float64",
                          "STWR":"float64","GWR_err2":"float64",
                          "STWR_err2":"float64"
                          },
                 skip_blank_lines = True,
                 keep_default_na = False)

df.info() 
all_loocv = df.values

numbs = all_loocv[:,0]
lop_GWR = all_loocv[:,1]
lop_STWR = all_loocv[:,2]

def demo(sty):
    mpl.style.use(sty)
    fig, ax = plt.subplots(figsize=(6, 6))

#    ax.set_title('style: {!r}'.format(sty), color='C0')
    ax.plot(numbs, lop_GWR, 'sb', label='AE of GWR')
    ax.plot(numbs, lop_STWR,'ro', label='AE of STWR')

    plt.ylabel('Absolute Error(AE) ',fontdict=font1)
    plt.xlabel('Point number',fontdict=font1)
    plt.title('(LOOCV) Absolute Error(AE) of prediction ',fontdict=font2)
   # ax.plot(th, np.cos(th), 'C1', label='C1')
    #ax.plot(th, np.sin(th), 'C2', label='C2')
    ax.legend()
#['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 
#'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette',
# 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 
#'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 
#'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
demo('seaborn')


all_loocv = all_loocv[:,1:3]
#all_loocv = all_loocv[:,3:5]
labels = ['GWR', 'STWR']
def demo(sty):
    mpl.style.use(sty)
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['blue', 'red']
    
    bplot = ax.boxplot(all_loocv,
               notch=True,
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=labels)  # will be used to label x-ticks
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.yaxis.grid(True)
    ax.set_title('(LOOCV) Absolute Error(AE) of prediction box plot',fontdict=font2)
#demo('seaborn-paper')
demo('default')




