import  numpy  as  np
from numpy import *
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

if __name__ == '__main__':
    list_ols_aic = []
    list_gwr_aicc = []
    list_stwr_aicc = []

    listgndvi_f = []
    listgpnbi_f = []
    listmoran_f = []
    #Define the time interval
    def procedure(second =1):
        time.sleep(second)
    #read data read The image is 354 * 355 in size #95

    r95gndvi = rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2000/X2_GNDVI.tif')
    b95gndvi = r95gndvi.read(1)
    b95gndvi_f = b95gndvi[1:-1,1:]
    listgndvi_f.append(b95gndvi_f)

    r95gpnbi  = rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2000/X1_GNDBI.tif')
    b95gpnbi = r95gpnbi.read(1)
    b95gpnbi_f = b95gpnbi[1:-1,1:]
    listgpnbi_f.append(b95gpnbi_f)

    r95moran =  rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2000/X3_LNDVI.tif')
    b95moran = r95moran.read(1)
    b95moran_f = b95moran[1:-1,1:]
    listmoran_f.append(b95moran_f)
    #2005

    r00gndvi = rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2005/X2_GNDVI.tif')
    b00gndvi = r00gndvi.read(1)
    b00gndvi_f = b00gndvi[1:-1,1:]
    listgndvi_f.append(b00gndvi_f)

    r00gpnbi  = rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2005/X1_GNDBI.tif')
    b00gpnbi = r00gpnbi.read(1)
    b00gpnbi_f = b00gpnbi[1:-1,1:]
    listgpnbi_f.append(b00gpnbi_f)

    r00moran =  rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2005/X3_LNDVI.tif')
    b00moran = r00moran.read(1)
    b00moran_f = b00moran[1:-1,1:]
    listmoran_f.append(b00moran_f)
    #2010

    r05gndvi = rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2010/X2_GNDVI.tif')
    b05gndvi = r05gndvi.read(1)
    b05gndvi_f = b05gndvi[1:-1,1:]
    listgndvi_f.append(b05gndvi_f)

    r05gpnbi  = rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2010/X1_GNDBI.tif')
    b05gpnbi = r05gpnbi.read(1)
    b05gpnbi_f = b05gpnbi[1:-1,1:]
    listgpnbi_f.append(b05gpnbi_f)

    r05moran =  rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2010/X3_LNDVI.tif')
    b05moran = r05moran.read(1)
    b05moran_f = b05moran[1:-1,1:]
    listmoran_f.append(b05moran_f)
    #2015

    r10gndvi = rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2015/X2_GNDVI.tif')
    b10gndvi = r10gndvi.read(1)
    b10gndvi_f = b10gndvi[1:-1,1:]
    listgndvi_f.append(b10gndvi_f)

    r10gpnbi  = rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2015/X1_GNDBI.tif')
    b10gpnbi = r10gpnbi.read(1)
    b10gpnbi_f = b10gpnbi[1:-1,1:]
    listgpnbi_f.append(b10gpnbi_f)

    r10moran =  rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2015/X3_LNDVI.tif')
    b10moran = r10moran.read(1)
    b10moran_f = b10moran[1:-1,1:]
    listmoran_f.append(b10moran_f)
    #2020
    r15gndvi = rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2020/X2_GNDVI.tif')
    b15gndvi = r15gndvi.read(1)
    b15gndvi_f = b15gndvi[1:-1,1:]
    listgndvi_f.append(b15gndvi_f)

    r15gpnbi  = rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2020/X1_GNDBI.tif')
    b15gpnbi = r15gpnbi.read(1)
    b15gpnbi_f = b15gpnbi[1:-1,1:]
    listgpnbi_f.append(b15gpnbi_f)

    r15moran =  rasterio.open('D:/STWR/Data-Run/ChaoFan20211110/2020/X3_LNDVI.tif')
    b15moran = r15moran.read(1)
    b15moran_f = b15moran[1:-1,1:]
    listmoran_f.append(b15moran_f)


    #############################
    pf = r95gndvi.profile
    transform =r95gndvi.profile['transform']
    nodata = pf['nodata']

    time_dif = [0.0,5.0,5.0,5.0,5.0]
    #############################
    np.random.seed(42)


    #beta0
    beta0 = np.ones((353,353))
    beta0 = 2*beta0
    beta_min_0 = np.amin(beta0)
    beta_max_0 = np.amax(beta0)
    beta_min = beta_min_0
    beta_max = beta_max_0
    beta_min_list =[beta_min_0]
    beta_max_list =[beta_max_0]
    #beta1
    beta1 = -2*np.ones((353,353))
    for i in range(353) :
                for j in range(353):
                    beta1[i,j] = -2+(i+j)/353                 
    #minmax
    beta_min_1 = np.amin(beta1)
    beta_max_1 = np.amax(beta1)
    beta_min_list.append(beta_min_1)
    beta_max_list.append(beta_max_1)
    if beta_min_1<beta_min:
        beta_min =  beta_min_1
    if beta_max_1> beta_max:
        beta_max =  beta_max_1

    #beta2
    beta2 = np.ones((353,353))
    for i in range(353) :
                for j in range(353):
                    beta2[i,j] = 1 +(3- (2- i/90)**2)*(3- (2- j/90)**2)/2
    #beta2minmax
    beta_min_2 = np.amin(beta2)
    beta_max_2 = np.amax(beta2)
    beta_min_list.append(beta_min_2)
    beta_max_list.append(beta_max_2)
    if beta_min_2<beta_min:
        beta_min =  beta_min_2
    if beta_max_2> beta_max:
        beta_max = beta_max_2

    #beta3
    beta3 = np.ones((353,353))
    for i in range(353) :
      for j in range(353):
              beta3[i,j]= 0.2*1*math.sin(1/100*np.pi*(j+i))*2.5**2
    #  beta3 minmax
    beta_min_3 = np.amin(beta3)
    beta_max_3 = np.amax(beta3)
    beta_min_list.append(beta_min_3)
    beta_max_list.append(beta_max_3)

    if np.amin(beta3)<beta_min:
        beta_min =  beta_min_3
    if np.amax(beta3)> beta_max:
        beta_max =  beta_max_3

    #add error
    mu_err, sigma_err = 0, 2
    err = np.random.normal(mu_err, sigma_err, 5*353*353*1)
    err = err.reshape((5,353*353,1))

    lst95 = beta0 + beta1 * b95gpnbi_f + beta2 * b95gndvi_f + beta3 * b95moran_f + np.random.choice((-1,1))*err[0].reshape((353,353))
    lst00 = beta0 + beta1 * b00gpnbi_f + beta2 * b00gndvi_f + beta3 * b00moran_f + np.random.choice((-1,1))*err[1].reshape((353,353))
    lst05 = beta0 + beta1 * b05gpnbi_f + beta2 * b05gndvi_f + beta3 * b05moran_f + np.random.choice((-1,1))*err[2].reshape((353,353))
    lst10 = beta0 + beta1 * b10gpnbi_f + beta2 * b10gndvi_f + beta3 * b10moran_f + np.random.choice((-1,1))*err[3].reshape((353,353))
    lst15 = beta0 + beta1 * b15gpnbi_f + beta2 * b15gndvi_f + beta3 * b15moran_f + np.random.choice((-1,1))*err[4].reshape((353,353))
    # lst20 = beta0 + beta1 * b20gpnbi_f + beta2 * b20gndvi_f + beta3 * b20moran_f + np.random.choice((-1,1))*err[5].reshape((353,353))


    with open('D:/STWR/Data-Run/results/case3/normal_errs.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(err.shape))
        for data_slice in err:
            np.savetxt(outfile, data_slice, fmt='%-7.7f')
            outfile.write('# New slice\n')

    ####################################################################################################
    #output  betas surfaces of each years
    with open('D:/STWR/Data-Run/results/case3/init_beta0.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(beta0.shape))
        np.savetxt(outfile, beta0, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/init_beta1.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(beta1.shape))
        np.savetxt(outfile, beta1, fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/init_beta2.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(beta2.shape))
        np.savetxt(outfile, beta2, fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/init_beta3.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(beta3.shape))
        np.savetxt(outfile, beta3, fmt='%-7.7f')


    lst95_min = np.amin(lst95)
    lst95_max = np.amax(lst95)
    lst_min = lst95_min
    lst_max = lst95_max

    lst00_min = np.amin(lst00)
    lst00_max = np.amax(lst00)
    if lst00_min<lst_min:
        lst_min =  lst00_min
    if lst00_max> lst_max:
        lst_max = lst00_max

    lst05_min = np.amin(lst05)
    lst05_max = np.amax(lst05)
    if lst05_min<lst_min:
        lst_min =  lst05_min
    if lst05_max> lst_max:
        lst_max = lst05_max

    lst10_min = np.amin(lst10)
    lst10_max = np.amax(lst10)
    if lst10_min<lst_min:
        lst_min =  lst10_min
    if lst10_max> lst_max:
        lst_max = lst10_max

    lst15_min = np.amin(lst15)
    lst15_max = np.amax(lst15)
    if lst15_min<lst_min:
        lst_min =  lst15_min
    if lst15_max> lst_max:
        lst_max = lst15_max

    #draw beta surfaces
    fig = plt.figure(figsize=(17,3),constrained_layout=True)
    gs = GridSpec(1, 4, figure=fig)
    jet  = plt.get_cmap('jet',256)
    psm1=0
    psm2=0
    psm3=0
    psm4=0

    ax1 = fig.add_subplot(gs[0,0])
    psm1 = ax1.pcolormesh(beta0, cmap=jet, rasterized=True, vmin=beta_min, vmax=beta_max)

    ax2 = fig.add_subplot(gs[0,1])
    psm2 = ax2.pcolormesh(beta1, cmap=jet, rasterized=True, vmin=beta_min, vmax=beta_max)

    ax3 = fig.add_subplot(gs[0,2])
    psm3 = ax3.pcolormesh(beta2, cmap=jet, rasterized=True, vmin=beta_min, vmax=beta_max)
    
    ax4 = fig.add_subplot(gs[0,3])
    psm4 = ax4.pcolormesh(beta3, cmap=jet, rasterized=True, vmin=beta_min, vmax=beta_max)
    fig.colorbar(psm4, ax = ax4)
    plt.show()

    #draw y surfaces
    fig2 = plt.figure(figsize=(20,3),constrained_layout=True)
    gs2 = GridSpec(1, 5, figure=fig2)
    jet  = plt.get_cmap('jet',256)
    psm1=0
    psm2=0
    psm3=0
    psm4=0
    psm5=0

 
    ax1 = fig2.add_subplot(gs2[0,0])
    psm1 = ax1.pcolormesh(lst95, cmap=jet, rasterized=True, vmin=lst_min, vmax=lst_max)

    ax2 = fig2.add_subplot(gs2[0,1])
    psm2 = ax2.pcolormesh(lst00, cmap=jet, rasterized=True, vmin=lst_min, vmax=lst_max)

    ax3 = fig2.add_subplot(gs2[0,2])
    psm3 = ax3.pcolormesh(lst05, cmap=jet, rasterized=True, vmin=lst_min, vmax=lst_max)

    ax4 = fig2.add_subplot(gs2[0,3])
    psm4 = ax4.pcolormesh(lst10, cmap=jet, rasterized=True, vmin=lst_min, vmax=lst_max)
    fig.colorbar(psm4, ax = ax4)
    ax4 = fig2.add_subplot(gs2[0,4])
    psm4 = ax4.pcolormesh(lst15, cmap=jet, rasterized=True, vmin=lst_min, vmax=lst_max)
    fig2.colorbar(psm4, ax = ax4)
    plt.show()

    ######################################################
    #2.Define Four type of heterogeneity trend functions#
    #draw the Four trend surfaces#
    values_change_dwr = np.ones([4,353,353])
    def Values_change_trend(values_change_trend,changecase,cur_tick=0, deta_t = 1,sita = 1,power = 1):
                if (changecase == 0):
                  for i in range(353):
                    for j in range(353):
                        values_change_trend[cur_tick,i,j] = values_change_trend[cur_tick,i,j] + sita*1*math.sin((j/40))*deta_t**power
                elif(changecase == 1):
                        for i in range(353):
                            for j in range(353):
                                values_change_trend[cur_tick,i,j] =values_change_trend[cur_tick,i,j] + sita*1*math.sin(1/100*np.pi*(i))*deta_t**power

                elif (changecase == 2):
                        for i in range(353):
                                for j in range(353):
                                        values_change_trend[cur_tick,i,j] = values_change_trend[cur_tick,i,j] + sita*1*math.sin(1/60*np.pi*(j+i))*deta_t**power
                elif (changecase == 3):
                                    for i in range(353):
                                            for j in range(353):
                                                values_change_trend[cur_tick,i,j] = values_change_trend[cur_tick,i,j] + sita*1*math.sin(1/160*np.pi*(j+i))*deta_t**power
    Values_change_trend(values_change_dwr,0,0,1,0.1)
    Values_change_trend(values_change_dwr,1,1,1,0.1)
    Values_change_trend(values_change_dwr,2,2,1,0.1)
    Values_change_trend(values_change_dwr,3,3,1,0.1)

    tred_max = [np.amax(values_change_dwr[0]),np.amax(values_change_dwr[1]),np.amax(values_change_dwr[2]),np.amax(values_change_dwr[3])]
    tred_min = [np.amin(values_change_dwr[0]),np.amin(values_change_dwr[1]),np.amin(values_change_dwr[2]),np.amin(values_change_dwr[3])]

    tredmax = max(tred_max)
    tredmin = min(tred_min)
    #####Generate the change thrend and draw #####
    fig_tred = plt.figure(figsize=(17,3),constrained_layout=True)
    gs_tred = GridSpec(1, 4, figure=fig_tred)
    jet_tred  = plt.get_cmap('jet',256)

    #add supplot
    ax_tred = fig_tred.add_subplot(gs_tred[0,0])
    psm_tred = ax_tred.pcolormesh(values_change_dwr[0], cmap=jet_tred, rasterized=True, vmin=tredmin, vmax=tredmax)

    ax_tred = fig_tred.add_subplot(gs_tred[0,1])
    psm_tred = ax_tred.pcolormesh(values_change_dwr[1], cmap=jet_tred, rasterized=True, vmin=tredmin, vmax=tredmax)

    ax_tred = fig_tred.add_subplot(gs_tred[0,2])
    psm_tred = ax_tred.pcolormesh(values_change_dwr[2], cmap=jet_tred, rasterized=True, vmin=tredmin, vmax=tredmax)

    ax_tred = fig_tred.add_subplot(gs_tred[0,3])
    psm_tred = ax_tred.pcolormesh(values_change_dwr[3], cmap=jet_tred, rasterized=True, vmin=tredmin, vmax=tredmax)

    fig_tred.colorbar(psm_tred, ax = ax_tred)
    plt.show()

    #Define the beta change through time#
    def betavaluechange(onebeta,timeintervel=1,pchange=0,sita=1,power = 1):
        result_beta = np.ones_like(onebeta)
        if power ==0:
            result_beta = onebeta
            return result_beta
        elif (pchange == 0):
           for i in range(353):
               for j in range(353):
                       result_beta[i,j] = onebeta[i,j] + sita*1*math.sin((j/40))*timeintervel**power
           return result_beta
        elif (pchange == 1):
            for i in range(353):
                   for j in range(353):
                       result_beta[i,j] =onebeta[i,j] + sita*1*math.sin(1/100*np.pi*(i))*timeintervel**power
            return result_beta
        elif (pchange == 2):
            for i in range(353):
                   for j in range(353):
                        result_beta[i,j] =  onebeta[i,j]+ sita*1*math.sin(1/60*np.pi*(j+i))*timeintervel**power
            return result_beta
        elif (pchange == 3):
            for i in range(353):
                   for j in range(353):
                        result_beta[i,j]  = onebeta[i,j] + sita*1*math.sin(1/160*np.pi*(j+i))*timeintervel**power
            return result_beta
        else:
            raise print(' None supported')
            return result_beta
    #Define the beta change through time#

    #Generate Betas surfaces throught time
    kticks = 5
    tick_times_intervel = time_dif
    delt_gtwr_intervel = np.zeros((kticks,1))


    beta_change = np.ones([kticks,4,353,353]);
    beta_min_list =  []
    beta_max_list =  []
    beta =  [beta0,beta1,beta2,beta3]
    beta_tick =beta.copy()

    beta_change[0,0] = beta0
    beta_change[0,1] = beta1
    beta_change[0,2] = beta2
    beta_change[0,3] = beta3
    beta_min_list.append(int(np.amin(beta_change[0])))
    beta_max_list.append(math.ceil(np.amax(beta_change[0])))

    for i in range((kticks-1)):
        #case 3 beta surfaces change over time
        beta_change[i+1,0]= betavaluechange(beta_tick[0],tick_times_intervel[i+1],0,0.1)
        beta_change[i+1,1]= betavaluechange(beta_tick[1],tick_times_intervel[i+1],1,0.1)
        beta_change[i+1,2]= betavaluechange(beta_tick[2],tick_times_intervel[i+1],2,0.1)
        beta_change[i+1,3]= betavaluechange(beta_tick[3],tick_times_intervel[i+1],3,0.1)
        #case 3 beta surfaces change over time
        beta_tick[0] = beta_change[i,0]
        beta_tick[1] = beta_change[i,1]
        beta_tick[2] = beta_change[i,2]
        beta_tick[3] = beta_change[i,3]
        #record the min and max value for draw the pics later.
        beta_min_list.append(int(np.amin(beta_change[i+1])))
        beta_max_list.append(math.ceil(np.amax(beta_change[i+1])))
    beta_min = min(beta_min_list)
    beta_max = max(beta_max_list)

    #2.Draw figures of different heterogeneity surfaces
    fig = plt.figure(figsize=(17,3),constrained_layout=True)
    gs = GridSpec(1, 4, figure=fig)
    jet  = plt.get_cmap('jet',256)
    psm =0
    for col in range(4):
        ax = fig.add_subplot(gs[0,col])
        psm = ax.pcolormesh(beta[col], cmap=jet, rasterized=True, vmin=beta_min, vmax=beta_max)
        #write true beta out to files
    fig.colorbar(psm, ax = fig.axes[2])
    plt.show()

    #draw the cofficients surfaces of different time stages.
    fig_time = plt.figure(figsize=(17,4*kticks),constrained_layout=True)
    gs_time = GridSpec(kticks, 4, figure=fig_time)
    jet  = plt.get_cmap('jet',256)
    psm_time =0
    for row in range(kticks):
        for col in range(4):
            ax_time = fig_time.add_subplot(gs_time[row,col])
            psm_time = ax_time.pcolormesh(beta_change[row,col], cmap=jet, rasterized=True, vmin=beta_min_list[row], vmax=beta_max_list[row])

             #output  betas surfaces of each years
            str_file_out = 'D:/STWR/Data-Run/results/case3/true_beta_{0}_{1}.txt'.format(row,col)
            with open(str_file_out, 'w') as outfile:
                outfile.write('# Array shape: {0}\n'.format( beta_change[row,col].shape))
                np.savetxt(outfile,beta_change[row,col], fmt='%-7.7f')
            fig_time.colorbar(psm_time, ax = ax_time)
    plt.show()


    ###########################################################
    lst95 = beta_change[0,0] + beta_change[0,1] * b95gpnbi_f + beta_change[0,2] * b95gndvi_f + beta_change[0,3] * b95moran_f + np.random.choice((-1,1))*err[0].reshape((353,353))
    lst00 = beta_change[1,0] + beta_change[1,1] * b00gpnbi_f + beta_change[1,2] * b00gndvi_f + beta_change[1,3] * b00moran_f + np.random.choice((-1,1))*err[1].reshape((353,353))
    lst05 = beta_change[2,0] + beta_change[2,1] * b05gpnbi_f + beta_change[2,2] * b05gndvi_f + beta_change[2,3] * b05moran_f + np.random.choice((-1,1))*err[2].reshape((353,353))
    lst10 = beta_change[3,0] + beta_change[3,1] * b10gpnbi_f + beta_change[3,2] * b10gndvi_f + beta_change[3,3] * b10moran_f + np.random.choice((-1,1))*err[3].reshape((353,353))
    lst15 = beta_change[4,0] + beta_change[4,1] * b15gpnbi_f + beta_change[4,2] * b15gndvi_f + beta_change[4,3] * b15moran_f + np.random.choice((-1,1))*err[4].reshape((353,353))

    #output  y surfaces of each years
    with open('D:/STWR/Data-Run/results/case3/y_surface_2000.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(lst95.shape))
        np.savetxt(outfile, lst95, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/y_surface_2005.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(lst00.shape))
        np.savetxt(outfile, lst00, fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/y_surface_2010.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(lst05.shape))
        np.savetxt(outfile, lst05, fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/y_surface_2015.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(lst10.shape))
        np.savetxt(outfile, lst10, fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/y_surface_2020.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(lst15.shape))
        np.savetxt(outfile, lst15, fmt='%-7.7f')


    lst95_min = np.amin(lst95)
    lst95_max = np.amax(lst95)
    lst_min = lst95_min
    lst_max = lst95_max

    lst00_min = np.amin(lst00)
    lst00_max = np.amax(lst00)
    if lst00_min<lst_min:
        lst_min =  lst00_min
    if lst00_max> lst_max:
        lst_max = lst00_max

    lst05_min = np.amin(lst05)
    lst05_max = np.amax(lst05)
    if lst05_min<lst_min:
        lst_min =  lst05_min
    if lst05_max> lst_max:
        lst_max = lst05_max

    lst10_min = np.amin(lst10)
    lst10_max = np.amax(lst10)
    if lst10_min<lst_min:
        lst_min =  lst10_min
    if lst10_max> lst_max:
        lst_max = lst10_max

    lst15_min = np.amin(lst15)
    lst15_max = np.amax(lst15)
    if lst15_min<lst_min:
        lst_min =  lst15_min
    if lst15_max> lst_max:
        lst_max = lst15_max

    #draw y surfaces
    fig2 = plt.figure(figsize=(20,3),constrained_layout=True)
    gs2 = GridSpec(1, 5, figure=fig2)
    jet  = plt.get_cmap('jet',256)
    psm1=0
    psm2=0
    psm3=0
    psm4=0
    psm5=0


    ax1 = fig2.add_subplot(gs2[0,0])
    psm1 = ax1.pcolormesh(lst95, cmap=jet, rasterized=True, vmin=lst_min, vmax=lst_max)

    ax2 = fig2.add_subplot(gs2[0,1])
    psm2 = ax2.pcolormesh(lst00, cmap=jet, rasterized=True, vmin=lst_min, vmax=lst_max)

    ax3 = fig2.add_subplot(gs2[0,2])
    psm3 = ax3.pcolormesh(lst05, cmap=jet, rasterized=True, vmin=lst_min, vmax=lst_max)

    ax4 = fig2.add_subplot(gs2[0,3])
    psm4 = ax4.pcolormesh(lst10, cmap=jet, rasterized=True, vmin=lst_min, vmax=lst_max)

    ax5 = fig2.add_subplot(gs2[0,4])
    psm5 = ax5.pcolormesh(lst15, cmap=jet, rasterized=True, vmin=lst_min, vmax=lst_max)
    fig2.colorbar(psm5, ax = ax5)
    plt.show()
    ######################################################################################################################


    #get all_coord from tif
    all_coords_list = []
    mask_r95gndvi = r95gndvi.dataset_mask()
    mask_r95gndvi = mask_r95gndvi[1:-1,1:]
    for row in range(mask_r95gndvi.shape[0]):
        for col in range (mask_r95gndvi.shape[1]):
            if(mask_r95gndvi[row,col]>0):
                all_coords_list.append(r95gndvi.xy(row+1,col+1))
    #############################
    ###3 random sample points from the simulation surfaces
    sample_num = 1000
    random_times = 100
    #repeated 100 times for predicted the mean results
    ols_predbeta_results = []
    ols_predy_results = []

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

    for rand_times in range(random_times):
                samples_ticks = np.ones(sample_num)
                samples_ticks = np.array_split(samples_ticks, 6, axis = 0)
                tick_sam_num_0 = len(samples_ticks[0])
                tick_sam_num_1 = len(samples_ticks[1])
                tick_sam_num_2 = len(samples_ticks[2])
                tick_sam_num_3 = len(samples_ticks[3])
                tick_sam_num_4 = len(samples_ticks[4])
                tick_sam_num_5 = len(samples_ticks[5])

                samples_ticks[0] = np.random.choice(range(353*353),tick_sam_num_0,replace=False)
                samples_ticks[1] = np.random.choice(range(353*353),tick_sam_num_1,replace=False)
                samples_ticks[2] = np.random.choice(range(353*353),tick_sam_num_2,replace=False)
                samples_ticks[3] = np.random.choice(range(353*353),tick_sam_num_3,replace=False)
                samples_ticks[4] = np.random.choice(range(353*353),tick_sam_num_4,replace=False)
                samples_ticks[5] = np.random.choice(range(353*353),tick_sam_num_5,replace=False)

                cal_coords_list = []
                cal_y_list = []
                cal_X_list = []

                pre_coords_list = []
                Pre_y_list = []
                Pre_X_list = []

                #95
                mask_ticks_95 = np.ones(353*353,dtype=bool).flatten()
                mask_ticks_95[samples_ticks[0]] = False
                cal_y_tick = lst95.flatten()[~mask_ticks_95]
                cal_y_list.append(cal_y_tick.reshape((-1,1)))
                b95gndvi_f_x2 = b95gndvi_f.flatten()[~mask_ticks_95].reshape((-1,1))
                b95gpnbi_f_x1 = b95gpnbi_f.flatten()[~mask_ticks_95].reshape((-1,1))
                b95moran_f_x3 = b95moran_f.flatten()[~mask_ticks_95].reshape((-1,1))
                cal_X_tick = np.concatenate((b95gpnbi_f_x1,b95gndvi_f_x2,b95moran_f_x3),axis=1)
                cal_X_list.append(cal_X_tick)
                all_coords_arr = np.asarray(all_coords_list)
                cal_coords_tick =  all_coords_arr[~mask_ticks_95]
                cal_coords_list.append(cal_coords_tick)

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
                mask_ticks_00[samples_ticks[1]] = False
                cal_y_tick = lst00.flatten()[~mask_ticks_00]
                cal_y_list.append(cal_y_tick.reshape((-1,1)))
                b00gndvi_f_x2 = b00gndvi_f.flatten()[~mask_ticks_00].reshape((-1,1))
                b00gpnbi_f_x1 = b00gpnbi_f.flatten()[~mask_ticks_00].reshape((-1,1))
                b00moran_f_x3 = b00moran_f.flatten()[~mask_ticks_00].reshape((-1,1))
                cal_X_tick = np.concatenate((b00gpnbi_f_x1,b00gndvi_f_x2,b00moran_f_x3),axis=1)
                cal_X_list.append(cal_X_tick)
                all_coords_arr = np.asarray(all_coords_list)
                cal_coords_tick =  all_coords_arr[~mask_ticks_00]
                cal_coords_list.append(cal_coords_tick)

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
                mask_ticks_05[samples_ticks[2]] = False
                cal_y_tick = lst05.flatten()[~mask_ticks_05]
                cal_y_list.append(cal_y_tick.reshape((-1,1)))
                b05gndvi_f_x2 = b05gndvi_f.flatten()[~mask_ticks_05].reshape((-1,1))
                b05gpnbi_f_x1 = b05gpnbi_f.flatten()[~mask_ticks_05].reshape((-1,1))
                b05moran_f_x3 = b05moran_f.flatten()[~mask_ticks_05].reshape((-1,1))
                cal_X_tick = np.concatenate((b05gpnbi_f_x1 ,b05gndvi_f_x2,b05moran_f_x3),axis=1)
                cal_X_list.append(cal_X_tick)
                all_coords_arr = np.asarray(all_coords_list)
                cal_coords_tick =  all_coords_arr[~mask_ticks_05]
                cal_coords_list.append(cal_coords_tick)

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
                mask_ticks_10[samples_ticks[3]] = False
                cal_y_tick = lst10.flatten()[~mask_ticks_10]
                cal_y_list.append(cal_y_tick.reshape((-1,1)))
                b10gndvi_f_x2 = b10gndvi_f.flatten()[~mask_ticks_10].reshape((-1,1))
                b10gpnbi_f_x1 = b10gpnbi_f.flatten()[~mask_ticks_10].reshape((-1,1))
                b10moran_f_x3 = b10moran_f.flatten()[~mask_ticks_10].reshape((-1,1))
                cal_X_tick = np.concatenate((b10gpnbi_f_x1,b10gndvi_f_x2,b10moran_f_x3),axis=1)
                cal_X_list.append(cal_X_tick)
                all_coords_arr = np.asarray(all_coords_list)
                cal_coords_tick =  all_coords_arr[~mask_ticks_10]
                cal_coords_list.append(cal_coords_tick)

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
                mask_ticks_15[samples_ticks[4]] = False
                cal_y_tick = lst15.flatten()[~mask_ticks_15]
                cal_y_list.append(cal_y_tick.reshape((-1,1)))
                b15gndvi_f_x2 = b15gndvi_f.flatten()[~mask_ticks_15].reshape((-1,1))
                b15gpnbi_f_x1 = b15gpnbi_f.flatten()[~mask_ticks_15].reshape((-1,1))
                b15moran_f_x3 = b15moran_f.flatten()[~mask_ticks_15].reshape((-1,1))
                cal_X_tick = np.concatenate((b15gpnbi_f_x1, b15gndvi_f_x2,b15moran_f_x3),axis=1)
                cal_X_list.append(cal_X_tick)
                all_coords_arr = np.asarray(all_coords_list)
                cal_coords_tick =  all_coords_arr[~mask_ticks_15]
                cal_coords_list.append(cal_coords_tick)

                pred_coords_tick = all_coords_arr[mask_ticks_15]
                pre_coords_list.append(pred_coords_tick)
                pred_y_tick =  lst15.flatten()[mask_ticks_15]
                Pre_y_list.append(pred_y_tick.reshape((-1,1)))
                b15gndvi_p_x2 = b15gndvi_f.flatten()[mask_ticks_15].reshape((-1,1))
                b15gpnbi_p_x1 = b15gpnbi_f.flatten()[mask_ticks_15].reshape((-1,1))
                b15moran_p_x3 = b15moran_f.flatten()[mask_ticks_15].reshape((-1,1))
                pre_X_tick = np.concatenate((b15gpnbi_p_x1,b15gndvi_p_x2,b15moran_p_x3),axis=1)
                Pre_X_list.append(pre_X_tick)
                #############################
                #4. test the prediction of beta surfaces
                #OLS
                ols_X = cal_X_list[-1]
                ols_X = np.concatenate((np.ones((ols_X.shape[0], 1)), ols_X),axis=1)
                ols_result = GLM(cal_y_list[-1],ols_X,constant=False,family=Gaussian()).fit()
                ols_predbeta_results.append(ols_result.params)
                list_ols_aic.append(ols_result.aic)
                #OLS
                #gwr
                gwr_selector = Sel_BW(cal_coords_list[-1], cal_y_list[-1], cal_X_list[-1], spherical = False)
                gwr_bw= gwr_selector.search(bw_min=2)
                gwr_model = GWR(cal_coords_list[-1], cal_y_list[-1], cal_X_list[-1], gwr_bw,spherical = False)
                gwr_results = gwr_model.fit()
                print(gwr_results.summary())
                gwr_scale = gwr_results.scale
                gwr_residuals = gwr_results.resid_response
                list_gwr_aicc.append(gwr_results.aicc)

                #stwr
                stwr_selector_ = Sel_Spt_BW(cal_coords_list, cal_y_list, cal_X_list,time_dif ,spherical = False)
                #F-STWR
                optalpha,optsita,opt_btticks,opt_gwr_bw0 = stwr_selector_.search(nproc = 12)
                stwr_model = STWR(cal_coords_list,cal_y_list,cal_X_list,time_dif,optsita,opt_gwr_bw0,tick_nums=opt_btticks,alpha =optalpha,spherical = False,recorded=1)#recorded = True)
                #F-STWR
                stwr_results = stwr_model.fit()
                print(stwr_results.summary())
                stwr_scale = stwr_results.scale
                stwr_residuals = stwr_results.resid_response
                list_stwr_aicc.append(stwr_results.aicc)

                ###########################################
                #5compare different methods  compare predicted Beta surfaces of latest time stages
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
                       predPointList = [pre_coords_list[-1]]
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
                        spl_coords_stwr = np.array_split(pre_coords_list[-1], spl_parts_stwr, axis = 0)
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
                        #row_c,col_c= r95gndvi.index(pre_coords_list[-1][j,0]+1,pre_coords_list[-1][j,1]+1)
                        row_c,col_c= r95gndvi.index(pre_coords_list[-1][j,0],pre_coords_list[-1][j,1])
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
                        row_c,col_c= r95gndvi.index(pre_coords_list[-1][j,0],pre_coords_list[-1][j,1])
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
                ###############OLS prediction############################
                prey_ols = ols_result.params[0]*np.ones_like(listgpnbi_f[-1]) + ols_result.params[1] *  listgpnbi_f[-1] + ols_result.params[2] * listgndvi_f[-1] + ols_result.params[3] * listmoran_f[-1]
                ols_predy_results.append(prey_ols)
                ###############OLS prediction############################
    #OLS Results#
    ols_betas = np.asarray(ols_predbeta_results)
    ols_betas_means = np.mean(ols_betas, axis=0)
    ols_betas_std = np.std(ols_betas, axis=0)

    ols_predy_arr = np.asarray(ols_predy_results)
    ols_predy_means = np.mean(ols_predy_arr, axis=0)
    ols_predy_std = np.std(ols_predy_arr, axis=0)
    #GWR Results#
    gwr_predbeta0_arr = np.asarray(gwr_predbeta0_results)
    gwr_predbeta0_means = np.mean(gwr_predbeta0_arr, axis=0)
    gwr_predbeta0_std = np.std(gwr_predbeta0_arr, axis=0)
    gwr_predbeta1_arr = np.asarray(gwr_predbeta1_results)
    gwr_predbeta1_means = np.mean(gwr_predbeta1_arr, axis=0)
    gwr_predbeta1_std = np.std(gwr_predbeta1_arr, axis=0)
    gwr_predbeta2_arr = np.asarray(gwr_predbeta2_results)
    gwr_predbeta2_means = np.mean(gwr_predbeta2_arr, axis=0)
    gwr_predbeta2_std = np.std(gwr_predbeta2_arr, axis=0)
    gwr_predbeta3_arr = np.asarray(gwr_predbeta3_results)
    gwr_predbeta3_means = np.mean(gwr_predbeta3_arr, axis=0)
    gwr_predbeta3_std = np.std(gwr_predbeta3_arr, axis=0)
    gwr_predy_arr = np.asarray(gwr_predy_results)
    gwr_predy_means = np.mean(gwr_predy_arr, axis=0)
    gwr_predy_std = np.std(gwr_predy_arr, axis=0)

    #STWR Results#
    stwr_predbeta0_arr = np.asarray(stwr_predbeta0_results)
    stwr_predbeta0_means = np.mean(stwr_predbeta0_arr, axis=0)
    stwr_predbeta0_std = np.std(stwr_predbeta0_arr,axis=0)
    stwr_predbeta1_arr = np.asarray(stwr_predbeta1_results)
    stwr_predbeta1_means = np.mean(stwr_predbeta1_arr, axis=0)
    stwr_predbeta1_std = np.std(stwr_predbeta1_arr,axis=0)
    stwr_predbeta2_arr = np.asarray(stwr_predbeta2_results)
    stwr_predbeta2_means = np.mean(stwr_predbeta2_arr, axis=0)
    stwr_predbeta2_std = np.std(stwr_predbeta2_arr,axis=0)
    stwr_predbeta3_arr = np.asarray(stwr_predbeta3_results)
    stwr_predbeta3_means = np.mean(stwr_predbeta3_arr, axis=0)
    stwr_predbeta3_std = np.std(stwr_predbeta3_arr,axis=0)
    stwr_predy_arr = np.asarray(stwr_predy_results)
    stwr_predy_means = np.mean(stwr_predy_arr, axis=0)
    stwr_predy_std = np.std(stwr_predy_arr,axis=0)

    #######save results to txt files
    with open('D:/STWR/Data-Run/results/case3/100_test_olsbeta_1000_hc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ols_betas.shape))
        np.savetxt(outfile, ols_betas, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/100_test_olspredy_1000_hc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ols_predy_arr.shape))
        for data_slice in ols_predy_arr:
            np.savetxt(outfile, data_slice, fmt='%-7.7f')
            outfile.write('# New slice\n')

    with open('D:/STWR/Data-Run/results/case3/100_test_gwr_predbeta0arr_1000_hc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(gwr_predbeta0_arr.shape))
        for data_slice in gwr_predbeta0_arr:
            np.savetxt(outfile, data_slice, fmt='%-7.7f')
            outfile.write('# New slice\n')
    with open('D:/STWR/Data-Run/results/case3/100_test_gwr_predbeta1arr_1000_hc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(gwr_predbeta1_arr.shape))
        for data_slice in gwr_predbeta1_arr:
            np.savetxt(outfile, data_slice, fmt='%-7.7f')
            outfile.write('# New slice\n')
    with open('D:/STWR/Data-Run/results/case3/100_test_gwr_predbeta2arr_1000_hc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(gwr_predbeta2_arr.shape))
        for data_slice in gwr_predbeta2_arr:
            np.savetxt(outfile, data_slice, fmt='%-7.7f')
            outfile.write('# New slice\n')
    with open('D:/STWR/Data-Run/results/case3/100_test_gwr_predbeta3arr_1000_hc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(gwr_predbeta3_arr.shape))
        for data_slice in gwr_predbeta3_arr:
            np.savetxt(outfile, data_slice, fmt='%-7.7f')
            outfile.write('# New slice\n')
    with open('D:/STWR/Data-Run/results/case3/100_test_gwr_predy_arr_1000_hc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(gwr_predy_arr.shape))
        for data_slice in gwr_predy_arr:
            np.savetxt(outfile, data_slice, fmt='%-7.7f')
            outfile.write('# New slice\n')
    #STWR
    with open('D:/STWR/Data-Run/results/case3/100_test_stwr_predbeta0arr_1000_hc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(stwr_predbeta0_arr.shape))
        for data_slice in stwr_predbeta0_arr:
            np.savetxt(outfile, data_slice, fmt='%-7.7f')
            outfile.write('# New slice\n')
    with open('D:/STWR/Data-Run/results/case3/100_test_stwr_predbeta1arr_1000_hc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(stwr_predbeta1_arr.shape))
        for data_slice in stwr_predbeta1_arr:
            np.savetxt(outfile, data_slice, fmt='%-7.7f')
            outfile.write('# New slice\n')
    with open('D:/STWR/Data-Run/results/case3/100_test_stwr_predbeta2arr_1000_hc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(stwr_predbeta2_arr.shape))
        for data_slice in stwr_predbeta2_arr:
            np.savetxt(outfile, data_slice, fmt='%-7.7f')
            outfile.write('# New slice\n')
    with open('D:/STWR/Data-Run/results/case3/100_test_stwr_predbeta3arr_1000_hc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(stwr_predbeta3_arr.shape))
        for data_slice in stwr_predbeta3_arr:
            np.savetxt(outfile, data_slice, fmt='%-7.7f')
            outfile.write('# New slice\n')
    with open('D:/STWR/Data-Run/results/case3/100_test_stwr_predy_arr_1000_hc.txt', 'w') as outfile:
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
    vmin_ytrue =np.amin(lst15)
    vmax_ytrue =np.amax(lst15)

    ax_true  = fig_pred_cmp_stwr.add_subplot(gs_stwr[0,0])
    psm_true = ax_true.pcolormesh(lst15, cmap=jet, rasterized=True, vmin=vmin_ytrue, vmax=vmax_ytrue)
    ax_stwr  = fig_pred_cmp_stwr.add_subplot(gs_stwr[0,1])
    psm_stwr = ax_stwr.pcolormesh(stwr_predy_means, cmap=jet, rasterized=True, vmin=vmin_ytrue, vmax=vmax_ytrue)
    ax_gwr  =  fig_pred_cmp_stwr.add_subplot(gs_stwr[0,2])
    psm_gwr =  ax_gwr.pcolormesh(gwr_predy_means, cmap=jet, rasterized=True, vmin=vmin_ytrue, vmax=vmax_ytrue)
    ax_ols =  fig_pred_cmp_stwr.add_subplot(gs_stwr[0,3])
    psm_ols=  ax_ols.pcolormesh(ols_predy_means, cmap=jet, rasterized=True, vmin=vmin_ytrue, vmax=vmax_ytrue)

    fig_pred_cmp_stwr.colorbar(psm_true, ax = fig_pred_cmp_stwr.axes[3])
    plt.show()

    with open('D:/STWR/Data-Run/results/case3/stwr_predy_means.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(stwr_predy_means.shape))
        np.savetxt(outfile, stwr_predy_means, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/gwr_predy_means.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(gwr_predy_means.shape))
        np.savetxt(outfile, gwr_predy_means, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ols_predy_means.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ols_predy_means.shape))
        np.savetxt(outfile, ols_predy_means, fmt='%-7.7f')


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
    psm_true = ax_true.pcolormesh(lst15, cmap=jet, rasterized=True, vmin=vmin_ytrue, vmax=vmax_ytrue)
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
    y_self_err = np.abs(lst15-lst15)
    y_pre_stwr_err = np.abs(lst15 - stwr_predy_means)
    y_pre_gwr_err = np.abs(lst15 - gwr_predy_means)
    y_pre_ols_err = np.abs(lst15- ols_predy_means)

    y_pre_stwr_eee = stwr_predy_means -lst15
    y_pre_gwr_eee  =  gwr_predy_means -lst15
    y_pre_ols_eee  =  ols_predy_means -lst15


    with open('D:/STWR/Data-Run/results/case3/y_pre_stwr_eee.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(y_pre_stwr_eee.shape))
        np.savetxt(outfile, y_pre_stwr_eee, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/y_pre_gwr_eee.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(y_pre_gwr_eee.shape))
        np.savetxt(outfile, y_pre_gwr_eee, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/y_pre_ols_eee.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ols_predy_means.shape))
        np.savetxt(outfile, y_pre_ols_eee, fmt='%-7.7f')


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

    with open('D:/STWR/Data-Run/results/case3/y_self_err.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(y_self_err.shape))
        np.savetxt(outfile, y_self_err, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/y_pre_stwr_err.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(y_pre_stwr_err.shape))
        np.savetxt(outfile, y_pre_stwr_err, fmt='%-7.7f')

    ####Fanchao 20220119
    with open('D:/STWR/Data-Run/results/case3/y_pre_gwr_err.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(y_pre_gwr_err.shape))
        np.savetxt(outfile, y_pre_gwr_err, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/y_pre_ols_eee.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(y_pre_ols_eee.shape))
        np.savetxt(outfile, y_pre_ols_eee, fmt='%-7.7f')


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

    with open('D:/STWR/Data-Run/results/case3/stwr_predbeta0_means.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(stwr_predbeta0_means.shape))
        np.savetxt(outfile, stwr_predbeta0_means, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/stwr_predbeta1_means.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(stwr_predbeta1_means.shape))
        np.savetxt(outfile, stwr_predbeta1_means, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/stwr_predbeta2_means.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(stwr_predbeta2_means.shape))
        np.savetxt(outfile, stwr_predbeta2_means, fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/stwr_predbeta3_means.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(stwr_predbeta3_means.shape))
        np.savetxt(outfile, stwr_predbeta3_means, fmt='%-7.7f')

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

    with open('D:/STWR/Data-Run/results/case3/gwr_predbeta0_means.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(gwr_predbeta0_means.shape))
        np.savetxt(outfile, gwr_predbeta0_means, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/gwr_predbeta1_means.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(gwr_predbeta1_means.shape))
        np.savetxt(outfile, gwr_predbeta1_means, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/gwr_predbeta2_means.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(gwr_predbeta2_means.shape))
        np.savetxt(outfile, gwr_predbeta2_means, fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/gwr_predbeta3_means.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(gwr_predbeta3_means.shape))
        np.savetxt(outfile, gwr_predbeta3_means, fmt='%-7.7f')

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


    with open('D:/STWR/Data-Run/results/case3/ols_betas_means0.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format((ols_betas_means[0]*np.ones_like(beta0)).shape))
        np.savetxt(outfile, ols_betas_means[0]*np.ones_like(beta0), fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ols_betas_means1.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format( (ols_betas_means[1]*np.ones_like(beta1)).shape))
        np.savetxt(outfile,  ols_betas_means[1]*np.ones_like(beta1), fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ols_betas_means2.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(( ols_betas_means[2]*np.ones_like(beta2)).shape))
        np.savetxt(outfile,  ols_betas_means[2]*np.ones_like(beta2), fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/ols_betas_means3.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format( (ols_betas_means[3]*np.ones_like(beta3)).shape))
        np.savetxt(outfile,  ols_betas_means[3]*np.ones_like(beta3), fmt='%-7.7f')



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

    with open('D:/STWR/Data-Run/results/case3/ppstwr_abm_err_beta0.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppstwr_err_beta0.shape))
        np.savetxt(outfile, ppstwr_err_beta0, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ppstwr_abm_err_beta1.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppstwr_err_beta1.shape))
        np.savetxt(outfile, ppstwr_err_beta1, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ppstwr_abm_err_beta2.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppstwr_err_beta2.shape))
        np.savetxt(outfile, ppstwr_err_beta2, fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/ppstwr_amb_err_beta3.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppstwr_err_beta3.shape))
        np.savetxt(outfile, ppstwr_err_beta3, fmt='%-7.7f')


    with open('D:/STWR/Data-Run/results/case3/ppgwr_abm_err_beta0.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppgwr_err_beta0.shape))
        np.savetxt(outfile, ppgwr_err_beta0, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ppgwr_abm_err_beta1.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppgwr_err_beta1.shape))
        np.savetxt(outfile, ppgwr_err_beta1, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ppgwr_abm_err_beta2.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppgwr_err_beta2.shape))
        np.savetxt(outfile, ppgwr_err_beta2, fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/ppgwr_abm_err_beta3.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppgwr_err_beta3.shape))
        np.savetxt(outfile, ppgwr_err_beta3, fmt='%-7.7f')


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

    with open('D:/STWR/Data-Run/results/case3/ols_abm_err_beta0.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(np.abs(beta0 - ols_result.params[0]*np.ones_like(b10gndvi_f)).shape))
        np.savetxt(outfile, np.abs(beta0 - ols_result.params[0]*np.ones_like(b10gndvi_f)), fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ols_abm_err_beta1.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(np.abs(beta1 - ols_result.params[1]*np.ones_like(b10gndvi_f)).shape))
        np.savetxt(outfile, np.abs(beta1 - ols_result.params[1]*np.ones_like(b10gndvi_f)), fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ols_abm_err_beta2.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(np.abs(beta1 - ols_result.params[2]*np.ones_like(b10gndvi_f)).shape))
        np.savetxt(outfile, np.abs(beta1 - ols_result.params[2]*np.ones_like(b10gndvi_f)), fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ols_abm_err_beta3.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(np.abs(beta1 - ols_result.params[3]*np.ones_like(b10gndvi_f)).shape))
        np.savetxt(outfile, np.abs(beta1 - ols_result.params[3]*np.ones_like(b10gndvi_f)), fmt='%-7.7f')


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


    with open('D:/STWR/Data-Run/results/case3/stwr_params0_mean_err.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppstwr_eee_beta0.shape))
        np.savetxt(outfile, ppstwr_eee_beta0, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/stwr_params1_mean_err.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppstwr_eee_beta1.shape))
        np.savetxt(outfile, ppstwr_eee_beta1, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/stwr_params2_mean_err.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppstwr_eee_beta2.shape))
        np.savetxt(outfile, ppstwr_eee_beta2, fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/stwr_params3_mean_err.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppstwr_eee_beta3.shape))
        np.savetxt(outfile, ppstwr_eee_beta3, fmt='%-7.7f')

    #FanChao20220119
    with open('D:/STWR/Data-Run/results/case3/gwr_params0_mean_err.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppgwr_eee_beta0.shape))
        np.savetxt(outfile, ppgwr_eee_beta0, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/gwr_params1_mean_err.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppgwr_eee_beta1.shape))
        np.savetxt(outfile, ppgwr_eee_beta1, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/gwr_params2_mean_err.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppgwr_eee_beta2.shape))
        np.savetxt(outfile, ppgwr_eee_beta2, fmt='%-7.7f')
    with open('D:/STWR/Data-Run/results/case3/gwr_params3_mean_err.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(ppgwr_eee_beta3.shape))
        np.savetxt(outfile, ppgwr_eee_beta3, fmt='%-7.7f')



    with open('D:/STWR/Data-Run/results/case3/ols_merr_beta0.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format((ols_result.params[0]*np.ones_like(b10gndvi_f) - beta0).shape))
        np.savetxt(outfile, (ols_result.params[0]*np.ones_like(b10gndvi_f) - beta0), fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ols_merr_beta1.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format((ols_result.params[1]*np.ones_like(b10gndvi_f) - beta1).shape))
        np.savetxt(outfile, (ols_result.params[1]*np.ones_like(b10gndvi_f) - beta1), fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ols_merr_beta2.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format((ols_result.params[2]*np.ones_like(b10gndvi_f) - beta2).shape))
        np.savetxt(outfile, (ols_result.params[2]*np.ones_like(b10gndvi_f) - beta0), fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/ols_merr_beta3.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format((ols_result.params[3]*np.ones_like(b10gndvi_f) - beta3).shape))
        np.savetxt(outfile, (ols_result.params[3]*np.ones_like(b10gndvi_f) - beta3), fmt='%-7.7f')

    #comparison of STWR,GWR,and OLS.
    sum_err2_stwr = np.zeros_like(lst15)
    for idx_py_stwr in stwr_predy_arr:
        sum_err2_stwr +=   (idx_py_stwr-lst15)*(idx_py_stwr-lst15)
    rmse_stwr_predy = np.sqrt(sum_err2_stwr/random_times)

    sum_err2_gwr = np.zeros_like(lst15)
    for idx_py_gwr in gwr_predy_arr:
        sum_err2_gwr +=   (idx_py_gwr-lst15)*(idx_py_gwr-lst15)
    rmse_gwr_predy = np.sqrt(sum_err2_gwr/random_times)

    sum_err2_ols = np.zeros_like(lst15)
    for idx_py_ols in ols_predy_arr :
        sum_err2_ols +=   (idx_py_ols-lst15)*(idx_py_ols-lst15)
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


    with open('D:/STWR/Data-Run/results/case3/rmse_stwr_predy.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_stwr_predy.shape))
        np.savetxt(outfile, rmse_stwr_predy, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/rmse_gwr_predy.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_gwr_predy.shape))
        np.savetxt(outfile, rmse_gwr_predy, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/rmse_ols_predy.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_ols_predy.shape))
        np.savetxt(outfile, rmse_ols_predy, fmt='%-7.7f')


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


    with open('D:/STWR/Data-Run/results/case3/rmse_stwr_beta0.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_stwr_beta0.shape))
        np.savetxt(outfile, rmse_stwr_beta0, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/rmse_gwr_beta0.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_gwr_beta0.shape))
        np.savetxt(outfile, rmse_gwr_beta0, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/rmse_ols_beta0.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_ols_beta0.shape))
        np.savetxt(outfile, rmse_ols_beta0, fmt='%-7.7f')


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

    with open('D:/STWR/Data-Run/results/case3/rmse_stwr_beta1.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_stwr_beta1.shape))
        np.savetxt(outfile, rmse_stwr_beta1, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/rmse_gwr_beta1.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_gwr_beta1.shape))
        np.savetxt(outfile, rmse_gwr_beta1, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/rmse_ols_beta2.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_ols_beta1.shape))
        np.savetxt(outfile, rmse_ols_beta1, fmt='%-7.7f')

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


    with open('D:/STWR/Data-Run/results/case3/rmse_stwr_beta2.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_stwr_beta2.shape))
        np.savetxt(outfile, rmse_stwr_beta2, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/rmse_gwr_beta2.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_gwr_beta2.shape))
        np.savetxt(outfile, rmse_gwr_beta2, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/rmse_ols_beta2.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_ols_beta2.shape))
        np.savetxt(outfile, rmse_ols_beta2, fmt='%-7.7f')



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

    with open('D:/STWR/Data-Run/results/case3/rmse_stwr_beta3.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_stwr_beta3.shape))
        np.savetxt(outfile, rmse_stwr_beta3, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/rmse_gwr_beta3.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_gwr_beta3.shape))
        np.savetxt(outfile, rmse_gwr_beta3, fmt='%-7.7f')

    with open('D:/STWR/Data-Run/results/case3/rmse_ols_beta3.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(rmse_ols_beta3.shape))
        np.savetxt(outfile, rmse_ols_beta3, fmt='%-7.7f')

    print("The Mean of AIC of OLS is :{}".format(mean(list_ols_aic)))
    print("The Mean of AIC of GWR is :{}".format(mean(list_gwr_aicc)))
    print("The Mean of AIC of STWR is :{}".format(mean(list_stwr_aicc)))





