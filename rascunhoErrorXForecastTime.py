import os, sys
import netCDF4 as nc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import pyresample
import math
import time
# matplotlib.use('Agg')
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.dates as mdates
import matplotlib.dates as dates
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
import skill_metrics as sm
from scipy.stats import gaussian_kde
from xarray import open_dataset
from bokeh.plotting import figure, output_file, show
from pylab import date2num, DateFormatter

from lib.errors import metrics

SL=13

f=nc.Dataset('ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc')
datm=f.variables['date_time'][:]
et=f.variables['cycletime'][:]
eft=f.variables['forecast_time'][:]
ensm=f.variables['ensmember'][:]
stations=f.variables['stationID'][:]
eatmp=f.variables['ecmwf_atmp'][:,:,:,:]
oatmp=f.variables['omega_atmp'][:,:,:]
erh=f.variables['ecmwf_rh'][:,:,:,:]
orh=f.variables['omega_rh'][:,:,:]
emsl=f.variables['ecmwf_msl'][:,:,:,:]
omsl=f.variables['omega_msl'][:,:,:]
ewspa=f.variables['ecmwf_wsp_AnemAinfer'][:,:,:,:]
owspa=f.variables['omega_wsp_AnemAinfer'][:,:,:]
ewdira=f.variables['ecmwf_wdir_AnemAinfer'][:,:,:,:]
owdira=f.variables['omega_wdir_AnemAinfer'][:,:,:]
ewspb=f.variables['ecmwf_wsp_AnemB'][:,:,:,:]
owspb=f.variables['omega_wsp_AnemB'][:,:,:]
ewspc=f.variables['ecmwf_wsp_AnemCsup'][:,:,:,:]
owspc=f.variables['omega_wsp_AnemCsup'][:,:,:]
ewdirc=f.variables['ecmwf_wdir_AnemCsup'][:,:,:,:]
owdirc=f.variables['omega_wdir_AnemCsup'][:,:,:]
f.close()

# Sellection of stations to plot
NvarAnh=['AnemA','AnemB','AnemC']
Nvaren=['AtmPressure','Temperature','RHumidity','WindSpeed','WindDir']

# Error versus Forecast time

hd = '      Bias             RMSE          N-Bias          N-RMSE          SCrmse            SI              HH              CC'
hd2 = ' Lines: ECMWFensembleMean: PressaoAtmos,Temperatura,Umidade,VelVenMedAnemA,DirVenMedAnemA,VelVenMedAnemB,VelVenMedAnemC,DirVenMedAnemA,'
hd3 = ' Lines: ECMWF(DailyMean)ensembleMean: PressaoAtmos,Temperatura,Umidade,VelVenMedAnemA,DirVenMedAnemA,VelVenMedAnemB,VelVenMedAnemC,DirVenMedAnemA,'
# loop through forecast lead time

observed_data = [omsl, oatmp, orh, owspa, owdira, owspb, owspc, owdirc]
ensemble_data = [emsl , eatmp, erh, ewspa, ewdira, ewspb, ewspc, ewdirc]

terrem=np.zeros((8,stations.shape[0],len(observed_data),int(eft[-1]/3600/24)),'f')*np.nan # array for error metrics, ensemble mean
terrms=np.zeros((8,stations.shape[0],len(observed_data),ensm.shape[0],int(eft[-1]/3600/24)),'f')*np.nan # array for error metrics, all members
terremd=np.zeros((8,stations.shape[0],len(observed_data),eft.shape[0]),'f')*np.nan # array for error metrics, ensemble mean from daily mean
for d in range(0,int(eft[-1]/3600/24)):

	ind = np.where((eft>=(d*24*3600)) & (eft<((d+1)*24*3600)))[0]

	for i in range(0,stations.shape[0]):

		for k in range(len(observed_data)):
			a = np.array(np.nanmean(ensemble_data[k][i,:,:,:], axis=0))[:,ind]
			b = observed_data[k][i,:,ind].T
			inde = np.where(a*b > -999)
			terrem[:,i,k,d] = metrics(a[inde[0],inde[1]],b[inde[0],inde[1]])

		for j in range(0,ensm.shape[0]):
			for k in range(len(observed_data)):
				a = ensemble_data[k][i,j,:,ind]
				b = observed_data[k][i,:,ind]
				inde=np.where(a*b> -999)
				terrms[:,i,k,j,d] = metrics(a[inde[0],inde[1]],b[inde[0],inde[1]])

		# Daily Mean
		# uniform_filter1d(gfs[ind],size=2),uniform_filter1d(obs[ind],size=2))

		# for k in range(len(observed_data)):
		# 	a = np.array(np.nanmean(ensemble_data[k][i,:,:,d], axis=0))
		# 	b = observed_data[k][i,:,d]
		# 	inde=np.where(a*b > -999)
		# 	terremd[:,i,k,d] = metrics(a[inde],b[inde])

		# # Table err metrics
		# fname = 'Table_ErrorECMWFemXForecastTime_'+stations[i]+'_D'+str(d+1).zfill(2)+'.txt'
		# ifile = open(fname,'w')
		# ifile.write('# '+hd2+' \n')
		# ifile.write('# '+hd+' \n')
		# np.savetxt(ifile,terrem[:,i,:,d].T,fmt="%12.3f",delimiter='	')
		# ifile.close()

		# fname = 'Table_ErrorECMWFemDMXForecastTime_'+stations[i]+'_D'+str(d+1).zfill(2)+'.txt'
		# ifile = open(fname,'w')
		# ifile.write('# '+hd3+' \n')
		# ifile.write('# '+hd+' \n')
		# np.savetxt(ifile,terremd[:,i,:,d].T,fmt="%12.3f",delimiter='	')
		# ifile.close()

	print('done day '+repr(d))


# Plot err X forecast time das 16 estacoes com mais dados

# Error metrics plot (error versus forecast time)
scolors=np.array(['c','royalblue','steelblue','blue','darkblue','cyan','dodgerblue','g','limegreen','lime',
	'chocolate','red','orangered','salmon','brown','saddlebrown'])
smarkers=np.array(['D','s','s','s','s','d','s','X','X','X','h','o','8','8','h','h'])
fnames=np.array(['301','302','304','305','306','307','308','502',
	'601','602','pba01','pdd01','ppa01','psa01','tbr01','tbr03'])
indst=np.array([0,1,3,4,5,6,7,9,10,11,18,19,20,21,22,23]).astype('int') # Index of stations with few problems
indstma=np.arange(0,10).astype('int') # Maranhao
indstpi=np.arange(10,16).astype('int') # Piaui
nerrm=['bias','RMSE','NBias','NRMSE','SCrmse','SI','HH','CC']
Nvar=[Nvaren[0],Nvaren[1],Nvaren[2],Nvaren[3]+NvarAnh[0],Nvaren[4]+NvarAnh[0],Nvaren[3]+NvarAnh[1],Nvaren[3]+NvarAnh[2],Nvaren[4]+NvarAnh[2]]
for i in range(0,np.size(Nvar)):
	for l in range(0,8):
		eft_i = list(range(0,int(eft[-1]/3600/24)))
		# Maranhao
		fig1 = plt.figure(1,figsize=(7,6)); ax = fig1.add_subplot(111)
		for j in range(0,np.size(stations[indst][indstma])):
			ax.plot(eft_i,gaussian_filter(terrem[l,indst[indstma][j],i,:], 1.), color=scolors[indstma[j]],label=fnames[indstma[j]], linewidth=2.,zorder=2)
			for k in range(0,ensm.shape[0]):
				ax.plot(eft_i,gaussian_filter(terrms[l,indst[indstma][j],i,k,:], 1.), color=scolors[indstma[j]],linewidth=0.1,zorder=1)

		plt.legend()
		ax.set_xlabel('Forecast Time (Days)',size=SL); ax.set_ylabel(nerrm[l]+' '+Nvar[i],size=SL)
		plt.tight_layout();plt.axis('tight')
		plt.grid(c='k', ls='-', alpha=0.3)
		plt.xticks(np.array([1,5,10,15,20,30,40])); plt.xlim(xmin = 0.9, xmax = eft[-1]+0.1)
		plt.savefig('ErrXForecastTime_Maranhao_'+nerrm[l]+'_'+Nvar[i]+'_leadsComparison.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
		plt.close(fig1)

		# Piaui
		fig1 = plt.figure(1,figsize=(7,6)); ax = fig1.add_subplot(111)
		for j in range(0,np.size(stations[indst][indstpi])):
			ax.plot(eft_i,gaussian_filter(terrem[l,indst[indstpi][j],i,:], 1.), color=scolors[indstpi[j]],label=fnames[indstpi[j]], linewidth=2.,zorder=2)
			for k in range(0,ensm.shape[0]):
				ax.plot(eft_i,gaussian_filter(terrms[l,indst[indstpi][j],i,k,:], 1.), color=scolors[indstpi[j]],linewidth=0.1,zorder=1)

		plt.legend()
		ax.set_xlabel('Forecast Time (Days)',size=SL); ax.set_ylabel(nerrm[l]+' '+Nvar[i],size=SL)
		plt.tight_layout();plt.axis('tight')
		plt.grid(c='k', ls='-', alpha=0.3)
		plt.xticks(np.array([1,5,10,15,20,30,40])); plt.xlim(xmin = 0.9, xmax = eft[-1]+0.1)
		plt.savefig('ErrXForecastTime_Piaui__'+nerrm[l]+'_'+Nvar[i]+'_leadsComparison.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
		plt.close(fig1)


	# Table Mean error All stations
	fname = 'Table2_MeanErrorECMWFemXForecastTime_'+Nvar[i]+'.txt'
	ifile = open(fname,'w')
	ifile.write('#   EnsembleMean \n')
	ifile.write('# Lines: '+hd+' \n')
	np.savetxt(ifile,np.atleast_2d(eft),fmt="%3i",delimiter=' ')
	ifile.write('# \n')
	for l in range(0,8):
		np.savetxt(ifile,np.atleast_2d(np.nanmean(terrem[l,indst,i,:],axis=0)),fmt="%12.5f",delimiter='	')

	ifile.close()


