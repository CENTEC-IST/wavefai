import os, sys
import netCDF4 as nc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import pyresample
import math
import time
import xarray
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

xdata = xarray.open_dataset('ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc')

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

# Error versus Forecast time

hd = '      Bias             RMSE          N-Bias          N-RMSE          SCrmse            SI              HH              CC'
hd2 = ' Lines: ECMWFensembleMean: PressaoAtmos,Temperatura,Umidade,VelVenMedAnemA,DirVenMedAnemA,VelVenMedAnemB,VelVenMedAnemC,DirVenMedAnemA,'
hd3 = ' Lines: ECMWF(DailyMean)ensembleMean: PressaoAtmos,Temperatura,Umidade,VelVenMedAnemA,DirVenMedAnemA,VelVenMedAnemB,VelVenMedAnemC,DirVenMedAnemA,'

# Variables Data
observed_data = [omsl, oatmp, orh, owspa, owdira, owspb, owspc, owdirc]
ensemble_data = [emsl , eatmp, erh, ewspa, ewdira, ewspb, ewspc, ewdirc]

# Variables names
Nvar = ['AtmPressure', 'Temperature', 'RHumidity', 'WindSpeed_AnemA', 'WindDir_AnemA', 'WindSpeed_AnemB', 'WindSpeed_AnemC', 'WindDir_AnemC']

eft_i = np.asarray(xdata.forecast_time.isel({'forecast_time':slice(0,180,4)})) # choose only the seconds that represent each day

terrem=np.zeros((8, stations.shape[0], len(observed_data), eft_i.size), 'f')*np.nan # array for error metrics, ensemble mean
terrms=np.zeros((8, stations.shape[0], len(observed_data), ensm.shape[0], eft_i.size), 'f')*np.nan # array for error metrics, all members
terremd=np.zeros((8, stations.shape[0], len(observed_data), eft.shape[0]), 'f')*np.nan # array for error metrics, ensemble mean from daily mean

for day in range(0,eft_i.size):

	print(f"\rProcessing day [{day}/{eft_i.size}]", end='')

	day_ftime = np.where((eft >= (day*24*3600)) & (eft < ((day+1)*24*3600)))[0] # forecast seconds for this day

	for station in range(0, xdata.stations.size): # iterate over each station

		for variable in range(len(observed_data)):
			a = np.array(np.nanmean(ensemble_data[variable][station,:,:,:], axis=0))[:,day_ftime]
			b = observed_data[variable][station,:,day_ftime].T
			inde = np.where(a*b > -999)
			terrem[:,station,variable,day] = metrics(a[inde[0],inde[1]],b[inde[0],inde[1]])

		for ens in range(0,ensm.size):
			for variable in range(len(observed_data)):
				a = ensemble_data[variable][station,ens,:,day_ftime]
				b = observed_data[variable][station,:,day_ftime]
				inde=np.where(a*b> -999)
				terrms[:,station,variable,ens,day] = metrics(a[inde[0],inde[1]],b[inde[0],inde[1]])

		# Daily Mean
		# uniform_filter1d(gfs[day_ftime],size=2),uniform_filter1d(obs[day_ftime],size=2))

		# for variable in range(len(observed_data)):
		# 	a = np.array(np.nanmean(ensemble_data[variable][station,:,:,day], axis=0))
		# 	b = observed_data[variable][station,:,day]
		# 	inde=np.where(a*b > -999)
		# 	terremd[:,station,variable,day] = metrics(a[inde],b[inde])

		# # Table err metrics
		# fname = 'Table_ErrorECMWFemXForecastTime_'+stations[station]+'_D'+str(day+1).zfill(2)+'.txt'
		# ifile = open(fname,'w')
		# ifile.write('# '+hd2+' \n')
		# ifile.write('# '+hd+' \n')
		# np.savetxt(ifile,terrem[:,station,:,day].T,fmt="%12.3f",delimiter='	')
		# ifile.close()

		# fname = 'Table_ErrorECMWFemDMXForecastTime_'+stations[station]+'_D'+str(day+1).zfill(2)+'.txt'
		# ifile = open(fname,'w')
		# ifile.write('# '+hd3+' \n')
		# ifile.write('# '+hd+' \n')
		# np.savetxt(ifile,terremd[:,station,:,day].T,fmt="%12.3f",delimiter='	')
		# ifile.close()


# Plot err X forecast time das 16 estacoes com mais dados

# Error metrics plot (error versus forecast time)
scolors = np.array(['c','royalblue','steelblue','blue','darkblue','cyan','dodgerblue','g','limegreen','lime',
	'chocolate','red','orangered','salmon','brown','saddlebrown'])
smarkers = np.array(['D','s','s','s','s','d','s','X','X','X','h','o','8','8','h','h'])
fnames = np.array(['301','302','304','305','306','307','308','502',
	'601','602','pba01','pdd01','ppa01','psa01','tbr01','tbr03'])
station_indexes = np.array([0,1,3,4,5,6,7,9,10,11,18,19,20,21,22,23]).astype('int') # Index of stations with few problems
stations_maranhao = np.arange(0,10).astype('int') # Maranhao
stations_piaui = np.arange(10,16).astype('int') # Piaui
error_type = ['bias','RMSE','NBias','NRMSE','SCrmse','SI','HH','CC']

for var in range(len(Nvar)):
	for err in range(len(error_type)):
		# Maranhao
		fig1 = plt.figure(1,figsize=(7,6)); ax = fig1.add_subplot(111)
		for station in range(0,np.size(stations[station_indexes][stations_maranhao])):
			ax.plot(eft_i,gaussian_filter(terrem[err,station_indexes[stations_maranhao][station],var,:], 1.), color=scolors[stations_maranhao[station]],label=fnames[stations_maranhao[station]], linewidth=2.,zorder=2)
			for k in range(0,ensm.shape[0]):
				ax.plot(eft_i,gaussian_filter(terrms[err,station_indexes[stations_maranhao][station],var,k,:], 1.), color=scolors[stations_maranhao[station]],linewidth=0.1,zorder=1)

		plt.legend()
		ax.set_xlabel('Forecast Time (Days)',size=SL); ax.set_ylabel(error_type[err]+' '+Nvar[var],size=SL)
		plt.tight_layout();plt.axis('tight')
		plt.grid(c='k', ls='-', alpha=0.3)
		plt.xticks(np.array([1,5,10,15,20,30,40])); plt.xlim(xmin = 0.9, xmax = eft[-1]+0.1)
		plt.savefig('ErrXForecastTime_Maranhao_'+error_type[err]+'_'+Nvar[var]+'_leadsComparison.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
		plt.close(fig1)

		# Piaui
		fig1 = plt.figure(1,figsize=(7,6)); ax = fig1.add_subplot(111)
		for station in range(0,np.size(stations[station_indexes][stations_piaui])):
			ax.plot(eft_i,gaussian_filter(terrem[err,station_indexes[stations_piaui][station],var,:], 1.), color=scolors[stations_piaui[station]],label=fnames[stations_piaui[station]], linewidth=2.,zorder=2)
			for k in range(0,ensm.shape[0]):
				ax.plot(eft_i,gaussian_filter(terrms[err,station_indexes[stations_piaui][station],var,k,:], 1.), color=scolors[stations_piaui[station]],linewidth=0.1,zorder=1)

		plt.legend()
		ax.set_xlabel('Forecast Time (Days)',size=SL); ax.set_ylabel(error_type[err]+' '+Nvar[var],size=SL)
		plt.tight_layout();plt.axis('tight')
		plt.grid(c='k', ls='-', alpha=0.3)
		plt.xticks(np.array([1,5,10,15,20,30,40])); plt.xlim(xmin = 0.9, xmax = eft[-1]+0.1)
		plt.savefig('ErrXForecastTime_Piaui__'+error_type[err]+'_'+Nvar[var]+'_leadsComparison.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
		plt.close(fig1)


	# Table Mean error All stations
	# fname = 'Table2_MeanErrorECMWFemXForecastTime_'+Nvar[var]+'.txt'
	# ifile = open(fname,'w')
	# ifile.write('#   EnsembleMean \n')
	# ifile.write('# Lines: '+hd+' \n')
	# np.savetxt(ifile,np.atleast_2d(eft),fmt="%3i",delimiter=' ')
	# ifile.write('# \n')
	# for err in len(error_type):
	# 	np.savetxt(ifile,np.atleast_2d(np.nanmean(terrem[err,station_indexes,var,:],axis=0)),fmt="%12.5f",delimiter='	')

	# ifile.close()


