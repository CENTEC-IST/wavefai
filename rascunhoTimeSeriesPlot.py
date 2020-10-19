import os
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
import merr
sl=13

matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl)
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})

f=nc.Dataset('ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc')
datm=f.variables['date_time'][:]; et=f.variables['cycletime'][:]; eft=f.variables['forecast_time'][:]
ensm=f.variables['ensmember'][:]; blat=f.variables['latitude'][:]; blon=f.variables['longitude'][:]; stations=f.variables['stationID'][:]
eatmp=f.variables['ecmwf_atmp'][:,:,:,:]; oatmp=f.variables['omega_atmp'][:,:,:]
erh=f.variables['ecmwf_rh'][:,:,:,:]; orh=f.variables['omega_rh'][:,:,:]
emsl=f.variables['ecmwf_msl'][:,:,:,:]; omsl=f.variables['omega_msl'][:,:,:]
ewspa=f.variables['ecmwf_wsp_AnemAinfer'][:,:,:,:]; owspa=f.variables['omega_wsp_AnemAinfer'][:,:,:]
ewdira=f.variables['ecmwf_wdir_AnemAinfer'][:,:,:,:]; owdira=f.variables['omega_wdir_AnemAinfer'][:,:,:]
ewspb=f.variables['ecmwf_wsp_AnemB'][:,:,:,:]; owspb=f.variables['omega_wsp_AnemB'][:,:,:]
ewspc=f.variables['ecmwf_wsp_AnemCsup'][:,:,:,:]; owspc=f.variables['omega_wsp_AnemCsup'][:,:,:]
ewdirc=f.variables['ecmwf_wdir_AnemCsup'][:,:,:,:]; owdirc=f.variables['omega_wdir_AnemCsup'][:,:,:]
f.close(); del f
# Sellection of stations to plot
ssp=np.array([1,23]).astype('int')
Nvarpt=['PressaoAtmos','Temperatura','Umidade','VelVenMed','DirVenMed']; NvarAnh=['AnemA','AnemB','AnemC']
Nvaren=['AtmPressure','Temperature','RHumidity','WindSpeed','WindDir']


# TIME SERIES PLOT ==========================
fnames=['OBS','ECMWF_EM','ECMWF_m1']; scolors=['k','lightsalmon','brown']

i=1 # Station 'ma_dt302_v'
k=0 # Pressure k=0 Nvaren[k]
indi=161 # initial time selected
x=np.zeros(eft.shape[0],'f')*np.nan
for j in range(0,eft.shape[0]):
	x[j]=date2num(datetime.fromtimestamp(et[indi]+eft[j]))

fig, ax1 = plt.subplots(1,figsize=(16,4), sharex=True, sharey=True)
# Observation
ax1.plot(x,gaussian_filter(omsl[i,indi,:],1.),color=scolors[0],label=fnames[0],linewidth=2.,zorder=2)
# Ensemble Mean
ax1.plot(x,gaussian_filter(np.nanmean(emsl[i,:,indi,:],axis=0),1.),color=scolors[2],label=fnames[1],linewidth=3.,zorder=3)
# Ensemble members
ax1.plot(x,gaussian_filter(emsl[i,0,indi,:],1.),color=scolors[1],label=fnames[2],linewidth=0.5,zorder=1)
for j in range(1,ensm.shape[0]):
	ax1.plot(x,gaussian_filter(emsl[i,j,indi,:],1.),color=scolors[1],linewidth=0.5,zorder=1)

plt.legend(loc='upper left')
plt.xlabel("Time (day/month)",size=sl)
plt.ylabel(Nvaren[k],size=sl)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5)); ax1.xaxis.set_major_formatter( DateFormatter('%d/%m') )
fig.text(0.015,0.05,stations[i], va='center', rotation='horizontal',size=sl)
plt.grid(); plt.tight_layout();plt.axis('tight')
plt.xlim(xmin=x[0]-0.1,xmax=x[-1]+0.1)
plt.savefig('TimeSeries.'+str(indi).zfill(3)+'_'+stations[i]+'_'+Nvaren[k]+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.close(fig); del fig
# Loop
i1=154; i2=223
c=0
for indi in range(i1,i2+1):
	c=c+1
	x=np.zeros(eft.shape[0],'f')*np.nan
	for j in range(0,eft.shape[0]):
		x[j]=date2num(datetime.fromtimestamp(et[indi]+eft[j]))

	fig, ax1 = plt.subplots(1,figsize=(16,4), sharex=True, sharey=True)
	# Observation
	ax1.plot(x,gaussian_filter(omsl[i,indi,:],1.),color=scolors[0],label=fnames[0],linewidth=2.,zorder=2)
	# Ensemble Mean
	ax1.plot(x,gaussian_filter(np.nanmean(emsl[i,:,indi,:],axis=0),1.),color=scolors[2],label=fnames[1],linewidth=3.,zorder=3)
	# Ensemble members
	ax1.plot(x,gaussian_filter(emsl[i,0,indi,:],1.),color=scolors[1],label=fnames[2],linewidth=0.5,zorder=1)
	for j in range(1,ensm.shape[0]):
		ax1.plot(x,gaussian_filter(emsl[i,j,indi,:],1.),color=scolors[1],linewidth=0.5,zorder=1)

	# plt.legend(loc='upper left')
	plt.xlabel("Time (day/month)",size=sl)
	plt.ylabel(Nvaren[k],size=sl)
	ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5)); ax1.xaxis.set_major_formatter( DateFormatter('%d/%m') )
	sdatec=repr(time.gmtime(et[indi])[0])+str(time.gmtime(et[indi])[1]).zfill(2)+str(time.gmtime(et[indi])[2]).zfill(2)
	fig.text(0.015,0.05,stations[i]+', cycle'+sdatec, va='center', rotation='horizontal',size=sl)
	fig.text(0.96,0.05,str(c).zfill(3), va='center', rotation='horizontal',size=sl)
	plt.grid(); plt.tight_layout();plt.axis('tight')
	plt.ylim(ymin=np.nanmin(omsl[i,i1:i2+1,:]),ymax=np.nanmax(omsl[i,i1:i2+1,:])); plt.xlim(xmin=x[0]-0.1,xmax=x[-1]+0.1)
	plt.savefig('TimeSeries_'+str(indi).zfill(3)+'_'+stations[i]+'_'+Nvaren[k]+'.png', dpi=100, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='jpg',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig); del fig

# convert -delay 100 -loop 0 TimeSeries_*_ma_dt302_v_AtmPressure.png TimeSeries_ma_dt302_v_AtmPressure.gif

k=3 # WindSpeed Nvaren[k]
Nvar = Nvaren[k]+NvarAnh[2]
indi=218
x=np.zeros(eft.shape[0],'f')*np.nan
for j in range(0,eft.shape[0]):
	x[j]=date2num(datetime.fromtimestamp(et[indi]+eft[j]))

fig, ax1 = plt.subplots(1,figsize=(16,4), sharex=True, sharey=True)
# Observation
ax1.plot(x,gaussian_filter(owspc[i,indi,:],1.),color=scolors[0],label=fnames[0],linewidth=2.,zorder=2)
# Ensemble Mean
ax1.plot(x,gaussian_filter(np.nanmean(ewspc[i,:,indi,:],axis=0),1.),color=scolors[2],label=fnames[1],linewidth=3.,zorder=3)
# Ensemble members
ax1.plot(x,gaussian_filter(ewspc[i,0,indi,:],1.),color=scolors[1],label=fnames[2],linewidth=0.5,zorder=1)
for j in range(1,ensm.shape[0]):
	ax1.plot(x,gaussian_filter(ewspc[i,j,indi,:],1.),color=scolors[1],linewidth=0.5,zorder=1)

plt.legend(loc='upper left')
plt.xlabel("Time (day/month)",size=sl)
plt.ylabel(Nvaren[k],size=sl)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5)); ax1.xaxis.set_major_formatter( DateFormatter('%d/%m') )
fig.text(0.015,0.05,stations[i], va='center', rotation='horizontal',size=sl)
plt.grid(); plt.tight_layout();plt.axis('tight')
plt.xlim(xmin=x[0]-0.1,xmax=x[-1]+0.1)
plt.savefig('TimeSeries.'+str(indi).zfill(3)+'_'+stations[i]+'_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.close(fig); del fig
# Loop
i1=154; i2=223
c=0
for indi in range(i1,i2+1):
	c=c+1
	x=np.zeros(eft.shape[0],'f')*np.nan
	for j in range(0,eft.shape[0]):
		x[j]=date2num(datetime.fromtimestamp(et[indi]+eft[j]))

	fig, ax1 = plt.subplots(1,figsize=(16,4), sharex=True, sharey=True)
	# Observation
	ax1.plot(x,gaussian_filter(owspc[i,indi,:],1.),color=scolors[0],label=fnames[0],linewidth=2.,zorder=2)
	# Ensemble Mean
	ax1.plot(x,gaussian_filter(np.nanmean(ewspc[i,:,indi,:],axis=0),1.),color=scolors[2],label=fnames[1],linewidth=3.,zorder=3)
	# Ensemble members
	ax1.plot(x,gaussian_filter(ewspc[i,0,indi,:],1.),color=scolors[1],label=fnames[2],linewidth=0.5,zorder=1)
	for j in range(1,ensm.shape[0]):
		ax1.plot(x,gaussian_filter(ewspc[i,j,indi,:],1.),color=scolors[1],linewidth=0.5,zorder=1)

	# plt.legend(loc='upper left')
	plt.xlabel("Time (day/month)",size=sl)
	plt.ylabel(Nvaren[k],size=sl)
	ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5)); ax1.xaxis.set_major_formatter( DateFormatter('%d/%m') )
	sdatec=repr(time.gmtime(et[indi])[0])+str(time.gmtime(et[indi])[1]).zfill(2)+str(time.gmtime(et[indi])[2]).zfill(2)
	fig.text(0.015,0.05,stations[i]+', cycle'+sdatec, va='center', rotation='horizontal',size=sl)
	fig.text(0.96,0.05,str(c).zfill(3), va='center', rotation='horizontal',size=sl)
	plt.grid(); plt.tight_layout();plt.axis('tight')
	plt.ylim(ymin=np.nanmin(owspc[i,i1:i2+1,:]),ymax=np.nanmax(owspc[i,i1:i2+1,:])); plt.xlim(xmin=x[0]-0.1,xmax=x[-1]+0.1)
	plt.savefig('TimeSeries_'+str(indi).zfill(3)+'_'+stations[i]+'_'+Nvar+'.png', dpi=100, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig); del fig

# convert -delay 100 -loop 0 TimeSeries_*_ma_dt302_v_WindSpeedAnemC.png TimeSeries_ma_dt302_v_WindSpeedAnemC.gif

# Error versus Forecast time

hd = '      Bias             RMSE          N-Bias          N-RMSE          SCrmse            SI              HH              CC'
hd2 = ' Lines: ECMWFensembleMean: PressaoAtmos,Temperatura,Umidade,VelVenMedAnemA,DirVenMedAnemA,VelVenMedAnemB,VelVenMedAnemC,DirVenMedAnemA,'
hd3 = ' Lines: ECMWF(DailyMean)ensembleMean: PressaoAtmos,Temperatura,Umidade,VelVenMedAnemA,DirVenMedAnemA,VelVenMedAnemB,VelVenMedAnemC,DirVenMedAnemA,'
terrem=np.zeros((8,stations.shape[0],8,eft.shape[0]),'f')*np.nan # array for error metrics, ensemble mean
terrms=np.zeros((8,stations.shape[0],8,ensm.shape[0],eft.shape[0]),'f')*np.nan # array for error metrics, all members
terremd=np.zeros((8,stations.shape[0],8,eft.shape[0]),'f')*np.nan # array for error metrics, ensemble mean from daily mean
# loop through forecast lead time
for d in range(0,eft.shape[0]):

	ind = np.where((eft>=(d*24*3600)) & (eft<((d+1)*24*3600)))[0]

	for i in range(0,stations.shape[0]):

		a=np.array(np.nanmean(emsl[i,:,:,:],axis=0))[:,ind]; b=omsl[i,:,ind].T; inde=np.where(a*b> -999)
		terrem[:,i,0,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
		a=np.array(np.nanmean(eatmp[i,:,:,:],axis=0))[:,ind]; b=oatmp[i,:,ind].T; inde=np.where(a*b> -999)
		terrem[:,i,1,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
		a=np.array(np.nanmean(erh[i,:,:,:],axis=0))[:,ind]; b=orh[i,:,ind].T; inde=np.where(a*b> -999)
		terrem[:,i,2,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
		a=np.array(np.nanmean(ewspa[i,:,:,:],axis=0))[:,ind]; b=owspa[i,:,ind].T; inde=np.where(a*b> -999)
		terrem[:,i,3,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
		a=np.array(np.nanmean(ewdira[i,:,:,:],axis=0))[:,ind]; b=owdira[i,:,ind].T; inde=np.where(a*b> -999)
		terrem[:,i,4,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
		a=np.array(np.nanmean(ewspb[i,:,:,:],axis=0))[:,ind]; b=owspb[i,:,ind].T; inde=np.where(a*b> -999)
		terrem[:,i,5,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
		a=np.array(np.nanmean(ewspc[i,:,:,:],axis=0))[:,ind]; b=owspc[i,:,ind].T; inde=np.where(a*b> -999)
		terrem[:,i,6,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
		a=np.array(np.nanmean(ewdirc[i,:,:,:],axis=0))[:,ind]; b=owdirc[i,:,ind].T; inde=np.where(a*b> -999)
		terrem[:,i,7,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b

		for j in range(0,ensm.shape[0]):

			a=emsl[i,j,:,ind]; b=omsl[i,:,ind]; inde=np.where(a*b> -999)
			terrms[:,i,0,j,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
			a=eatmp[i,j,:,ind]; b=oatmp[i,:,ind]; inde=np.where(a*b> -999)
			terrms[:,i,1,j,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
			a=erh[i,j,:,ind]; b=orh[i,:,ind]; inde=np.where(a*b> -999)
			terrms[:,i,2,j,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
			a=ewspa[i,j,:,ind]; b=owspa[i,:,ind]; inde=np.where(a*b> -999)
			terrms[:,i,3,j,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
			a=ewdira[i,j,:,ind]; b=owdira[i,:,ind]; inde=np.where(a*b> -999)
			terrms[:,i,4,j,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
			a=ewspb[i,j,:,ind]; b=owspb[i,:,ind]; inde=np.where(a*b> -999)
			terrms[:,i,5,j,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
			a=ewspc[i,j,:,ind]; b=owspc[i,:,ind]; inde=np.where(a*b> -999)
			terrms[:,i,6,j,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b
			a=ewdirc[i,j,:,ind]; b=owdirc[i,:,ind]; inde=np.where(a*b> -999)
			terrms[:,i,7,j,d] = merr.metrics(a[inde[0],inde[1]],b[inde[0],inde[1]]); del inde,a,b

		# Daily Mean
		# uniform_filter1d(gfs[ind],size=2),uniform_filter1d(obs[ind],size=2))
		a=np.array(np.nanmean(emsl[i,:,:,d],axis=0)); b=omsl[i,:,d]; inde=np.where(a*b> -999)
		terremd[:,i,0,d] = merr.metrics(a[inde],b[inde]); del inde,a,b
		a=np.array(np.nanmean(eatmp[i,:,:,d],axis=0)); b=oatmp[i,:,d]; inde=np.where(a*b> -999)
		terremd[:,i,1,d] = merr.metrics(a[inde],b[inde]); del inde,a,b
		a=np.array(np.nanmean(erh[i,:,:,d],axis=0)); b=orh[i,:,d]; inde=np.where(a*b> -999)
		terremd[:,i,2,d] = merr.metrics(a[inde],b[inde]); del inde,a,b
		a=np.array(np.nanmean(ewspa[i,:,:,d],axis=0)); b=owspa[i,:,d]; inde=np.where(a*b> -999)
		terremd[:,i,3,d] = merr.metrics(a[inde],b[inde]); del inde,a,b
		a=np.array(np.nanmean(ewdira[i,:,:,d],axis=0)); b=owdira[i,:,d]; inde=np.where(a*b> -999)
		terremd[:,i,4,d] = merr.metrics(a[inde],b[inde]); del inde,a,b
		a=np.array(np.nanmean(ewspb[i,:,:,d],axis=0)); b=owspb[i,:,d]; inde=np.where(a*b> -999)
		terremd[:,i,5,d] = merr.metrics(a[inde],b[inde]); del inde,a,b
		a=np.array(np.nanmean(ewspc[i,:,:,d],axis=0)); b=owspc[i,:,d]; inde=np.where(a*b> -999)
		terremd[:,i,6,d] = merr.metrics(a[inde],b[inde]); del inde,a,b
		a=np.array(np.nanmean(ewdirc[i,:,:,d],axis=0)); b=owdirc[i,:,d]; inde=np.where(a*b> -999)
		terremd[:,i,7,d] = merr.metrics(a[inde],b[inde]); del inde,a,b

		# # Table err metrics
		# fname = 'Table_ErrorECMWFemXForecastTime_'+stations[i]+'_D'+str(d+1).zfill(2)+'.txt'
		# ifile = open(fname,'w')
		# ifile.write('# '+hd2+' \n')
		# ifile.write('# '+hd+' \n')
		# np.savetxt(ifile,terrem[:,i,:,d].T,fmt="%12.3f",delimiter='	')
		# ifile.close(); del ifile, fname

		# fname = 'Table_ErrorECMWFemDMXForecastTime_'+stations[i]+'_D'+str(d+1).zfill(2)+'.txt'
		# ifile = open(fname,'w')
		# ifile.write('# '+hd3+' \n')
		# ifile.write('# '+hd+' \n')
		# np.savetxt(ifile,terremd[:,i,:,d].T,fmt="%12.3f",delimiter='	')
		# ifile.close(); del ifile

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

		# Maranhao
		fig1 = plt.figure(1,figsize=(7,6)); ax = fig1.add_subplot(111)
		for j in range(0,np.size(stations[indst][indstma])):
			ax.plot(eft,gaussian_filter(terrem[l,indst[indstma][j],i,:], 1.), color=scolors[indstma[j]],label=fnames[indstma[j]], linewidth=2.,zorder=2)
			for k in range(0,ensm.shape[0]):
				ax.plot(eft,gaussian_filter(terrms[l,indst[indstma][j],i,k,:], 1.), color=scolors[indstma[j]],linewidth=0.1,zorder=1)

		plt.legend()
		ax.set_xlabel('Forecast Time (Days)',size=sl); ax.set_ylabel(nerrm[l]+' '+Nvar[i],size=sl)
		plt.tight_layout();plt.axis('tight')
		plt.grid(c='k', ls='-', alpha=0.3)
		plt.xticks(np.array([1,5,10,15,20,30,40])); plt.xlim(xmin = 0.9, xmax = eft[-1]+0.1)
		plt.savefig('ErrXForecastTime_Maranhao_'+nerrm[l]+'_'+Nvar[i]+'_leadsComparison.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
		plt.close(fig1); del fig1, ax

		# Piaui
		fig1 = plt.figure(1,figsize=(7,6)); ax = fig1.add_subplot(111)
		for j in range(0,np.size(stations[indst][indstpi])):
			ax.plot(eft,gaussian_filter(terrem[l,indst[indstpi][j],i,:], 1.), color=scolors[indstpi[j]],label=fnames[indstpi[j]], linewidth=2.,zorder=2)
			for k in range(0,ensm.shape[0]):
				ax.plot(eft,gaussian_filter(terrms[l,indst[indstpi][j],i,k,:], 1.), color=scolors[indstpi[j]],linewidth=0.1,zorder=1)

		plt.legend()
		ax.set_xlabel('Forecast Time (Days)',size=sl); ax.set_ylabel(nerrm[l]+' '+Nvar[i],size=sl)
		plt.tight_layout();plt.axis('tight')
		plt.grid(c='k', ls='-', alpha=0.3)
		plt.xticks(np.array([1,5,10,15,20,30,40])); plt.xlim(xmin = 0.9, xmax = eft[-1]+0.1)
		plt.savefig('ErrXForecastTime_Piaui__'+nerrm[l]+'_'+Nvar[i]+'_leadsComparison.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
		plt.close(fig1); del fig1, ax


	# Table Mean error All stations
	fname = 'Table2_MeanErrorECMWFemXForecastTime_'+Nvar[i]+'.txt'
	ifile = open(fname,'w')
	ifile.write('#   EnsembleMean \n')
	ifile.write('# Lines: '+hd+' \n')
	np.savetxt(ifile,np.atleast_2d(eft),fmt="%3i",delimiter=' ')
	ifile.write('# \n')
	for l in range(0,8):
		np.savetxt(ifile,np.atleast_2d(np.nanmean(terrem[l,indst,i,:],axis=0)),fmt="%12.5f",delimiter='	')

	ifile.close(); del ifile, fname


