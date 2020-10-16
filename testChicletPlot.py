#!/home/rmc/progs/python/anaconda3/bin/ipython --pylab=auto

import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

from lib.plots import chicklet_plot

f=nc.Dataset('ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc')
datm=f.variables['date_time'][:]
eft=f.variables['forecast_time'][:]
stations=f.variables['stationID'][:]
eatmp=f.variables['ecmwf_atmp'][:,:,:,:]
oatmp=f.variables['omega_atmp'][:,:,:]
erh=f.variables['ecmwf_rh'][:,:,:,:]
orh=f.variables['omega_rh'][:,:,:]
emsl=f.variables['ecmwf_msl'][:,:,:,:]
omsl=f.variables['omega_msl'][:,:,:]
ewspa=f.variables['ecmwf_wsp_AnemAinfer'][:,:,:,:]
owspa=f.variables['omega_wsp_AnemAinfer'][:,:,:]
f.close()

# Sellection of stations to plot
ssp=np.array([1,23]).astype('int')
Nvarpt=['PressaoAtmos','Temperatura','Umidade','VelVenMed','DirVenMed']
NvarAnh=['AnemA','AnemB','AnemC']
Nvaren=['AtmPressure','Temperature','RHumidity','WindSpeed','WindDir']

# Pressure
Nvar = Nvaren[0]
lev= np.linspace(1001.,np.nanmax(np.append(omsl,np.nanmean(emsl,axis=1))),100)
lev2= np.linspace(0.,1.9,100)
lev3= np.linspace(-4.6,4.6,101)
for i in ssp:
	auxo=np.copy(omsl[i,:,:].T)
	auxm=np.copy(np.nanmean(emsl[i,:,:,:],axis=0).T)
	auxm2=np.copy(emsl[i,0,:,:].T)
	auxmsprd=np.copy(np.std(emsl[i,:,:,:],axis=0).T)
	# Observation
	img_name = 'ChicletChart_Observation_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxo, datm, eft, lev)
	# Ensemble Mean
	img_name = 'ChicletChart_ModelEM_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm, datm, eft, lev)
	# First Ensemble Member
	img_name = 'ChicletChart_Modelm1_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm2, datm, eft, lev)
	# Spread
	img_name = 'ChicletChart_ModelSpread_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxmsprd, datm, eft, lev2, extend='max')
	# Differences
	img_name = 'ChicletChart_DiffModelEMObs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm - auxo, datm, eft, lev3, color_palette=plt.cm.RdBu_r)
	img_name = 'ChicletChart_DiffModelm1Obs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm2 - auxo, datm, eft, lev3, color_palette=plt.cm.RdBu_r)

# Temperature
Nvar = Nvaren[1]
lev= np.linspace(23,31,100)
lev2= np.linspace(0.,1.3,100)
lev3= np.linspace(-3.1,3.1,101)
for i in ssp:
	auxo=np.copy(oatmp[i,:,:].T)
	auxm=np.copy(np.nanmean(eatmp[i,:,:,:],axis=0).T)
	auxm2=np.copy(eatmp[i,0,:,:].T)
	auxmsprd=np.copy(np.std(eatmp[i,:,:,:],axis=0).T)
	# Observation
	img_name = 'ChicletChart_Observation_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxo, datm, eft, lev)
	# Ensemble Mean
	img_name = 'ChicletChart_ModelEM_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm, datm, eft, lev)
	# First Ensemble Member
	img_name = 'ChicletChart_Modelm1_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm2, datm, eft, lev)
	# Spread
	img_name = 'ChicletChart_ModelSpread_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxmsprd, datm, eft, lev2, extend='max')
	# Differences
	img_name = 'ChicletChart_DiffModelEMObs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm - auxo, datm, eft, lev3, color_palette=plt.cm.RdBu_r)
	img_name = 'ChicletChart_DiffModelm1Obs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm2 - auxo, datm, eft, lev3, color_palette=plt.cm.RdBu_r)

# Humidade
Nvar = Nvaren[2]
lev= np.linspace(65,100,100)
lev2= np.linspace(0.,8.,100)
lev3= np.linspace(-15.1,15.1,101)
for i in ssp:
	auxo=np.copy(orh[i,:,:].T)
	auxm=np.copy(np.nanmean(erh[i,:,:,:],axis=0).T)
	auxm2=np.copy(erh[i,0,:,:].T)
	auxmsprd=np.copy(np.std(erh[i,:,:,:],axis=0).T)
	# Observation
	img_name = 'ChicletChart_Observation_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxo, datm, eft, lev, extend='min')
	# Ensemble Mean
	img_name = 'ChicletChart_ModelEM_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm, datm, eft, lev, extend='min')
	# First Ensemble Member
	img_name = 'ChicletChart_Modelm1_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm2, datm, eft, lev, extend='min')
	# Spread
	img_name = 'ChicletChart_ModelSpread_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxmsprd, datm, eft, lev2, extend='max')
	# Differences
	img_name = 'ChicletChart_DiffModelEMObs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm - auxo, datm, eft, lev3, color_palette=plt.cm.RdBu_r)
	img_name = 'ChicletChart_DiffModelm1Obs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm2 - auxo, datm, eft, lev3, color_palette=plt.cm.RdBu_r)

# Wind Speed
Nvar = Nvaren[3]+NvarAnh[0]
lev= np.linspace(0,13.,100)
lev2= np.linspace(0.,2.8,100)
lev3= np.linspace(-4.6,4.6,101)
for i in ssp:
	auxo=np.copy(owspa[i,:,:].T)
	auxm=np.copy(np.nanmean(ewspa[i,:,:,:],axis=0).T)
	auxm2=np.copy(ewspa[i,0,:,:].T)
	auxmsprd=np.copy(np.std(ewspa[i,:,:,:],axis=0).T)
	# Observation
	img_name = 'ChicletChart_Observation_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxo, datm, eft, lev, extend='max')
	# Ensemble Mean
	img_name = 'ChicletChart_ModelEM_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm, datm, eft, lev, extend='max')
	# First Ensemble Member
	img_name = 'ChicletChart_Modelm1_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm2, datm, eft, lev, extend='max')
	# Spread
	img_name = 'ChicletChart_ModelSpread_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxmsprd, datm, eft, lev2, extend='max')
	# Differences
	img_name = 'ChicletChart_DiffModelEMObs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm - auxo, datm, eft, lev3, color_palette=plt.cm.RdBu_r)
	img_name = 'ChicletChart_DiffModelm1Obs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
	chicklet_plot(img_name, auxm2 - auxo, datm, eft, lev3, color_palette=plt.cm.RdBu_r)

