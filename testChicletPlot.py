import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

from lib.plots import chicklet_plot

f=nc.Dataset('ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc')
datm=f.variables['date_time'][:]
et=f.variables['cycletime'][:]
eft=f.variables['forecast_time'][:]
ensm=f.variables['ensmember'][:]
blat=f.variables['latitude'][:]
blon=f.variables['longitude'][:]
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
    chicklet_plot(img_name, auxo, lev)
    # Ensemble Mean
    img_name = 'ChicletChart_ModelEM_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm, lev)
    # First Ensemble Member
    img_name = 'ChicletChart_Modelm1_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm2, lev)
    # Spread
    img_name = 'ChicletChart_ModelSpread_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxmsprd, lev2, extend='max')
    # Differences
    img_name = 'ChicletChart_DiffModelEMObs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm - auxo, lev3, cmap=plt.cm.RdBu_r)
    img_name = 'ChicletChart_DiffModelm1Obs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm2 - auxo, lev3, cmap=plt.cm.RdBu_r)

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
    chicklet_plot(img_name, auxo, lev)
    # Ensemble Mean
    img_name = 'ChicletChart_ModelEM_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm, lev)
    # First Ensemble Member
    img_name = 'ChicletChart_Modelm1_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm2, lev)
    # Spread
    img_name = 'ChicletChart_ModelSpread_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxmsprd, lev2, extend='max')
    # Differences
    img_name = 'ChicletChart_DiffModelEMObs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm - auxo, lev3, cmap=plt.cm.RdBu_r)
    img_name = 'ChicletChart_DiffModelm1Obs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm2 - auxo, lev3, cmap=plt.cm.RdBu_r)

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
    chicklet_plot(img_name, auxo, lev, extend='min')
    # Ensemble Mean
    img_name = 'ChicletChart_ModelEM_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm, lev, extend='min')
    # First Ensemble Member
    img_name = 'ChicletChart_Modelm1_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm2, lev, extend='min')
    # Spread
    img_name = 'ChicletChart_ModelSpread_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxmsprd, lev2, extend='max')
    # Differences
    img_name = 'ChicletChart_DiffModelEMObs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm - auxo, lev3, cmap=plt.cm.RdBu_r)
    img_name = 'ChicletChart_DiffModelm1Obs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm2 - auxo, lev3, cmap=plt.cm.RdBu_r)

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
    chicklet_plot(img_name, auxo, lev, extend='max')
    # Ensemble Mean
    img_name = 'ChicletChart_ModelEM_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm, lev, extend='max')
    # First Ensemble Member
    img_name = 'ChicletChart_Modelm1_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm2, lev, extend='max')
    # Spread
    img_name = 'ChicletChart_ModelSpread_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxmsprd, lev2, extend='max')
    # Differences
    img_name = 'ChicletChart_DiffModelEMObs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm - auxo, lev3, cmap=plt.cm.RdBu_r)
    img_name = 'ChicletChart_DiffModelm1Obs_'+stations[i]+'_'+Nvar+'XFtimeXTime.png'
    chicklet_plot(img_name, auxm2 - auxo, lev3, cmap=plt.cm.RdBu_r)

