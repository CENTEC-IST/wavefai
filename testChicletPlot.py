import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

from lib.plots import chiclet_plot

OUTPUT_DIR = 'output/'

if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)


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
ssp=[1,23]

# Pressure
levels = [np.linspace(1001.,np.nanmax(np.append(omsl,np.nanmean(emsl,axis=1))),100), np.linspace(0.,1.9,100), np.linspace(-4.6,4.6,101)]

print('Plotting AtmPressure')
for i in ssp:
	omsl_i = omsl[i,:,:].T
	emsl_m0 = emsl[i,0,:,:].T
	emsl_mean = np.nanmean(emsl[i,:,:,:],axis=0).T
	emsl_spread = np.std(emsl[i,:,:,:],axis=0).T

	# Observation
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_Observation_{stations[i]}_AtmPressure.png",
			omsl_i, datm, eft, levels[0])
	# Ensemble Mean
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_ModelEM_{stations[i]}_AtmPressure.png",
			emsl_mean, datm, eft, levels[0])
	# First Ensemble Member
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_Modelm1_{stations[i]}_AtmPressure.png",
			emsl_m0, datm, eft, levels[0])
	# Spread
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_ModelSpread_{stations[i]}_AtmPressure.png",
			emsl_spread, datm, eft, levels[1], extend='max')
	# Differences
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_DiffModelEMObs_{stations[i]}_AtmPressure.png",
			emsl_mean - omsl_i, datm, eft, levels[2], color_palette=plt.cm.RdBu_r)
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_DiffModelm1Obs_{stations[i]}_AtmPressure.png",
			emsl_m0 - omsl_i, datm, eft, levels[2], color_palette=plt.cm.RdBu_r)

# Temperature
levels = [np.linspace(23,31,100), np.linspace(0.,1.3,100), np.linspace(-3.1,3.1,101)]


print('Plotting Temperature')
for i in ssp:
	oatmp_i = oatmp[i,:,:].T
	eatmp_m0 = eatmp[i,0,:,:].T
	eatmp_mean = np.nanmean(eatmp[i,:,:,:],axis=0).T
	eatmp_spread = np.std(eatmp[i,:,:,:],axis=0).T

	# Observation
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_Observation_{stations[i]}_Temperature.png",
			oatmp_i, datm, eft, levels[0])
	# Ensemble Mean
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_ModelEM_{stations[i]}_Temperature.png",
			eatmp_mean, datm, eft, levels[0])
	# First Ensemble Member
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_Modelm1_{stations[i]}_Temperature.png",
			eatmp_m0, datm, eft, levels[0])
	# Spread
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_ModelSpread_{stations[i]}_Temperature.png",
			eatmp_spread, datm, eft, levels[1], extend='max')
	# Differences
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_DiffModelEMObs_{stations[i]}_Temperature.png",
			eatmp_mean - oatmp_i, datm, eft, levels[2], color_palette=plt.cm.RdBu_r)
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_DiffModelm1Obs_{stations[i]}_Temperature.png",
			eatmp_m0 - oatmp_i, datm, eft, levels[2], color_palette=plt.cm.RdBu_r)

# Humidity
levels = [np.linspace(65,100,100), np.linspace(0.,8.,100), np.linspace(-15.1,15.1,101)]

print('Plotting Humidity')
for i in ssp:
	orh_i = orh[i,:,:].T
	erh_m0 = erh[i,0,:,:].T
	erh_mean = np.nanmean(erh[i,:,:,:],axis=0).T
	erh_spread = np.std(erh[i,:,:,:],axis=0).T

	# Observation
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_Observation_{stations[i]}_RHumidity.png",
			orh_i, datm, eft, levels[0], extend='min')
	# Ensemble Mean
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_ModelEM_{stations[i]}_RHumidity.png",
			erh_mean, datm, eft, levels[0], extend='min')
	# First Ensemble Member
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_Modelm1_{stations[i]}_RHumidity.png",
			erh_m0, datm, eft, levels[0], extend='min')
	# Spread
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_ModelSpread_{stations[i]}_RHumidity.png",
			erh_spread, datm, eft, levels[1], extend='max')
	# Differences
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_DiffModelEMObs_{stations[i]}_RHumidity.png",
			erh_mean - orh_i, datm, eft, levels[2], color_palette=plt.cm.RdBu_r)
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_DiffModelm1Obs_{stations[i]}_RHumidity.png",
			erh_m0 - orh_i, datm, eft, levels[2], color_palette=plt.cm.RdBu_r)

# Wind Speed
levels = [np.linspace(0,13.,100), np.linspace(0.,2.8,100), np.linspace(-4.6,4.6,101)]

print('Plotting Wind Speed')
for i in ssp:
	owspa_i = owspa[i,:,:].T
	ewspa_m0 = ewspa[i,0,:,:].T
	ewspa_mean = np.nanmean(ewspa[i,:,:,:],axis=0).T
	ewspa_spread = np.std(ewspa[i,:,:,:],axis=0).T

	# Observation
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_Observation_{stations[i]}_WindSpeed.png",
			owspa_i, datm, eft, levels[0], extend='max')
	# Ensemble Mean
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_ModelEM_{stations[i]}_WindSpeed.png",
			ewspa_mean, datm, eft, levels[0], extend='max')
	# First Ensemble Member
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_Modelm1_{stations[i]}_WindSpeed.png",
			ewspa_m0, datm, eft, levels[0], extend='max')
	# Spread
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_ModelSpread_{stations[i]}_WindSpeed.png",
			ewspa_spread, datm, eft, levels[1], extend='max')
	# Differences
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_DiffModelEMObs_{stations[i]}_WindSpeed.png",
			ewspa_mean - owspa_i, datm, eft, levels[2], color_palette=plt.cm.RdBu_r)
	chiclet_plot(f"{OUTPUT_DIR}/ChicletChart_DiffModelm1Obs_{stations[i]}_WindSpeed.png",
			ewspa_m0 - owspa_i, datm, eft, levels[2], color_palette=plt.cm.RdBu_r)

