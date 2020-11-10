import os
import numpy as np
import xarray

from lib.errors import metrics
from lib.plots import errors_plot

OUTPUT_DIR = 'output/'

if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)


xdata = xarray.open_dataset('../data/ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc')

# f=nc.Dataset('../data/ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc')
datm=xdata.date_time
et=xdata.cycletime
eft=xdata.forecast_time
ensm=xdata.ensmember
stations=xdata.stationID
eatmp=xdata.ecmwf_atmp
oatmp=xdata.omega_atmp
erh=xdata.ecmwf_rh
orh=xdata.omega_rh
emsl=xdata.ecmwf_msl
omsl=xdata.omega_msl
ewspa=xdata.ecmwf_wsp_AnemAinfer
owspa=xdata.omega_wsp_AnemAinfer
ewdira=xdata.ecmwf_wdir_AnemAinfer
owdira=xdata.omega_wdir_AnemAinfer
ewspb=xdata.ecmwf_wsp_AnemB
owspb=xdata.omega_wsp_AnemB
ewspc=xdata.ecmwf_wsp_AnemCsup
owspc=xdata.omega_wsp_AnemCsup
ewdirc=xdata.ecmwf_wdir_AnemCsup
owdirc=xdata.omega_wdir_AnemCsup

# Variables Data
observed_data = [omsl, oatmp, orh, owspa, owdira, owspb, owspc, owdirc]
ensemble_data = [emsl , eatmp, erh, ewspa, ewdira, ewspb, ewspc, ewdirc]

# Variables names
eft_i = xdata.forecast_time.isel({'forecast_time':slice(0,180,4)}).data # choose only the seconds that represent each day

terrem=np.zeros((8, stations.size, len(observed_data), eft_i.size), 'f')*np.nan # array for error metrics, ensemble mean
terrms=np.zeros((8, stations.size, len(observed_data), ensm.size, eft_i.size), 'f')*np.nan # array for error metrics, all members

for day in range(0,eft_i.size):

	print(f"\rProcessing day [{day}/{eft_i.size-1}]", end='')

	day_ftime = np.where((eft >= (day*24*3600)) & (eft < ((day+1)*24*3600)))[0] # forecast seconds for this day

	for station in range(0, xdata.stations.size): # iterate over each station

		for variable in range(len(observed_data)):
			a = np.array(np.nanmean(ensemble_data[variable].data[station,:,:,:], axis=0))[:,day_ftime]
			b = observed_data[variable].data[station,:,day_ftime].T
			inde = np.where(a*b > -999)
			terrem[:,station,variable,day] = metrics(a[inde[0],inde[1]],b[inde[0],inde[1]])

		for ens in range(0,ensm.size):
			for variable in range(len(observed_data)):
				a = ensemble_data[variable].data[station,ens,:,day_ftime]
				b = observed_data[variable].data[station,:,day_ftime]
				inde=np.where(a*b > -999)
				terrms[:,station,variable,ens,day] = metrics(a[inde[0],inde[1]],b[inde[0],inde[1]])

print()

# Plot err X forecast time das 16 estacoes com mais dados

# Error metrics plot (error versus forecast time)
scolors = np.array(['c','royalblue','steelblue','blue','darkblue','cyan','dodgerblue','g','limegreen','lime',
	'chocolate','red','orangered','salmon','brown','saddlebrown'])
fnames = np.array(['301','302','304','305','306','307','308','502',
	'601','602','pba01','pdd01','ppa01','psa01','tbr01','tbr03'])
station_indexes = np.array([0,1,3,4,5,6,7,9,10,11,18,19,20,21,22,23]).astype('int') # Index of stations with few problems
error_type = ['bias','RMSE','NBias','NRMSE','SCrmse','SI','HH','CC']

stations_maranhao = np.arange(0,10).astype('int') # Maranhao
stations_piaui = np.arange(10,16).astype('int') # Piaui

Nvar = ['AtmPressure', 'Temperature', 'RHumidity', 'WindSpeed_AnemA', 'WindDir_AnemA', 'WindSpeed_AnemB', 'WindSpeed_AnemC', 'WindDir_AnemC']

for var in range(len(Nvar)):
	for err in range(len(error_type)):
		# Maranhao

		errors_plot(f"{OUTPUT_DIR}/ErrXForecastTime_Maranhao_{error_type[err]}_{Nvar[var]}.png",
				terrms[err,:,var,:,:], terrem[err,:,var,:], eft_i, station_indexes = station_indexes[stations_maranhao],
				colors=scolors[stations_maranhao], labels=fnames[stations_maranhao],
				ylabel=error_type[err]+' '+Nvar[var])

		errors_plot(f"{OUTPUT_DIR}/ErrXForecastTime_Piaui_{error_type[err]}_{Nvar[var]}.png",
				terrms[err,:,var,:,:], terrem[err,:,var,:], eft_i, station_indexes = station_indexes[stations_piaui],
				colors=scolors[stations_piaui], labels=fnames[stations_piaui],
				ylabel=error_type[err]+' '+Nvar[var])


