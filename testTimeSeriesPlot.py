import os, sys
import netCDF4 as nc
import time

from lib.plots import time_series_plot

f=nc.Dataset('ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc')
et=f.variables['cycletime'][:]
eft=f.variables['forecast_time'][:]
stations=f.variables['stationID'][:]
eatmp=f.variables['ecmwf_atmp'][:,:,:,:]
oatmp=f.variables['omega_atmp'][:,:,:]
ewspc=f.variables['ecmwf_wsp_AnemCsup'][:,:,:,:]
owspc=f.variables['omega_wsp_AnemCsup'][:,:,:]
f.close()

# Sellection of stations to plot
NvarAnh=['AnemA','AnemB','AnemC']
Nvaren=['AtmPressure','Temperature','RHumidity','WindSpeed','WindDir']


# TIME SERIES PLOT ==========================

i=1 # Station 'ma_dt302_v'
i1=154; i2=223
for indi in range(i1,i2+1):
	# Pressure
	sdatec=time.strftime('%Y%m%d', time.gmtime(et[indi]))
	time_series_plot(f"TimeSeries_{indi:03}_{stations[i]}_AtmPressure.png",
				oatmp[i,indi,:],
				eatmp[i,:,indi,:],
				et[indi], eft,
				variable_name = 'AtmPressure',
				text = f"{stations[i]}, cycle{sdatec}  {indi - i1 + 1}")

	# WindSpeed
	time_series_plot(f"TimeSeries_{indi:03}_{stations[i]}_WindSpeed AnemC.png",
				owspc[i,indi,:],
				ewspc[i,:,indi,:],
				et[indi], eft,
				variable_name = 'WindSpeed AnemC',
				text = f"{stations[i]}, cycle{sdatec}  {str(indi - i1 + 1)}")

# convert -delay 100 -loop 0 TimeSeries_*_ma_dt302_v_AtmPressure.png TimeSeries_ma_dt302_v_AtmPressure.gif
# convert -delay 100 -loop 0 TimeSeries_*_ma_dt302_v_WindSpeedAnemC.png TimeSeries_ma_dt302_v_WindSpeedAnemC.gif

