import os
import xarray

from lib.plots import qq_plot

OUTPUT_DIR = 'output/'

if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

scolors = ['c','royalblue','steelblue','blue','darkblue','cyan']

d = xarray.open_dataset('ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc')

station = 0
forecast_index = 10

obs = d.omega_atmp[station,:,forecast_index]
ens1 = d.ecmwf_atmp[station,0,:,forecast_index] # ensemble 1
ens2= d.ecmwf_atmp[station,1,:,forecast_index] # ensemble 2
ens3 = d.ecmwf_atmp[station,2,:,forecast_index] # ensemble 3

qq_plot(f"{OUTPUT_DIR}/QQplot_{str(d.stationID[station].data)}.png", obs, [ens1, ens2, ens3],
		ens_colors=scolors, ens_names=['ens1', 'ens2', 'ens3'])

