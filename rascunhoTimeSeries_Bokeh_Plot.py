import os
import netCDF4 as nc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from bokeh.plotting import figure, output_file, show
import xarray as xr
import pandas as pd
matplotlib.use('Agg')
import pickle
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.dates as mdates
import matplotlib.dates as dates
from scipy import signal
from scipy.ndimage.filters import gaussian_filter, uniform_filter1d
from scipy.stats import gaussian_kde
import xarray
from calendar import timegm
from lib.plots import time_series_plot

sl=14
matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl)
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})

# Filter functions
def butter_lowpass(cutoff, nyq_freq, order=4):
	normal_cutoff = float(cutoff) / nyq_freq
	b, a = signal.butter(order, normal_cutoff, btype='lowpass')
	return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
	nyq_freq = nyq_freq * 0.5
	b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
	y = signal.filtfilt(b, a, data)
	return y

def butter_highpass(interval, sampling_rate, cutoff, order=5):
	nyq = sampling_rate * 0.5
	stopfreq = float(cutoff)
	cornerfreq = 0.4 * stopfreq  # (?)
	ws = cornerfreq/nyq
	wp = stopfreq/nyq
	N, wn = signal.buttord(wp, ws, 3, 16)	# (?)
	b, a = signal.butter(N, wn, btype='high')	# should 'high' be here for bandpass?
	sf = signal.lfilter(b, a, interval)
	return sf


# PLOT TIME SERIES
d = xarray.open_dataset('ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc')
station = 0
fc_time = 0
time_series_plot(f'TimeSeriesObsvForecasts_{station}_D.png',
		d.omega_atmp[station,:,fc_time],
		[d.ecmwf_atmp[station,0,:,fc_time]], d.date_time) # Make a list of

# # Bokeh fig
# 	p = figure(plot_width=700, plot_height=600)
# 	p.title.text = 'Click on legend entries to hide the corresponding lines'
# 	k=0;de = gaussian_kde(obs[ind])
# 	p.line(px, gaussian_filter(de(px),1.), line_width=2, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	k=1;de = gaussian_kde(gfs[ind]);
# 	p.line(px, gaussian_filter(de(px),1.), line_width=2, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	k=2;de = gaussian_kde(ecm[ind]);
# 	p.line(px, gaussian_filter(de(px),1.), line_width=2, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	k=3;de = gaussian_kde(twc[ind]);
# 	p.line(px, gaussian_filter(de(px),1.), line_width=2, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	k=4;de = gaussian_kde(mlv1[ind]);
# 	p.line(px, gaussian_filter(de(px),1.), line_width=2, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	k=5;de = gaussian_kde(mlv2[ind]);
# 	p.line(px, gaussian_filter(de(px),1.), line_width=2, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	k=6;de = gaussian_kde(mixp[ind]);
# 	p.line(px, gaussian_filter(de(px),1.), line_width=2, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	p.legend.location = "top_left" ; p.legend.click_policy = "hide"
# 	p.xaxis.axis_label = 'VelVenMedSup'; p.yaxis.axis_label = 'Densidade de Probabilidade'
# 	output_file('PDF_'+stname+'_D'+str(d+1).zfill(2)+'.html'); show(p)

# # QQ-plots
# # a=np.nanmin(np.c_[obs[ind],gfs[ind],ecm[ind],mlv1[ind]]); b=np.nanmax(np.c_[obs[ind],gfs[ind],ecm[ind],mlv1[ind]]); p = np.arange(1,99,1)
# 	a=np.nanmin(np.c_[gfs[ind],ecm[ind],mlv1[ind]]); b=np.nanmax(np.c_[gfs[ind],ecm[ind],mlv1[ind]]); p = np.arange(1,99,1)
# 	px=np.linspace(a,b,100)
# 	aux=np.linspace(a-0.01*np.abs(b-a),b+0.01*np.abs(b-a),p.shape[0])
# 	fig1 = plt.figure(1,figsize=(8,6)); ax1 = fig1.add_subplot(111)
# 	p = figure(plot_width=700, plot_height=600, x_range = (aux.min(), aux.max()), y_range = (aux.min(), aux.max()))
# 	p.title.text = 'Click on legend entries to hide the corresponding lines'
# 	ax1.plot(aux,aux,'k',linewidth=1.)
# 	p.line(aux,aux,line_width=2,color='black',alpha=0.8)
# 	k=1; ax1.plot(np.sort(obs[ind]),np.sort(gfs[ind]),'.',color=scolors[k],label=fnames[k])
# 	p.circle(np.sort(obs[ind]), np.sort(gfs[ind]), size=5, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	k=2; ax1.plot(np.sort(obs[ind]),np.sort(ecm[ind]),'.',color=scolors[k],label=fnames[k])
# 	p.circle(np.sort(obs[ind]), np.sort(ecm[ind]), size=5, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	k=3; ax1.plot(np.sort(obs[ind]),np.sort(twc[ind]),'.',color=scolors[k],label=fnames[k])
# 	p.circle(np.sort(obs[ind]), np.sort(twc[ind]), size=5, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	k=4; ax1.plot(np.sort(obs[ind]),np.sort(mlv1[ind]),'.',color=scolors[k],label=fnames[k])
# 	p.circle(np.sort(obs[ind]), np.sort(mlv1[ind]), size=5, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	k=5; ax1.plot(np.sort(obs[ind]),np.sort(mlv2[ind]),'.',color=scolors[k],label=fnames[k])
# 	p.circle(np.sort(obs[ind]), np.sort(mlv2[ind]), size=5, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	k=6; ax1.plot(np.sort(obs[ind]),np.sort(mixp[ind]),'.',color=scolors[k],label=fnames[k])
# 	p.circle(np.sort(obs[ind]), np.sort(mixp[ind]), size=5, color=scolors2[k], alpha=0.8, legend_label=fnames[k])
# 	plt.ylim(ymax = aux.max(), ymin = aux.min()); xticks(np.arange(0,aux.max(),3))
# 	plt.xlim(xmax = aux.max(), xmin = aux.min()); yticks(np.arange(0,aux.max(),3))
# 	plt.locator_params(axis='y', nbins=7) ; plt.locator_params(axis='x', nbins=7)
# 	plt.legend()
# 	ax1.set_xlabel('Medicoes ',size=sl); ax1.set_ylabel('Modelo ',size=sl);
# 	plt.tight_layout(); # plt.axis('tight')
# 	plt.grid(c='k', ls='-', alpha=0.3)
# 	plt.savefig('QQplot_'+stname+'_D'+str(d+1).zfill(2)+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
# 	plt.close(fig1)
# 	p.legend.location = "top_left" ; p.legend.click_policy = "hide"
# 	p.xaxis.axis_label = 'Medicoes'; p.yaxis.axis_label = 'Modelo'
# 	output_file('QQplot_'+stname+'_D'+str(d+1).zfill(2)+'.html'); show(p)



