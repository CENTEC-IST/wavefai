import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
matplotlib.use('Agg')
from scipy import signal
import xarray

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

# QQ-plots

scolors = np.array(['c','royalblue','steelblue','blue','darkblue','cyan','dodgerblue','g','limegreen','lime',
	'chocolate','red','orangered','salmon','brown','saddlebrown'])

d = xarray.open_dataset('ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc')

station = 0
obs = d.omega_atmp[station,:,:]
ens1 = d.ecmwf_atmp[station,0,:,:] # ensembler 1
ens2= d.ecmwf_atmp[station,1,:,:] # ensembler 2
ens3 = d.ecmwf_atmp[station,2,:,:] # ensembler 3

ind = 10

a=np.nanmin(np.c_[ens1[ind],ens2[ind],ens3[ind]])
b=np.nanmax(np.c_[ens1[ind],ens2[ind],ens3[ind]])
px=np.linspace(a,b,100)
aux=np.linspace(a-0.01*np.abs(b-a),b+0.01*np.abs(b-a),np.arange(1,99,1).shape[0])
fig1 = plt.figure(1,figsize=(8,6))
ax1 = fig1.add_subplot(111)

ax1.plot(aux,aux,'k',linewidth=1.)

ax1.plot(np.sort(obs[ind]), np.sort(ens1[ind]),'.',color=scolors[1],label='ensembler 1')
ax1.plot(np.sort(obs[ind]), np.sort(ens2[ind]),'.',color=scolors[2],label='ensembler 2')
ax1.plot(np.sort(obs[ind]), np.sort(ens3[ind]),'.',color=scolors[3],label='ensembler 3')

# GENERATE PNG
plt.ylim(ymax = aux.max(), ymin = aux.min())
plt.xlim(xmax = aux.max(), xmin = aux.min())
plt.xticks(np.arange(aux.min(),aux.max(),(aux.max() - aux.min()) / 10))
plt.yticks(np.arange(aux.min(),aux.max(),(aux.max() - aux.min()) / 10))
plt.locator_params(axis='y', nbins=7)
plt.locator_params(axis='x', nbins=7)
plt.legend()
ax1.set_xlabel('Medicoes ',size=sl)
ax1.set_ylabel('Modelo ',size=sl)
plt.tight_layout()
# plt.axis('tight')
plt.grid(c='k', ls='-', alpha=0.3)
plt.savefig('QQplot_'+str(d.stationID[station].data)+'_.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.close(fig1)


# GENERATE HTML
p = figure(plot_width=700, plot_height=600)
p.title.text = 'Click on legend entries to hide the corresponding lines'
p.line(aux,aux,line_width=2,color='black',alpha=0.8)
p.circle(np.sort(obs[ind]), np.sort(ens1[ind]), size=5, color=scolors[1], alpha=0.8, legend_label='ensembler 1')
p.circle(np.sort(obs[ind]), np.sort(ens2[ind]), size=5, color=scolors[2], alpha=0.8, legend_label='ensembler 2')
p.circle(np.sort(obs[ind]), np.sort(ens3[ind]), size=5, color=scolors[3], alpha=0.8, legend_label='ensembler 3')
p.legend.location = "top_left"
p.legend.click_policy = "hide"
p.xaxis.axis_label = 'Medicoes'
p.yaxis.axis_label = 'Modelo'
output_file('QQplot_'+str(d.stationID[station].data)+'_.html')
show(p)



