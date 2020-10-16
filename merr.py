import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib.dates as mdates
from matplotlib.mlab import *
import matplotlib.ticker
from matplotlib.dates import DateFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import cm
palette = plt.cm.jet
palette.set_bad('aqua', 10.0)
import pickle
from matplotlib import ticker
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import gaussian_kde
import sys
import warnings; warnings.filterwarnings("ignore")
import scipy.stats
import lmoments3 as lm
# pip install git+https://github.com/OpenHydrology/lmoments3.git

# Table summary statistics
def smrstat(*args):
	'''
	Summary Statistics
	Input: one array of interest
	Output: mean, variance, skewness, kurtosis, min, max, percentile80, percentile90, percentile95, percentile99, percentile99.9, LCV, Lskew, Lkurtosis
	'''
	vmin=-np.inf; vmax=np.inf
	if len(args) == 1:
		x=copy.copy(args[0])
	elif len(args) == 2:
		x=copy.copy(args[0]); vmin=copy.copy(args[1])
	elif len(args) == 3:
		x=copy.copy(args[0]); vmin=copy.copy(args[1]); vmax=copy.copy(args[2])
	elif len(args) > 3:
		sys.exit(' Too many inputs')

	ind=np.where((np.isnan(x)==False) & (x>vmin) & (x<vmax))
	if np.any(ind):
		x=np.copy(x[ind[0]])
	else:
		sys.exit(' Array without valid numbers.')

	ferr=np.zeros((14),'f')*np.nan
	ferr[0] = np.mean(x)
	ferr[1] = np.var(x)
	ferr[2] = scipy.stats.skew(x)
	ferr[3] = scipy.stats.kurtosis(x)
	ferr[4] = np.min(x)
	ferr[5] = np.max(x)
	ferr[6] = np.percentile(x,80)
	ferr[7] = np.percentile(x,90)
	ferr[8] = np.percentile(x,95)
	ferr[9] = np.percentile(x,99)
	ferr[10] = np.percentile(x,99.9)
	# Hosking & Wallis L-moment ratios
	# pip install git+https://github.com/OpenHydrology/lmoments3.git
	hwlm = lm.lmom_ratios(x, nmom=5)
	ferr[11] = hwlm[1]/hwlm[0]
	ferr[12] = hwlm[2]
	ferr[13] = hwlm[3]

	return ferr


# Error metrics =============================

def metrics(*args):
	'''
	Error Metrics. Mentaschi et al. (2013)
	Input: two arrays of model and observation, respectively.
		They must have the same size
	Output: ferr array with shape equal to 8
		bias, RMSE, NBias, NRMSE, SCrmse, SI, HH, CC
	'''

	vmin=-np.inf; vmax=np.inf; maxdiff=np.inf
	if len(args) < 2:
		sys.exit(' Need two arrays with model and observations.')
	elif len(args) == 2:
		model=copy.copy(args[0]); obs=copy.copy(args[1])
	elif len(args) == 3:
		model=copy.copy(args[0]); obs=copy.copy(args[1])
		vmin=copy.copy(args[2])
	elif len(args) == 4:
		model=copy.copy(args[0]); obs=copy.copy(args[1])
		vmin=copy.copy(args[2]); vmax=copy.copy(args[3])
	elif len(args) == 5:
		model=copy.copy(args[0]); obs=copy.copy(args[1])
		vmin=copy.copy(args[2]); vmax=copy.copy(args[3]); maxdiff=copy.copy(args[4]);
	elif len(args) > 5:
		sys.exit(' Too many inputs')

	model=np.atleast_1d(model); obs=np.atleast_1d(obs)
	if model.shape != obs.shape:
		sys.exit(' Model and Observations with different size.')
	if vmax<=vmin:
		sys.exit(' vmin cannot be higher than vmax.')

	ind=np.where((np.isnan(model)==False) & (np.isnan(obs)==False) & (model>vmin) & (model<vmax) & (obs>vmin) & (obs<vmax) & (np.abs(model-obs)<=maxdiff) )
	if np.any(ind) or model.shape[0]==1:
		model=np.copy(model[ind[0]]); obs=np.copy(obs[ind[0]])
	else:
		sys.exit(' Array without valid numbers.')

	ferr=np.zeros((8),'f')*np.nan
	ferr[0] = model.mean()-obs.mean() # Bias
	ferr[1] = (((model-obs)**2).mean())**0.5 # RMSE
	if obs.mean()!=0.:
		ferr[2] = ferr[0] / np.abs(obs.mean()) # Normalized Bias
	ferr[3] = ( ((model-obs)**2).sum() / (obs**2).sum() )**0.5  # Normalized RMSE
	# ferr[4] = ((((model-model.mean())-(obs-obs.mean()))**2).mean())**0.5   # Scatter Component of RMSE
	if ( (ferr[1]**2) - (ferr[0]**2) ) >= 0.:
		ferr[4] = ( (ferr[1]**2) - (ferr[0]**2) )**0.5
	ferr[5] = ( (((model-model.mean())-(obs-obs.mean()))**2).sum() / (obs**2).sum() )**0.5  # Scatter Index
	ferr[6] = ( ((model - obs)**2).sum() / (model * obs).sum() )**0.5  # HH
	ferr[7]=np.corrcoef(model,obs)[0,1]  #  Correlation Coefficient

	return ferr

def imetrics(*args):
	'''
	Error Metrics at each individual instant. Same equations of Mentaschi et al. (2013), but using n equal to one.
	Input: two arrays of model and observation, respectively.
		They must have the same size
	Output: ferr array with shape equal to [inpShape,8]
		bias, RMSE, NBias, NRMSE, SCrmse, SI, HH, CC
	'''

	vmin=-np.inf; vmax=np.inf; maxdiff=np.inf
	if len(args) < 2:
		sys.exit(' Need two arrays with model and observations.')
	elif len(args) == 2:
		model=copy.copy(args[0]); obs=copy.copy(args[1])
	elif len(args) == 3:
		model=copy.copy(args[0]); obs=copy.copy(args[1])
		vmin=copy.copy(args[2])
	elif len(args) == 4:
		model=copy.copy(args[0]); obs=copy.copy(args[1])
		vmin=copy.copy(args[2]); vmax=copy.copy(args[3])
	elif len(args) == 5:
		model=copy.copy(args[0]); obs=copy.copy(args[1])
		vmin=copy.copy(args[2]); vmax=copy.copy(args[3]); maxdiff=copy.copy(args[4]);
	elif len(args) > 5:
		sys.exit(' Too many inputs')

	model=np.atleast_1d(model); obs=np.atleast_1d(obs)
	if model.shape != obs.shape:
		sys.exit(' Model and Observations with different size.')
	if vmax<=vmin:
		sys.exit(' vmin cannot be higher than vmax.')

	ind=np.where((np.isnan(model)==False) & (np.isnan(obs)==False) & (model>vmin) & (model<vmax) & (obs>vmin) & (obs<vmax))
	if model.shape[0]>1 and np.any(ind)==False:
		sys.exit(' Array without valid numbers.')

	ferr=np.zeros((model.shape[0],8),'f')*np.nan
	for i in range(0,model.shape[0]):
		if model[i]>vmin and model[i]<vmax and obs[i]>vmin and obs[i]<vmax and abs(model[i]-obs[i]) < maxdiff:
			ferr[i,0] = model[i] - obs[i]  # bias
			ferr[i,1] = np.sqrt(((model[i] - obs[i]) ** 2)) # RMSE
			ferr[i,2] = ferr[i,0] / obs[i] # NBias
			ferr[i,3] = ferr[i,1] / obs[i] # NRMSE
			ferr[i,4] = np.sqrt(((model[i]-np.nanmean(model[ind[0]])) - (obs[i]-np.nanmean(obs[ind[0]])))**2) # SCrmse, Mentaschi et al. (2013)
			ferr[i,5] = np.sqrt(((model[i]-np.nanmean(model[ind[0]])) - (obs[i]-np.nanmean(obs[ind[0]])))**2)/obs[i] # SI, Mentaschi et al. (2013)
			if model[i]*obs[i]>0.001 and model[i]*obs[i]<9999.:
				ferr[i,6] = np.sqrt(((model[i] - obs[i])**2) / (model[i]*obs[i]))  # HH, Mentaschi et al. (2013)
			ferr[i,7]=1. # Correlation coefficient between two single values

	aux=np.where(ferr==np.inf)
	if np.any(aux):
		ferr[aux[0],aux[1]]=np.nan
	return ferr


# Plots ==========================

# Error X Forecast time
def errXftime(timeAhead,merr,ccol,mmark,llinst,Nvar,Nmetric,fpath,fid,sl):

	ccol = np.atleast_1d(ccol) ; mmark = np.atleast_1d(mmark) ; llinst = np.atleast_1d(llinst)
	merr = np.atleast_2d(merr)
	# color='green', marker='o', linestyle='dashed'
	fig1 = plt.figure(1,figsize=(5,5)); ax1 = fig1.add_subplot(111)
	plt.grid()
	# each model
	for i in range(0,merr.shape[0]):
		ax1.plot(timeAhead/(3600.*24.),gaussian_filter(merr[i,:], 0.8), color=ccol[i], marker=mmark[i], linestyle=llinst[i], linewidth=1.)
	# ensemble average (exclude control)
	ax1.set_xlabel("Forecast Time (Days)",size=sl); ax1.set_ylabel(Nmetric+" "+Nvar,size=sl);
	plt.tight_layout();plt.axis('tight')
	ax1.set_xlim(left=0., right=(timeAhead/(3600.*24.)).max())
	plt.locator_params(axis='y', nbins=7) ; plt.locator_params(axis='x', nbins=7)
	plt.savefig(fpath+fid+Nmetric+'XFtime_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)

# Plot Error X Percentiles, for diff forecast times
def errXpercentile(nvepe,vepe,merr,ccol,mmark,llinst,Nvar,Nmetric,fpath,fid,sl):

	ccol = np.atleast_1d(ccol) ; mmark = np.atleast_1d(mmark) ; llinst = np.atleast_1d(llinst)
	vepe = np.atleast_2d(vepe) ; merr = np.atleast_2d(merr) ;

	fig1 = plt.figure(1,figsize=(5,5)) ; ax11 = fig1.add_subplot(111)
	plt.grid()
	#if Nmetric=='Bias' or Nmetric=='NBias':
	#	ax11.axhline(y=0, color='w')
	for i in range(0,merr.shape[0]):
		ax11.plot(nvepe,gaussian_filter(merr[i,:], 0.8), color=ccol[i], marker=mmark[i], linestyle=llinst[i], linewidth=1.)

	plt.figure(1)
	ax11.set_xlabel('Quantiles of '+Nvar,size=sl); ax11.set_ylabel(Nmetric,size=sl)
	plt.locator_params(axis='y', nbins=7) ; plt.locator_params(axis='x', nbins=7)
	plt.savefig(fpath+fid+Nmetric+'Xpercentile_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)

# Contourf of Error X forecast time X time
def cerrXftimetime(dates,timeAhead,merr,palt,Nvar,Nmetric,fpath,fid,sl):

	fig, ax1 = plt.subplots(1,figsize=(12,3.5), sharex=True, sharey=True)
	plt.xlabel("Time (month/year)",size=sl)
	if Nmetric=='Bias' or Nmetric=='NBias':
		lev= np.linspace(-np.nanpercentile(abs(merr),98.),np.nanpercentile(abs(merr),98.),100)
		im1 = ax1.contourf(dates,timeAhead/(3600.*24.),merr.T,lev,cmap=palt,extend="both")
	else:
		lev= np.linspace(np.nanmin(merr),np.nanpercentile(merr,98.),100)
		im1 = ax1.contourf(dates,timeAhead/(3600.*24.),merr.T,lev,cmap=palt,extend="max")

	divider = make_axes_locatable(ax1); cax = divider.append_axes("right", size="2%", pad=0.2); cb = plt.colorbar(im1, cax=cax); tick_locator = ticker.MaxNLocator(nbins=7)
	cb.locator = tick_locator; cb.update_ticks()
	ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax1.xaxis.set_major_formatter( DateFormatter('%m/%Y') )
	plt.tight_layout();plt.axis('tight')
	fig.text(0.008, 0.53, 'Forecast Time (Days)', va='center', rotation='vertical',size=sl)
	plt.savefig(fpath+fid+Nmetric+'XFtimeTime_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig)

# Contourf of Error X Forecast Time X Quaniles
def errXftimeXQuantile(*args):

	if len(args) == 8:
		meval=copy.copy(args[0]); nvepe=copy.copy(args[1]); timeAhead=copy.copy(args[2]); Nvar=copy.copy(args[3]); Nmetric=copy.copy(args[4]); fpath=copy.copy(args[5]); fid=copy.copy(args[6]); sl=copy.copy(args[7]);
	elif len(args) == 9:
		meval=copy.copy(args[0]); nvepe=copy.copy(args[1]); timeAhead=copy.copy(args[2]); Nvar=copy.copy(args[3]); Nmetric=copy.copy(args[4]); fpath=copy.copy(args[5]); fid=copy.copy(args[6]); sl=copy.copy(args[7]); lev=copy.copy(args[8]);

	meval = gaussian_filter(meval, 0.8) # smooth
	fig,ax1 = plt.subplots(figsize=(6,5))
	if Nmetric=='Bias' or Nmetric=='NBias':
		try:
			lev
		except:
			lev=np.linspace(-np.nanpercentile(np.abs(meval[meval[:,:]>-999.]),99.9),np.nanpercentile(np.abs(meval[meval[:,:]>-999.]),99.9),100)

		palette = plt.cm.RdBu_r
		im2 = ax1.contourf(nvepe, (timeAhead/(3600*24)),meval,lev,cmap=palette,extend="both")
	elif Nmetric=='CC':
		try:
			lev
		except:
			lev=np.linspace(np.nanmin(meval[meval[:,:]>-999.]),1.,100)

		palette = plt.cm.gist_stern
		im2 = ax1.contourf(nvepe, (timeAhead/(3600*24)),meval,lev,cmap=palette,extend="min")
	else:
		try:
			lev
		except:
			lev=np.linspace(np.nanmin(meval[meval[:,:]>-999.]),np.nanpercentile(meval[meval[:,:]>-999.],99.9),100)

		palette = plt.cm.gist_stern_r
		im2 = ax1.contourf(nvepe, (timeAhead/(3600*24)),meval,lev,cmap=palette,extend="max")

	ax1.set_xlabel('Quantile of '+Nvar,size=sl); ax1.set_ylabel("Forecast Time (Days)",size=sl); plt.grid()
	divider = make_axes_locatable(ax1); cax = divider.append_axes("right", size="4%", pad=0.2); cb = plt.colorbar(im2, cax=cax); tick_locator = ticker.MaxNLocator(nbins=7)
	cb.locator = tick_locator; cb.update_ticks()
	plt.locator_params(axis='y', nbins=7) ; plt.locator_params(axis='x', nbins=7)
	plt.tight_layout();plt.axis('tight')
	ax1.xaxis.grid(True, zorder=1)
	plt.grid(c='k', ls='-', alpha=0.3)
	plt.savefig(fpath+fid+Nmetric+'XForecastTimeXQuantile_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig); del fig, ax1,im2

# QQ-plots
def eqqplot(obs,model,ccol,mmark,llinst,Nvar,fpath,fid,sl):

	ccol = np.atleast_1d(ccol) ; mmark = np.atleast_1d(mmark) ; llinst = np.atleast_1d(llinst)
	model = np.atleast_2d(model)

	p = np.arange(1,99,1)
	qobs = np.zeros((model.shape[0],p.shape[0]),'f')*np.nan
	qm = np.zeros((model.shape[0],p.shape[0]),'f')*np.nan

	for i in range(0,p.shape[0]):
		for j in range(0,model.shape[0]):
			qobs[j,i] = np.nanpercentile(obs[:],p[i])
			qm[j,i] = np.nanpercentile(model[j,:],p[i])

	a=np.nanmin([qobs,qm]) ; b=np.nanmax([qobs,qm])
	aux=np.linspace(a-0.5*a,b+0.5*a,p.shape[0])

	fig1 = plt.figure(1,figsize=(5,4.5)); ax1 = fig1.add_subplot(111)
	ax1.plot(aux,aux,'k',linewidth=1.)
	for i in range(0,model.shape[0]):
		ax1.plot(qobs[i,:],qm[i,:], color=ccol[i], marker=mmark[i], linestyle=llinst[i], linewidth=1.)

	plt.ylim(ymax = aux.max(), ymin = aux.min())
	plt.xlim(xmax = aux.max(), xmin = aux.min())
	plt.locator_params(axis='y', nbins=7) ; plt.locator_params(axis='x', nbins=7)
	ax1.set_xlabel("Observation "+Nvar,size=sl); ax1.set_ylabel("Model "+Nvar,size=sl);
	plt.tight_layout(); # plt.axis('tight')
	plt.grid(c='k', ls='-', alpha=0.3)
	plt.savefig(fpath+fid+'QQplot_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)

# Probability Density Function plots
def epdf(obs,model,ccol,mmark,llinst,Nvar,fpath,fid,sl):

	ccol = np.atleast_1d(ccol) ; mmark = np.atleast_1d(mmark) ; llinst = np.atleast_1d(llinst)
	model = np.atleast_2d(model) ; obs = np.atleast_2d(obs)

	px=np.linspace(np.nanmin([np.nanmin(obs),np.nanmin(model)]),np.nanmax([np.nanmax(obs),np.nanmax(model)])*0.95,100)

	fig1 = plt.figure(1,figsize=(5,5)); ax = fig1.add_subplot(111)
	# https://matplotlib.org/examples/color/named_colors.html

	for i in range(0,obs.shape[0]):
		dx = gaussian_kde(obs[i,:])
		ax.fill_between(px, 0., gaussian_filter(dx(px), 1.), color='tan', alpha=0.7)

	for i in range(0,model.shape[0]):
		de = gaussian_kde(model[i,:])
		ax.plot(px,gaussian_filter(de(px), 1.), color=ccol[i], marker=mmark[i], linestyle=llinst[i], linewidth=0.5)
		del de

	ax.set_xlabel(Nvar,size=sl); ax.set_ylabel("Probability Density",size=sl);
	plt.tight_layout();plt.axis('tight')
	plt.grid(c='k', ls='-', alpha=0.3)
	plt.savefig(fpath+fid+'PDF_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)

# Probability Density Function plots - using log scale
def epdflogs(obs,model,ccol,mmark,llinst,Nvar,fpath,fid,sl):

	ccol = np.atleast_1d(ccol) ; mmark = np.atleast_1d(mmark) ; llinst = np.atleast_1d(llinst)
	model = np.atleast_2d(model) ; obs = np.atleast_2d(obs)

	px=np.linspace(np.nanmin([np.nanmin(obs),np.nanmin(model)]),np.nanmax([np.nanmax(obs),np.nanmax(model)])*0.95,100)

	fig1 = plt.figure(1,figsize=(5,5)); ax = fig1.add_subplot(111)
	# https://matplotlib.org/examples/color/named_colors.html

	for i in range(0,obs.shape[0]):
		dx = gaussian_kde(obs[i,:])
		ax.fill_between(px, 0., gaussian_filter(dx(px), 1.), color='tan', alpha=0.7)

	for i in range(0,model.shape[0]):
		de = gaussian_kde(model[i,:])
		ax.plot(px,gaussian_filter(de(px), 1.), color=ccol[i], marker=mmark[i], linestyle=llinst[i], linewidth=0.5)
		del de

	ax.set_yscale( "log" )
	ax.set_xlabel(Nvar,size=sl); ax.set_ylabel("Probability Density",size=sl);
	plt.tight_layout();plt.axis('tight');plt.ylim(ymin = 0.001);plt.ylim(ymax = 1.)
	plt.grid(c='k', ls='-', alpha=0.3)
	plt.savefig(fpath+fid+'PDFlogs_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)

# Error X Latitude
def errXlat(lat,merr,ccol,mmark,llinst,Nvar,Nmetric,fpath,fid,sl,latick):

	ccol = np.atleast_1d(ccol) ; mmark = np.atleast_1d(mmark) ; llinst = np.atleast_1d(llinst)
	merr = np.atleast_2d(merr)

	fig1 = plt.figure(1,figsize=(6,4)); ax1 = fig1.add_subplot(111)
	# each model
	for i in range(0,merr.shape[0]):
		ax1.plot(lat,gaussian_filter(merr[i,:], 0.8), color=ccol[i], marker=mmark[i], linestyle=llinst[i], linewidth=0.5)

	ax1.set_xlabel("Latitude",size=sl); ax1.set_ylabel(Nmetric+" "+Nvar,size=sl);
	plt.tight_layout();plt.axis('tight')
	ax1.set_xlim(left=np.nanmin(lat), right=np.nanmax(lat))
	plt.grid(c='k', ls='-', alpha=0.3)
	plt.xticks(latick)
	plt.savefig(fpath+fid+Nmetric+'XLatitude_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)


