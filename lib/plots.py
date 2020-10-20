import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import DateFormatter, date2num
from datetime import datetime

from scipy.ndimage.filters import gaussian_filter
from scipy.stats import gaussian_kde

FONT_SIZE = 13

# SET some default values
plt.rc('font', size=FONT_SIZE)
matplotlib.rc('xtick', labelsize=FONT_SIZE)
matplotlib.rc('ytick', labelsize=FONT_SIZE)
matplotlib.rcParams.update({'font.size': FONT_SIZE})
matplotlib.use('Agg')

DEFAULT_PALETTE = plt.cm.jet
# DEFAULT_PALETTE.set_bad('aqua', 10.0) # XXX Was used in merr.py

def chiclet_plot(img_name, data, date_time, forecast_time, lev,
		color_palette=DEFAULT_PALETTE, extend="both", sl=FONT_SIZE, img_format='png'):
	'''Chiclet plot function... TODO
	Prameters:
		img_name -- name of the output image file. The image format can be controlled with `img_format`
		data -- TODO
		date_time -- TODO
		forecast_time -- forecast timestamps
		lev -- TODO
	Optional:
		color_palette -- matplotlib.cm object representing color palette. Defaults to plt.cm.jet
		extend -- TODO
		sl -- text size
		img_format -- file type to produce ('jpg' or 'png'). Defaults to 'png'
	'''
	fig, ax1 = plt.subplots(1,figsize=(16,4), sharex=True, sharey=True)
	plt.xlabel("Time (month/year)",size=sl)
	im1 = ax1.contourf(date_time,(forecast_time/3600.)/24., data, lev, cmap=color_palette, extend=extend)
	divider = make_axes_locatable(ax1)
	cax = divider.append_axes("right", size="2%", pad=0.2)
	cb = plt.colorbar(im1, cax=cax)
	tick_locator = ticker.MaxNLocator(nbins=7)
	cb.locator = tick_locator
	cb.update_ticks()
	ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
	ax1.xaxis.set_major_formatter( DateFormatter('%m/%Y') )
	plt.tight_layout()
	plt.axis('tight')
	fig.text(0.000, 0.53, 'Forecast Time (Days)', va='center', rotation='vertical',size=sl)
	plt.savefig(img_name, dpi=300, facecolor='w', edgecolor='w',orientation='portrait',
			papertype=None, format=img_format,transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig)



def time_series_plot(img_name, observation_data, ensemble_members, cycle_time, forecast_time,
		variable_name = None, text=None, sl=FONT_SIZE, img_format='png'):
	'''This function generates a plot for .. TODO
	Parameters:
		img_name -- name of the output image file. The image format can be controlled with `img_format`
		observation_data -- 1D array containing the observed data along the forecast time
		ensemble_members -- 2D array containing each ensembler data along the forecast time
		cycle_time -- timestamp of the specific day to plot
		forecast_time -- forecast timestamps
	Optional:
		variable_name -- name to put in YY axis
		text -- text to put at the bottom of the graph
		sl -- text size
		img_format -- file type to produce ('jpg' or 'png'). Defaults to 'png'
	'''

	n_ensemblers = ensemble_members.shape[0]
	pred_time = np.asarray([date2num(datetime.fromtimestamp(cycle_time + forecast_time[j])) for j in range(forecast_time.shape[0])])

	fig, ax1 = plt.subplots(1,figsize=(16,4), sharex=True, sharey=True)

	# Observation
	ax1.plot(pred_time,gaussian_filter(observation_data,1.),color='k',label='OBS',linewidth=2.,zorder=2)
	# Ensemble Mean
	ax1.plot(pred_time,gaussian_filter(np.nanmean(ensemble_members, axis=0), 1.),color='brown',label='ECMWF_EM',linewidth=3.,zorder=3)
	# Ensemble members
	for j in range(n_ensemblers):
		ax1.plot(pred_time, gaussian_filter(ensemble_members[j,:],1.),color='lightsalmon',linewidth=0.5,zorder=1)

	plt.legend(loc='upper left')
	plt.xlabel("Time (day/month)",size=sl)
	if variable_name:
		plt.ylabel(variable_name,size=sl)
	ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
	ax1.xaxis.set_major_formatter( DateFormatter('%d/%m') )
	if text:
		fig.text(0.015,0.05, text, va='center', rotation='horizontal',size=sl)
	plt.grid()
	plt.tight_layout()
	plt.axis('tight')
	plt.xlim(xmin=pred_time[0]-0.1,xmax=pred_time[-1]+0.1)
	plt.savefig(img_name, dpi=300, facecolor='w', edgecolor='w',orientation='portrait',
			papertype=None, format=img_format,transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig)




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TODO test the code bellow -- it was copied straight from Ricardo's merr.py

# Error X Forecast time
def errXftime(img_name, timeAhead, merr, ccol, mmark, llinst, Nvar, Nmetric, sl=FONT_SIZE):

	ccol = np.atleast_1d(ccol) ; mmark = np.atleast_1d(mmark) ; llinst = np.atleast_1d(llinst)
	merr = np.atleast_2d(merr)
	# color='green', marker='o', linestyle='dashed'
	fig1 = plt.figure(1,figsize=(5,5)); ax1 = fig1.add_subplot(111)
	plt.grid()
	# each model
	for i in range(0,merr.shape[0]):
		ax1.plot(timeAhead/(3600.*24.),gaussian_filter(merr[i,:], 0.8), color=ccol[i], marker=mmark[i], linestyle=llinst[i], linewidth=1.)
	# ensemble average (exclude control)
	ax1.set_xlabel("Forecast Time (Days)",size=sl)
	ax1.set_ylabel(Nmetric+" "+Nvar,size=sl);
	plt.tight_layout()
	plt.axis('tight')
	ax1.set_xlim(left=0., right=(timeAhead/(3600.*24.)).max())
	plt.locator_params(axis='y', nbins=7) ; plt.locator_params(axis='x', nbins=7)
	plt.savefig(img_name+Nmetric+'XFtime_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)

# Plot Error X Percentiles, for diff forecast times
def errXpercentile(img_name, nvepe, vepe, merr, ccol, mmark, llinst, Nvar, Nmetric, sl=FONT_SIZE):

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
	plt.savefig(img_name+Nmetric+'Xpercentile_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)

# Contourf of Error X forecast time X time
def cerrXftimetime(img_name, dates, timeAhead, merr, Nvar, Nmetric, sl=FONT_SIZE, color_palette=DEFAULT_PALETTE):

	fig, ax1 = plt.subplots(1,figsize=(12,3.5), sharex=True, sharey=True)
	plt.xlabel("Time (month/year)",size=sl)
	if Nmetric=='Bias' or Nmetric=='NBias':
		lev= np.linspace(-np.nanpercentile(abs(merr),98.),np.nanpercentile(abs(merr),98.),100)
		im1 = ax1.contourf(dates,timeAhead/(3600.*24.),merr.T,lev,cmap=color_palette,extend="both")
	else:
		lev= np.linspace(np.nanmin(merr),np.nanpercentile(merr,98.),100)
		im1 = ax1.contourf(dates,timeAhead/(3600.*24.),merr.T,lev,cmap=color_palette,extend="max")

	divider = make_axes_locatable(ax1); cax = divider.append_axes("right", size="2%", pad=0.2); cb = plt.colorbar(im1, cax=cax); tick_locator = ticker.MaxNLocator(nbins=7)
	cb.locator = tick_locator; cb.update_ticks()
	ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax1.xaxis.set_major_formatter( DateFormatter('%m/%Y') )
	plt.tight_layout();plt.axis('tight')
	fig.text(0.008, 0.53, 'Forecast Time (Days)', va='center', rotation='vertical',size=sl)
	plt.savefig(img_name+Nmetric+'XFtimeTime_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig)

# Contourf of Error X Forecast Time X Quaniles
def errXftimeXQuantile(img_name, meval, nvepe, timeAhead, Nvar, Nmetric, sl=FONT_SIZE, lev=None):
	# TODO finish fixing this function
	meval = gaussian_filter(copy.copy(meval), 0.8) # smooth
	fig,ax1 = plt.subplots(figsize=(6,5))
	if Nmetric=='Bias' or Nmetric=='NBias':
		if not lev:
			lev=np.linspace(-np.nanpercentile(np.abs(meval[meval[:,:]> -999.]),99.9),np.nanpercentile(np.abs(meval[meval[:,:]> -999.]),99.9),100)

		palette = plt.cm.RdBu_r
		im2 = ax1.contourf(nvepe, (timeAhead/(3600*24)),meval,lev,cmap=palette,extend="both")
	elif Nmetric=='CC':
		if not lev:
			lev=np.linspace(np.nanmin(meval[meval[:,:]> -999.]),1.,100)

		palette = plt.cm.gist_stern
		im2 = ax1.contourf(nvepe, (timeAhead/(3600*24)),meval,lev,cmap=palette,extend="min")
	else:
		if not lev:
			lev=np.linspace(np.nanmin(meval[meval[:,:]> -999.]),np.nanpercentile(meval[meval[:,:]> -999.],99.9),100)

		palette = plt.cm.gist_stern_r
		im2 = ax1.contourf(nvepe, (timeAhead/(3600*24)),meval,lev,cmap=palette,extend="max")

	ax1.set_xlabel('Quantile of '+Nvar,size=sl); ax1.set_ylabel("Forecast Time (Days)",size=sl); plt.grid()
	divider = make_axes_locatable(ax1); cax = divider.append_axes("right", size="4%", pad=0.2); cb = plt.colorbar(im2, cax=cax); tick_locator = ticker.MaxNLocator(nbins=7)
	cb.locator = tick_locator; cb.update_ticks()
	plt.locator_params(axis='y', nbins=7) ; plt.locator_params(axis='x', nbins=7)
	plt.tight_layout();plt.axis('tight')
	ax1.xaxis.grid(True, zorder=1)
	plt.grid(c='k', ls='-', alpha=0.3)
	plt.savefig(img_name+Nmetric+'XForecastTimeXQuantile_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig); del fig, ax1,im2

# QQ-plots
def eqqplot(img_name, obs, model, ccol, mmark, llinst, Nvar, sl=FONT_SIZE):

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
	plt.savefig(img_name+'QQplot_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)

# Probability Density Function plots
def epdf(img_name, obs, model, ccol, mmark, llinst, Nvar, sl=FONT_SIZE):

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
	plt.savefig(img_name+'PDF_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)

# Probability Density Function plots - using log scale
def epdflogs(img_name, obs, model, ccol, mmark, llinst, Nvar, sl=FONT_SIZE):

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
	plt.savefig(img_name+'PDFlogs_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)

# Error X Latitude
def errXlat(img_name, lat, merr, ccol, mmark, llinst, Nvar, Nmetric, latick, sl=FONT_SIZE):

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
	plt.savefig(img_name+Nmetric+'XLatitude_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)
