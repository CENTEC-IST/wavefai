import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import DateFormatter

from scipy.ndimage.filters import gaussian_filter
from scipy.stats import gaussian_kde

SL = 13

plt.rc('font', size=SL)
matplotlib.rc('xtick', labelsize=SL)
matplotlib.rc('ytick', labelsize=SL)
matplotlib.rcParams.update({'font.size': SL})
matplotlib.use('Agg')
# plt.ioff()

DEFAULT_PALETTE = plt.cm.jet
DEFAULT_PALETTE.set_bad('aqua', 10.0)

def chicklet_plot(img_name, data, date_time, forecast_time, lev, color_palette=DEFAULT_PALETTE, extend="both", sl=SL):
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
	plt.savefig(img_name, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig)

# Error X Forecast time
def errXftime(img_name, timeAhead, merr, ccol, mmark, llinst, Nvar, Nmetric, sl=SL):

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
def errXpercentile(img_name, nvepe, vepe, merr, ccol, mmark, llinst, Nvar, Nmetric, sl=SL):

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
def cerrXftimetime(img_name, dates, timeAhead, merr, Nvar, Nmetric, sl=SL, color_palette=DEFAULT_PALETTE):

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
def errXftimeXQuantile(img_name, meval, nvepe, timeAhead, Nvar, Nmetric, sl=SL, lev=None):
	# TODO finish fixing this function
	meval = gaussian_filter(copy.copy(meval), 0.8) # smooth
	fig,ax1 = plt.subplots(figsize=(6,5))
	if Nmetric=='Bias' or Nmetric=='NBias':
		if not lev:
			lev=np.linspace(-np.nanpercentile(np.abs(meval[meval[:,:]>-999.]),99.9),np.nanpercentile(np.abs(meval[meval[:,:]>-999.]),99.9),100)

		palette = plt.cm.RdBu_r
		im2 = ax1.contourf(nvepe, (timeAhead/(3600*24)),meval,lev,cmap=palette,extend="both")
	elif Nmetric=='CC':
		if not lev:
			lev=np.linspace(np.nanmin(meval[meval[:,:]>-999.]),1.,100)

		palette = plt.cm.gist_stern
		im2 = ax1.contourf(nvepe, (timeAhead/(3600*24)),meval,lev,cmap=palette,extend="min")
	else:
		if not lev:
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
	plt.savefig(img_name+Nmetric+'XForecastTimeXQuantile_'+Nvar+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig); del fig, ax1,im2

# QQ-plots
def eqqplot(img_name, obs, model, ccol, mmark, llinst, Nvar, sl=SL):

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
def epdf(img_name, obs, model, ccol, mmark, llinst, Nvar, sl=SL):

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
def epdflogs(img_name, obs, model, ccol, mmark, llinst, Nvar, sl=SL):

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
def errXlat(img_name, lat, merr, ccol, mmark, llinst, Nvar, Nmetric, latick, sl=SL):

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
