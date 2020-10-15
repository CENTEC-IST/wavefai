import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from pylab import DateFormatter

SL = 13

matplotlib.rcParams.update({'font.size': SL})
plt.rc('font', size=SL)
matplotlib.rc('xtick', labelsize=SL)
matplotlib.rc('ytick', labelsize=SL)
matplotlib.rcParams.update({'font.size': SL})

def chicklet_plot(img_name, data, date_time, forecast_time, lev, cmap=plt.cm.jet, extend="both"):
	fig, ax1 = plt.subplots(1,figsize=(16,4), sharex=True, sharey=True)
	plt.xlabel("Time (month/year)",size=SL)
	im1 = ax1.contourf(date_time,(forecast_time/3600.)/24.,data,lev,cmap=cmap,extend=extend)
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
	fig.text(0.000, 0.53, 'Forecast Time (Days)', va='center', rotation='vertical',size=SL)
	plt.savefig(img_name, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig)
