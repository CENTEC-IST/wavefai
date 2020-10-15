import matplotlib.pyplot as plt
import matplotlib.dates as mdates

matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl)
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})

def chicklet_plot(img_name, data, lev, cmap=plt.cm.jet, extend="both"):
    fig, ax1 = plt.subplots(1,figsize=(16,4), sharex=True, sharey=True)
    plt.xlabel("Time (month/year)",size=sl)
    im1 = ax1.contourf(datm,(eft/3600.)/24.,data,lev,cmap=cmap,extend=extend)
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
