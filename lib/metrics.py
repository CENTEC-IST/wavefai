
# Table summary statistics
def smrstat(*args):
	'''
	Summary Statistics
	Input: one array of interest
	Output: mean, variance, skewness, kurtosis, min, max, percentile80, percentile90, percentile95, percentile99, percentile99.9, LCV, Lskew, Lkurtosis
	'''

	# TODO do this without copy.copy
	# TODO remove argument list
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

