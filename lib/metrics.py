import numpy as np
import lmoments3 as lm

# Table summary statistics
def smrstat(x, vmin=-np.inf, vmax=np.inf):
	'''
	Summary Statistics
	Input: one array of interest
	Output: mean, variance, skewness, kurtosis, min, max, percentile80, percentile90, percentile95, percentile99, percentile99.9, LCV, Lskew, Lkurtosis
	'''

	ind=np.where((np.isnan(x)==False) & (x>vmin) & (x<vmax))
	if np.any(ind):
		x_0=[ind[0]]
	else:
		sys.exit(' Array without valid numbers.')

	ferr=np.zeros((14),'f')*np.nan
	ferr[0] = np.mean(x_0)
	ferr[1] = np.var(x_0)
	ferr[2] = scipy.stats.skew(x_0)
	ferr[3] = scipy.stats.kurtosis(x_0)
	ferr[4] = np.min(x_0)
	ferr[5] = np.max(x_0)
	ferr[6] = np.percentile(x_0,80)
	ferr[7] = np.percentile(x_0,90)
	ferr[8] = np.percentile(x_0,95)
	ferr[9] = np.percentile(x_0,99)
	ferr[10] = np.percentile(x_0,99.9)
	# Hosking & Wallis L-moment ratios
	# pip install git+https://github.com/OpenHydrology/lmoments3.git
	hwlm = lm.lmom_ratios(x_0, nmom=5)
	ferr[11] = hwlm[1]/hwlm[0]
	ferr[12] = hwlm[2]
	ferr[13] = hwlm[3]

	return ferr

