import numpy as np

def metrics(model, obs, vmin=-np.inf, vmax=np.inf, maxdiff=np.inf):
	'''
	Error Metrics. Mentaschi et al. (2013)
	Input: two arrays of model and observation, respectively.
		They must have the same size
	Output: ferr array with shape equal to 8
		bias, RMSE, NBias, NRMSE, SCrmse, SI, HH, CC
	'''

	model=np.atleast_1d(model); obs=np.atleast_1d(obs)
	if model.shape != obs.shape:
		raise ValueError('Model and Observations with different size.')
	if vmax<=vmin:
		sys.exit(' vmin cannot be higher than vmax.')

	ind=np.where((np.isnan(model)==False) & (np.isnan(obs)==False) & (model>vmin) & (model<vmax) & (obs>vmin) & (obs<vmax) & (np.abs(model-obs)<=maxdiff) )
	if np.any(ind) or model.shape[0]==1:
		model = model[ind[0]]
		obs = obs[ind[0]]
	else:
		raise ValueError('Array without valid numbers.')

	ferr=np.zeros((8),'f')*np.nan
	ferr[0] = model.mean() - obs.mean() # Bias
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


def imetrics(model, obs, vmin=-np.inf, vmax=np.inf, maxdiff=np.inf):
	'''
	Error Metrics at each individual instant. Same equations of Mentaschi et al. (2013), but using n equal to one.
	Input: two arrays of model and observation, respectively.
		They must have the same size
	Output: ferr array with shape equal to [inpShape,8]
		bias, RMSE, NBias, NRMSE, SCrmse, SI, HH, CC
	'''

	model=np.atleast_1d(model); obs=np.atleast_1d(obs)
	if model.shape != obs.shape:
		sys.exit(' Model and Observations with different size.')
	if vmax<=vmin:
		sys.exit(' vmin cannot be higher than vmax.')

	ind=np.where((np.isnan(model)==False) & (np.isnan(obs)==False) & (model>vmin) & (model<vmax) & (obs>vmin) & (obs<vmax))
	if model.shape[0]>1 and np.any(ind)==False:
		raise ValueError('Array without valid numbers.')

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
