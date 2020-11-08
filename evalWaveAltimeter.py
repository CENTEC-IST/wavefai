import numpy as np
import os.path
import netCDF4 as nc
from os import path
from scipy import interpolate
from pylab import *
import pandas as pd
from calendar import timegm
from time import strptime, gmtime, strftime
from scipy.ndimage.filters import gaussian_filter
from geopy import distance
import pyresample
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm, interp
import merr
colormap = cm.GMT_polar
palette = plt.cm.jet
palette.set_bad('aqua', 10.0)
import matplotlib.colors as colors

# Font size and style
sl=15
matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl) 
matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); matplotlib.rcParams.update({'font.size': sl})

# Number of cores without hyperthread for pyresample
npcs=2

# weight function
dlim = 51000.
def wf(pdist):
	a=(1 - pdist / 52000.0)
	return (abs(a)+a)/2

# weight function
dlim = 101000.
def wf2(pdist):
	a=(1 - pdist / 102000.0)
	return (abs(a)+a)/2


# cyclone radius of influence (in km)
crof=900

#   Satellites selected
# Altimeters:
altsats=['CRYOSAT2','SARAL','JASON3','SENTINEL3A']; altsatsn=['CRYOSAT-2','SARAL','JASON-3','JASON-3']
aodnap='/home/rmc/hdd1/rmc/work/ist/EXWAV/Lorenzo/proc/showdata/AODN_altm/'
# Scatterometers:
scatsats=['METOPA_N','METOPB_N']; scatsatsn=['METOP-A','METOP-B']
aodnsp='/media/degas/satellite/scatterometer/AODN_scat/'

# Altimeter Quality Control parameters
max_swh_rms = 1.5  # Max RMS of the band significant wave height
min_swh_numval = 17. # Min number of valid points used to compute significant wave height (Saral automaticaly multiplies this number by 2, internally)
max_sig0_rms = 0.8 # Max RMS of the backscatter coefficient
max_swh_qc = 2.0 # Max SWH Ku band quality control
mbd=200. # Minimum Water Depth (m)
mdfc=30. # Minimum Distance from the coast (km)
hsmax=20.; wspmax=60.
maxti = 5401.  # maximum time distance between regular grid and satellite data  run1(3h,500km,2degr), run2(0.5h,25km,0.5degr)
# maximum distance (in meters) between saletellite and grid point
dlim = 25000.; dgrw=0.5  #


# ibtracs.Lorenzo
ibtr = pd.read_csv("/home/rmc/hdd1/rmc/work/ist/EXWAV/Lorenzo/ibtracs.Lorenzo.csv") 
auxt=ibtr.values[:,6]; etype=ibtr.values[:,7]; lat=ibtr.values[:,8]; lon=ibtr.values[:,9]
# select tropical storm only "TS"
ind=np.where((etype=='TS') | (etype=='ET')); auxt=np.copy(auxt[ind]); lat=np.array(lat[ind]).astype('float'); lon=np.array(lon[ind]).astype('float'); del ind, etype
# organize time
t=np.zeros(auxt.shape[0],'double')
for i in range(0,auxt.shape[0]):
		t[i] = float(timegm( time.strptime(auxt[i][0:-6], '%Y-%m-%d %H') ))

# Interpolate ibtracs to 1h
x = np.arange(0, t.shape[0]); xnew=np.arange(x[0],x[-1]+0.1,0.5)
fibt = interpolate.interp1d(x,t); ibtime = fibt(xnew); del fibt
fibt = interpolate.interp1d(x,lat); iblat = fibt(xnew); del fibt
fibt = interpolate.interp1d(x,lon); iblon = fibt(xnew); del fibt, x, xnew, t, auxt, lat, lon
# --------
# Select only a segment of the Hurricane
ind = np.where((iblat>=20)); ibtime=np.copy(ibtime[ind]); iblat=np.copy(iblat[ind]); iblon=np.copy(iblon[ind]); del ind

# Grid mask determining the influence of the cyclone
print(' - Building Cyclone Mask ...')
clat=np.arange( np.round((np.min(iblat)-crof/100.-1)), np.round((np.max(iblat)+crof/100.+1))+0.1, 0.5 )
clon=np.arange( np.round((np.min(iblon)-crof/100.-1)), np.round((np.max(iblon)+crof/100.+1))+0.1, 0.5 )
cmask=np.zeros((ibtime.shape[0],clat.shape[0],clon.shape[0]),'i')
for k in range(0,ibtime.shape[0]):
	aux=np.zeros((cmask.shape[1],cmask.shape[2]),'f')*np.nan
	for i in range(0,clat.shape[0]):
		for j in range(0,clon.shape[0]):
			coords_1 = (clat[i], clon[j]); coords_2 = (iblat[k], iblon[k])
			aux[i,j] = np.float(distance.great_circle(coords_1, coords_2).km)
			del coords_1, coords_2

	ind = np.where(aux<=crof)
	cmask[k,ind[0],ind[1]] = 1
	print('       done '+repr(k))

print(' - Grid Mask done')
# --------

# Selecte AODN Altimeter tracks within the cyclone ===========
auxlat = np.array(np.arange(clat.min()-2,clat.max()+2.1,1)).astype('int')
auxlon = np.array(np.arange(clon.min()-2,clon.max()+2.1,1)).astype('int')
ast=[]; aslat=[]; aslon=[]; ahsk=[]; awnd=[]; asig0kunstd=[]; aswhkunobs=[]; aswhkunstd=[]; aswhkuqc=[]; abd=[]; adfc=[]
for i in range(0,auxlat.shape[0]):

	if auxlat[i]>=0:
		hem='N'
	else:
		hem='S'

	for j in range(0,auxlon.shape[0]):
		if auxlon[j]<0:
			alon=np.copy(np.array(auxlon[j]+360.))
		else:
			alon=np.copy(np.array(auxlon[j]+360.))

		for s in range(0,size(altsats)):
			sfname = aodnap+altsats[s]+'/IMOS_SRS-Surface-Waves_MW_'+altsatsn[s]+'_FV02_'+str(np.int(np.abs(auxlat[i]))).zfill(3)+hem+'-'+str(np.int(alon)).zfill(3)+'E-DM00.nc'
			if path.isfile(sfname) == True:
				# print('ok '+sfname)
				try:
					fu=nc.Dataset(sfname)
				except:
					print(' Attention, '+sfname+' crashed.')
				else:
					st=np.double(fu.variables['TIME'][:])
					if size(st)>10:
						slat=fu.variables['LATITUDE'][:]
						slon=fu.variables['LONGITUDE'][:]
						wnd=fu.variables['WSPD_CAL'][:]
						bd=fu.variables['BOT_DEPTH'][:]*(-1.)
						dfc=fu.variables['DIST2COAST'][:]						
						try: 
							hsk=fu.variables['SWH_KU_CAL'][:]
							sig0kunstd=fu.variables['SIG0_KU_std_dev'][:]
							swhkunobs=fu.variables['SWH_KU_num_obs'][:]
							swhkunstd=fu.variables['SWH_KU_std_dev'][:]
							swhkuqc=fu.variables['SWH_KU_quality_control'][:]
						except:
							# print(' error reading KU, pick KA')
							hsk=fu.variables['SWH_KA_CAL'][:]
							sig0kunstd=fu.variables['SIG0_KA_std_dev'][:]
							swhkunobs=fu.variables['SWH_KA_num_obs'][:]
							swhkunstd=fu.variables['SWH_KA_std_dev'][:]
							swhkuqc=fu.variables['SWH_KA_quality_control'][:]

						ast=np.double(np.append(ast,st)); aslat=np.append(aslat,slat); aslon=np.append(aslon,slon)
						abd=np.append(abd,bd); adfc=np.append(adfc,dfc)
						ahsk=np.append(ahsk,hsk); awnd=np.append(awnd,wnd); 
						asig0kunstd=np.append(asig0kunstd,sig0kunstd); aswhkunobs=np.append(aswhkunobs,swhkunobs)
						aswhkunstd=np.append(aswhkunstd,swhkunstd); aswhkuqc=np.append(aswhkuqc,swhkuqc)
						del st,slat,slon,hsk,wnd,sig0kunstd,swhkunobs,swhkunstd,swhkuqc
						fu.close(); del fu


		print('        Done Lon '+repr(alon))
		del alon

	print('    Done Lat '+repr(auxlat[i]))
	del hem


# Quality Control of Satellite data ----
ast=np.array(np.copy(ast)*24.*3600.+double(timegm( time.strptime('1985010100', '%Y%m%d%H') ))).astype('double')
indq = np.where( (abd>=mbd) & (adfc>=mdfc) & (aswhkunstd<=max_swh_rms) & (asig0kunstd<=max_sig0_rms) & (aswhkunobs>=min_swh_numval) & (aswhkuqc<=max_swh_qc) & (ahsk>0.01) & (ahsk<hsmax) & (awnd>0.01) & (awnd<wspmax) & (ast<=ibtime[-1]+3600.) & (ast>=ibtime[0]-3600.))     
if np.any(indq):
	del asig0kunstd,aswhkunobs,aswhkunstd,aswhkuqc,abd,adfc
	st=np.array(np.copy(ast[indq[0]])).astype('double'); slat=np.array(np.copy(aslat[indq[0]])); slon=np.array(np.copy(aslon[indq[0]]))
	shs=np.array(np.copy(ahsk[indq[0]])); swnd=np.array(np.copy(awnd[indq[0]]))
	del awnd, ahsk, ast, aslat, aslon, indq

slon[slon>180.]=slon[slon>180.]-360.
fid=[] # Select satellite inside the cyclone
for i in range(0,ibtime.shape[0]):
			ind = np.where( (np.abs(st-ibtime[i])<=maxti) & (np.abs(slat-iblat[i])<=((crof/100.))) & (np.abs(slon-iblon[i])<=((crof/100.))) )
			if np.any(ind):
				fid=np.array(np.append(fid,np.array(ind[0]).astype('int'))).astype('int')
				del ind

print(' Total of '+repr(size(fid))+' altimeter data for the model/satellite pairing/matchups, in the next step')
st=np.array(np.copy(st[fid])).astype('double'); slat=np.array(np.copy(slat[fid])); slon=np.array(np.copy(slon[fid]))
shs=np.array(np.copy(shs[fid])); swnd=np.array(np.copy(swnd[fid])); del fid, auxlat, auxlon
# =============================================================

# Final time array where the assessment will be conducted
ftime = np.array(np.arange(timegm(strptime("2019100100",'%Y%m%d%H%M')),timegm(strptime("2019100400",'%Y%m%d%H%M'))+1,3*3600)).astype('double')
# i=0; ind=np.where(np.abs(ftime[i]-st)<=maxti)


#  ========== Model ===================================================================
itimec = np.array(np.arange(ftime[0]-(7*24*3600),ftime[-1]+1,24*3600)).astype('double')

pathn="/home/rmc/hdd1/rmc/work/ist/EXWAV/Lorenzo/dados_model/"
# final forecast time desired
ffct = np.double(np.arange(0,180*3600+1,3*3600))
# Ensemble members
ensmb=np.array(np.arange(0,20+1))

# First info setup: WW3
f=nc.Dataset(pathn+"ww3.2019092500/ww3.2019092500.nc") # HTSGW_surface, PERPW_surface
aux=f.variables['HTSGW_surface'][:,0,:,:]; latww3=f.variables['latitude'][:]; lonww3=f.variables['longitude'][:]
fctww3=np.array(np.arange(0,180+1,3)*3600.).astype('float'); tww3=np.array(f.variables['time'][:]+fctww3).astype('double')
indftww3=[]
for k in range(0,ffct.shape[0]):
	indftww3 = np.append(indftww3,np.where(ffct[k]==fctww3))

indftww3=np.copy(indftww3.astype('int'))
indgww3 = np.where(aux[0,:,:]>0.)
f.close(); del f,aux

# First info setup: GWES
f=nc.Dataset(pathn+"gwes.2019092500/gwes00.glo_30m.2019092500.nc") # HTSGW_surface, PERPW_surface
aux=f.variables['HTSGW_surface'][:,:,:]; latgwes=f.variables['latitude'][:]; longwes=f.variables['longitude'][:]
tgwes=np.array(f.variables['time'][:]).astype('double')
fctgwes=np.array(tgwes-tgwes[0]).astype('float')
indftgwes=[]
for k in range(0,ffct.shape[0]):
	indftgwes = np.append(indftgwes,np.where(ffct[k]==fctgwes))

indftgwes=np.copy(indftgwes.astype('int'))
indggwes = np.where(aux[0,:,:]>0.)
orig_def_gwes = pyresample.geometry.SwathDefinition(lons=longwes[indggwes[1]], lats=latgwes[indggwes[0]])
f.close(); del f,aux

# First info setup: NFCENS , 6h of resolution only
f=nc.Dataset(pathn+"gwes.2019092500/nfcens00.glo_30m.2019092500.nc") # HTSGW_surface only
aux=f.variables['HTSGW_surface'][:,:,:]; latnfcens=f.variables['latitude'][:]; lonnfcens=f.variables['longitude'][:]
tnfcens=np.array(f.variables['time'][:]).astype('double')
fctnfcens=np.array(tnfcens-tnfcens[0]).astype('float')
indftnfcens=[]
for k in range(0,ffct.shape[0]):
	indftnfcens = np.append(indftnfcens,np.where(ffct[k]==fctnfcens))

indftnfcens2=np.array(0)
for k in range(0,fctnfcens.shape[0]):
	ind=np.where(fctnfcens[k]==ffct)
	if np.any(ind):
		indftnfcens2 = np.append(indftnfcens2,ind[0])
		del ind

indftnfcens=np.copy(indftnfcens.astype('int')); indftnfcens2=np.copy(indftnfcens2.astype('int'))
indgnfcens = np.where(aux[0,:,:]>0.)
orig_def_nfcens = pyresample.geometry.SwathDefinition(lons=lonnfcens[indgnfcens[1]],lats=latnfcens[indgnfcens[0]])
f.close(); del f,aux

# Arrays to be allocated
ww3=np.zeros((itimec.shape[0],ffct.shape[0],latww3.shape[0],lonww3.shape[0]),'f')*np.nan
gwes=np.zeros((itimec.shape[0],ffct.shape[0],latgwes.shape[0],longwes.shape[0]),'f')*np.nan
nfcens=np.zeros((itimec.shape[0],ffct.shape[0],latnfcens.shape[0],lonnfcens.shape[0]),'f')*np.nan

# Main loop to read and allocate data
c1=0
for i in range(0,itimec.shape[0]):
	fdata=repr(time.gmtime(itimec[i])[0])+str(time.gmtime(itimec[i])[1]).zfill(2)+str(time.gmtime(itimec[i])[2]).zfill(2)+'00'

	# WW3  ===================================
	efname = pathn+'ww3.'+fdata+'/ww3.'+fdata+'.nc'
	# check if file exists
	if path.isfile(efname) == True:
		try:
			f=nc.Dataset(efname)
			if 'HTSGW_surface' in f.variables.keys():
				ww3[i,:,:,:] = np.array(f.variables['HTSGW_surface'][:,0,:,:])

			f.close(); del f

		except:
			print(' -- Crashed:  '+efname)

		else:
			c1=c1+1

	del efname # -----------------------------


	# GWES ===================================
	aux = np.zeros((21,gwes[i,:,:,:].shape[0],gwes[i,:,:,:].shape[1],gwes[i,:,:,:].shape[2]),'f')*np.nan
	for e in range(0,ensmb.shape[0]):
		efname = pathn+'gwes.'+fdata+'/gwes'+str(e).zfill(2)+'.glo_30m.'+fdata+'.nc'
		# check if file exists
		if path.isfile(efname) == True:
			try:
				f=nc.Dataset(efname)
				if 'HTSGW_surface' in f.variables.keys():
					aux[e,:,:,:] = np.array(f.variables['HTSGW_surface'][:,:,:])[indftgwes,:,:]

				f.close(); del f

			except:
				print(' -- Crashed:  '+efname)

			else:
				c1=c1+1

		del efname # -----------------------------

	gwes[i,:,:,:] = np.array(np.nanmean(aux,axis=0)); del aux


	# NFCENS ===================================
	aux = np.zeros((21,nfcens[i,:,:,:].shape[0],nfcens[i,:,:,:].shape[1],nfcens[i,:,:,:].shape[2]),'f')*np.nan
	for e in range(0,ensmb.shape[0]):
		efname = pathn+'gwes.'+fdata+'/nfcens'+str(e).zfill(2)+'.glo_30m.'+fdata+'.nc'
		# check if file exists
		if path.isfile(efname) == True:
			try:
				f=nc.Dataset(efname)
				if 'HTSGW_surface' in f.variables.keys():
					aux[e,indftnfcens2,:,:] = np.array(f.variables['HTSGW_surface'][:,:,:])[indftnfcens,:,:]

				f.close(); del f

			except:
				print(' -- Crashed:  '+efname)

			else:
				c1=c1+1

		del efname # -----------------------------

	# Interpolate NFCENS from 6h to 3h of time resolution
	for e in range(0,ensmb.shape[0]):
		for j in range(1,ffct.shape[0]-1):
			if np.isnan(np.nanmean(aux[e,j,:,:])):
				aux[e,j,:,:] = np.array( np.nanmean( aux[e,(j-1):(j+2),:,:],axis=0 ) )

	nfcens[i,:,:,:] = np.array(np.nanmean(aux,axis=0)); del aux

	del fdata
	print(' Done '+repr(i))



# Interpolate NFCENS from 1degree to 0.5 degree
[mnlonw,mnlatw]=np.meshgrid(longwes,latgwes)
nfcensi=np.zeros(gwes.shape,'f')*np.nan
for i in range(0,nfcens.shape[0]):
	for j in range(0,nfcens.shape[1]):
		if np.isnan(np.nanmean(nfcens[i,j,:,:]))==False:
			nfcensi[i,j,:,:] = interp(nfcens[i,j,:,:],lonnfcens,latnfcens,mnlonw,mnlatw,checkbounds=False, masked=False, order=1)

# Matching lat/lon
nfcens=np.copy(nfcensi[:,:,0:-5,:]); gwes=np.copy(gwes[:,:,0:-5,:]); del latgwes, longwes, latnfcens, lonnfcens

# Time array
motime=np.zeros((itimec.shape[0],ffct.shape[0]),'d')
for i in range(0,itimec.shape[0]):
	for j in range(0,ffct.shape[0]):
		motime[i,j]=np.double(itimec[i]+ffct[j])

taux = np.array(np.arange(0,7*24*3600+1,24*3600)).astype('double')
ft=[]; fit=[]; fct=[]; flat=[]; flon=[]; fshs=[]; fmhsww3=[]; fmhsgwes=[]; fmhsnfcens=[]
[mnlonww3,mnlatww3]=np.meshgrid(lonww3,latww3)
# Build matchups satellite / model
for i in range(0,ftime.shape[0]):
	# Satellite
	indt = np.where( np.abs(st-ftime[i])<=maxti )
	if np.any(indt):
		print(repr(i))
		targ_def = pyresample.geometry.SwathDefinition(lons=slon[indt], lats=slat[indt])

		# Model
		indm = np.where( np.abs(motime-(ftime[i]))<10.)
		for j in range(0,size(indm[0])):

			ind=np.where(ww3[indm[0][j],indm[1][j],:,:]>0); orig_def_ww3 = pyresample.geometry.SwathDefinition(lons=mnlonww3[ind[0],ind[1]], lats=mnlatww3[ind[0],ind[1]])
			auxww3 = np.array(pyresample.kd_tree.resample_custom(orig_def_ww3,ww3[indm[0][j],indm[1][j],ind[0],ind[1]],targ_def,radius_of_influence=dlim,weight_funcs=wf,fill_value=-999., nprocs=1))
			ind=np.where(gwes[indm[0][j],indm[1][j],:,:]>0); orig_def_gwes = pyresample.geometry.SwathDefinition(lons=mnlonww3[ind[0],ind[1]], lats=mnlatww3[ind[0],ind[1]])
			if np.any(ind):
				auxgwes = np.array(pyresample.kd_tree.resample_custom(orig_def_gwes,gwes[indm[0][j],indm[1][j],ind[0],ind[1]],targ_def,radius_of_influence=dlim,weight_funcs=wf,fill_value=-999., nprocs=1))
			else:
				auxgwes = np.zeros((auxww3.shape[0]),'f')*np.nan

			ind=np.where(nfcens[indm[0][j],indm[1][j],:,:]>0); orig_def_nfcens = pyresample.geometry.SwathDefinition(lons=mnlonww3[ind[0],ind[1]], lats=mnlatww3[ind[0],ind[1]])
			if np.any(ind):
				auxnfcens = np.array(pyresample.kd_tree.resample_custom(orig_def_nfcens,nfcens[indm[0][j],indm[1][j],ind[0],ind[1]],targ_def,radius_of_influence=dlim,weight_funcs=wf2,fill_value=-999., nprocs=1))
			else:
				auxnfcens = np.zeros((auxww3.shape[0]),'f')*np.nan

			ft=np.append(ft,np.zeros((size(indt)),'d')+ftime[i])
			fit=np.append(fit,np.zeros((size(indt)),'d')+itimec[indm[0][j]])
			fct=np.append(fct,np.zeros((size(indt)),'d')+ffct[indm[1][j]])
			flat=np.append(flat,slat[indt])
			flon=np.append(flon,slon[indt])
			fshs=np.append(fshs,shs[indt])
			fmhsww3=np.append(fmhsww3,auxww3)
			fmhsgwes=np.append(fmhsgwes,auxgwes)
			fmhsnfcens=np.append(fmhsnfcens,auxnfcens)

			del auxww3, auxgwes, auxnfcens, orig_def_ww3, orig_def_gwes, orig_def_nfcens, ind


ind=np.where((fshs>0.)&(fshs<20.)&(fmhsww3>0.)&(fmhsgwes>0.)&(fmhsnfcens>0.))
ft=np.copy(ft[ind]); fit=np.copy(fit[ind]); fct=np.copy(fct[ind]); flat=np.copy(flat[ind]); flon=np.copy(flon[ind]); fshs=np.copy(fshs[ind]); 
fmhsww3=np.copy(fmhsww3[ind]); fmhsgwes=np.copy(fmhsgwes[ind]); fmhsnfcens=np.copy(fmhsnfcens[ind]) 


hd = '      Bias            RMSE         N-Bias          N-RMSE          SCrmse            SI              HH              CC'
hd2 = ' Lines: Forecast Days'
terrww3=np.zeros((8,7),'f')*np.nan
terrgwes=np.zeros((8,7),'f')*np.nan
terrnfcens=np.zeros((8,7),'f')*np.nan

for i in range(1,8):
	ind = np.where( (fct>=ffct[::8][i-1]) & (fct<=ffct[::8][i]))
	# Err X Forecast Time
	terrww3[:,i-1] = merr.metrics(fmhsww3[ind],fshs[ind])
	terrgwes[:,i-1] = merr.metrics(fmhsgwes[ind],fshs[ind])
	terrnfcens[:,i-1] = merr.metrics(fmhsnfcens[ind],fshs[ind])
		

fdays=np.array(np.arange(1,7+1)).astype('int')
nerrm=['Bias','RMSE','NBias','NRMSE','SCrmse','SI','HH','CC']
scolors=np.array(['b','forestgreen','darkviolet']); fnames=np.array(['NCEPWW3D','NCEPGWES_EM','FNMOC_EM'])
for l in range(0,8):
	fig1 = plt.figure(1,figsize=(7,6)); ax = fig1.add_subplot(111)
	ax.plot(fdays,gaussian_filter(terrww3[l,:], 1.), color=scolors[0],label=fnames[0],linestyle='-',linewidth=2,zorder=3)
	ax.plot(fdays,gaussian_filter(terrgwes[l,:], 1.), color=scolors[1],label=fnames[1],linestyle='--',linewidth=2,zorder=3)
	ax.plot(fdays,gaussian_filter(terrnfcens[l,:], 1.), color=scolors[2],label=fnames[2],linestyle='-.',linewidth=2,zorder=3)
	plt.legend()
	ax.set_xlabel('Forecast Time (Days)',size=sl); ax.set_ylabel(nerrm[l],size=sl)
	plt.tight_layout();plt.axis('tight') 
	plt.grid(c='k', ls='-', alpha=0.3)
	plt.savefig('ErrXForecastTime_'+nerrm[l]+'_Hs_leadsComparison.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1); del fig1, ax


# QQplot
p = np.arange(1,99,1)
for i in range(1,8):
	ind = np.where( (fct>=ffct[::8][i-1]) & (fct<=ffct[::8][i]))

	qobs = np.zeros((p.shape[0]),'f')*np.nan
	qww3 = np.zeros((p.shape[0]),'f')*np.nan
	qgwes = np.zeros((p.shape[0]),'f')*np.nan	
	qnfcens = np.zeros((p.shape[0]),'f')*np.nan
	for j in range(0,p.shape[0]):
		qobs[j] = np.nanpercentile(fshs[ind],p[j])
		qww3[j] = np.nanpercentile(fmhsww3[ind],p[j])
		qgwes[j] = np.nanpercentile(fmhsgwes[ind],p[j])
		qnfcens[j] = np.nanpercentile(fmhsnfcens[ind],p[j])

	aux=np.linspace(1,15,p.shape[0])

	fig1 = plt.figure(1,figsize=(7,6)); ax1 = fig1.add_subplot(111)
	ax1.plot(aux,aux,'k',linewidth=1.)
	ax1.plot(qobs,gaussian_filter(qww3, 1.), color=scolors[0], label=fnames[0], marker='.', linewidth=1.)
	ax1.plot(qobs,gaussian_filter(qgwes, 1.), color=scolors[1], label=fnames[1], marker='.', linewidth=1.)
	ax1.plot(qobs,gaussian_filter(qnfcens, 1.), color=scolors[2], label=fnames[2], marker='.', linewidth=1.)
	plt.ylim(ymax = 15, ymin = 1)
	plt.xlim(xmax = 15, xmin = 1)
	plt.locator_params(axis='y', nbins=7) ; plt.locator_params(axis='x', nbins=7) 
	plt.legend()
	ax1.set_xlabel("Observation",size=sl); ax1.set_ylabel("Model",size=sl);
	plt.tight_layout(); # plt.axis('tight') 
	plt.grid(c='k', ls='-', alpha=0.3)
	plt.savefig('QQplot_d'+repr(i)+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig1)


# Track Plot around the max
ind=np.where(fshs==fshs.max())[0]
for i in range(0,size(ind)):

	fig, ax1 = plt.subplots(1,figsize=(7,4), sharex=True, sharey=True)
	# Observation	
	ax1.plot(gaussian_filter(flat[ind[i]-70:ind[i]+101],0.5),gaussian_filter(fshs[ind[i]-70:ind[i]+101],1),'k.',label='Sat',linewidth=3.,zorder=3)
	# WW3
	ax1.plot(gaussian_filter(flat[ind[i]-70:ind[i]+101],0.5),gaussian_filter(fmhsww3[ind[i]-70:ind[i]+101],1), 'b.',label=fnames[0],linewidth=1.,zorder=3)
	# GWES
	ax1.plot(gaussian_filter(flat[ind[i]-70:ind[i]+101],0.5),gaussian_filter(fmhsgwes[ind[i]-70:ind[i]+101],1), color=scolors[1], marker='.', LineStyle='none', label=fnames[1],linewidth=1.,zorder=3)
	# Nfcens
	ax1.plot(gaussian_filter(flat[ind[i]-70:ind[i]+101],0.5),gaussian_filter(fmhsnfcens[ind[i]-70:ind[i]+101],1), color=scolors[2], marker='.', LineStyle='none', label=fnames[2],linewidth=1.,zorder=3)
	plt.legend(fontsize=12)
	plt.xlabel("Latitude",size=sl)
	plt.ylabel("Significant Wave Height (m)",size=sl)
	plt.grid(); plt.tight_layout();plt.axis('tight')
	plt.xlim(xmin=30,xmax=46); plt.ylim(ymin=2.,ymax=20.)
	plt.savefig('AltTrackHs_'+repr(np.round(fct[ind[i]]/(3600*24)).astype('int'))+'.png', dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig); del fig


