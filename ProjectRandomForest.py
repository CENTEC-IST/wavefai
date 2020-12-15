#!/usr/bin/env python
# coding: utf-8

import netCDF4
from netCDF4 import Dataset
import numpy as np
import pandas as pd
np.warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from io import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from collections import OrderedDict 
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from datetime import timedelta
import sys
import copy
from sklearn.ensemble import RandomForestClassifier
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import RepeatedStratifiedKFold
import plotly
import plotly.graph_objs as go
from iteration_utilities import deepflatten
import sklearn.metrics as metrics

#lOAD DATA
WW3_N= 'WW3_selectionPointNearest_20190925_20200701.nc'
nc_N0=Dataset(WW3_N,'r')
GWES_N= 'GWES_selectionPointNearest_20190925_20200701.nc'
nc_N1=Dataset(GWES_N,'r')
GFS_N='GFS_selectionPointNearest_20190925_20200701.nc'
nc_N2=Dataset(GFS_N,'r')

#deterministic
ft0=nc_N0['fctime']
t0=netCDF4.num2date(nc_N0.variables['time'][:],nc_N0.variables['time'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
Dp0=nc_N0['Dp']
Hs0=nc_N0['Hs']
Tp0=nc_N0['Tp']
Ds10=nc_N0['Dswell1']
Ds20=nc_N0['Dswell2']
Hs10=nc_N0['Hswell1']
Hs20=nc_N0['Hswell2']
Ts10=nc_N0['Tswell1']
Ts20=nc_N0['Tswell2']
Dw0=nc_N0['Dwsea']
Hw0=nc_N0['Hwsea']
Tw0=nc_N0['Twsea']

#ensemble
ft1=nc_N1['fctime']
t1=netCDF4.num2date(nc_N1.variables['time'][:],nc_N1.variables['time'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
Dp1=nc_N1['Dp']
Hs1=nc_N1['Hs']
Tp1=nc_N1['Tp']
Ds11=nc_N1['Dswell1']
Ds21=nc_N1['Dswell2']
Hs11=nc_N1['Hswell1']
Hs21=nc_N1['Hswell2']
Ts11=nc_N1['Tswell1']
Ts21=nc_N1['Tswell2']
Dw1=nc_N1['Dwsea']
Hw1=nc_N1['Hwsea']
Tw1=nc_N1['Twsea']

#gfs
u=nc_N2['U10m']
v=nc_N2['V10m']
t2=netCDF4.num2date(nc_N2['time'][:],nc_N2['time'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)

#buoys 
buoys='NDBC_selection_deepWaters_20190925_20200701.nc'
buoys=Dataset(buoys,'r')
buoy=[None]*len(buoys['buoyID'])
for i in range(len(buoys['buoyID'])):
    a=buoys['WVHT'][i]
    a=pd.DataFrame(data=np.transpose(np.array(a)))
    t=netCDF4.num2date(buoys['time'][:],buoys['time'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
    buoy[i]=a.set_index(pd.DatetimeIndex(pd.Series(t)))
    buoy[i].columns=['Hs']


f0=0 #nowcast
f5=np.where(ft0[:] == 432000)[0][0] #day 5
fl=60 #last index with prediction, which is 7.5
forecast=[f0,f5,fl]

plt.rcParams.update({'font.size': 17}) #set general fontsize of plots


#time series plot of Hs in the different buoys
with plt.style.context('seaborn-colorblind'):
    plt.figure(figsize=(16, 5))
    for i in range(len(buoy)):
        plt.plot(buoy[i].index,buoy[i]['Hs'],label= buoys['buoyID'][i])    
    
    plt.legend(loc='upper right',frameon=False,ncol=2)
    plt.xlabel('Time')
    plt.ylabel('Hs')
plt.margins(0)
#plt.savefig('buoysM.png', bbox_inches='tight')#,dpi=80)

#FUNCTIONS
def metricsB(*args):
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
        print(' Array without valid numbers.')
        return np.zeros((8),'f')*np.nan # TODO is this correct

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

ranks = {}
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

#CREATE THE DATAFRAMES OF THE STUDY
def data(Dp0, Hs0, Tp0, Ds10, Ds20, Hs10, Hs20, Ts10, 
         Ts20, Dw0, Hw0, Tw0, t0, Dp1, Hs1, Tp1, Ds11, Ds21, Hs11, Hs21, Ts11, 
         Ts21, Dw1, Hw1, Tw1, t1, u2, v2, t2, ft, s, obs, b):
    '''
    s: seconds to shift from nowcast
    ft: index where the value s is
    b: index of buoy
    obs: dataframe with hs values in buoy[obs] and time as index
    '''
    if not np.logical_and( (t0==t1).all(), (t1==t2).all() ):
        sys.exit('Models must have same times')
        
    #m0 (ww3)
    df0=[Dp0[b,:,ft],Hs0[b,:,ft],Tp0[b,:,ft],Ds10[b,:,ft], Ds20[b,:,ft], Hs10[b,:,ft], Hs20[b,:,ft], Ts10[b,:,ft], 
         Ts20[b,:,ft], Dw0[b,:,ft], Hw0[b,:,ft], Tw0[b,:,ft]]
    df0=pd.DataFrame(data=np.transpose(np.array(df0)))    
    df0.apply(pd.to_numeric)
    ti0=t0+timedelta(seconds=s)
    model0=df0.set_index(pd.DatetimeIndex(pd.Series(ti0)))
    
    #m1 (gwes)
    dfa=[]    
    for j in range(1,21):
        df1=[Dp1[b,j,:,ft],Hs1[b,j,:,ft],Tp1[b,j,:,ft],Ds11[b,j,:,ft], Ds21[b,j,:,ft], Hs11[b,j,:,ft], Hs21[b,j,:,ft], 
             Ts11[b,j,:,ft], Ts21[b,j,:,ft], Dw1[b,j,:,ft], Hw1[b,j,:,ft], Tw1[b,j,:,ft]]
        df1=pd.DataFrame(data=np.transpose(np.array(df1)))
        dfa.append(df1)
        
    df1a=pd.concat(dfa)
    vHs=df1a[df1a.columns[1]].groupby(df1a.index).var()
    df1a=df1a.groupby(df1a.index).mean()  #data is the mean of all elements per time  
    df1a['varHs']=vHs
    df1a.apply(pd.to_numeric)
    ti1=t1+timedelta(seconds=s)
    model1=df1a.set_index(pd.DatetimeIndex(pd.Series(ti1)))
    
    df2=[u2[b,:,ft],v2[b,:,ft]]
    df2=pd.DataFrame(data=np.transpose(np.array(df2)))
    df2.apply(pd.to_numeric)
    ti2=t2+timedelta(seconds=s)
    model2=df2.set_index(pd.DatetimeIndex(pd.Series(ti2)))
    
    model0.columns=['Dp0','Hs0','Tp0','Ds10','Ds20','Hs10','Hs20','Ts10','Ts20','Dw0','Hw0','Tw0']
    model1.columns=['Dp1','Hs1','Tp1','Ds11','Ds21','Hs11','Hs21','Ts11','Ts21','Dw1','Hw1','Tw1','varHs1']
    model2.columns=['U','V']    
    
    #convert all direction features
    model0['sinD0']=np.sin(2*np.pi*model0[['Dp0']]/360)
    model0['cosD0']=np.cos(2*np.pi*model0[['Dp0']]/360)
    model0['sinDs10']=np.sin(2*np.pi*model0[['Ds10']]/360)
    model0['cosDs10']=np.cos(2*np.pi*model0[['Ds10']]/360)
    model0['sinDs20']=np.sin(2*np.pi*model0[['Ds20']]/360)
    model0['cosDs20']=np.cos(2*np.pi*model0[['Ds20']]/360)
    model0['sinDw0']=np.sin(2*np.pi*model0[['Dw0']]/360)
    model0['cosDw0']=np.cos(2*np.pi*model0[['Dw0']]/360)
    model0=model0.round(decimals=3)
    
    model1['sinD1']=np.sin(2*np.pi*model1[['Dp1']]/360)
    model1['cosD1']=np.cos(2*np.pi*model1[['Dp1']]/360)
    model1['sinDs11']=np.sin(2*np.pi*model1[['Ds11']]/360)
    model1['cosDs11']=np.cos(2*np.pi*model1[['Ds11']]/360)
    model1['sinDs21']=np.sin(2*np.pi*model1[['Ds21']]/360)
    model1['cosDs21']=np.cos(2*np.pi*model1[['Ds21']]/360)
    model1['sinDw1']=np.sin(2*np.pi*model1[['Dw1']]/360)
    model1['cosDw1']=np.cos(2*np.pi*model1[['Dw1']]/360)
    model1=model1.round(decimals=3)
    
    l0=len(model0) #model0 and model1 have same size (must)
    
    #Add hs prediction errors
    errorw=np.empty((0)) #error hs0
    errorg=np.empty((0)) #error hs1
    for h in range(l0):        
        ind = np.where(obs.index == model0.index[h]) #find position where times match 
        if len(ind[0])!=0: #if a position was found
            errorw = np.append(errorw, abs(model0['Hs0'][h] - obs.iloc[ind])) #append to array of errors abs of difference
            errorg = np.append(errorg, abs(model1['Hs1'][h] - obs.iloc[ind]))
        else: #if no position was found
            errorw = np.append(errorw, np.nan)
            errorg = np.append(errorg, np.nan)

    X=pd.concat([model2,model0,model1],axis=1)#all variables as columns of dataframe
    X=X.drop(['Dp0','Ds10','Ds20','Dw0','Dp1','Ds11','Ds21','Dw1'],axis=1)  #drop directions
    y=[]
    #creating target class with variable choice
    choice=np.array(errorw)-np.array(errorg) 
    
    #if choice[i] is positive, at time i, error of prediction is higher in model0 (errorw[i]) --> assign class 1 (lower errorg[i])    
    for i in range(l0):
        #choice has nan values if matching position is not found previously (obs index)
        if not np.isnan(choice[i]):
            if choice[i]>0:
                y.append(1)#gwes
            elif choice[i]<0:
                y.append(0)#ww3
            elif s<432000:#when error is the same, if ft is less than 5 days (seconds = 432000): ww3, else: gwes 
                y.append(0)
            else:
                y.append(1)
        else: #choice is nan
              y.append(np.nan)
                
    X['buoy']=[b]*len(X) #variable indicating buoy index
    X['y']=y 
    
    X=X.dropna()
    X['y']=X['y'].astype('int64')
    
    return X, errorw,errorg

#Feature importance and ranking
def selection(X_train,y_train):
    
    names=X_train.columns
    rank=[]
    clf=RandomForestClassifier(random_state=42, max_features='sqrt',criterion='entropy',
                              max_depth=10, min_samples_split=30, max_leaf_nodes=10) #hyperparameters were set manually (suboptimal)
    permuter=PermutationImportance(clf, n_iter=10, random_state=42, cv=3, scoring='accuracy') #reduce n_iter --> lower computational cost
    selector= RFECV(permuter,scoring='accuracy',cv=3).fit(X_train, y_train)
    rank=ranking(list(map(float, selector.ranking_)), names  , order=-1)#ranking: function created above to organize in dictionary
    
    return(rank)

#S for simple, predict with suboptimal (simpler) classifier 
def decisionS(X_train,y_train,X_test,y_test, df_rank,n):
    '''
    df_rank: ordered ranking of features in dataframe without reseting index 
    n: number of features to include in dataset
    '''
    selected_features=list(df_rank.index[0:n])   
    
    X_train_sel = X_train.iloc[:,selected_features]
    X_test_sel = X_test.iloc[:,selected_features]    
        
    clf=RandomForestClassifier(random_state=42, max_features='sqrt',criterion='entropy',
                              max_depth=10, min_samples_split=30, max_leaf_nodes=10)#hyperparameters were set manually (suboptimal)
    clf.fit(X=X_train_sel, y=y_train)                     
    y_pred=clf.predict(X_test_sel)
    acc1=accuracy_score(y_test,y_pred)      
    return (acc1)

#predict with optimal set of hyperparameters
def decision(X_train_sel,X_test_sel,y_train,y_test):       

    param_grid = {"criterion": ["gini",'entropy'],
                'n_estimators': [400],
                "min_samples_split": list(range(35,51)),
                "max_depth": list(range(2,15)),
                "max_leaf_nodes": list(range(3,20)),
                'max_features':['sqrt']}   #range of values set manually
    
    clf=RandomForestClassifier(random_state=42)
    clf = GridSearchCV(clf, param_grid, n_jobs=-1,cv=3) #test all possible combinations of hyperparameters
    clf.fit(X=X_train_sel, y=y_train)
    tree_model = clf.best_estimator_ #the best model - set of hyperparameters
                     
    y_pred=tree_model.predict(X_test_sel)
    acc0=accuracy_score(y_train, tree_model.predict(X_train_sel)) #accuracy of training set
    acc1=accuracy_score(y_test,y_pred) #accuracy of test set
   
    return (acc0,acc1,tree_model,y_pred)    
    

def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    if dict2==[]:
        return dict1
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = [value , dict1[key]]
    return dict3

def flatten(lst):
    for x in lst:
        if isinstance(x, list):
            for y in flatten(x): 
                yield y           
        else:
            yield x
            
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df):
    '''List of top correlated features'''
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr

#Block for creating chiclet plots and metrics plots
#Errors of hs for all forecast times (61) between nowcast and day 7.5

M0=[] #set of matrices of model0 errors, each matrix is from a buoy
M1=[] # same for model1
Dset=[] # set of dataframes of predicted hs, each is from a combination buoy/forecast time
Bmatch=[] #set of dataframes of hs observations matching times in Dset

for b in range(len(buoy)):
    obs=buoy[b]
    m0=[]
    m1=[]
    for i in range(0,61):    
        df=Hs0[b,:,i]
        df=pd.DataFrame(data = df)
        df.apply(pd.to_numeric)
        ti0=t0+timedelta(seconds=ft0[i].item(0))
        model0=df.set_index(pd.DatetimeIndex(pd.Series(ti0)))
        model0=pd.DataFrame(model0)
        model0.columns=['Hs']
        l0=len(model0)
        model0['model']=[0]*l0

        #ensemble average of GWES
        dfa=[]    
        for j in range(1,21):
            df=Hs1[b,j,:,i]
            df=pd.DataFrame(data = df)
            df.apply(pd.to_numeric)
            dfa.append(df)

        df1a=pd.concat(dfa)
        df1a=df1a.groupby(df1a.index).mean()
        model1=df1a.set_index(pd.DatetimeIndex(pd.Series(ti0)))
        model1=pd.DataFrame(model1)
        model1.columns=['Hs']
        model1['model']=[1]*l0

        match=[]
        #buoy errors
        errorw=np.empty((0))
        errorg=np.empty((0))
        for h in range(l0):             
            ind = np.where(obs.index == model0.index[h])              
            if len(ind[0])!=0:
                errorw = np.append(errorw, model0['Hs'][h] - obs.iloc[ind])
                errorg = np.append(errorg, model1['Hs'][h] - obs.iloc[ind])
            else:
                errorw = np.append(errorw, np.nan)
                errorg = np.append(errorg, np.nan)

            match.append(obs.iloc[ind].values)
        #merge 0 dataframes
        result = model0.append(model1)
        models=pd.DataFrame(result)
        models=models.round(decimals=1)
        Dset.append(models) #each dataframe with 2 columns:Hs and model
        Bmatch.append(match)
        m0.append(errorw)#matrix WW1
        m1.append(errorg)#matrix GWES

    M0.append(np.array(m0)) #each matrix is len(ft) x len(t)
    M1.append(np.array(m1))

#Plot
#Chiclet

X,Y = np.meshgrid(np.arange(M0[0].shape[1]),np.arange(M0[0].shape[0]))#every matrix has the same shape
date = [x.date().isoformat() for x in t0]#times in 0(ww3) are equal to times in 1(gwes)
#---block-for-tick-labels-------
dt = np.array(date)
ft = np.array(ft0[0:61]).astype(int)
dts=[]
for i in dt[::50]:
    dts.append(i)
fts=[]
for i in ft[::20]:
    fts.append(i/60/60/24)    
#-------------------------------
for i in range(len(buoy)):    
    #WW3
    figure = plt.figure(figsize=(16,2) )
    ax = plt.subplot() 
    m0m=ax.scatter(X,Y,c=M0[i], s=15, linewidth=0, marker="s",cmap='RdYlBu')
    figure.colorbar(m0m, ax=ax,shrink=0.9, aspect=7,pad=.01)
    ax.margins(0)
    #figure.canvas.draw()    
    ax.set_xticks(np.arange(0,M0[i].shape[1],50))
    ax.set_xticklabels(dts)
    ax.set_yticks(np.arange(0,M0[i].shape[0],20))
    ax.set_yticklabels(fts)
    ax.set_xlabel('Time')
    ax.set_ylabel('Forecast Time (days)')
    ax.yaxis.set_label_coords(-0.07, 0.5)
    #if buoys['buoyID'][i] == '44008':
        #plt.savefig('chiclet{}'.format(0), bbox_inches='tight')
    #GWES    
    figure = plt.figure(figsize=(16,2) )
    ax = plt.subplot() 
    m1m=ax.scatter(X,Y,c=M1[i], s=15, linewidth=0, marker="s",cmap='RdYlBu')#,vmin=-4,vmax=4)
    figure.colorbar(m1m, ax=ax, shrink=0.9, aspect=7,pad=.01)
    ax.margins(0)
    #figure.canvas.draw()
    ax.set_xticks(np.arange(0,M1[i].shape[1],50))
    ax.set_xticklabels(dts)
    ax.set_yticks(np.arange(0,M1[i].shape[0],20))
    ax.set_yticklabels(fts)
    ax.set_xlabel('Time')
    ax.set_ylabel('Forecast Time (days)')
    ax.yaxis.set_label_coords(-0.07, 0.5)
    #if buoys['buoyID'][i] == '44008':
        #plt.savefig('chiclet{}'.format(1), bbox_inches='tight')   

#Plot
#Metrics

for i in range(len(buoy)):
    Mw=[]
    Mg=[]
    for j in range(0,61):
        l=len(Dset[j+61*i].loc[Dset[j+61*i]['model'] == 0])
        wm=Dset[j+61*i][['Hs']].iloc[0:l] #array WW3
        gm=Dset[j+61*i][['Hs']].iloc[l:] #array GWES
        Bmatch[j+61*i]=[np.array([np.nan]) if len(Bmatch[j+61*i][q])==0 else Bmatch[j+61*i][q] for q in range(l)]
        mw=metricsB(np.array(wm).flatten(),np.array(list(deepflatten(Bmatch[j+61*i]))))
        mg=metricsB(np.array(gm).flatten(),np.array(list(deepflatten(Bmatch[j+61*i]))))    
        Mw.append(mw)
        Mg.append(mg)
    Mw=np.vstack(Mw)
    Mw=np.transpose(Mw)
    Mg=np.vstack(Mg)
    Mg=np.transpose(Mg)
    figure = plt.figure(figsize=(5,5)) 
    ax = plt.subplot()    
    ax.plot(ft0[0:61]/60/60/24,Mw[2],c='green',label='0 (WW3)')
    ax.plot(ft1[0:61]/60/60/24,Mg[2],c='blue',label='1 (GWES)') 
    ax.set_xlabel('Forecast Time (days)')
    ax.set_ylabel('NBias')   
    plt.legend(frameon=False)
    #if buoys['buoyID'][i] == '44008':
     #   plt.savefig('metrics{}'.format('NBias'), bbox_inches='tight')   
    
    figure = plt.figure(figsize=(5,5)) 
    ax = plt.subplot()
    ax.plot(ft0[0:61]/60/60/24,Mw[5],c='green',label='0 (WW3)')
    ax.plot(ft1[0:61]/60/60/24,Mg[5],c='blue',label='1 (GWES)')    
    ax.set_xlabel('Forecast Time (days)')
    ax.set_ylabel('SI')
    plt.legend(frameon=False)
    #if buoys['buoyID'][i] == '44008':
     #   plt.savefig('metrics{}'.format('SI'), bbox_inches='tight')           
    
    figure = plt.figure(figsize=(5,5))   
    ax = plt.subplot()
    ax.plot(ft0[0:61]/60/60/24,Mw[1],c='green',label='0 (WW3)')
    ax.plot(ft1[0:61]/60/60/24,Mg[1],c='blue',label='1 (GWES)')
    ax.set_ylabel('RMSE')
    plt.legend(frameon=False)
    #if buoys['buoyID'][i] == '44008':
     #   plt.savefig('metrics{}'.format('RMSE'), bbox_inches='tight')   
        
    figure = plt.figure(figsize=(5,5)) 
    ax = plt.subplot()
    ax.plot(ft0[0:61]/60/60/24,Mw[7],c='green',label='0 (WW3)')
    ax.plot(ft1[0:61]/60/60/24,Mg[7],c='blue',label='1 (GWES)')
    ax.set_xlabel('Forecast Time (days)')
    ax.set_ylabel('CC') 
    plt.legend(frameon=False)
    #if buoys['buoyID'][i] == '44008':
     #   plt.savefig('metrics{}'.format('CC'), bbox_inches='tight')   

#Merge data from all buoys for specific forecast time
Xx=[None]*3
ew=[None]*3
eg=[None]*3
h=0
for i in forecast: 
    x=[]
    eww=[]
    egg=[]
    for j in range(len(buoy)):
        X,errorw,errorg = data(Dp0, Hs0, Tp0, Ds10, Ds20, Hs10, Hs20, Ts10, 
             Ts20, Dw0, Hw0, Tw0, t0, Dp1, Hs1, Tp1, Ds11, Ds21, Hs11, Hs21, Ts11, 
             Ts21, Dw1, Hw1, Tw1, t1, u, v, t2, i, ft0[i].item(0),buoy[j], j)
        x.append(X)
        eww.append(errorw)
        egg.append(errorg)
    Xx[h]=pd.concat(x)
    ew[h]=np.concatenate(eww)
    eg[h]=np.concatenate(egg)
    h+=1
    
#Check balance of output
for i in range(3):
    print(Xx[i]['y'].value_counts(normalize=True) * 100)

#Plot
#Balance: target values per buoy 
Yy=[None]*3

for i in range(3):
    fig = plt.figure(figsize=(7,4))
    ax = plt.subplot()#1, 3, j+1   
    sns.countplot(x="buoy",  hue="y", data=Xx[i], ax=ax,palette=['green','blue'])   
    ax.set_title('Forecast day {}'.format(ft0[forecast[i]].item(0)/60/60/24))
    ax.set_xlabel('Buoy')
    ax.set_ylabel('Count')
    ax.set_xticklabels(buoys['buoyID'][:])
    legend_labels, _= ax.get_legend_handles_labels()
    ax.legend(legend_labels, ['0 (WW3)','1 (GWES)'],loc='lower right' )
    
    Yy[i]=Xx[i]['y'] #just here, separate target from rest of dataframe
    Yy[i]=Yy[i].values.reshape((len(Yy[i]),1)).astype('int64') 
    Xx[i]=Xx[i].drop(['y','buoy'],axis=1) #drop buoy information also    
    Xx[i]=Xx[i].astype(float) 
    plt.savefig('countlabel{}'.format(i))

#Plot
#Heatmaps of correlation

for i in range(3):
    fig = plt.figure(figsize=(10,9))
    cbar_ax = fig.add_axes([.9,.4,.015,.4])
    ax = plt.subplot()
    corr=Xx[i].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask,ax=ax,                
                vmin=-1, vmax=1,cbar_ax=cbar_ax,
                xticklabels=True, yticklabels=True)
    ax.set_title('Forecast day {}'.format(ft0[forecast[i]].item(0)/60/60/24))
    #--lines-to-separate-groups-of-features----------------
    ax.hlines([2,18], [0,0],[1,18],colors='w')
    ax.vlines([2,18],[2,18],[35,35],colors='w')
  
    #plt.savefig('corr{}'.format(i))

#Remove features with Pearson's correlation higher than threshold (one each pair)
#not executed because RFE is supposed to identify important variables anyways
threshold=0.9
print('Features with correlation score higher than {} with other features'.format(threshold))
for i in range(3):
    corr_matrix=Xx[i].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    #Xx[i]=Xx[i].drop(to_drop,axis=1)
    print(to_drop)

#Split train and test sets
X_train=[None]*3
X_test=[None]*3
y_train=[None]*3
y_test=[None]*3

for i in range(3):    
    X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(Xx[i], Yy[i], test_size=0.15, random_state=42)


#Balance after split

percentagew=[None]*3
for i in range(3):
    l=list(y_test[i])
    total=len(l)
    cw=l.count(0)
    percentagew[i]=(cw/total)*100
print('Test set percentage of WW3 for each forecast time: {}'.format(percentagew))

percentagew=[None]*3
for i in range(3):
    l=list(y_train[i])
    total=len(l)
    cw=l.count(0)
    percentagew[i]=(cw/total)*100
print('Train set percentage of WW3 for each forecast time: {}'.format(percentagew))


#Plot
#Hyperparameters behavior
i=0 #choose which dataset to train with

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train[i], y_train[i], test_size=0.25, random_state=42)

#---N-trees-and-max-features-----------
fig = plt.figure(figsize=(6,4))
ax = plt.subplot()   
trees = np.arange(50, 900)
ensemble_clfs = [("max_features='sqrt'", RandomForestClassifier( warm_start=True,
                            max_features="sqrt", random_state=42)), # in this case, max_features is 5 (floor of sqrt(35))
                ("max_features=None", RandomForestClassifier( warm_start=True,
                            max_features=None, random_state=42))] #max_features is 35

colors=['green','orange']
c=0
for label, dt in ensemble_clfs:
    test1_results = []
    for i in trees:        
        dt.set_params(n_estimators=i)
        dt.fit(X_train1, y_train1)           
        accuracy = accuracy_score(y_test1, dt.predict(X_test1))
        test1_results.append(accuracy)    

    line2, = ax.plot(trees, test1_results, colors[c], label= label)
    c+=1
ax.legend(frameon=False)
ax.set_xlabel('n_estimators')
ax.set_ylabel('Test accuracy')
#plt.savefig('n_estimators')

#----Max-depth----------------------------
fig = plt.figure(figsize=(6,4))
ax = plt.subplot()
max_depths = np.linspace(1, 50, 50, endpoint=True)
train1_results = []
test1_results = []

for i in max_depths:    
    dt = RandomForestClassifier(warm_start=True,max_depth=i)
    dt.fit(X_train1, y_train1)    
    accuracy = accuracy_score(y_train1,dt.predict(X_train1))
    train1_results.append(accuracy)
    accuracy = accuracy_score(y_test1, dt.predict(X_test1))
    test1_results.append(accuracy)

line1, = ax.plot(max_depths, train1_results, 'b', label='Train accuracy')
line2, = ax.plot(max_depths, test1_results, 'r', label= 'Test accuracy')
ax.legend(frameon=False)
ax.set_xlabel('max_depth')
#plt.savefig('max_depth')

#----Samples-split-------------------------
fig = plt.figure(figsize=(6,4))
ax=plt.subplot()
samples_split = np.arange(2,51)
train1_results = []
test1_results = []

for i in samples_split:    
    dt = RandomForestClassifier(warm_start=True,min_samples_split=i)
    dt.fit(X_train1, y_train1)    
    accuracy = accuracy_score(y_train1,dt.predict(X_train1))
    train1_results.append(accuracy)
    accuracy = accuracy_score(y_test1, dt.predict(X_test1))
    test1_results.append(accuracy)

line1, = ax.plot(samples_split, train1_results, 'b', label='Train accuracy')
line2, = ax.plot(samples_split, test1_results, 'r', label= 'Test accuracy')
ax.legend(frameon=False)
ax.set_xlabel('min_samples_split')
#plt.savefig('min_samples_split')
#----Leaf-nodes------------------------------
fig = plt.figure(figsize=(6,4))
ax=plt.subplot()
leaf_nodes = np.arange(2,100)
train1_results = []
test1_results = []

for i in leaf_nodes:    
    dt = RandomForestClassifier(warm_start=True,max_leaf_nodes=i)
    dt.fit(X_train1, y_train1)    
    accuracy = accuracy_score(y_train1,dt.predict(X_train1))
    train1_results.append(accuracy)
    accuracy = accuracy_score(y_test1, dt.predict(X_test1))
    test1_results.append(accuracy)

line1, = ax.plot(leaf_nodes, train1_results, 'b', label='Train accuracy')
line2, = ax.plot(leaf_nodes, test1_results, 'r', label= 'Test accuracy')
ax.legend(frameon=False)
ax.set_xlabel('max_leaf_nodes')
#plt.savefig('max_leaf_nodes')

#---Samples-leaf-------------------------------
fig = plt.figure(figsize=(6,4))
ax=plt.subplot()
samples_leaf=np.arange(1,51)
train1_results = []
test1_results = []

for i in samples_leaf:    
    dt = RandomForestClassifier(warm_start=True,min_samples_leaf=i)
    dt.fit(X_train1, y_train1)    
    accuracy = accuracy_score(y_train1,dt.predict(X_train1))
    train1_results.append(accuracy)
    accuracy = accuracy_score(y_test1, dt.predict(X_test1))
    test1_results.append(accuracy)

line1, = ax.plot(samples_leaf, train1_results, 'b', label='Train accuracy')
line2, = ax.plot(samples_leaf, test1_results, 'r', label= 'Test accuracy')
ax.legend(frameon=False)
ax.set_xlabel('min_samples_leaf')
#plt.savefig('samples_leaf')


#Selection of best features
df=[]
final=[None]*3
for i in range(3):   #for each dataframe split training set multiple times
    kfold = RepeatedStratifiedKFold(n_repeats=3, n_splits=5, random_state=42)#0.1 for test. reduce n_repeats and n_plits --> less computational cost
    d = []
    for train, test in kfold.split(X_train[i], y_train[i]):
        dict1=selection(X_train[i].iloc[train,:], y_train[i][train])
        dict2=d
        d= mergeDict(dict1,dict2) #merge dictionaries of ranking found in each iteration
    df.append( {k: list(flatten(v)) for k, v in d.items()})

    m={}
    sd={}
    for name in list(df[i].keys()):
        m[name] = round(np.mean(df[i][name]), 2) #mean ranking of each feature
    for name in list(df[i].keys()):
        sd[name] = round(np.std(df[i][name]), 2) #standard deviation of each feature's ranking

    meanplot = pd.DataFrame(list(m.items()), columns= ['Feature','Mean Ranking'])
    sdplot=pd.DataFrame(list(sd.items()), columns= ['Feature_','Sd'])
    final[i]=pd.concat([meanplot,sdplot],axis=1)
    final[i] = final[i].sort_values('Mean Ranking', ascending=False)  #final list of ordered ranking


#Plot 
#Ranking of features

for i in range(3): 
    fig = plt.figure(figsize=(15,4))
    ax = plt.subplot() 
    sns.barplot(x='Feature', y='Mean Ranking', data = final[i], palette='coolwarm',ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    labels = list(final[i]['Sd'])
    j=0
    #each bar with sd information
    for j, (label, height) in enumerate(zip(labels, final[i]['Mean Ranking'])):         
        ax.text(j, height, ' ' + str(label), ha='center', va='top',rotation=90)
    ax.set_title('Forecast day {}'.format(ft0[forecast[i]].item(0)/60/60/24))

    #plt.savefig('ranking_v{}'.format(i))

#Plot 
#Accuracy of simple model depending on number of features selected

a=[]
#----samples---
n_repeats=10
n_splits=5
#--------------
Fsamples=n_repeats*n_splits 
for i in range(3):
    fig = plt.figure(figsize=(8,4))
    kfold = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=42)#0.1 for test. same splits as before
    ax=plt.subplot()
    ax.set_title('Forecast day {}'.format(ft0[forecast[i]].item(0)/60/60/24))
    
    for train, test in kfold.split(X_train[i], y_train[i]): #for each sample in the training set
        a1=[]
        
        for n in np.arange(1,len(X_train[i].columns)+1): #adding a feature based on ranking at a time
            dd=decisionS(X_train[i].iloc[train,:], y_train[i][train],X_train[i].iloc[test,:], y_train[i][test],final[i],n)
            a1.append(dd)
        a.append(a1)    
        ax.plot(np.arange(1,len(X_train[i].columns)+1),a1,'k--',color='grey',label="Samples' accuracy") #each line is the accuracy of a sample using n features
    loc=len(a)
    ax.errorbar(np.arange(1,len(X_train[i].columns)+1),np.array(a[i*Fsamples:loc]).mean(axis=0),
                yerr=np.array(a[i*Fsamples:loc]).std(axis=0),color='black',label='Mean accuracy')
    ax.set_xlabel('n')    
    #--Block-for-legend------------------------
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),frameon=False)
    #-------------------------------------------
    #plt.savefig('meanAcc{}'.format(i))

#Argmax of accuracy. Number of features to select is argmax+1
n=[None]*3
n[0]=np.argmax(np.array(a[0:Fsamples]).mean(axis=0))
n[1]=np.argmax(np.array(a[Fsamples:Fsamples*2]).mean(axis=0))
n[2]=np.argmax(np.array(a[Fsamples*2:Fsamples*3]).mean(axis=0))
print(n)


#Dataframe with resulting mean and standard deviation of accuracy per number of features
#To get the precise values for analysis
g= {'Mean 0': np.array(a[0:Fsamples]).mean(axis=0), 'Std 0': np.array(a[0:Fsamples]).std(axis=0),
    'Mean 5':np.array(a[Fsamples:Fsamples*2]).mean(axis=0),'Std 5':np.array(a[Fsamples:Fsamples*2]).std(axis=0),
    'Mean 7.5':np.array(a[Fsamples*2:Fsamples*3]).mean(axis=0),'Std 7.5': np.array(a[Fsamples*2:Fsamples*3]).std(axis=0)}
g=pd.DataFrame(data=g)
g=g.round(3)


#Selecting best features by hand starts here
#Optimal number of features before manually optimize
for i in range(3):
    selected=list(final[i].index[0:(n[i]+1)]) 
    print(X_train[i].columns[selected])



#Top correlated features to select for testing insertion in model
i=2 # chosing dataframe, 0, 1 or 2
print("Top Absolute Correlations")
print(get_top_abs_correlations(X_train[i])[0:20])



#Find chosen feature in final array
np.where(final[i]['Feature']=='Hw0')



#Insert and remove features by hand in dataset i

kfold = RepeatedStratifiedKFold(n_repeats=10, n_splits=5, random_state=42)#0.1 for test
a1=[]
for train, test in kfold.split(X_train[i], y_train[i]):

    X_train1, y_train1, X_test1, y_test1= X_train[i].iloc[train,:], y_train[i][train], X_train[i].iloc[test,:], y_train[i][test]    
    selected_features=list(final[i].index[0:(n[i]+1)])  
    #add feature:
    selected_features.append(final[i].index[11])
   
    X_train_sel = X_train1.iloc[:,selected_features]
    X_test_sel = X_test1.iloc[:,selected_features]  
    
    #drop feature:
    X_train_sel=X_train_sel.drop(['Ts11'],axis=1)
    X_test_sel=X_test_sel.drop(['Ts11'],axis=1)

    clf=RandomForestClassifier(random_state=42, max_features='sqrt',criterion='entropy',
                              max_depth=10, min_samples_split=30, max_leaf_nodes=10)
    clf.fit(X=X_train_sel, y=y_train1)
                     
    y_pred=clf.predict(X_test_sel)
    acc1=accuracy_score(y_test1,y_pred)
    a1.append(acc1)

np.mean(a1) #check whether the accuracy increases or decreases



#FINAL: first time predicting in X_test with selected features

selected_features=[['Hs1','Hs21','sinD0','varHs1','cosD0','sinDs21','cosD1','Tp0','cosDs10'],
                   ['sinD0','Hs0','Ts20','cosDw0','Tp0','Hw0','cosDs21'],
                   ['Tw0','cosDw0','cosDs11','Hs1','Tp0']] 

tree_model=[None]*3
y_pred=[None]*3
for i in range(3):
    X_train_sel=X_train[i][selected_features[i]]
    X_test_sel=X_test[i][selected_features[i]]
    acc0,acc1,tree_model[i],y_pred[i]=decision(X_train_sel,X_test_sel,y_train[i],y_test[i])
    print(acc0,acc1)



#Plot
#Confusion matrix of results
for i in range(3):
    fig = plt.figure(figsize=(3,3))
    cbar_ax = fig.add_axes([.93,.3,.03,.4]) 
    ax=plt.subplot()
    cnf_matrix = metrics.confusion_matrix(y_test[i], y_pred[i])
    group_names = ['True 0','False 1','False 0','True 1']
    group_counts = ['{0:0.0f}'.format(value) for value in cnf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cnf_matrix.flatten()/np.sum(cnf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=labels, fmt='',ax=ax, cmap='coolwarm',cbar_ax=cbar_ax)

    #fig.tight_layout()  
    #plt.savefig('confusion_matrix{}'.format(i))



#Plot
#Relation between 2 most important features

for i in range(3):
    fig = plt.figure(figsize=(5,5))
    s=selected_features[i][0:2]
    w=Xx[i].loc[Yy[i] == 0] #ww3
    plt.scatter(w[s[0]],w[s[1]],c='green',label='0 (WW3)',s=70,linewidth=0,alpha=0.9)
    g=Xx[i].loc[Yy[i] == 1] #gwes
    plt.scatter(g[s[0]],g[s[1]],c='blue',label='1 (GWES)',s=70,linewidth=0,alpha=0.6)
    plt.legend(bbox_to_anchor=(0.91,1),frameon=False)
    plt.xlabel(s[0])
    plt.ylabel(s[1])
    plt.title('Forecast day {}'.format(ft0[forecast[i]].item(0)/60/60/24))
    
    #plt.savefig('3d{}'.format(i), bbox_inches='tight')


#Plot
#5d plot of relation between 4 features and output
for i in range(3):
    #--just-for-output-name---
    if i==0:
        j=0
    elif i==1:
        j=5
    else:
        j=7.5
    #-------------------------
    s=selected_features[i]
    data = np.array(Xx[i][s[3]]).reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(3, 21))
    scaler.fit(data)
    markersize = list(scaler.transform(data).flatten())

    df = pd.DataFrame({'label': Yy[i].astype(str).flatten(),
                       s[0]: Xx[i][s[0]],
                       s[1]: Xx[i][s[1]],
                       s[2]: Xx[i][s[2]]})

    color = { '0' : 'green',
              '1' : 'blue'}     

    fig = go.Figure()
    for lbl in df['label'].unique():
        dfp = df[df['label']==lbl]
        fig.add_traces(go.Scatter3d(x=dfp[s[0]], y=dfp[s[1]],z=dfp[s[2]], mode='markers',
                                 name=lbl, marker = dict(color=color[lbl], size = markersize,opacity=1, 
                                 line=dict(width=0.001)) ))

    fig.update_layout(title='Four most important features at forecast day {}'.format(j),legend_title="Legend",
                      scene=dict(xaxis=dict(title=s[0]),yaxis=dict( title=s[1]), zaxis=dict(title=s[2]),
                                annotations=[dict(
                                        x=np.min(Xx[i][s[0]]),
                                        y=np.min(Xx[i][s[1]]),
                                        z=np.max(Xx[i][s[2]]),
                                        text='markersize: {}'.format(s[3]),
                                        showarrow=False)]))

    #plotly.offline.plot({"data": fig}, auto_open=True,filename=('5D Plot{}'.format(i)))


#Plot
#Single tree from random forest to visualize
tree=0
i=0
dot_data = StringIO()
export_graphviz(tree_model[i].estimators_[tree], out_file=dot_data, feature_names =selected_features[i],
                class_names = [str(x) for x in [ int(x) for x in tree_model[i].classes_ ]],
                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png("dt.png")
Image(graph.create_png())

