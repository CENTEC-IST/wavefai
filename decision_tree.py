#!/usr/bin/env python
# coding: utf-8

# In[2]:


import netCDF4
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import time
np.warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from io import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from collections import OrderedDict 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import _tree
import matplotlib.pyplot as plt
from datetime import timedelta
import sys
import copy
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.ensemble import RandomForestClassifier
import timeit
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

WW3_N= 'WW3_selectionPointNearest_20190925_20200701.nc'
nc_N2=Dataset(WW3_N,'r')
GWES_N= 'GWES_selectionPointNearest_20190925_20200701.nc'
nc_N3=Dataset(GWES_N,'r')
GFS_N='GFS_selectionPointNearest_20190925_20200701.nc'
nc_N1=Dataset(GFS_N,'r')

#deterministic
ft2=nc_N2['fctime']
times2=nc_N2.variables['time']
t2=netCDF4.num2date(times2[:],times2.units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
ID2=nc_N2['buoyID']
Dp2=nc_N2['Dp']
Hs2=nc_N2['Hs']
Tp2=nc_N2['Tp']
Ds12=nc_N2['Dswell1']
Ds22=nc_N2['Dswell2']
Hs12=nc_N2['Hswell1']
Hs22=nc_N2['Hswell2']
Ts12=nc_N2['Tswell1']
Ts22=nc_N2['Tswell2']
Dw2=nc_N2['Dwsea']
Hw2=nc_N2['Hwsea']
Tw2=nc_N2['Twsea']
l2=len(t2)
out2 = ['WW3_N'] * l2 

#ensemble
ft3=nc_N3['fctime']
times3=nc_N3.variables['time']
t3=netCDF4.num2date(times3[:],times3.units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
ID3=nc_N3['buoyID']
Dp3=nc_N3['Dp']#direciton wave
Hs3=nc_N3['Hs']#wave height
Tp3=nc_N3['Tp']#peak wave period
Ds13=nc_N3['Dswell1']#direction of swell waves (???) part 1 and 2
Ds23=nc_N3['Dswell2']
Hs13=nc_N3['Hswell1']#height ...
Hs23=nc_N3['Hswell2']
Ts13=nc_N3['Tswell1']#period ...
Ts23=nc_N3['Tswell2']
Dw3=nc_N3['Dwsea']#wind sea 
Hw3=nc_N3['Hwsea']
Tw3=nc_N3['Twsea']
l3=len(t3)
out3 = ['GWES_N'] * l3

u=nc_N1['U10m']
v=nc_N1['V10m']
t1=netCDF4.num2date(nc_N1['time'][:],nc_N1['time'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)

buoy0 = pd.read_table('NDBC_historical_stdmet_41047.txt', sep='\s+')
val=buoy0.iloc[0,:]
buoy0=buoy0[~buoy0.iloc[:,0].str.contains("#")]
buoy0=buoy0.drop_duplicates()
buoy0=buoy0.apply(pd.to_numeric)
date = buoy0[["#YY", "MM", "DD",'hh','mm']].copy()
date.columns = ["year", "month", "day",'hour','minute']
buoy0=buoy0.set_index(pd.to_datetime(date))
buoy0=buoy0.drop(["#YY", "MM", "DD",'hh','mm'],axis=1)
buoy0=buoy0.drop(buoy0.loc[buoy0['WVHT']==99].index)
buoy0=buoy0.drop(buoy0.loc[buoy0['MWD']==999].index)
buoy0['sinD']=np.sin(2*np.pi*buoy0[['MWD']]/360)  
buoy0['cosD']=np.cos(2*np.pi*buoy0[['MWD']]/360)


buoy1 = pd.read_table('NDBC_historical_stdmet_41048.txt', sep='\s+')
buoy1=buoy1[~buoy1.iloc[:,0].str.contains("#", na=False)]
buoy1=buoy1.drop_duplicates()
buoy1=buoy1.apply(pd.to_numeric)
date = buoy1[["#YY", "MM", "DD",'hh','mm']].copy()
date.columns = ["year", "month", "day",'hour','minute']
buoy1=buoy1.set_index(pd.to_datetime(date))
buoy1=buoy1.drop(["#YY", "MM", "DD",'hh','mm'],axis=1)
buoy1=buoy1.drop(buoy1.loc[buoy1['WVHT']==99].index)
#buoy1=buoy1.drop(buoy1.loc[buoy1['MWD']==999].index)
buoy1['sinD']=np.sin(2*np.pi*buoy1[['MWD']]/360)  
buoy1['cosD']=np.cos(2*np.pi*buoy1[['MWD']]/360)


buoy2 = pd.read_table('NDBC_historical_stdmet_41049.txt', sep='\s+')
buoy2=buoy2[~buoy2.iloc[:,0].str.contains("#")]
buoy2=buoy2.drop_duplicates()
buoy2=buoy2.apply(pd.to_numeric)
date = buoy2[["#YY", "MM", "DD",'hh','mm']].copy()
date.columns = ["year", "month", "day",'hour','minute']
buoy2=buoy2.set_index(pd.to_datetime(date))
buoy2=buoy2.drop(["#YY", "MM", "DD",'hh','mm'],axis=1)
buoy2=buoy2.drop(buoy2.loc[buoy2['WVHT']==99].index)
buoy2=buoy2.drop(buoy2.loc[buoy2['MWD']==999].index)
buoy2['sinD']=np.sin(2*np.pi*buoy2[['MWD']]/360)  
buoy2['cosD']=np.cos(2*np.pi*buoy2[['MWD']]/360)

buoy3 = pd.read_table('NDBC_historical_stdmet_44008.txt', sep='\s+')
buoy3=buoy3[~buoy3.iloc[:,0].str.contains("#")]
buoy3=buoy3.drop_duplicates()
buoy3=buoy3.apply(pd.to_numeric)
date = buoy3[["#YY", "MM", "DD",'hh','mm']].copy()
date.columns = ["year", "month", "day",'hour','minute']
buoy3=buoy3.set_index(pd.to_datetime(date))
buoy3=buoy3.drop(["#YY", "MM", "DD",'hh','mm'],axis=1)
buoy3=buoy3.drop(buoy3.loc[buoy3['WVHT']==99].index)
buoy3=buoy3.drop(buoy3.loc[buoy3['MWD']==999].index)
buoy3['sinD']=np.sin(2*np.pi*buoy3[['MWD']]/360)  
buoy3['cosD']=np.cos(2*np.pi*buoy3[['MWD']]/360)

buoy4 = pd.read_table('NDBC_historical_stdmet_44009.txt', sep='\s+')
buoy4=buoy4[~buoy4.iloc[:,0].str.contains("#")]
buoy4=buoy4.drop_duplicates()
buoy4=buoy4.apply(pd.to_numeric)
date = buoy4[["#YY", "MM", "DD",'hh','mm']].copy()
date.columns = ["year", "month", "day",'hour','minute']
buoy4=buoy4.set_index(pd.to_datetime(date))
buoy4=buoy4.drop(["#YY", "MM", "DD",'hh','mm'],axis=1)
buoy4=buoy4.drop(buoy4.loc[buoy4['WVHT']==99].index)
buoy4=buoy4.drop(buoy4.loc[buoy4['MWD']==999].index)
buoy4['sinD']=np.sin(2*np.pi*buoy4[['MWD']]/360)  
buoy4['cosD']=np.cos(2*np.pi*buoy4[['MWD']]/360)

buoy=[buoy0,buoy1,buoy2,buoy3,buoy4]
f0=0 #nowcast
f5=np.where(ft2 == 432000)[0][0] #day 5
fl=60 #last index with prediction, which is 7.5


# In[22]:


with plt.style.context('seaborn-colorblind'):#
    plt.figure(figsize=(16, 5))
    plt.plot(buoy[0].index,buoy[0].iloc[:,3],label= '41047')
    plt.plot(buoy[1].index,buoy[1].iloc[:,3],label= '41048')
    plt.plot(buoy[2].index,buoy[2].iloc[:,3],label= '41049')
    plt.plot(buoy[3].index,buoy[3].iloc[:,3],label= '44008')
    plt.plot(buoy[4].index,buoy[4].iloc[:,3],label= '44009')
    plt.legend(loc='upper left',frameon=False,ncol=2)
    plt.xlabel('Time')
    plt.ylabel('Hs')
    #plt.xticks(fontsize=9)
    #plt.yticks(fontsize=9)
plt.margins(0)
plt.savefig('buoys.png', bbox_inches='tight')#,dpi=80)
plt.show()


# In[1]:


def data(Dp0, Hs0, Tp0, t0, Dp1, Hs1, Tp1, t1, u2, v2, t2, ft, s, obs, b):
    '''
2 wave models: 0 and 1 
1 atmospheric model: 2
t: time arrays
ft: forecast time indice
s: ft seconds
obs: hs of buoy with time as index
b is the buoy indice for the model variables
    '''
    
    if not np.logical_and( (t0==t1).all(), (t1==t2).all() ):
        sys.exit('Models must have same times')
        
    #m0 (ww3)
    df0=[Dp0[b,:,ft],Hs0[b,:,ft],Tp0[b,:,ft]]
    df0=pd.DataFrame(data=np.transpose(np.array(df0)))    
    df0.apply(pd.to_numeric)
    ti0=t0+timedelta(seconds=s)
    model0=df0.set_index(pd.DatetimeIndex(pd.Series(ti0)))
    
    #m1 (gwes)
    dfa=[]    
    for j in range(1,21):
        df1=[Dp1[0,j,:,0],Hs1[0,j,:,0],Tp1[0,j,:,0]]
        df1=pd.DataFrame(data=np.transpose(np.array(df1)))
        dfa.append(df1)
        
    df1a=pd.concat(dfa)
    vHs=df1a[[df1a.columns[1]]].groupby(df1a.index).var()
    df1a=df1a.groupby(df1a.index).mean()    
    df1a['varHs']=vHs
    df1a.apply(pd.to_numeric)
    ti1=t1+timedelta(seconds=s)
    model1=df1a.set_index(pd.DatetimeIndex(pd.Series(ti1)))
    
    df2=[u2[b,:,ft],v2[b,:,ft]]
    df2=pd.DataFrame(data=np.transpose(np.array(df2)))
    df2.apply(pd.to_numeric)
    ti2=t2+timedelta(seconds=s)
    model2=df2.set_index(pd.DatetimeIndex(pd.Series(ti2)))
    
    names=['Dp','Hs','Tp','varHs','U','V']
    model0.columns=names[0:3]
    model1.columns=names[0:4]
    model2.columns=names[-2:]    
    
    model0['sinD']=np.sin(2*np.pi*model0[['Dp']]/360)
    model0['cosD']=np.cos(2*np.pi*model0[['Dp']]/360)
    model0=model0.round(decimals=3)
    
    model1['sinD']=np.sin(2*np.pi*model1[['Dp']]/360)
    model1['cosD']=np.cos(2*np.pi*model1[['Dp']]/360)
    model1=model1.round(decimals=3)

    l0=len(model0)
    #buoy errors
    errorw=[]
    errorg=[]
    for h in range(l0):
            ind = np.where( abs(obs.index - model0.index[h]).total_seconds() <= 2400)  
            errorw.append(abs(model0['Hs'][h] - np.nanmean(obs.iloc[ind])))
            errorg.append(abs(model1['Hs'][h] - np.nanmean(obs.iloc[ind])))        
    
    
    x=[model0['Hs'],model1['Hs'],model0['Tp'],model1['Tp'],model0['sinD'],model1['sinD'],
       model0['cosD'],model1['cosD'],model2['U'],model2['V']]
    X=pd.concat(x,axis=1)
    X.columns=['hsw','hsg','tpw','tpg','sw','sg','cw','cg','u','v']
    y=[]
    choice=np.array(errorw)-np.array(errorg)
    for i in range(len(X)):
        if np.isnan(choice[i]):
            y.append(np.nan)
        elif choice[i]>0:
            y.append('GWES')
        elif choice[i]<0:
            y.append('WW3')
        elif s<432000:#when error is the same, if ft is less than 5 days, ww3, else gwes 
            y.append('WW3')
        else:
            y.append('GWES')
    X['y']=y
    X=X.dropna()    

    return X


# In[85]:


xx=data(Dp2, Hs2, Tp2, t2, Dp3, Hs3, Tp3, t3, u, v, t1, 0, ft2[0], buoy[0]['WVHT'],0)

plt.hist([xx['hsw'],xx['hsg']], bins = 30,density=True,
         color = ['green','blue'], label=['WW3','GWES'])

# Plot formatting
plt.legend()


# In[86]:


def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le.classes_


# In[115]:


def decision(Dp0, Hs0, Tp0, t0, Dp1, Hs1, Tp1, t1, u2, v2, t2, ft, s, buoys):
    '''
2 wave models: 0 and 1 
1 atmospheric model: 2
t: time arrays
ft: forecast time indice
s: ft seconds
buoys: list of buoys data
    '''
    Xx=[]
    for i in range(len(buoys)):
        X = data(Dp0, Hs0, Tp0, t0, Dp1, Hs1, Tp1, t1, u2, v2, t2, ft, s, buoys[i]['WVHT'], i)
        Xx.append(X)
    
    Xx=pd.concat(Xx)
    Yy=Xx['y']
    Xx=Xx.drop('y',axis=1)
    Xx=Xx.astype(float)
    Yy=Yy.values.reshape((len(Yy),1))    
    
    X_train, X_test, y_train, y_test = train_test_split(Xx, Yy, test_size=0.25)
    y_train_enc, y_test_enc, classes = prepare_targets(y_train, y_test)
        
    #Choose best decision tree model 
    param_grid = {"criterion": ["gini", "entropy"],
                'n_estimators': [1500],
                "min_samples_split": [2,6,10],
                "max_depth": [ 10,7, 5, 3],
                "min_samples_leaf": [2,5],
                "max_leaf_nodes": [10,15,30,40],
                'max_features':['sqrt']}
    
        
    clf=RandomForestClassifier(random_state=42)
    clf = GridSearchCV(clf, param_grid, n_jobs=-1,cv=5)
    clf.fit(X=X_train, y=y_train_enc)
    tree_model = clf.best_estimator_
                     
    y_pred=tree_model.predict(X_test)
    acc0=accuracy_score(y_train_enc, tree_model.predict(X_train))
    acc1=accuracy_score(y_test_enc,y_pred)
    report=classification_report(y_test_enc, y_pred)
      
    return ( tree_model,  Xx,Yy, acc0,acc1)
    


# In[112]:


#esta nao demora muito a correr porque e com randomforest standard
def decision(Dp0, Hs0, Tp0, t0, Dp1, Hs1, Tp1, t1, u2, v2, t2, ft, s, buoys):
    '''
2 wave models: 0 and 1 
1 atmospheric model: 2
t: time arrays
ft: forecast time indice
s: ft seconds
buoys: list of buoys data
    '''
    Xx=[]
    for i in range(len(buoys)):
        X = data(Dp0, Hs0, Tp0, t0, Dp1, Hs1, Tp1, t1, u2, v2, t2, ft, s, buoys[i]['WVHT'], i)
        Xx.append(X)
        
    Xx=pd.concat(Xx)
    Yy=Xx['y']
    Xx=Xx.drop('y',axis=1)
    Xx=Xx.astype(float)
    Yy=Yy.values.reshape((len(Yy),1))    
    
    X_train, X_test, y_train, y_test = train_test_split(Xx, Yy, test_size=0.25)
    y_train_enc, y_test_enc, classes = prepare_targets(y_train, y_test)

        
    clf=RandomForestClassifier(random_state=42)

    clf.fit(X=X_train, y=y_train_enc)

                     
    y_pred=clf.predict(X_test)
    acc0=accuracy_score(y_train_enc, clf.predict(X_train))
    acc1=accuracy_score(y_test_enc,y_pred)
    report=classification_report(y_test_enc, y_pred)
      
    return (   Xx,Yy, acc0,acc1)
    


# In[117]:


D1=decision(Dp2, Hs2, Tp2,t2,Dp3, Hs3, Tp3,t3, u, v, t1, f5, ft2[f5], buoy)


# In[37]:


sns.heatmap(D[0].corr())


# In[58]:



plt.figure(figsize=(20,20))
w=D[0].loc[D[1] == 'WW3']
plt.scatter(w.iloc[:,0],w.iloc[:,1],label='WW3',color='green',s=10)
g=D[0].loc[D[1] == 'GWES']
plt.scatter(g.iloc[:,0],g.iloc[:,1],label='GWES',color='blue',s=10,alpha=0.8)
plt.xlabel('hsw')
plt.ylabel('hsg')
plt.legend()
plt.show()


# In[5]:


M2=[]
M3=[]
Dset=[]
Bmatch=[]


for b in range(len(buoy)):
    m2=[]
    m3=[]
    for i in range(0,61):
    
        data2=Hs2[b,:,i]
        df2=pd.DataFrame(data = data2)
        df2.apply(pd.to_numeric)
        ti2=t2+timedelta(seconds=ft2[i])
        df2=df2.set_index(pd.DatetimeIndex(pd.Series(ti2)))
        df2['out']=out2
        #ensemble average of GWES
        dfa=[]    
        for j in range(1,21):
            group=range(l3)
            df=[Hs3[b,j,:,i],group]
            df=np.transpose(np.array(df))
            df=pd.DataFrame(data = df)
            df.apply(pd.to_numeric)
            dfa.append(df)
 
        df3a=pd.concat(dfa)
        df3a=df3a.groupby(df3a.columns[1]).mean()
        df3a=df3a.set_index(pd.DatetimeIndex(pd.Series(ti2)))
        df3a['out']= out3
                 
        #merge 2 dataframes
        result = df2.append(df3a)
        df2df3a=pd.DataFrame(result)
        names=['Hs','out']
        df2df3a.columns=names
        df2df3a=df2df3a.round(decimals=3)
        Dset.append(df2df3a)
        error=[]
        match=[]
        for j in range(0,len(df2df3a)):
            ind = np.where( abs(buoy0.index - df2df3a.index[j]).total_seconds() <= 2400)
            error.append(df2df3a['Hs'][j] - np.nanmean(buoy0['WVHT'].iloc[ind]))
            match.append(np.nanmean(buoy0['WVHT'].iloc[ind]))
            
        l=len(df2df3a.loc[df2df3a['out'] == 'WW3_N'])
        Bmatch.append(match[0:l])#no need for repetition of buoy values
        m2.append(error[0:l])#matrix WW3
        m3.append(error[l:])#matrix aGWES
 
    m2=np.array(m2)
    m3=np.array(m3)
    M2.append(m2)
    M3.append(m3)


# In[28]:


X,Y = np.meshgrid(np.arange(M2[0].shape[1]),np.arange(M2[0].shape[0]))
date = [x.date().isoformat() for x in t2]#times in 2(ww3) are equal to times in 3(gwes)
dt = np.array(date)
ft = np.array(ft2[0:61]).astype(int)
dts=[]
for i in dt[::50]:
    dts.append(i)
fts=[]
for i in ft[::20]:
    fts.append(i/60/60/24)
    
figure = plt.figure(figsize=(10,10)) 
g=1
for i in range(len(buoy)):
    ax = plt.subplot(len(buoy)*2, 1, g)
    ax.set_title(ID2[i],fontsize=9,pad=1) 
    m2m=ax.scatter(X,Y,c=M2[i], s=15, linewidth=0, marker="s",cmap='RdYlBu')#,vmin=-4,vmax=4)
    figure.colorbar(m2m, ax=ax,shrink=0.9, aspect=7,pad=.01)
    ax.margins(0)
    figure.canvas.draw()
    
    ax.set_xticks([])

    ax.set_yticks(np.arange(0,M2[i].shape[0],20))
    ax.set_yticklabels(fts,fontsize=9)
    
    g+=1
    ax = plt.subplot(len(buoy)*2, 1, g)
    
    m3m=ax.scatter(X,Y,c=M3[i], s=15, linewidth=0, marker="s",cmap='RdYlBu')#,vmin=-4,vmax=4)
    figure.colorbar(m3m, ax=ax, shrink=0.9, aspect=7,pad=.01)
    ax.margins(0)
    figure.canvas.draw()
    if i == len(buoy)-1:
        ax.set_xticks(np.arange(0,M3[i].shape[1],50))
        ax.set_xticklabels(dts,fontsize=9)
    else: 
        ax.set_xticks([])

    ax.set_yticks(np.arange(0,M2[i].shape[0],20))
    ax.set_yticklabels(fts,fontsize=9)

    g+=1

#ax1.set_title('WW3',fontsize=9)
#ax2.set_title('GWES',fontsize=9)
figure.text(0.45,0.09,'Time',ha='center',va='center')
figure.text(0.09, 0.5, 'Forecast Time (days)', ha='center', va='center', rotation='vertical')
plt.subplots_adjust(wspace=0, hspace=0.22)
#plt.savefig('chiclet', bbox_inches='tight')
plt.show()


# In[31]:


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


# In[138]:


figure = plt.figure(figsize=(13,17)) 
g=1
for i in range(len(buoy)):
    Mw=[]
    Mg=[]
    for j in range(0,61):
        wm=Dset[j+61*i][['Hs']].iloc[0:l]#array WW3
        gm=Dset[j+61*i][['Hs']].iloc[l:]#array aGWES
        
        mw=metrics(wm,pd.DataFrame(Bmatch[j+61*i]))
        mg=metrics(gm,pd.DataFrame(Bmatch[j+61*i]))    
        Mw.append(mw)
        Mg.append(mg)
    Mw=np.vstack(Mw)
    Mw=np.transpose(Mw)
    Mg=np.vstack(Mg)
    Mg=np.transpose(Mg)
    
    ax = plt.subplot(len(buoy), 3, g)    
    ax.plot(ft2[0:61]/60/60/24,Mw[2],c='green',label='WW3')
    ax.plot(ft3[0:61]/60/60/24,Mg[2],c='blue',label='GWES') 
    
    if g==1:
        ax.set_title('NBias')
        ax.legend(loc='upper left',frameon=False)    
    g+=1
    
    ax = plt.subplot(len(buoy), 3, g)
    ax.plot(ft2[0:61]/60/60/24,Mw[5],c='green',label='WW3')
    ax.plot(ft3[0:61]/60/60/24,Mg[5],c='blue',label='GWES')    
    if g==2:
        ax.set_title('SI')
    g+=1
        
    ax = plt.subplot(len(buoy), 3, g)
    ax.plot(ft2[0:61]/60/60/24,Mw[1],c='green',label='WW3')
    ax.plot(ft3[0:61]/60/60/24,Mg[1],c='blue',label='GWES')
    if g==3:
        ax.set_title('RMSE')    
    if g==(3*i+3):        
        ax.set_ylabel(ID2[i]) 
        ax.yaxis.set_label_position("right")
    g+=1

figure.text(0.5,0.11,'Forecast Time (days)',ha='center',va='center')
plt.margins(0)
plt.subplots_adjust(wspace=0.17, hspace=0.13)
plt.savefig('metrics', bbox_inches='tight')
plt.show()


# In[4]:



forecast=[f0,f5,fl]
trees=[]
tree_params=[]
bpy=[]
D=[]

for b in range(len(buoy)):
    bp=buoy[b][['WVHT','DPD','sinD','cosD']]
    bp['errorHS']=np.zeros(len(bp))

    for i in forecast:
        data2=[Dp2[b,:,i],Hs2[b,:,i],Tp2[b,:,i],Ds12[b,:,i],Ds22[b,:,i],Hs12[b,:,i],Hs22[b,:,i],Ts12[b,:,i],Ts22[b,:,i],Dw2[b,:,i],Hw2[b,:,i],Tw2[b,:,i]]
        data2=np.transpose(np.array(data2))
        df2=pd.DataFrame(data = data2)
        df2.apply(pd.to_numeric)
        df2['out']=out2
        ti2=t2+timedelta(seconds=ft2[i])
        df2=df2.set_index(pd.DatetimeIndex(pd.Series(ti2)))
    
        dfa=[]    
        for j in range(1,21):
            group=range(l3)
            df=[Dp3[b,j,:,i],Hs3[b,j,:,i],Tp3[b,j,:,i],Ds13[b,j,:,i],Ds23[b,j,:,i],Hs13[b,j,:,i],Hs23[b,j,:,i],Ts13[b,j,:,i],Ts23[b,j,:,i],Dw3[b,j,:,i],Hw3[b,j,:,i],Tw3[b,j,:,i],group]
            d=np.transpose(np.array(df))
            df=pd.DataFrame(data = d)
            df.apply(pd.to_numeric)
            dfa.append(df)
        
        df3a=pd.concat(dfa)
        df3a=df3a.groupby(df3a.columns[12]).mean()
        df3a['out']= out3
        ti3=t3+timedelta(seconds=ft3[i])
        df3a=df3a.set_index(pd.DatetimeIndex(pd.Series(ti3)))
        #merge 2 dataframes
        result = df2.append(df3a)
        df2df3a=pd.DataFrame(result)
        names=['Dp','Hs','Tp','Ds1','Ds2','Hs1','Hs2','Ts1','Ts2','Dw','Hw','Tw','out']
        df2df3a.columns=names
        df2df3a['sinD']=np.sin(2*np.pi*df2df3a[['Dp']]/360)
        df2df3a['cosD']=np.cos(2*np.pi*df2df3a[['Dp']]/360)
        df2df3a=df2df3a.round(decimals=3)
    
        #buoy errors
        error=[]
        for h in range(0,len(df2df3a)):
            ind = np.where( abs(buoy[b].index - df2df3a.index[h]).total_seconds() <= 2400) # vai dar o ind de tempo no array do tempo da boia 
            error.append(np.mean(buoy[b]['WVHT'].iloc[ind]) - df2df3a['Hs'][h])
        
        X=df2df3a[['Hs','Tp','sinD','cosD']]
        X['errorHS']=list(map(abs,error))
        X['out']=df2df3a.iloc[:,12]
        X=X.dropna()
        D.append(X)
        y=X[['out']]
        X=X.drop(columns=['out'])
        names=['Hs','Tp','sinD','cosD','errorHS']

        #Choose best decision tree model 
        param_grid = {"criterion": ["gini", "entropy"],
              'n_estimators': [500],
              "min_samples_split": [2, 10],
              "max_depth": [None, 10],
              "min_samples_leaf": [1],
              "max_leaf_nodes": [None, 20],
              'max_features': [ None ] }
        clf=RandomForestClassifier(random_state=42)
        clf = GridSearchCV(clf, param_grid, n_jobs=-1,cv=5)
        clf.fit(X=X, y=y.values.ravel())
        tree_model = clf.best_estimator_
        tree_params.append(clf.best_params_)
        trees.append(tree_model)
                  
        y_pred=tree_model.predict(bp)
        bpy.append(y_pred)


# In[145]:


plt.rcParams.update({'font.size': 18})   
fig, ax = plt.subplots(5, 6)#, gridspec_kw={'width_ratios': [2.5, 1.5,2.5,1.5,2.5,1.5]})
fig.set_figheight(27)
fig.set_figwidth(35)
nrow=0
k=0
for b in range(len(buoy)): 
    ncol=0
    for i in forecast:
        #ax = figure.add_subplot(spec[nrow, ncol])      
            
        indw=np.where(bpy[k]=='WW3_N')
        ax[nrow,ncol].scatter(buoy[b].iloc[indw][['DPD']],buoy[b].iloc[indw][['WVHT']],c='green',s=10,label='WW3',linewidth=0)
        indg=np.where(bpy[k]=='GWES_N')
        ax[nrow,ncol].scatter(buoy[b].iloc[indg][['DPD']],buoy[b].iloc[indg][['WVHT']],c='blue',s=10,label='GWES',linewidth=0)
        ax[nrow,ncol].margins(0)
        if nrow==0 and ncol==0:            
            ax[nrow,ncol].legend(loc='upper left',frameon=False)
        if b==0:
            if i == forecast[0]:
                ax[nrow,ncol].set_title("Forecast Day 0 (Hs vs Tp)")                
            elif i==forecast[1]:
                ax[nrow,ncol].set_title("Forecast Day 5 (Hs vs Tp)")
            elif i==forecast[2]:
                ax[nrow,ncol].set_title("Forecast Day 7.5 (Hs vs Tp)")
        
        ncol+=1
        #ax = figure.add_subplot(spec[nrow, ncol])
        indw=np.where(bpy[k]=='WW3_N')
        ax[nrow,ncol].scatter(buoy[b].iloc[indw][['cosD']],buoy[b].iloc[indw][['WVHT']],c='green',s=10,label='WW3',linewidth=0)
        indg=np.where(bpy[k]=='GWES_N')
        ax[nrow,ncol].scatter(buoy[b].iloc[indg][['cosD']],buoy[b].iloc[indg][['WVHT']],c='blue',s=10,label='GWES',linewidth=0)
        ax[nrow,ncol].margins(0)
        if nrow==0 and ncol==0:            
            ax[nrow,ncol].legend(loc='upper left',frameon=False)
        if b==0:
            if i == forecast[0]:
                ax[nrow,ncol].set_title("Forecast Day 0 (Hs vs cosD)")                
            elif i==forecast[1]:
                ax[nrow,ncol].set_title("Forecast Day 5 (Hs vs cosD)")
            elif i==forecast[2]:
                ax[nrow,ncol].set_title("Forecast Day 7.5 (Hs vs cosD)")   
        if i==forecast[2]:
            ax[nrow,ncol].set_ylabel(ID2[b])
            ax[nrow,ncol].yaxis.set_label_position("right")
       
        ncol+=1
        k+=1
    nrow+=1 
plt.subplots_adjust(wspace=0.1,hspace=0.1)
plt.savefig('predictions_newnew.png', bbox_inches='tight')
plt.show()


# In[148]:


plt.rcParams.update({'font.size': 9})
fig, ax = plt.subplots(5, 3)#, gridspec_kw={'width_ratios': [2.5, 1.5,2.5,1.5,2.5,1.5]})
fig.set_figheight(15)
fig.set_figwidth(10)
nrow=0
k=0
for b in range(len(buoy)): 
    ncol=0
    for i in forecast:
        importance = trees[k].feature_importances_
        std = np.std([tree.feature_importances_ for tree in trees[k].estimators_], axis=0)
        X=D[k].copy()
        X=X.drop(columns=['out'])
        indices = np.argsort(-1*importance)[:X.shape[1]]
        features=X.columns        
        ax[nrow,ncol].bar(range(X.shape[1]), importance[indices], width=0.5,color='orange', yerr=std[indices], align="center")
        ax[nrow,ncol].set_xticks(np.arange(len(features)))
        ax[nrow,ncol].set_xticklabels(features[indices] )            
        if b==0:
            if i == forecast[0]:
                ax[nrow,ncol].set_title("Forecast Day 0")                
            elif i==forecast[1]:
                ax[nrow,ncol].set_title("Forecast Day 5")
            elif i==forecast[2]:
                ax[nrow,ncol].set_title("Forecast Day 7.5")
        if i==forecast[2]:
            ax[nrow,ncol].set_ylabel(ID2[b])
            ax[nrow,ncol].yaxis.set_label_position("right")
        
        ncol+=1
        k+=1
    nrow+=1
    
plt.subplots_adjust(wspace=0.15,hspace=0.1)
plt.savefig('predictions_importance.png', bbox_inches='tight')
plt.show()


# In[52]:


#errors in m for wave in ft time selected

error=[]
indf=[]
#eTp=[]
#eS=[]
#eC=[]
for i in range(0,len(df2df3a)):
    ind = np.where( abs(buoy0.index - df2df3a.index[i]).total_seconds() <= 2400) # vai dar o ind de tempo no array do tempo da boia 
    error.append(np.mean(buoy0['WVHT'].iloc[ind]) - df2df3a['Hs'][i])
   # eTp.append(np.mean(buoy0['DPD'].iloc[ind])-df2df3a['Tp'][i])
   # eS.append(np.mean(buoy0['sin'].iloc[ind])-df2df3a['sin'][i])
   # eC.append(np.mean(buoy0['cos'].iloc[ind])-df2df3a['cos'][i])
    indf.append(ind)
    


# In[77]:


#relation between error and Hs 
w=errorHs.loc[errorHs['out'] == 'WW3_N']
plt.scatter(w.iloc[:,0],w.iloc[:,2],c='green',label='WW3_N',linewidth=0)
g=errorHs.loc[errorHs['out'] == 'GWES_N']
plt.scatter(g.iloc[:,0],g.iloc[:,2],c='blue',label='GWES_N',linewidth=0)
plt.legend()
plt.xlabel('Hs')
plt.ylabel('Residue of Hs')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.savefig('eHsl.png', bbox_inches='tight')#,dpi=80)
plt.show()


# In[80]:


#behavior of Hs depending on model ft0
plotdf=df2df3a[[df2df3a.columns[1],df2df3a.columns[12]]]
plotdf=plotdf.reset_index()



plt.figure(figsize=(12, 3))
w=plotdf.loc[plotdf['out'] == 'WW3_N']
plt.plot(w.iloc[:,0],w.iloc[:,1],c='green', label='WW3_N')
g=plotdf.loc[plotdf['out'] == 'GWES_N']
plt.plot(g.iloc[:,0],g.iloc[:,1],c='blue',label='GWES_N')

plt.plot(buoy0f0.iloc[:,0],buoy0f0.iloc[:,4],c='orange',label= 'buoy41047', linestyle='dashed')#,s=15,linewidth=0)
plt.legend()
plt.xlabel('time')
plt.ylabel('Hs')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.savefig('wavel.png', bbox_inches='tight')#,dpi=80)
plt.show()


# In[67]:


#plot wave period values of both buoy and models to see if they coincide and whether there are outliers or not
d0t=df2df3a[[df2df3a.columns[2],df2df3a.columns[12]]]
d0t=d0t.reset_index()

plt.scatter(d0t.iloc[:,0],d0t.iloc[:,1],s=10)
w=d0t.loc[d0t['out'] == 'WW3_N']
plt.scatter(w.iloc[:,0],w.iloc[:,1],c='green',s=10,label='WW3_N')
g=d0t.loc[d0t['out'] == 'GWES_N']
plt.scatter(g.iloc[:,0],g.iloc[:,1],c='blue',s=10,label='GWES_N')
plt.legend()
plt.xlabel('date')
plt.ylabel('Tp')
plt.show()


# In[81]:


#decision tree focusing on wave height:

a=df2df3a[[df2df3a.columns[1],df2df3a.columns[12]]]
a=a.dropna()
X=a.iloc[:,0]
X=X.values.reshape(-1,1)
names=['Hs']
y=a.iloc[:,1]
y=y.values.reshape(-1,1)

#Choose best decision tree model 
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
clf = DecisionTreeClassifier(random_state=1)
clf = GridSearchCV(clf, param_grid, n_jobs=4,cv=10)
clf.fit(X=X, y=y)
tree_model = clf.best_estimator_


# In[43]:


#plot of chosen model for each hs value of buoy if decision tree with just hs
#delete outliers 
b0p=buoy0['WVHT']
b0p=b0p.drop(b0p.loc[b0p==99].index)
b0p = pd.DataFrame(b0p)
b0p.columns=['Hs']
y_pred=tree_model.predict(b0p)
b0p['y_pred']= y_pred
b0p=b0p.reset_index()

b0pw=b0p.loc[b0p['y_pred'] == 'WW3_N']
plt.scatter(b0pw.iloc[:,0],b0pw.iloc[:,1],c='green',s=10,label='WW3_N',linewidth=0)
b0pg=b0p.loc[b0p['y_pred'] == 'GWES_N']
plt.scatter(b0pg.iloc[:,0],b0pg.iloc[:,1],c='blue',s=10,label='GWES_N',linewidth=0)
plt.legend()
plt.xlabel('time')
plt.ylabel('Hs')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.savefig('predHs5.png', bbox_inches='tight')
plt.show()


# In[101]:


#decisiontree of hs and error associated 

errorHs=df2df3a[[df2df3a.columns[1],df2df3a.columns[12]]]
errorHs=errorHs.reset_index()
errorHs['errorHS']=list(map(abs, error)) 
errorHs=errorHs.dropna()
X=errorHs[['Hs','errorHS']]
names=['Hs', 'errorHS']

y=errorHs[['out']]


#Choose best decision tree model 
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
clf = DecisionTreeClassifier(random_state=1)
clf = GridSearchCV(clf, param_grid, n_jobs=4,cv=10)
clf.fit(X=X, y=y)
tree_model = clf.best_estimator_


# In[85]:


#predicted model for each hs value with errors =0
b0p=buoy0['WVHT']
b0p=b0p.drop(b0p.loc[b0p==99].index)
b0p = pd.DataFrame(b0p)
b0p.columns=['Hs']
b0p['error']=np.zeros(len(b0p))#errors are all zero
y_pred=tree_model.predict(b0p)
b0p['y_pred']= y_pred
b0p=b0p.reset_index()

b0pw=b0p.loc[b0p['y_pred'] == 'WW3_N']
plt.scatter(b0pw.iloc[:,0],b0pw.iloc[:,1],c='green',s=10,label='WW3_N',linewidth=0)
b0pg=b0p.loc[b0p['y_pred'] == 'GWES_N']
plt.scatter(b0pg.iloc[:,0],b0pg.iloc[:,1],c='blue',s=10,label='GWES_N',linewidth=0)
plt.legend()
plt.xlabel('time')
plt.ylabel('Hs')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.savefig('predHsel.png', bbox_inches='tight')
plt.show()


# In[96]:


#decisiontree of hs, Tp and error associated 
errorHsT=df2df3a.iloc[:,1:3]
errorHsT=pd.DataFrame(errorHsT)
errorHsT['errorHS']=list(map(abs,error))
errorHsT['out']=df2df3a.iloc[:,12]
errorHsT=errorHsT.dropna()


X=errorHsT[['Hs','Tp','errorHS']]
names=['Hs','Tp', 'errorHS']
y=errorHsT[['out']]


#Choose best decision tree model 
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
clf = DecisionTreeClassifier(random_state=1)
clf = GridSearchCV(clf, param_grid, n_jobs=4,cv=10)
clf.fit(X=X, y=y)
tree_model = clf.best_estimator_


# In[87]:


#predicted model for each hs value with errors =0 and Tp
b0p=buoy0[['WVHT','DPD']]
b0p = pd.DataFrame(b0p)
b0p.columns=['Hs','Tp']
b0p['error']=np.zeros(len(b0p))#errors are all zero
y_pred=tree_model.predict(b0p)
b0p['y_pred']= y_pred
b0p=b0p.reset_index()

b0pw=b0p.loc[b0p['y_pred'] == 'WW3_N']
plt.scatter(b0pw.iloc[:,0],b0pw.iloc[:,1],c='green',s=10,label='WW3_N',linewidth=0)
b0pg=b0p.loc[b0p['y_pred'] == 'GWES_N']
plt.scatter(b0pg.iloc[:,0],b0pg.iloc[:,1],c='blue',s=10,label='GWES_N',linewidth=0)
plt.legend()
plt.xlabel('time')
plt.ylabel('Hs')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.savefig('predHseTl.png', bbox_inches='tight')
plt.show()


# In[88]:


#decisiontree of hs, Tp and error associated and Dp
errorHsTD=df2df3a.iloc[:,1:3]
errorHsTD=pd.DataFrame(errorHsTD)
errorHsTD['sin']=np.sin(2*np.pi*df2df3a[['Dp']]/360)
errorHsTD['cos']=np.cos(2*np.pi*df2df3a[['Dp']]/360)
errorHsTD['errorHS']=list(map(abs,error))
errorHsTD['out']=df2df3a.iloc[:,12]
errorHsTD=errorHsTD.dropna()


X=errorHsTD[['Hs','Tp','sin', 'cos','errorHS']]
names=['Hs','Tp', 'sin', 'cos','errorHS']
y=errorHsTD[['out']]


#Choose best decision tree model 
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
clf = DecisionTreeClassifier(random_state=1)
clf = GridSearchCV(clf, param_grid, n_jobs=4,cv=10)
clf.fit(X=X, y=y)
tree_model = clf.best_estimator_


# In[89]:


#predicted model for each hs value with errors =0 and Tp and Dp
b0p=buoy0[['WVHT','DPD']]
b0p['sin']=np.sin(2*np.pi*buoy0[['MWD']]/360)
b0p['cos']=np.cos(2*np.pi*buoy0[['MWD']]/360)
b0p = pd.DataFrame(b0p)
b0p['error']=np.zeros(len(b0p))#errors are all zero
y_pred=tree_model.predict(b0p)
b0p['y_pred']= y_pred
b0p=b0p.reset_index()

b0pw=b0p.loc[b0p['y_pred'] == 'WW3_N']
plt.scatter(b0pw.iloc[:,0],b0pw.iloc[:,1],c='green',s=10,label='WW3_N',linewidth=0)
b0pg=b0p.loc[b0p['y_pred'] == 'GWES_N']
plt.scatter(b0pg.iloc[:,0],b0pg.iloc[:,1],c='blue',s=10,label='GWES_N',linewidth=0)
plt.legend()
plt.xlabel('time')
plt.ylabel('Hs')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.savefig('predHseTDl.png', bbox_inches='tight')
plt.show()


# In[57]:


#NAO FUNCIONA
#decisiontree of hs, Tp and error associated and Dp and varHS
errorHsTD=df2df3a.iloc[:,1:3]
errorHsTD=pd.DataFrame(errorHsTD)
errorHsTD['sin']=np.sin(2*np.pi*df2df3a[['Dp']]/360)
errorHsTD['cos']=np.cos(2*np.pi*df2df3a[['Dp']]/360)
errorHsTD['errorHS']=list(map(abs,error))
errorHsTD['varHS']=df2df3a.iloc[:,13]
errorHsTD['out']=df2df3a.iloc[:,12]
errorHsTD=errorHsTD.dropna()


X=errorHsTD[['Hs','Tp','sin', 'cos','errorHS','varHS']]
names=['Hs','Tp', 'sin', 'cos','errorHS','varHS']
y=errorHsTD[['out']]


#Choose best decision tree model 
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
clf = DecisionTreeClassifier(random_state=1)
clf = GridSearchCV(clf, param_grid, n_jobs=4,cv=10)
clf.fit(X=X, y=y)
tree_model = clf.best_estimator_


# In[103]:


#experiments
#dataframe of ww3
b=1
f=fl
data2=[Dp2[b,:,f],Hs2[b,:,f],Tp2[b,:,f],Ds12[b,:,f],Ds22[b,:,f],Hs12[b,:,f],Hs22[b,:,f],Ts12[b,:,f],Ts22[b,:,f],Dw2[b,:,f],Hw2[b,:,f],Tw2[b,:,f]]
data2=np.transpose(np.array(data2))
df2=pd.DataFrame(data = data2)
df2.apply(pd.to_numeric)
df2['out']=out2
df2['varHs']=np.zeros(len(df2))
times2=nc_N2.variables['time']
ti2=netCDF4.num2date(times2[:],times2.units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
ti2=ti2+timedelta(seconds=ft2[f])
df2=df2.set_index(pd.DatetimeIndex(pd.Series(ti2)))
#extract rows with nan values if there's any
#nan=np.where(np.isnan(df2.iloc[:,0:12]))
#ind = list(OrderedDict.fromkeys(nan[0])) 
#df2=df2.drop(df2.index[ind])

#dataframe of ensemble average of GWES
dfa=[]    
for i in range(1,21):
    group=range(l3)
    df=[Dp3[b,i,:,f],Hs3[b,i,:,f],Tp3[b,i,:,f],Ds13[b,i,:,f],Ds23[b,i,:,f],Hs13[b,i,:,f],Hs23[b,i,:,f],Ts13[b,i,:,f],Ts23[b,i,:,f],Dw3[b,i,:,f],Hw3[b,i,:,f],Tw3[b,i,:,f],group]
    d=np.transpose(np.array(df))
    df=pd.DataFrame(data = d)
    df.apply(pd.to_numeric)
    dfa.append(df)
 
df3a=pd.concat(dfa)
vHs=df3a[[df3a.columns[1],df3a.columns[12]]]
vHs=vHs.groupby(df3a.columns[12]).var()
df3a=df3a.groupby(df3a.columns[12]).mean()
df3a['out']= out3
df3a['varHs']=vHs
times3=nc_N3.variables['time']
ti3=netCDF4.num2date(times3[:],times3.units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
ti3=ti3+timedelta(seconds=ft3[f])
df3a=df3a.set_index(pd.DatetimeIndex(pd.Series(ti3)))
#extract rows with nan values if there's any
#nan=np.where(np.isnan(df3a.iloc[:,0:12]))
#ind = list(OrderedDict.fromkeys(nan[0])) 
#df3a=df3a.drop(df3a.index[ind])

#merge 2 dataframes
result = df2.append(df3a)
#result = result.reset_index(drop=True)
df2df3a=pd.DataFrame(result)
names=['Dp','Hs','Tp','Ds1','Ds2','Hs1','Hs2','Ts1','Ts2','Dw','Hw','Tw','out','varHs']
df2df3a.columns=names
df2df3a=df2df3a.round(decimals=3)  
df2df3a['sinD']=np.sin(2*np.pi*df2df3a[['Dp']]/360)
df2df3a['cosD']=np.cos(2*np.pi*df2df3a[['Dp']]/360)

#errors in m for wave in ft time selected

error=[]
indf=[]
#eTp=[]
#eS=[]
#eC=[]
for i in range(0,len(df2df3a)):
    ind = np.where( abs(buoy[b].index - df2df3a.index[i]).total_seconds() <= 2400) # vai dar o ind de tempo no array do tempo da boia 
    error.append(np.mean(buoy[b]['WVHT'].iloc[ind]) - df2df3a['Hs'][i])
   # eTp.append(np.mean(buoy0['DPD'].iloc[ind])-df2df3a['Tp'][i])
   # eS.append(np.mean(buoy0['sin'].iloc[ind])-df2df3a['sin'][i])
   # eC.append(np.mean(buoy0['cos'].iloc[ind])-df2df3a['cos'][i])
    indf.append(ind)


# In[107]:


#indf=list(indf)
indf=np.hstack(indf).squeeze()
buoy0f0=buoy[b].iloc[indf,:]
buoy0f0= buoy0f0[~buoy0f0.index.duplicated(keep='first')]
buoy0f0


# In[33]:



plt.scatter(df2df3a[['Hs']],df2df3a[['Tp']],s=10,label='sinD',color='c')
plt.scatter(np.arange(df2df3a.shape[0]),df2df3a[['cosD']],s=10,label='cosD',color='orange')
plt.legend()
plt.show()


# In[65]:


#plt.figure(figsize=(3,9))
w=df2df3a.loc[df2df3a['out'] == 'WW3_N']
plt.scatter(w[['Tp']],w[['Hs']],c='green',s=10,label='WW3_N')
g=df2df3a.loc[df2df3a['out'] == 'GWES_N']
plt.scatter(g[['Tp']],g[['Hs']],c='blue',s=10,label='GWES_N')
plt.legend()
plt.ylabel('Hs')
plt.xlabel('Tp')
plt.show()


# In[66]:


w=df2df3a.loc[df2df3a['out'] == 'WW3_N']
plt.scatter(w[['Hs']],w[['Tp']],c='green',s=10,label='WW3_N')
g=df2df3a.loc[df2df3a['out'] == 'GWES_N']
plt.scatter(g[['Hs']],g[['Tp']],c='blue',s=10,label='GWES_N')
plt.legend()
plt.ylabel('Tp')
plt.xlabel('Hs')
plt.show()


# In[83]:


X=df2df3a[['Hs','Tp','sinD','cosD']]
X['errorHS']=list(map(abs,error))
#X['eS']=list(map(abs,eS))
#X['eC']=list(map(abs,eC))
X['out']=df2df3a.iloc[:,12]
X=X.dropna()
y=X[['out']]
X=X.drop(columns=['out'])

names=['Hs','Tp', 'sinD', 'cosD','errorHS']#,'eS','eC']


#Choose best decision tree model 
param_grid = {"criterion": ["gini", "entropy"],
              'n_estimators': [100,200,300, 500],
              "min_samples_split": [2, 10],
              "max_depth": [None, 10],
              "min_samples_leaf": [1],
              "max_leaf_nodes": [None, 20],
              'max_features': [ None]}
start = timeit.default_timer()
clf=RandomForestClassifier(random_state=42)
clf = GridSearchCV(clf, param_grid, n_jobs=-1,cv=5)
clf.fit(X=X, y=y.values.ravel())
stop = timeit.default_timer()
runt= stop - start
tree_model = clf.best_estimator_
tree_params=clf.best_params_


# In[69]:


bp=buoy[b][['WVHT','DPD','sinD','cosD']]
bp['errorHS']=np.zeros(len(bp))
y_pred=tree_model.predict(bp)
bpy=bp.copy()
bpy['y_pred']= y_pred
bpw=bpy.loc[bpy['y_pred'] == 'WW3_N']
plt.scatter(bpw[['DPD']],bpw[['WVHT']],c='green',s=10,label='WW3',linewidth=0)
bpg=bpy.loc[bpy['y_pred'] == 'GWES_N']
plt.scatter(bpg[['DPD']],bpg[['WVHT']],c='blue',s=10,label='GWES',linewidth=0)
plt.show()


# In[99]:


plt.scatter(buoy[4][['MWD']],buoy[4][['WVHT']])
plt.show()


# In[125]:


fig=plt.figure()
ax=plt.subplot(1,1,1)
importance = tree_model.feature_importances_

std = np.std([tree.feature_importances_
                            for tree in tree_model.estimators_], axis=0)

indices = np.argsort(importance)
features=X.columns

ax.barh(range(X.shape[1]), importance[indices], color='orange', xerr=std[indices], align="center")#,width=0.5)
#ax.set_yticks(np.arange(len(features)))
#ax.set_yticklabels(features[indices] )
#ax.tick_params(axis="y",direction="in", pad=-50)
ax.set_xlabel('Relative Importance')
ax.set_ylabel('Features')
#plt.savefig('featimp0')

plt.show()


# In[85]:


#random forest
dot_data = StringIO()
export_graphviz(clf.estimators_[0], out_file=dot_data, #feature_names =X.columns, #trees[3*3+2].estimators_[0]
                            filled=True,class_names = classes, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())


# In[53]:


b=3
bp=buoy[b][['WVHT','DPD','sinD','cosD']]
bp['errorHS']=np.zeros(len(bp))


# In[67]:


# empty all nodes, i.e.set color to white and number of samples to zero
for node in graph.get_node_list():
    if node.get_attributes().get('label') is None:
        continue
    if 'samples = ' in node.get_attributes()['label']:
        labels = node.get_attributes()['label'].split('<br/>')
        for i, label in enumerate(labels):
            if label.startswith('samples = '):
                labels[i] = 'samples = 0'
        node.set('label', '<br/>'.join(labels))
        node.set_fillcolor('white')

samples = bp
decision_paths = trees[3*3+2].estimators_[0].decision_path(samples)

for decision_path in decision_paths:
    for n, node_value in enumerate(decision_path.toarray()[0]):
        if node_value == 0:
            continue
        node = graph.get_node(str(n))[0]            
        node.set_fillcolor('orange')
        labels = node.get_attributes()['label'].split('<br/>')
        for i, label in enumerate(labels):
            if label.startswith('samples = '):
                labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

        node.set('label', '<br/>'.join(labels))
        

graph.write_png("dt32.png")
Image(graph.create_png())


# In[126]:


def tree_to_code(tree, feature_names):
    tree = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.feature
    ]
    print ('def tree({}):'.format(','.join(feature_names)))

    def recurse(node, depth):
        indent = '  ' * depth
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree.threshold[node]
            print ('{}if {} <= {}:'.format(indent, name, threshold))
            recurse(tree.children_left[node], depth + 1)
            print ('{}else:  # if {} > {}'.format(indent, name, threshold))
            recurse(tree.children_right[node], depth + 1)
        else:
            print ('{}return {}'.format(indent, tree.value[node]))

    recurse(0, 1)
    
tree_to_code(tree_model,names)


# In[31]:


b0p.pivot_table(index=['y_pred'], aggfunc='size')


# In[29]:


#analysis of forecast time errors
#buoy0=buoy0.drop(buoy0.loc[buoy0['WVHT']==99].index)
m2=[]
m3=[]
for i in range(0,61):#61 because from there, there are no predictions, which is from ft 7.5
    #WW3
    data2=Hs2[0,:,i]
    df2=pd.DataFrame(data = data2)
    df2.apply(pd.to_numeric)
    times2=nc_N2.variables['time']
    ti2=netCDF4.num2date(times2[:],times2.units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
    ti2=ti2+timedelta(seconds=ft2[i])
    df2=df2.set_index(pd.DatetimeIndex(pd.Series(ti2)))

    #ensemble average of GWES
    dfa=[]    
    for j in range(1,21):
        group=range(l3)
        df=[Hs3[0,j,:,i],group]
        df=np.transpose(np.array(df))
        df=pd.DataFrame(data = df)
        df.apply(pd.to_numeric)
        dfa.append(df)
 
    df3a=pd.concat(dfa)
    df3a=df3a.groupby(df3a.columns[1]).mean()
    df3a=df3a.set_index(pd.DatetimeIndex(pd.Series(ti2)))


    error=['nan']*i   
    n1=['nan']*7
    for j in range(0,len(df2)):
        if len(error) >= len(df2)*8-7:
            break
        else:
            ind = np.where( abs(buoy0.index - df2.index[j]).total_seconds() <= 2400) # vai dar o ind de tempo no array do tempo da boia 
            e=abs(np.mean(buoy0['WVHT'].iloc[ind]) - df2.iloc[j,0])
            x=np.append(e,n1)
            error=np.append(error,x)
    error=error[0:2241]
    m2.append(error)#matrix WW3
    
    error=['nan']*i   
    n1=['nan']*7
    for j in range(0,len(df3a)):
        if len(error) >= len(df3a)*8-7:
            break
        else:
            ind = np.where( abs(buoy0.index - df3a.index[j]).total_seconds() <= 2400) # vai dar o ind de tempo no array do tempo da boia 
            e=abs(np.mean(buoy0['WVHT'].iloc[ind]) - df3a.iloc[j,0])
            x=np.append(e,n1)
            error=np.append(error,x)
    error=error[0:2241]
    m3.append(error)#matrix aGWES
 
m2=np.array(m2)    
m3=np.array(m3)    


# In[15]:


m2=pd.DataFrame(m2)
m2=m2.astype(float)
plt.matshow(m2,fignum=100)
plt.colorbar()
plt.gca().set_aspect('auto')
#plt.savefig('m2spaced.png', dpi=600)
plt.show()


# In[12]:


X,Y = np.meshgrid(np.arange(m2.shape[1]),np.arange(m2.shape[0]))
colors = ["green", "lime",'cyan','blue','darkorange','red','darkred','black']
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
fig, ax = plt.subplots()
m2spaced= ax.scatter(X,Y,c=m2, s=35, linewidth=0, marker="s",cmap=cmap1)
#ax.set_aspect("equal")
ax.margins(0)
plt.colorbar(m2spaced)
plt.savefig('m2spaced2.png', dpi=600)
plt.show()

