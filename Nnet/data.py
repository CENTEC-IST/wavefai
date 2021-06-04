import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import datetime
import re
import glob
import matplotlib.dates as dates

def BuoysFrame(path):
    '''
    Create DataFrame from txt file with multiindex columns: buoyID and time.
    
    path - string containing a path specification with all the downloaded NDBC files, 
    where name of files have to contain 'stdmet_*', * being buoy's ID.
    Example of file name: NDBC_historical_stdmet_41044_2020.txt (ID 41044).
    '''
    buoys=[]
    for file in glob.glob(path):
        f=pd.read_table(file,delim_whitespace=True,dtype='unicode')
        #search buoyID in name
        b = re.search(r'stdmet_(\d+)',file)
        if b is not None:
            b=b.group(1)
        else:
            print("%s is not a valid file -- not included"%(file))
            continue            
        f['buoy']=b
        f=f[~f['#YY'].str.contains("#",na=False)]
        f['time']=f.apply(lambda x:datetime.datetime.strptime
                          ("{0}-{1}-{2} {3}:{4}".format(x['#YY'],x['MM'], x['DD'], x['hh'], x['mm']), "%Y-%m-%d %H:%M"),axis=1)
        f=f.drop(f.columns[0:5],1)
        buoys.append(f)
    buoys=pd.concat(buoys)
    buoys=buoys.set_index(['buoy','time']).astype(float)    
    return(buoys)

def Plot_checkTimes(dataframe,years,figsize=(40,40),fontsize=20,s=3):
    '''
    Plots of buoys' data per year. Each row a buoy. Each column a year.
    
    dataframe - DataFrame with multiindex of buoyID and times in datetime with names buoy and time, respectively.
    years - array of years to include in ascending order.
    figsize, fontsize and s are parameters of matplotlib (scatter plot).
    '''
    b=dataframe.index.get_level_values('buoy').unique()
    print('Rows:',list(b))
    cols=years[-1]-years[0]+1
    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(len(b),cols,figsize=figsize)    
    for i in range(len(b)):
        #row-buoy
        data=dataframe.iloc[dataframe.index.get_level_values('buoy') == b[i]]
        year=np.unique([d.year for d in data.index.get_level_values('time')])
        year=[x for x in year if x >= years[0]]#if years in dataframe start earlier than the input years, drop them
        y=0
        for j in range(cols): #column-year
            if year[y]==years[j]: #in case there is no datapoints in some year

                datad=data[data.index.get_level_values('time').year == year[y]]     
                dic_t={}
                dic_d={}

                #x axis is time of day 
                #if a specific time is missing in all days, x range will shorten
                values=np.unique([d.time() for d in datad.index.get_level_values('time')])
                keys=np.arange(0,len(values))#if time is hourly, keys' range is from 0 to 23
                for w in keys:
                    dic_t[w]=values[w]                
                dic_t = dict((v,k) for k,v in dic_t.items())

                #y axis is day of year
                #if a whole day is missing, y range will shorten
                values=np.unique([d.date() for d in datad.index.get_level_values('time')])
                keys=np.arange(1,len(values)+1)#if all days of year are present, key's range is from 1 to 365(366)            
                for w in keys:
                    dic_d[w]=values[w-1]
                dic_d = dict((v,k) for k,v in dic_d.items())

                f=pd.DataFrame(data={'t':[dic_t.get(x).astype(int) for x in [d.time() for d in datad.index.get_level_values('time')]],'d': [dic_d.get(x).astype(int) for x in [d.date() for d in datad.index.get_level_values('time')]]})

                axs[i,j].scatter(f.t,f.d,s=s,c='black')
                #if range of years is longer
                if len(year)-1 > y:
                    y+=1               
            if i==0:
                axs[i,j].title.set_text(str(years[j]))
            axs[i,j].margins(0)
    fig.text(0.5,0.11,'Time',ha='center',va='center')
    fig.text(0.1,0.5,'Days',ha='center',va='center',rotation='vertical')
    
    
def ERA5frame(path):
    '''
    Create DataFrame from txt file with multiindex columns: buoyID and time.
    
    path - string containing a path specification with all the downloaded ERA5 files, 
    where name of files have to contain 'ERA5_*.txt', * being buoy's ID.
    '''
    reanalysis=[]
    for file in glob.glob(path):
        f=pd.read_table(file,delim_whitespace=True,header=1) 
        #search buoyID in name
        b=re.search(r'ERA5_(\d+).txt',file)
        if b is not None:
            b=b.group(1)
        else:
            print("%s is not a valid file -- not included"%(file))
            continue  
        f['buoy']=b
        f=f.set_index(['buoy'])
        reanalysis.append(f)
    reanalysis=pd.concat(reanalysis)
    #the columns' names were shifted by 1 because of string %
    columns=list(reanalysis.columns)
    #----shift back----
    columns.pop(0)
    reanalysis=reanalysis.drop(reanalysis.columns[-1],axis=1)
    reanalysis.columns=columns
    #--------------------
    reanalysis['time']=reanalysis.apply(lambda x:datetime.datetime.strptime("{0}-{1}-{2} {3}".format
                    (x['YEAR'].astype(int),x['MONTH'].astype(int), x['DAY'].astype(int), x['HOUR'].astype(int)), "%Y-%m-%d %H"),axis=1)
    reanalysis=reanalysis.drop(reanalysis.columns[0:4],1)
    reanalysis=reanalysis.set_index(['time'],append=True)
    
    #add julian days
    years=np.unique(reanalysis.index.get_level_values('time').year)
    aux=[0]*len(reanalysis)
    for i in range(len(reanalysis)):
        for j in years:
            if (reanalysis.index.get_level_values('time')[i].year == j):   
                #subtract each time with january 1st at 00h of the same year
                aux[i]=reanalysis.index.get_level_values('time')[i]-datetime.datetime(j,1,1,0,0)
                break

    aux=np.array([x.total_seconds() for x in aux],dtype=np.float32)
    reanalysis['cost']=np.cos(aux*np.pi*2/(np.max(aux)-np.min(aux)))
    reanalysis['sint']=np.sin(aux*np.pi*2/(np.max(aux)-np.min(aux)))
    reanalysis=reanalysis.astype(float)

    return(reanalysis)

def directionComp( X, directionVars, speedVars):
    '''
    Input: 
        - X - dataframe
        - directionVars - list of direction variables' names  
        - speedVars - list of corresponding speed variables' names. When inexistent = None for each entrance of list
    Output: X with u & v components     
    '''  
    if len(directionVars) != len(speedVars):
        error='List of variables do not match in size.'
        return error
    
    for i in range(0,len(directionVars)):
        if any(var == directionVars[i] for var in X.columns): #confirm if that variable is in X
            X=X.dropna(subset=[directionVars[i]]) #remove nans from dataframe            
            if speedVars[i] != None:
                X=X.dropna(subset=[speedVars[i]]) #remove nans from dataframe
                d=X[directionVars[i]].values
                s=X[speedVars[i]].values
                
                u=-s*np.sin(np.pi*d/180)
                v=-s*np.cos(np.pi*d/180)  
                
                del X[directionVars[i]]
                del X[speedVars[i]]
                X['u%s'%(directionVars[i])]=u
                X['v%s'%(directionVars[i])]=v
            else:
                d=X[directionVars[i]].values
                
                u=np.sin(np.pi*d/180)
                v=np.cos(np.pi*d/180)   
                
                del X[directionVars[i]]
                X['u%s'%(directionVars[i])]=u
                X['v%s'%(directionVars[i])]=v
        else: 
            print('Direction variable %s is not in X'%(directionVars[i]))    
    return X

def target(X,Buoy,ex_vars,eb_vars,fctime=0):
    '''
    Create the output variables -- error of predictions (model-real) -- for one or more locations (buoys).
    
    X (numerical model) with multiindex names: 
        - buoy 
        - time 

    buoys data with multiindex:
        - buoy 
        - time
    
    ex_vars - list of the variables' names in X wanted for the error prediction.
    eb_vars - list of the variables' names in Buoy wanted for the error prediction, ordered as ex_vars.
    fctime - int forecast time.
    
    '''
    if len(eb_vars)!= len(ex_vars):
        return ('List of variables do not match in size.')    
    #list of error arrays
    elist=[[]]*len(ex_vars) 
    T=[]
    B=[]
    b=np.unique(X.index.get_level_values('buoy'))  
    for i in b:
        model=X.iloc[np.where(X.index.get_level_values('buoy') == i) ]
        real=Buoy.iloc[np.where(Buoy.index.get_level_values('buoy') == i) ]
        l=len(model.index)
        for j in range(l):    
            time=pd.Timestamp(model.index.get_level_values('time')[j])+timedelta(seconds=fctime)
            ind=np.where(np.logical_and(real.index.get_level_values('time') >= (time - timedelta(minutes=40)).strftime('%Y-%m-%d %H:%M:%S'), real.index.get_level_values('time') <= (time + timedelta(minutes=40)).strftime('%Y-%m-%d %H:%M:%S')))

            if len(ind[0])!=0: #if a position was found, append to array of errors abs of difference
                for w in range(len(ex_vars)):                    
                    elist[w] = np.append(elist[w], model[ex_vars[w]].values[j] - np.mean(real[eb_vars[w]].values[ind])) 

            else: #if no position was found
                for w in range(len(ex_vars)):
                    elist[w] = np.append(elist[w],np.nan)
           
        B.extend(model.index.get_level_values('buoy'))
        T.extend(model.index.get_level_values('time'))
        
    y=pd.DataFrame(elist).transpose()
    y.columns=['e_%s' % i for i in ex_vars]
    y['time']=T
    y['buoy']=B
    y.set_index(['buoy','time'],inplace=True)

    return y

def split_years(X,y,yearsTrain,yearsTest):
    '''
    X - dataframe of inputs.
    y - corresponding outputs.
    yearsTrain/yearsTest - list of years of data selected for training set/test set. 
    '''
    X_train=X.loc[np.isin(pd.DatetimeIndex(X.index.get_level_values('time')).year.astype(int),list(map(int, yearsTrain)))]
    y_train=y.loc[np.isin(pd.DatetimeIndex(y.index.get_level_values('time')).year.astype(int),list(map(int, yearsTrain)))]

    X_test=X.loc[np.isin(pd.DatetimeIndex(X.index.get_level_values('time')).year.astype(int),list(map(int, yearsTest)))]
    y_test=y.loc[np.isin(pd.DatetimeIndex(y.index.get_level_values('time')).year.astype(int),list(map(int, yearsTest)))]
    return (X_train, X_test, y_train, y_test)


def scaled_dataset(input_scaler, output_scaler,X_train,X_test,y_train,y_test):
    
    if input_scaler is not None:
        input_scaler.fit(X_train)
        X_train = input_scaler.transform(X_train)
        X_test = input_scaler.transform(X_test)
        
    if output_scaler is not None:
        output_scaler.fit(y_train)
        y_train = output_scaler.transform(y_train)
        y_test = output_scaler.transform(y_test)
        
    return X_train, X_test, y_train, y_test, output_scaler


def e_metrics(model,obs,vmin=-np.inf,vmax=np.inf,maxdiff=np.inf):
    '''
    Error Metrics. Mentaschi et al. (2013)
    Input: two arrays of model and observation, respectively.
        They must have the same size
    Output: ferr array with shape equal to 8
        bias, RMSE, NBias, NRMSE, SCrmse, SI, HH, CC
    '''
    if model.shape != obs.shape:
        return(' Model and Observations with different shape.')
    if vmax<=vmin:
        return(' vmin cannot be higher than vmax.')

    ind=np.where((np.isnan(model)==False) & (np.isnan(obs)==False) & (model>vmin) & (model<vmax) & (obs>vmin) & (obs<vmax) & (np.abs(model-obs)<=maxdiff) )
    
    if np.any(ind) or model.shape[0]==1:
        model=np.copy(model[ind[0]]); obs=np.copy(obs[ind[0]])
    else:
        return(' Array without valid numbers.')

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






