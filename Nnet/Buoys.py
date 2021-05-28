import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import datetime
import re
import glob
import matplotlib.dates as dates

def createBuoysFrame(path):
    '''
    Create DataFrame with multiindex columns: buoyID and time.
    
    path - string containing a path specification to downloaded NDBC files, 
    where name of files have to contain 'stdmet_*' * being buoy's ID.
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

                f=pd.DataFrame(data={'t':[dic_t.get(x).astype(int) for x in [d.time() for d in datad.index.get_level_values('time')]],
                                'd': [dic_d.get(x).astype(int) for x in [d.date() for d in datad.index.get_level_values('time')]]})

                axs[i,j].scatter(f.t,f.d,s=s,c='black')
                #if range of years is longer
                if len(year)-1 > y:
                    y+=1               
            if i==0:
                axs[i,j].title.set_text(str(years[j]))
            axs[i,j].margins(0)
    fig.text(0.5,0.11,'Time',ha='center',va='center')
    fig.text(0.1,0.5,'Days',ha='center',va='center',rotation='vertical')

