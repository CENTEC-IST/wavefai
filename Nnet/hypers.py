#nohup python hypers.py &

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import keras
from keras import models, layers, callbacks 
from datetime import datetime, timedelta
import random
import time
import datetime
import talos 
from talos.model.normalizers import lr_normalizer
from talos.model.early_stopper import early_stopper
import tensorflow as tf

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10, 
                        inter_op_parallelism_threads=10,allow_soft_placement=True)

session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def target(X,buoy):
    '''
    X numerical model with multiindex names: 
        - fctime (seconds)
        - buoy
        - time 
    buoy Hs, U and Tp data with multiindex:
        - buoy 
        - time
    '''
    l=len(X.index)   
    
    errorh=np.empty((0))
    #errort=np.empty((0))
    erroru=np.empty((0))
    for i in range(l):    
        time=datetime.datetime.strptime(X.index.get_level_values('time')[i], '%Y-%m-%d %H:%M:%S') #+timedelta(seconds=fctime)
        ind = np.where(
            np.logical_and(buoy.index.get_level_values('time') >= (time - timedelta(minutes=40)).strftime('%Y-%m-%d %H:%M:%S'), 
            buoy.index.get_level_values('time') <= (time + timedelta(minutes=40)).strftime('%Y-%m-%d %H:%M:%S'))) 

        if len(ind[0])!=0: #if a position was found, append to array of errors abs of difference
            
            errorh = np.append(errorh, X['swh'].values[i] - np.mean(buoy['WVHT'].values[ind])) 
            erroru = np.append(erroru, X['uwnd'].values[i] - np.mean(buoy['uWDIR'].values[ind])) 
        else: #if no position was found
            
            errorh = np.append(errorh, np.nan)
            erroru = np.append(erroru, np.nan)
    y = pd.DataFrame(list(zip(errorh,erroru)),
                    columns =['errorh','erroru'])
    return y

def directionComp( X, directionVars, speedVars):
    '''
    Input: 
        - X - dataframe
        - directionVars - list of direction variables' names  
        - speedVars - list of corresponding speed variables' names, when inexistent = None 
    Output: X with u & v components 
    
    '''  
    if len(directionVars) != len(speedVars):
        error='Variable lists must have same length'
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

def generate_dataset(input_scaler, output_scaler,X,y):

    '''
    scaler can be MinMaxScaler(), StandardScaler() or None
    X - DataFrame with inputs
    '''    
    # split into train and test
    test_size=int(0.2*len(X)) #split can't be shuffled because of wt
    X_train=X[test_size:]
    y_train=y[test_size:]
    X_test=X[:test_size]
    y_test=y[:test_size]

    # scale inputs
    if input_scaler is not None:
        input_scaler.fit(X_train)
        X_train = input_scaler.transform(X_train)
        X_test = input_scaler.transform(X_test)
        
    if output_scaler is not None:
        output_scaler.fit(y_train)
        y_train = output_scaler.transform(y_train)
        y_test = output_scaler.transform(y_test)
    return X_train, X_test, y_train, y_test


#---data-------------------------------------------------------------
buoysdata=pd.read_csv('buoysdata1516.txt', sep=" ", header=0,index_col=[0,1])
reanalysis=pd.read_csv('era5_1516.txt', sep=" ", header=0,index_col=[0,1,2])
reanalysis.rename(index={41004:41044}, level='buoy', inplace=True) #a mistake downloading
buoyslist=np.unique(buoysdata.index.get_level_values('buoy'))

k=2
B=directionComp(buoysdata.iloc[buoysdata.index.get_level_values('buoy')==buoyslist[k]],['WDIR'],['WSPD']).sort_values(by='time')
R=directionComp(reanalysis.iloc[reanalysis.index.get_level_values('buoy')==buoyslist[k]],
                ['mwd','mwd1','mwd2','mwd3','mdww','mdts'],[None,None,None,None,None,None]).sort_values(by='time')
#---julian days-----------------
aux=[0]*len(R)
year=np.unique([datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year for x in R.index.get_level_values('time')])
for i in range(len(R)):
    for j in year:
        if (datetime.datetime.strptime(R.index.get_level_values('time')[i],'%Y-%m-%d %H:%M:%S').year == j):   
            aux[i]=datetime.datetime.strptime(R.index.get_level_values('time')[i],'%Y-%m-%d %H:%M:%S')-datetime.datetime(j,1,1,0,0)
aux=np.array([x.total_seconds() for x in aux],dtype=np.float32)
R['cost']=np.cos(aux*np.pi*2/(np.max(aux)-np.min(aux)))
R['sint']=np.sin(aux*np.pi*2/(np.max(aux)-np.min(aux)))
#---target-----------------------
Y=target(R,B)
index=np.unique(np.where(pd.isnull(Y))[0]) 
Y=Y.drop(index) 
X=R.drop(R.index[index])

y=Y.copy()
window=3
y.iloc[:,0]=y.iloc[:,0].rolling(window, min_periods=1, center=False, win_type=None, on=None, axis=0, closed=None).mean()
y.iloc[:,1]=y.iloc[:,1].rolling(window, min_periods=1, center=False, win_type=None, on=None, axis=0, closed=None).mean()

X_train, X_test, y_train, y_test = generate_dataset(StandardScaler(),MinMaxScaler(),X,y)
#---train split into train/validation--
X_trainv, X_testv,y_trainv, y_testv = train_test_split(X_train,y_train,test_size=0.2,random_state=1)

pca=PCA(.95) #95% of variance explained
pca.fit(X_trainv)
X_trainv=pca.transform(X_trainv)
X_testv=pca.transform(X_testv)
#-----------------------------------------------------------------------------------
#---optimization---

# (1) Define dict of parameters to try
p = {'first_neuron':[5,15,50,100,200,500],
     'activation':['relu','sigmoid','tanh'],
     'hidden_layers':[0,1,2],
     'shapes': ['brick', 'funnel'],
     'optimizer': ['Adam','SGD'],
     'kernel_initializer': ['glorot_uniform','glorot_normal','orthogonal','he_normal'],
     'dropout': [0.0,0.25,0.5],
     'lr':[10**-3, 10**-2, 10**-1],
     'momentum':[0.9,0.5,0.2]}

# (2) create a function which constructs a compiled keras model object
def nnet_model(X_train, y_train, X_val, y_val, params):

    model = models.Sequential()    
    
    # initial layer
    model.add(layers.Dense(params['first_neuron'], input_dim=X_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer = params['kernel_initializer'] ))
    model.add(layers.Dropout(params['dropout']))
    
    # hidden layers
    talos.utils.hidden_layers(model, params, y_train.shape[1])
    
    
    # final layer
    model.add(layers.Dense(y_train.shape[1], 
                    kernel_initializer=params['kernel_initializer']))
    
    if params['optimizer']=="Adam":
        opt=keras.optimizers.Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999)
    if params['optimizer']=="SGD":
        opt=keras.optimizers.SGD(lr=params['lr'], momentum=params['momentum'], nesterov=True)
    
    model.compile(loss='mean_squared_error',optimizer=opt)
    
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        batch_size=32,
                        epochs=1000,
                        verbose=1,
                        callbacks=[early_stopper(epochs=1000, 
                                                mode='strict', 
                                                monitor='val_loss')])
    return history, model


# (3) Run a "Scan" using the params and function created above
start_time = time.time()
t = talos.Scan(x=X_trainv,
            y=y_trainv,
            x_val=X_testv,
            y_val=y_testv,
            model=nnet_model,
            params=p,
            fraction_limit=0.50,
            experiment_name='nnet_opt')
s=(time.time() - start_time)
print("--- %s seconds ---" % s)

