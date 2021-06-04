from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

def series_to_supervised(train,test,outvars,scaler=MinMaxScaler(),n_in=1,n_out=1,dropnan=True):
    '''
    Data reframe for LSTM.
    
    Train/Test - DataFrames (train and test sets) of input variables merged with output variables (if existing) 
    scaler - a scaler function or None
    n_in - number of previous time steps in input
    n_out - number of time steps to predict
    outvars - list of variables' names to predict
    
    '''
    if set(train.columns) != set(test.columns):
        return('Train and Test sets must have same col names')
    if scaler != None:
        scaler.fit(Train)
        train=pd.DataFrame(scaler.transform(train),columns=train.columns)
        test=pd.DataFrame(scaler.transform(test),columns=train.columns)

    #predicting with prior n_in time steps will leave the last n_in steps of training out, which will be moved to test set to predict first outputs
    move=train.iloc[-n_in:,:]    
    names,colsT,colst = list(), list(), list()
    
    #input sequence (t-n_in, ... t-1)
    for i in range(n_in, 0, -1):
        colsT.append(train.shift(i))
        colst.append(test.shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in train.columns]
        
    #forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        colsT.append(train[outvars].shift(-i))
        colst.append(test[outvars].shift(-i))
        if i == 0:
            names += [('%s(t)' % (j)) for j in outvars]
        else:
            names += [('%s(t+%d)' % (j, i)) for j in outvars]
    #put it all together
    aggT = pd.concat(colsT, axis=1)
    aggT.columns = names
    aggt=pd.concat(colst,axis=1)
    aggt.columns=names
    
    #adding the data moved from training to test
    for i in range(0,n_in):
        aggt.iloc[i,0:len(train.columns)]=move.iloc[i,:].values
        if i<n_in-1:
            aggt.iloc[i,len(train.columns):-len(outvars)]=move.iloc[i+1,:].values
            
    #drop rows with NaN values after reframe
    if dropnan:
        aggT.dropna(inplace=True)
        aggt.dropna(inplace=True)
        
    if scaler == None:
        return aggT,aggt
    else:
        return aggT,aggt, scaler