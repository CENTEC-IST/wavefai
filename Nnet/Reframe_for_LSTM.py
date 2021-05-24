def series_to_supervised(data,threshold=0.8,scaler=MinMaxScaler(), n_in=1, n_out=1, outvars=['errorh','erroru'],dropnan=True):
    '''
    Data reframe for LSTM.
    
    data - matrix of input variables merged with output variables (if existing)
    threshold - percentage of data for training set
    n_in - number of previous time steps in input
    n_out - number of time steps to predict
    outvars - list of the names of variables to predict
    
    '''
    df = pd.DataFrame(data)
    train_size=int(threshold*len(data))
    train=data[:train_size]
    test=data[train_size:]
    
    #avoid leakage by fitting scaler only on training
    scale=scaler.fit(train)
    train=pd.DataFrame(scale.transform(train),columns=df.columns)
    test=pd.DataFrame(scale.transform(test),columns=df.columns)
    
    #predicting with prior n_in time steps will leave the last n_in steps of training out
    #that will be moved to test set to predict first output
    move=train.iloc[-n_in:,:] 
    
    names,colsT,colst = list(), list(), list()
    
    #input sequence (t-n_in, ... t-1)
    for i in range(n_in, 0, -1):
        colsT.append(train.shift(i))
        colst.append(test.shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in df.columns]
        
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
        aggt.iloc[i,0:len(df.columns)]=move.iloc[i,:].values
        if i<n_in-1:
            aggt.iloc[i,len(df.columns):-len(outvars)]=move.iloc[i+1,:].values
            
    #drop rows with NaN values after reframe
    if dropnan:
        aggT.dropna(inplace=True)
        aggt.dropna(inplace=True)
        
    return aggT,aggt,scale

