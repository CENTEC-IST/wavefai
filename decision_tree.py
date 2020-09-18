#!/usr/bin/env python
# coding: utf-8

# In[1]:


import netCDF4
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import time
np.warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from io import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from collections import OrderedDict 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import _tree
import matplotlib.pyplot as plt


WW3_N= 'WW3_selectionPointNearest_20190925_20200701.nc'
nc_N2=Dataset(WW3_N,'r')
GWES_N= 'GWES_selectionPointNearest_20190925_20200701.nc'
nc_N3=Dataset(GWES_N,'r')

#deterministic
lat2=nc_N2['latitude'][:]
lon2=nc_N2['longitude'][:]
ft2=nc_N2['fctime'][:]
t2=nc_N2['time'][:]
date2=nc_N2['datetime'][:]
ID2=nc_N2['buoyID'][:]

Dp2=nc_N2['Dp'][:]
Hs2=nc_N2['Hs'][:]
Tp2=nc_N2['Tp'][:]
Ds12=nc_N2['Dswell1'][:]
Ds22=nc_N2['Dswell2'][:]
Hs12=nc_N2['Hswell1'][:]
Hs22=nc_N2['Hswell2'][:]
Ts12=nc_N2['Tswell1'][:]
Ts22=nc_N2['Tswell2'][:]
Dw2=nc_N2['Dwsea'][:]
Hw2=nc_N2['Hwsea'][:]
Tw2=nc_N2['Twsea'][:]
l2=len(t2)
out2 = ['WW3_N'] * l2 

#ensemble
lat3=nc_N3['latitude'][:]
lon3=nc_N3['longitude'][:]
ft3=nc_N3['fctime'][:]
t3=nc_N3['time'][:]
date3=nc_N3['datetime'][:]
ID3=nc_N3['buoyID'][:]
ens3=nc_N3['nensembles'][:]

Dp3=nc_N3['Dp'][:]#direciton wave
Hs3=nc_N3['Hs'][:]#wave height
Tp3=nc_N3['Tp'][:]#peak wave period
Ds13=nc_N3['Dswell1'][:]#direction of swell waves (???) part 1 and 2
Ds23=nc_N3['Dswell2'][:]
Hs13=nc_N3['Hswell1'][:]#height ...
Hs23=nc_N3['Hswell2'][:]
Ts13=nc_N3['Tswell1'][:]#period ...
Ts23=nc_N3['Tswell2'][:]
Dw3=nc_N3['Dwsea'][:]#wind sea 
Hw3=nc_N3['Hwsea'][:]
Tw3=nc_N3['Twsea'][:]
l3=len(t3)
out3 = ['GWES_N'] * l3 

#BUOY 0
#nowcast ft 0
#dataframe of ww3
data2=[Dp2[0,:,0],Hs2[0,:,0],Tp2[0,:,0],Ds12[0,:,0],Ds22[0,:,0],Hs12[0,:,0],Hs22[0,:,0],Ts12[0,:,0],Ts22[0,:,0],Dw2[0,:,0],Hw2[0,:,0]
,Tw2[0,:,0]]
data2=np.transpose(np.array(data2))
df2=pd.DataFrame(data = data2)
df2.apply(pd.to_numeric)
df2['out']=out2
times2=nc_N2.variables['time']
ti2=netCDF4.num2date(times2[:],times2.units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
df2=df2.set_index(pd.DatetimeIndex(pd.Series(ti2)))
#extract rows with nan values if there's any
nan=np.where(np.isnan(df2.iloc[:,0:12]))
ind = list(OrderedDict.fromkeys(nan[0])) 
df2=df2.drop(df2.index[ind])
#dataframe of ensemble average of GWES
dfa=[]    
for i in range(1,21):
    group=range(l3)
    df=[Dp3[0,i,:,0],Hs3[0,i,:,0],Tp3[0,i,:,0],Ds13[0,i,:,0],Ds23[0,i,:,0],Hs13[0,i,:,0],Hs23[0,i,:,0],Ts13[0,i,:,0],Ts23[0,i,:,0],Dw3[0,i,:,0],Hw3[0,i,:,0]
,Tw3[0,i,:,0],group]
    d=np.transpose(np.array(df))
    df=pd.DataFrame(data = d)
    df.apply(pd.to_numeric)
    dfa.append(df)
 
df3a=pd.concat(dfa)
df3a=df3a.groupby(df3a.columns[12]).mean()
df3a['out']= out3
times3=nc_N3.variables['time']
ti3=netCDF4.num2date(times3[:],times3.units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
df3a=df3a.set_index(pd.DatetimeIndex(pd.Series(ti3)))
#extract rows with nan values if there's any
nan=np.where(np.isnan(df3a.iloc[:,0:12]))
ind = list(OrderedDict.fromkeys(nan[0])) 
df3a=df3a.drop(df3a.index[ind])

#merge 2 dataframes
result = df2.append(df3a)
#result = result.reset_index(drop=True)
df2df3a=pd.DataFrame(result)
names=['Dp','Hs','Tp','Ds1','Ds2','Hs1','Hs2','Ts1','Ts2','Dw','Hw','Tw','out']
df2df3a.columns=names
df2df3a=df2df3a.round(decimals=3)  
names.pop() #for decision tree with feature names only

buoy0 = pd.read_table('NDBC_historical_stdmet_41047.txt', sep='\s+')
val=buoy0.iloc[0,:]
buoy0=buoy0[~buoy0.iloc[:,0].str.contains("#")]
buoy0=buoy0.drop_duplicates()
buoy0=buoy0.apply(pd.to_numeric)
buoy0=buoy0.reset_index(drop=True)
date = buoy0[["#YY", "MM", "DD",'hh','mm']].copy()
date.columns = ["year", "month", "day",'hour','minute']
buoy0=buoy0.set_index(pd.to_datetime(date))
buoy0=buoy0.drop(["#YY", "MM", "DD",'hh','mm'],axis=1)
l=len(buoy0)
#buoy0.reset_index().plot.scatter(x='index',y='WDIR')


# In[10]:


#errors for wave
buoy0=buoy0.drop(buoy0.loc[buoy0['WVHT']==99].index)
error=[]
for i in range(0,len(df2df3a)):
    ind = np.where( abs(buoy0.index - df2df3a.index[i]).total_seconds() <= 2400) # vai dar o ind de tempo no array do tempo da boia 
    error.append(abs(np.mean(buoy0['WVHT'].iloc[ind]) - df2df3a['Hs'][i]))
 


# In[11]:


errorHs=df2df3a.iloc[:,1]
errorHs=pd.DataFrame(errorHs)
errorHs['errorHS']=error
errorHs['out']=df2df3a.iloc[:,12]


# In[12]:


nan=np.where(np.isnan(errorHs.iloc[:,1]))
nan


# In[13]:


errorHs


# In[84]:


np.where( abs(buoy0.index - df2df3a.index[181]).total_seconds() <= 2400) 


# In[29]:


plote=errorHs[[errorHs.columns[1],errorHs.columns[2]]]
plote=plote.reset_index()

w=plote.loc[plote['out'] == 'WW3_N']
plt.scatter(w.iloc[:,0],w.iloc[:,1],c='green',s=10,label='WW3_N')
g=plote.loc[plote['out'] == 'GWES_N']
plt.scatter(g.iloc[:,0],g.iloc[:,1],c='blue',s=10,label='GWES_N')
plt.legend()
plt.xlabel('dates')
plt.ylabel('errorHS')
     
plt.show()


# In[37]:


print(np.mean(w.iloc[:,1]),np.mean(g.iloc[:,1]))


# In[107]:


#Decision tree dataset all features
X=df2df3a.iloc[:,0:12]
y=df2df3a.iloc[:,12]
 #features

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
#print (clf.best_score_, clf.best_params_) 


# In[2]:


#focusing on wave height:

X=df2df3a.iloc[:,1]
X=X.values.reshape(-1,1)#because it's 1D
y=df2df3a.iloc[:,12]


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


#y_pred=tree_model.predict(buoy0)


# In[14]:


#decisiontree of hs and error associated 

nan=np.where(np.isnan(errorHs.iloc[:,1]))
ind = list(OrderedDict.fromkeys(nan[0])) 
errorHs=errorHs.drop(errorHs.index[ind])

X=errorHs[['Hs','errorHS']]


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
#print (clf.best_score_, clf.best_params_) 


# In[ ]:


b0p=buoy0['WVHT']
b0p=b0p.drop(b0p.loc[b0p==99].index)
b0p = pd.DataFrame(b0p)
b0p.columns=['Hs']
b0p['error']=np.zeros(len(b0p))#errors are all zero
y_pred=tree_model.predict(b0p)
b0p['y_pred']= y_pred
b0p=b0p.reset_index(drop=True)
b0p=b0p.reset_index()

b0pw=b0p.loc[b0p['y_pred'] == 'WW3_N']
plt.scatter(b0pw.iloc[:,0],b0pw.iloc[:,1],c='green',s=10,label='WW3_N')
b0pg=b0p.loc[b0p['y_pred'] == 'GWES_N']
plt.scatter(b0pg.iloc[:,0],b0pg.iloc[:,1],c='blue',s=10,label='GWES_N')
plt.legend()
plt.xlabel('index')
plt.ylabel('Hs')
plt.show()


# In[ ]:


X=df2df3a.iloc[:,1]
X=X.values.reshape(-1,1)#because it's 1D
y=df2df3a.iloc[:,12]


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


# In[38]:


#plot decision tree
dot_data = StringIO()
export_graphviz(tree_model, out_file=dot_data, feature_names =['Hs', 'errorHS'],# [names[1]],#focusing on Hs
                            filled=True,class_names = clf.classes_, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[47]:


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

samples = b0p[['Hs','error']]
decision_paths = tree_model.decision_path(samples)

for decision_path in decision_paths:
    for n, node_value in enumerate(decision_path.toarray()[0]):
        if node_value == 0:
            continue
        node = graph.get_node(str(n))[0]            
        node.set_fillcolor('green')
        labels = node.get_attributes()['label'].split('<br/>')
        for i, label in enumerate(labels):
            if label.startswith('samples = '):
                labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

        node.set('label', '<br/>'.join(labels))
        
Image(graph.create_png())


# In[98]:


clf.best_params_


# In[48]:





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


# In[51]:



plotdf=df2df3a[[df2df3a.columns[1],df2df3a.columns[12]]]
plotdf=plotdf.reset_index()

w=plotdf.loc[plotdf['out'] == 'WW3_N']
plt.scatter(w.iloc[:,0],w.iloc[:,1],c='green',s=10,label='WW3_N')
g=plotdf.loc[plotdf['out'] == 'GWES_N']
plt.scatter(g.iloc[:,0],g.iloc[:,1],c='blue',s=10,label='GWES_N')
plt.legend()
plt.xlabel('dates')
plt.ylabel('Hs')
plt.show()


# In[44]:


#plot of chosen model for each hs value of buoy if decision tree with just hs
#delete outliers 
b0p=buoy0['WVHT']
b0p=b0p.drop(b0p.loc[b0p==99].index)
b0p = pd.DataFrame(b0p)
b0p.columns=['Hs']
y_pred=tree_model.predict(b0p)
b0p['y_pred']= y_pred
b0p=b0p.reset_index(drop=True)
b0p=b0p.reset_index()

b0pw=b0p.loc[b0p['y_pred'] == 'WW3_N']
plt.scatter(b0pw.iloc[:,0],b0pw.iloc[:,1],c='green',s=10,label='WW3_N')
b0pg=b0p.loc[b0p['y_pred'] == 'GWES_N']
plt.scatter(b0pg.iloc[:,0],b0pg.iloc[:,1],c='blue',s=10,label='GWES_N')
plt.legend()
plt.xlabel('index')
plt.ylabel('Hs')
plt.show()


# In[45]:


b0p.pivot_table(index=['y_pred'], aggfunc='size')


# In[28]:


b0p.pivot_table(index=['y_pred'], aggfunc='size')


# In[ ]:




