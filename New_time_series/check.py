# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:00:03 2019

@author: vkovvuru
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:14:46 2019

@author: vkovvuru
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:43:28 2019

@author: vkovvuru
"""
import easygui
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt # this is used for the plot the graph 
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error,r2_score
import sys
import io
import base64
sys.setrecursionlimit(3000)
#path='D:/time_series/check_data.txt'

#no_of_step_lags=1
#no_of_steps_ahead=1
#li=[]
a=''
#path=''

def data_loader(path):
    path=path
    data=pd.read_csv(path, sep=';', 
                     parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                     low_memory=False, na_values=['nan','?'], index_col='dt')
    li=list(data.columns)
    #li.append(li)
    return li#,data
#columns,data_frame=data_loader()

def data_frame():
    data=pd.read_csv("check_data.txt",sep=';', 
                     parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                     low_memory=False, na_values=['nan','?'], index_col='dt')
    return data

#li=data_frame().columns

def fun(lis,a):
    tar_var=[]
    for each in lis:
        if each ==a:
          tar_var.append(each)
    target_index=lis.index(''.join(tar_var))
    return target_index   
'''
def build_graph(x_coordinates, y_coordinates):
    img = io.BytesIO()
    plt.plot(x_coordinates, y_coordinates)
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

'''
'''
def step_lags(no_of_step_lags):
    return no_of_step_lags

def steps_ahead(no_of_steps_ahead):
    return no_of_steps_ahead        '''
'''
def train():
    droping_list_all=[]
    for j in range(0,len(data_frame().columns)):
        if not data_frame().iloc[:, j].notnull().all():
            droping_list_all.append(j)        
            #print(df.iloc[:,j].unique())
    droping_list_all
    
    for j in range(0,len(data_frame().columns)):        
            data_frame().iloc[:,j]=data_frame().iloc[:,j].fillna(data_frame().iloc[:,j].mean())
            
    data_resample = data_frame().resample('h').mean() 
    values = data_resample.values 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    def series_to_supervised(data2, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data2) is list else data2.shape[1]
        dff = pd.DataFrame(data2)
        cols, names = list(), list()
        	# input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
        	cols.append(dff.shift(i))
        	names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        	# forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
        	cols.append(dff.shift(-i))
        	if i == 0:
        		names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        	else:
        		names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        	# put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        	# drop rows with NaN values
        if dropnan:
        	agg.dropna(inplace=True)
        return agg 
       	
    reframed = series_to_supervised(scaled,step_lags(no_of_step_lags),steps_ahead(no_of_steps_ahead))
    values = reframed.values    
    n_train_time = round(len(data_resample)*0.5)
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    ##test = values[n_train_time:n_test_time, :]
    # split into input and outputs
    train_X, train_y = train[:,:step_lags(no_of_step_lags)*len(data_frame().columns)], train[:, step_lags(no_of_step_lags)*len(data_frame().columns)+fun(li,a)+(len(data_frame().columns)*(steps_ahead(no_of_steps_ahead)-1))]
    test_X, test_y = test[:, :step_lags(no_of_step_lags)*len(data_frame().columns)], test[:, step_lags(no_of_step_lags)*len(data_frame().columns)+fun(li,a)+(len(data_frame().columns)*(steps_ahead(no_of_steps_ahead)-1))]
        
    train_X = train_X.reshape((train_X.shape[0],step_lags(no_of_step_lags), len(data_frame().columns)))
    test_X = test_X.reshape((test_X.shape[0], step_lags(no_of_step_lags), len(data_frame().columns)))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
    
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    #    model.add(LSTM(70))
    #    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return "model trained"
'''
    
'''    
#history = model.fit(train_X, train_y, epochs=10, batch_size=70, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
#target=li.index(str1)

# make a prediction
yhat = model.predict(test_X)
#plt.plot(yhat,'r')
#plt.plot(test_act,'g')
test_X = test_X.reshape((test_X.shape[0], step_lags*len(data.columns)))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, (1-len(data.columns)):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual

test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, (1-len(data.columns)):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


aa=[x for x in range(100)]
plt.plot(aa, inv_y[:100], marker='.', label="actual")
plt.plot(aa, inv_yhat[:100], 'r',marker='.', label="prediction")
plt.ylabel(tar_var, size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()
'''
