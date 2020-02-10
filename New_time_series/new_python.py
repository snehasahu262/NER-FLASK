# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 18:12:05 2019

@author: vkovvuru
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:00:18 2019

@author: vkovvuru
"""

import os
#import magic
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template,session,send_file
from werkzeug.utils import secure_filename
from check import data_loader,fun,data_frame
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt # this is used for the plot the graph 
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
import io
import plotly
import json
#from plot import do_plot
from sklearn.metrics import mean_squared_error,r2_score
sys.setrecursionlimit(3000)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('page/modal/forecast.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			var_path=app.config['UPLOAD_FOLDER']+'/'+filename
			data_real=data_loader(var_path)
			session['columns']=data_real
			#flash(data_real)
			session['data']=data_real
			keys=list(range(0, len(session['data'])))
			values=session['data']
			jsonList = []
			for i in range(0,len(session['data'])):
			    jsonList.append({"id" : keys[i], "name" : values[i]})
			with open('static/assets/json/data.json', 'w') as outfile: 
			    json.dump(jsonList, outfile,indent=1)
			#flash(data_real)
			flash('File successfully uploaded')	
			return redirect('/')
		else:
			flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif,csv')
			return redirect(request.url)

@app.route('/train_model', methods=['POST'])
def train_model():
    if request.method == 'POST':
        a=request.form['text']
        b=request.form['text1']
        c=request.form['text2']
        target=fun(session['data'],a)
        session['target']=target
        session['steps_back']=int(b)
        session['steps_forward']=int(c) 
    if request.method == 'POST':
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
       	
    reframed = series_to_supervised(scaled,session['steps_back'],session['steps_forward'])
    values = reframed.values    
    n_train_time = round(len(data_resample)*0.5)
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    ##test = values[n_train_time:n_test_time, :]
    # split into input and outputs
    train_X, train_y = train[:,:session['steps_back']*len(data_frame().columns)], train[:, session['steps_back']*len(data_frame().columns)+session['target']+(len(data_frame().columns)*(session['steps_forward']-1))]
    test_X, test_y = test[:, :session['steps_back']*len(data_frame().columns)], test[:, session['steps_back']*len(data_frame().columns)+session['target']+(len(data_frame().columns)*(session['steps_forward']-1))]
        
    train_X = train_X.reshape((train_X.shape[0],session['steps_back'], len(data_frame().columns)))
    test_X = test_X.reshape((test_X.shape[0], session['steps_back'], len(data_frame().columns)))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
    
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    #    model.add(LSTM(70))
    #    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=10, batch_size=70, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    #model.save('timeseries.h5')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
        # make a prediction
    yhat = model.predict(test_X)
    #plt.plot(yhat,'r')
    #plt.plot(test_act,'g')
    test_X = test_X.reshape((test_X.shape[0], session['steps_back']*len(data_frame().columns)))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, (1-len(data_frame().columns)):]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    #predicted=inv_yhat
    inv_yhat=inv_yhat.tolist()
    #session['inv_yhat']=inv_yhat
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, (1-len(data_frame().columns)):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    #actual=inv_y
    inv_y=inv_y.tolist()
    #session['inv_y']=inv_y[:100]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
   # return "model trained'
    aa=[x for x in range(100)]
    #session['aa']=aa
    plt.plot(aa, inv_y[:100], marker='.', label="actual")
    plt.plot(aa, inv_yhat[:100], 'r',marker='.', label="prediction")
    plt.ylabel(a, size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    plt.show()
    '''forecast_data=pd.DataFrame({'actual':inv_y,'predicted':inv_yhat})
    forecast_data=forecast_data.to_json()'''
    jsonList_forecast = []
    for i in range(0,len(inv_yhat)):
        jsonList_forecast.append({"id" :i+1 ,"actual" :inv_y[i] , "predicted" :inv_yhat[i]})
    with open('static/assets/json/forecast_data.json', 'w') as outfile: 
			    json.dump(jsonList_forecast, outfile,indent=1)
  
    flash('Model Trained')
    return redirect('/')


if __name__ == "__main__":
    app.run()
