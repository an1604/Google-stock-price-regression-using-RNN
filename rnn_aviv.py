# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:38:25 2023

@author: adina
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense ,LSTM , Dropout

# data preprocessing 
#importing the dataset
dataset_tarin = pd.read_csv('Google_Stock_Price_Train.csv')
training_set =  dataset_tarin.iloc[: , 1:2].values 

#feature scaling 
sc = MinMaxScaler(feature_range= (0,1))
training_set_scaled = sc.fit_transform(training_set)

#create a data structure wuth some timesteps to look back to 
X_train=[]
y_train=[]

#timestep is 60...
for i in range(60,1258):
    #taking the last 60 prices to X_train
    X_train.append(training_set_scaled[i-60:i,0])
    #taking the price of the next day (i) to y_train in the exact index (i)
    y_train.append(training_set_scaled[i,0])

X_train , y_train = np.array(X_train ) , np.array(y_train )

#reshaping- making reshape to 3D - (btach size , timessteps , indicators ) , respectivly 
X_train = np.reshape(X_train , newshape= (X_train.shape[0] , X_train.shape[1] , 1)) 

#bulding the rnn 
regressor = Sequential()

#adding the first LSTM layer + dropout 
regressor.add(LSTM(units = 50 , return_sequences=True , input_shape =( X_train.shape[1] , 1) ))
# 20% of the neurons will drop out from the network each iteration of the training 
regressor.add(Dropout(rate= 0.2))

# add more 3 layers like the last one 
regressor.add(LSTM(units = 50 , return_sequences=True))
regressor.add(Dropout(rate= 0.2))




regressor.add(LSTM(units = 50 , return_sequences=True))
regressor.add(Dropout(rate= 0.2))

# in the last one we change return_sequences=False beacause its the last layer before the output layer 
regressor.add(LSTM(units = 50 , return_sequences=False))
regressor.add(Dropout(rate= 0.2))


# output layer
regressor.add(Dense(units =1 ))

# compiling the rnn
regressor.compile(optimizer='adam' , loss ='mean_squared_error')

# fit
regressor.fit(x=X_train,y=y_train,batch_size=32 , epochs= 100)

#getting the test set
dataset_test= pd.read_csv('Google_Stock_Price_Test.csv')
test_set =  dataset_test.iloc[: , 1:2].values 

#concatenate the both train and test set into one dataframe, and we just take our indicaor, 
# which is open price...
dataset_total = pd.concat((dataset_tarin['Open'] , dataset_test['Open']),axis=0)

# getting all the prices for year 2017 and shaping in the right format for the rnn
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
#getting the X_test like X_train...
X_test = []
#timestep is 60...
for i in range(60,80):
    #taking the last 60 prices to X_train
    X_test.append(inputs[i-60:i , 0])
    
X_test= np.array(X_test)

#reshaping- making reshape to 3D - (btach size , timessteps , indicators ) , respectivly 
X_test= np.reshape(X_test, newshape= (X_test.shape[0] , X_test.shape[1] , 1)) 

# Predict 
predicted_stock_price = regressor.predict(X_test)

# Returning the stock price to the real price before the scaling
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plotting the predicted stock prices
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()