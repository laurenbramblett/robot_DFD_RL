# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 21:04:58 2022

@author: qbr5kx
"""

# first neural network with keras tutorial

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

#DNN
filename = os.getcwd() + "\\ObsRecordUniformRandom.csv"
df = pd.read_csv(filename,header=None)
#observations stored as [distance_readings,self.x,self.y,goal_dist_x,goal_dist_y,force_x,force_y]
lidar_list = ["Lidar"+str(x) for x in range(0,32)]
headers = lidar_list + ["xPos","yPos","goalDistX","goalDistY","forceX","forceY"]
df.columns = headers

#Normalize the results
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dfNorm = pd.DataFrame(x_scaled)

X = np.array(dfNorm.iloc[:,0:-2])
y = np.array(dfNorm.iloc[:,-2:])

# define the keras model
model = Sequential()
model.add(Dense(64, input_dim=len(headers)-2, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
# fit the keras model on the dataset
model.fit(X, y, epochs=100, batch_size=64)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.7f' % (accuracy*100))
