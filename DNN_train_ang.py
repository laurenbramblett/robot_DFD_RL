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
import joblib

#DNN
filename = os.getcwd() + "\\ObsRecordUniformRandom_v5.csv"
df = pd.read_csv(filename,header=None)
#observations stored as [distance_readings,goal_dist,goal_ang,force_x,force_y]
lidar_list = ["Lidar"+str(x) for x in range(0,32)]
headers = lidar_list + ["goalDist","goalAng","forceAng","world"]
df.columns = headers

#Normalize the results
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dfNorm = pd.DataFrame(x_scaled)
dfNorm.columns = headers

X = np.array(dfNorm.iloc[:,0:-2])
y = np.array(dfNorm.iloc[:,-2:-1])

# define the keras model
model = Sequential()
model.add(Dense(64, input_dim=len(headers)-2, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='linear'))

# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
# fit the keras model on the dataset
model.fit(X, y, epochs=100, batch_size=64)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.7f' % (accuracy*100))
model.save('force_model_ang')
joblib.dump(min_max_scaler, 'min_max_scaler_ang')