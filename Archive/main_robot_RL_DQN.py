# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 08:44:09 2022

@author: Lauren Bramblett
Instructions:
    Press run to run as is. 
    
"""
import pathlib, os
os.chdir(pathlib.Path(__file__).parent.resolve())
import math as m
from robotClass_DQN_collisionAvoidance import learnEnv,robotEnv

import itertools
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pygame
import cv2
import numpy
from PIL import Image
from matplotlib import pyplot,colors


f = 'grid_files/grid_0.npy'
data = numpy.load(f)
cushion = numpy.transpose(numpy.tile(numpy.concatenate(([1],numpy.repeat(0,data.shape[0]-2),[1])),(10,1)))
data2 = numpy.column_stack((numpy.ones((data.shape[0],1)),cushion,data))
datafig = pyplot.figure(figsize=(30,40))
colormap = colors.ListedColormap(["white","black"])
pyplot.imshow(data2,cmap = colormap)
pyplot.yticks([])
pyplot.xticks([])
pyplot.rcParams['axes.linewidth'] = 20
pyplot.savefig('pictures/matrixFile.png',bbox_inches='tight',pad_inches = 0)
pyplot.close()

map_dims = (1175,1175)
start = (75,500)
goal = (1100,500)
#sensor
sensor_range = 250,m.radians(180/2)
angle_space = m.pi/8
map_matrix = cv2.imread('pictures/matrixFile.png')
map_matrix = numpy.transpose(map_matrix, (1,0,2)) 

env_configs = {
    'dimensions': map_dims,
    'drawing': False,
    'robot_img_path': 'pictures/robot_img.png',
    'map_matrix': map_matrix,
    'sensor_range': sensor_range,
    'goal': goal,
    'start': start,
    'angle_space': angle_space,
    'width': 0.01*3779.52,
    'goal_img_path': 'pictures/starPNG.png',
    'obs_space_len': 10}

# env = robotEnv(env_config = env_configs)

DATA_DIR = './data'


# how often to do performance tests
TEST_THRESHOLD = 500
if __name__ == '__main__':
    env = robotEnv(env_config = env_configs)

    # Grid parameters
    num_learns = [1]
    epochs = [1]
    lookbacks = [30000]
    batch_sizes = [64]

    for num_learn, epoch, lookback, batch_size in itertools.product(num_learns, epochs, lookbacks, batch_sizes):

        # create model from a scratch
        model = Sequential()
        model.add(Dense(256, input_shape=env.observation_space.shape, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(env.action_space.n, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))

        # create agent
        rl = learnEnv(
            model=model,
            env=env,
            data_dir=DATA_DIR,
            name=f'll-128relu-64relu-4linear-nl{num_learn}-epochs{epoch}-lb{lookback}-bs{batch_size}'
        )

        # let's learn
        rewards = rl.train(
            num_episodes=2000,
            num_learn=num_learn,
            epochs=epoch,
            eps=1,
            min_eps=0.1,
            epsilon_decay=0.998,
            verbose=1,
            lookback=lookback,
            batch_size=batch_size,
            gamma=0.99,
        )
        
        
pygame.quit()





