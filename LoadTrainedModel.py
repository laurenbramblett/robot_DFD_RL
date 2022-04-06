# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 08:07:01 2022

@author: qbr5kx
"""
from robotClass_DQN_collisionAvoidance import learnEnv,robotEnv
from tensorflow import keras
from datetime import datetime
import math as m
import cv2, numpy

time = datetime.now()

#Load trained configs
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
    'drawing': True,
    'robot_img_path': 'pictures/robot_img.png',
    'map_matrix': map_matrix,
    'sensor_range': sensor_range,
    'goal': goal,
    'start': start,
    'angle_space': angle_space,
    'width': 0.01*3779.52,
    'goal_img_path': 'pictures/starPNG.png',
    'obs_space_len': 10}

#----------LOAD SAVED MODEL-------------
model = keras.models.load_model('data/ll-128relu-64relu-4linear-nl1-epochs1-lb30000-bs64-model-1010.h5')
print("Loaded model from disk")
#Load environment    
env = robotEnv(env_config = env_configs)
DATA_DIR = 'data'
# evaluate loaded model on test data
model.compile(loss='mean_squared_error', optimizer='Adam')
rl = learnEnv(
    model=model,
    env=env,
    data_dir=DATA_DIR,
    name=f'test-{time}'
)
rewards, last_rewards = rl.train(num_episodes=10, eps=0, min_eps=0,
                                   epsilon_decay=0, verbose=0, num_learn=0)
