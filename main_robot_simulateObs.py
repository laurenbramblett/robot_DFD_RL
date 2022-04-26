# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 08:44:09 2022

@author: qbr5kx
"""

import pygame
# import csv
import math as m
from robotClass_simulateObs import Graphics,Robot,LaserScan,distance
from draw_background import draw_background
from tensorflow import keras
import joblib
import numpy as np
f = 'grid_files/grid_5.npy'
mode = 'dnn' #Use 'human' for human play, 'force' for Potential field, ...
#'dnn' for predicting using a DNN and anything else for random walk

#Define starting criteria
map_dims = (900,1200)  
goal=(1150,450)
map_matrix = draw_background(f,map_dims)
drawing = True

#EnvironmentGraphics
gfx = Graphics(map_dims,'pictures/rosbot.png','pictures/starPNG.png',map_matrix,drawing,goal)
pygame.init()

#Initalize robot
start = (100,500)
robot = Robot(start,0.01*3779.52,goal)

#Initialize sensor
sensor_range = 250,m.radians(180)
angle_space = m.pi/16
laser = LaserScan(sensor_range,angle_space,map_matrix)

running = True
frame = 0
dt = 0.01
observation_data = []
keys = [False, False, False, False]
if mode == 'dnn':
    model = keras.models.load_model('force_model_ang')
    scaler = joblib.load('min_max_scaler_ang') 
    
while running:# and frame<10000:
    frame += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key==pygame.K_UP:
                keys[0]=True
            elif event.key==pygame.K_LEFT:
                keys[1]=True
            elif event.key==pygame.K_DOWN:
                keys[2]=True
            elif event.key==pygame.K_RIGHT:
                keys[3]=True
        if event.type == pygame.KEYUP:
            if event.key==pygame.K_UP:
                keys[0]=False
            elif event.key==pygame.K_LEFT:
                keys[1]=False
            elif event.key==pygame.K_DOWN:
                keys[2]=False
            elif event.key==pygame.K_RIGHT:
                keys[3]=False

    gfx.draw_map()
    
    point_cloud,ray_points = laser.sense_obstacles(robot.x,robot.y,robot.heading)
    observations = robot.collect_observations(point_cloud)
    if mode == 'human':
        robot.play(keys,dt) #Move robot (user)
        robot.kinematics(dt)
    elif mode == 'force':
        force = observations[-2:]
        angle = m.atan2(force[1],force[0])
        robot.move_forces(angle)
        robot.kinematics(dt)
    elif mode == 'dnn':       
        obs = scaler.transform((np.concatenate((observations,[1]))).reshape(1,-1))
        tmp = np.tile(np.transpose(obs[0][:-2]), (64,1))
        angle = model.predict(tmp)[0,:]
        f_inv = scaler.inverse_transform(np.concatenate((obs[0][:-2],angle,[1])).reshape(1,-1))
        robot.move_forces(f_inv[0,-2])#f_inv[:,-3:-1][0])
        robot.kinematics(dt)
    else:        
        robot.kinematics(dt) #Move robot (randomly)
        robot.avoid_obstacles(point_cloud,dt)
    
    #Draw sensor data & robot
    gfx.draw_robot(robot.x,robot.y,robot.heading)
    gfx.draw_sensor_data(point_cloud)
    gfx.draw_lidar_rays(ray_points)
    
    #Collect data
    
    observation_data.append(robot.collect_observations(point_cloud))
    #observations stored as [distance_readings,self.x,self.y,goal_dist_x,goal_dist_y,force_x,force_y]
    if distance((robot.x,robot.y),goal)<70:
        running = False
    pygame.display.update()
    
    
pygame.quit()
# with open("obsHumanOrRandomWalk.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(observation_data)
# f.close()
