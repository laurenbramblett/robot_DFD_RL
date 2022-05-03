# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:27:33 2022

@author: qbr5kx
"""
import pygame
# import csv
import math as m
from robotClass_testBed import Graphics,Robot,LaserScan,distance
from draw_background import draw_background
from tensorflow import keras
from tensorflow.compat.v1 import disable_eager_execution
import joblib
import numpy as np
disable_eager_execution()

def run_vanilla_sim(model,scaler,world,drawing):
    f = 'grid_files/grid_%d.npy' % world
    #Define starting criteria
    map_dims = (900,1200)  
    goal=(1150,450)
    map_matrix = draw_background(f,map_dims)
    if drawing:
        pygame.init()
    gfx = Graphics(map_dims,'pictures/rosbot.png','pictures/starPNG.png',map_matrix,drawing,goal)
    
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
            
    while running:# and frame<10000:
        frame += 1
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if drawing:
            gfx.draw_map()
        
        point_cloud,ray_points = laser.sense_obstacles(robot.x,robot.y,robot.heading)
        observations = robot.collect_observations(point_cloud)
      
        obs = scaler.transform((np.concatenate((observations,[1]))).reshape(1,-1))
        tmp = np.tile(np.transpose(obs[0][:-2]), (64,1))
        angle = model.predict(tmp)[0,:]
        f_inv = scaler.inverse_transform(np.concatenate((obs[0][:-2],angle,[1])).reshape(1,-1))
        robot.move_forces(f_inv[0,-2])#f_inv[:,-3:-1][0])
        robot.kinematics(dt)
        
        #Draw sensor data & robot
        if drawing:
            gfx.draw_robot(robot.x,robot.y,robot.heading)
            gfx.draw_sensor_data(point_cloud)
            gfx.draw_lidar_rays(ray_points)
            pygame.display.update()
        
        #Collect data
        
        observation_data.append(robot.collect_observations(point_cloud))
        #observations stored as [distance_readings,self.x,self.y,goal_dist_x,goal_dist_y,force_x,force_y]
        if distance((robot.x,robot.y),goal)<70:
            success = 1
            running = False
        elif any(observations[:32]<50):
            success = 0
            running = False
    pygame.quit()
    return frame, success

#Example
model = keras.models.load_model('force_model_ang') 
scaler = joblib.load('min_max_scaler_ang')
frame,success = run_vanilla_sim(model, scaler, 1, True)
print("Num Control Iterations: {}\nSuccess: {}".format(frame,success))