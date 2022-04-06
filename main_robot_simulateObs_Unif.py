# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 08:44:09 2022

@author: qbr5kx
"""

# import pygame
import csv
import math as m
from robotClass_simulateObs import Robot,LaserScan
from draw_background import draw_background
from numpy.random import uniform

grid_num = 0; num_iterations = 1000
for grid_num in range(0,251):
    print(grid_num)
    f = 'grid_files/grid_%d.npy' % grid_num
    
    #Define starting criteria
    map_dims = (900,1200)  
    goal=(1150,450)
    map_matrix = draw_background(f,map_dims)
    drawing = False
    
    
    #Initialize sensor
    sensor_range = 250,m.radians(180)
    angle_space = m.pi/16
    obs = round(sensor_range[1]*2/angle_space)+6 #see below observation_data.append for addtl 6
    laser = LaserScan(sensor_range,angle_space,map_matrix)
    
    frame = 0
    observation_data = []
    
    while frame<num_iterations:
        frame += 1
    
        #Randomly sample starting positions
        start = (uniform(50,850),uniform(50,1150))
        #Initialize robot in starting position
        robot = Robot(start,0.01*3779.52,goal)
        # robot.heading = uniform(-m.pi,m.pi)
        point_cloud,ray_points = laser.sense_obstacles(robot.x,robot.y,robot.heading)
        
        #Collect data
        observations = robot.collect_observations(point_cloud)
        if len(observations) == obs:
            observation_data.append(robot.collect_observations(point_cloud))
        #observations stored as [distance_readings,self.x,self.y,goal_dist_x,goal_dist_y,force_x,force_y]
    
    with open("ObsRecordUniformRandom.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(observation_data)
    f.close()
