# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 08:44:09 2022

@author: qbr5kx
"""

import pygame
import csv
import math as m
from robotClass_simulateObs import Graphics,Robot,LaserScan
from draw_background import draw_background
f = 'grid_files/grid_0.npy'
mode = 'human' #Use 'human' for human play

#Define starting criteria
map_dims = (900,1200)  
goal=(1150,450)
map_matrix = draw_background(f,map_dims)
drawing = True

#EnvironmentGraphics
gfx = Graphics(map_dims,'pictures/rosbot.png','pictures/starPNG.png',map_matrix,drawing,goal)
pygame.init()

#Initalize robot
start = (200,200)
robot = Robot(start,0.01*3779.52,goal)

#Initialize sensor
sensor_range = 250,m.radians(180/2)
angle_space = m.pi/16
laser = LaserScan(sensor_range,angle_space,map_matrix)

running = True
frame = 0
dt = 0.01
observation_data = []
keys = [False, False, False, False]
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
    if mode == 'human':
        robot.play(keys,dt) #Move robot (user)
        robot.kinematics(dt)
    else:        
        robot.kinematics(dt) #Move robot (randomly)
        robot.avoid_obstacles(point_cloud,dt)
    
    #Draw sensor data & robot
    gfx.draw_robot(robot.x,robot.y,robot.heading)
    gfx.draw_sensor_data(point_cloud)
    gfx.draw_lidar_rays(ray_points)
    
    #Collect data
    observations = robot.collect_observations(point_cloud)
    observation_data.append(robot.collect_observations(point_cloud))
    #observations stored as [distance_readings,self.x,self.y,goal_dist_x,goal_dist_y,force_x,force_y]
    
    pygame.display.update()
    
    
pygame.quit()
with open("obsHumanOrRandomWalk.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(observation_data)
f.close()
