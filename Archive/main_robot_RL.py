# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 08:44:09 2022

@author: qbr5kx
"""

import pygame
import math as m
from robot_RL import Graphics,Robot,LaserScan
# from pygame_recorder import ScreenRecorder

map_dims = (600,1200)


#EnvironmentGraphics
gfx = Graphics(map_dims,'rosbot.png','obstacleMap.png')

start = (200,200)
goal = (550,600)
robot = Robot(start,0.01*3779.52,goal)

#sensor
sensor_range = 250,m.radians(90/2)
angle_space = 40
laser = LaserScan(sensor_range,angle_space,gfx.map)

dt = 0
last_time = pygame.time.get_ticks()


running = True
frame = 0

while running:# and frame<10000:
    frame += 1
    action = -1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    dt = (pygame.time.get_ticks()-last_time)/1000
    last_time = pygame.time.get_ticks()
    
    gfx.map.blit(gfx.map_img,(0,0))
    
    #step
    robot.step(action,dt)
    gfx.draw_robot(robot.x,robot.y,robot.heading)
    point_cloud = laser.sense_obstacles(robot.x,robot.y,robot.heading)
    #Reward
    reward, observations, done = robot.get_reward(point_cloud)
    
    #Draw new
    gfx.draw_sensor_data(point_cloud)
    
    pygame.display.update()
    
    
pygame.quit()
