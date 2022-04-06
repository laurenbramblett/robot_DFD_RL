# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 08:44:09 2022

@author: qbr5kx
"""

import pygame
import math as m
from robotClass_DFD_v2 import Graphics,Robot,LaserScan
import cv2
import numpy
# from pygame_recorder import ScreenRecorder

map_dims = (600,1200)
map_matrix = cv2.imread('pictures/obstacleMap.png')
map_matrix = numpy.transpose(map_matrix, (1,0,2))    
drawing = True
#EnvironmentGraphics
gfx = Graphics(map_dims,'pictures/rosbot.png',map_matrix,drawing)
pygame.init()

start = (200,200)
robot = Robot(start,0.01*3779.52)

#sensor
sensor_range = 250,m.radians(90/2)
angle_space = 40
laser = LaserScan(sensor_range,angle_space,drawing,map_matrix,gfx.map)

dt = 0
last_time = pygame.time.get_ticks()
surf = pygame.surfarray.make_surface(map_matrix)

running = True
frame = 0

while running:# and frame<10000:
    frame += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    dt = (pygame.time.get_ticks()-last_time)/1000
    last_time = pygame.time.get_ticks()

    gfx.map.blit(surf,(0,0))
    
    robot.kinematics(dt)
    gfx.draw_robot(robot.x,robot.y,robot.heading)
    point_cloud = laser.sense_obstacles(robot.x,robot.y,robot.heading,drawing)
    robot.avoid_obstacles(point_cloud,dt)
    gfx.draw_sensor_data(point_cloud)
    
    pygame.display.update()
    
    
pygame.quit()
