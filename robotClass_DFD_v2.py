# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 07:30:39 2022

@author: qbr5kx
"""
import numpy as np
import math as m
import pygame

def distance(point1,point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1-point2)

class Robot:
    def __init__(self,startpos,width):
        self.m2p = 3779.52
        self.w = width
        self.x = startpos[0]
        self.y = startpos[1]
        self.heading = 0
        
        self.lin_v = 0.01*self.m2p
        self.ang_v = 0
        self.maxspeed = 0.02*self.m2p
        self.minspeed = 0.01*self.m2p
        
        self.min_obs_dist = 100
        self.count_down = 5
    
    def avoid_obstacles(self,point_cloud,dt):
        closest_obs = None
        dist = np.inf
        
        if len(point_cloud)>1:
            for point in point_cloud:
                if dist>distance([self.x,self.y],point):
                    dist = distance([self.x,self.y],point)
                    closest_obs = (point,dist)
                    
            if closest_obs[1]<self.min_obs_dist and self.count_down>0:
                self.count_down -=dt
                self.move_backward()
                
            else:
                self.count_down = 5
                self.move_forward()
    def move_backward(self):
        self.lin_v = 0
        self.ang_v = self.minspeed
        
    def move_forward(self):
        self.lin_v = self.minspeed
        self.ang_v = 0
        
    def kinematics(self,dt):
        self.x += (self.lin_v)*m.cos(self.heading)*dt
        self.y-= (self.lin_v)*m.sin(self.heading)*dt
        self.heading += (self.ang_v)/self.w*dt
        
        if self.heading>2*m.pi or self.heading<-2*m.pi:
            self.heading = 0
            
        self.lin_v = max(min(self.maxspeed,self.lin_v),self.minspeed)
        # self.vl = max(min(self.maxspeed,self.vl),self.minspeed)
        
class Graphics:
    def __init__(self,dimensions, robot_img_path,map_matrix,drawing):
        pygame.init()
        #Colors
        self.black = (0,0,0)
        self.grey = (70,70,70)
        self.blue = (0,0,255)
        self.green = (0,255,0)
        self.red = (255,0,0)
        self.white = (255,255,255)
        
        # --------------------MAP ------------
        #dimensions
        self.height,self.width = dimensions
        #window settings
        pygame.display.set_caption("Obstacle Avoidance")
        if drawing:
            self.map = pygame.display.set_mode((self.width,self.height))
            surf = pygame.surfarray.make_surface(map_matrix)

            self.map.blit(surf,(0,0))
            #load imgs
            self.robot = pygame.image.load(robot_img_path).convert_alpha()
            self.robot =  pygame.transform.scale(self.robot, (60, 60))

        
        

        
    def draw_robot(self,x,y,heading):
       
        rotated = pygame.transform.rotozoom(self.robot,m.degrees(heading),1)
        rect = rotated.get_rect(center = (x,y))
        self.map.blit(rotated,rect)
        
    def draw_sensor_data(self,point_cloud):
        for point in point_cloud:
            pygame.draw.circle(self.map,self.red,point,3,0)


class LaserScan:
    def __init__(self,sensor_range,angle_space,drawing,map = [],map_img = []):
        self.sensor_range = sensor_range
        self.map_width,self.map_height = map.shape[0:2]
        self.map = map
        self.angle_space = angle_space
        if drawing:
            self.map_img = map_img
        
    def sense_obstacles(self,x,y,heading,drawing):
        obstacles = []
        x1,y1 = x,y
        start_angle = heading - self.sensor_range[1]
        finish_angle = heading + self.sensor_range[1]
        for angle in np.linspace(start_angle,finish_angle,self.angle_space,False):
            x2 = x1 +self.sensor_range[0]*m.cos(angle)
            y2 = y1 - self.sensor_range[0]*m.sin(angle)
            for i in range(0,100):
                u = i/100
                x = int(x2*u+x1*(1-u))
                y = int(y2*u+y1*(1-u))
                if 0<x<self.map_width and 0<y<self.map_height:
                    color = self.map[x,y,:]
                    if drawing:
                        self.map_img.set_at((x,y),(0,208,255))
                    if (color[0],color[1],color[2]) == (0,0,0):
                        obstacles.append([x,y])
                        break
                    
        return obstacles
        
            
        