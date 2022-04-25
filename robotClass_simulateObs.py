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
    def __init__(self,startpos,width,goal):
        self.m2p = 3779.52
        self.w = width
        self.x = startpos[0]
        self.y = startpos[1]
        self.heading = 0
        self.kp = 0.5
        self.lin_v = 0.01*self.m2p
        self.ang_v = 0
        self.maxspeed = 0.02*self.m2p
        self.minspeed = 0.01*self.m2p
        
        self.min_obs_dist = 100
        self.count_down = 5
        self.goal = goal
    
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
    def play(self,keys,dt):
        self.lin_v = 0; self.ang_v = 0
        if keys[1]:
            self.ang_v = self.minspeed
        if keys[2]:
            self.lin_v = -self.maxspeed
        if keys[3]:
            self.ang_v = -self.minspeed
        if keys[0]:
            self.lin_v = self.maxspeed  
    def move_forces(self,forces):
        force_x = forces[0]
        force_y = forces[1]
        self.ang_v = m.atan2(force_y,force_x)-self.heading
        
    def kinematics(self,dt):
        self.x += (self.lin_v)*m.cos(self.heading)*dt
        self.y -= (self.lin_v)*m.sin(self.heading)*dt
        self.heading += (self.ang_v)*self.kp*dt
        
        if self.heading>2*m.pi or self.heading<-2*m.pi:
            self.heading = 0
            
        self.lin_v = max(min(self.maxspeed,self.lin_v),self.minspeed)
        # self.vl = max(min(self.maxspeed,self.vl),self.minspeed)
    def check_obstacles(self,point_cloud):
        closest_obs = np.inf
        distance_readings = []
        dist = np.inf
        
        if len(point_cloud)>1:
            for point in point_cloud:
                dist2obs = distance([self.x,self.y],point)
                distance_readings.append(dist2obs) 
                if dist>dist2obs:
                    dist = dist2obs
                    closest_obs = (point,dist)
                
        return closest_obs, distance_readings
    
    def collect_observations(self,point_cloud):
        observations = []
        closest_obs,distance_readings = self.check_obstacles(point_cloud)
        goal_dist = distance([self.x,self.y],self.goal)
        obs_forces_x = []
        obs_forces_y = []
        for idx in range(0,len(distance_readings)):
            if distance_readings[idx]<100:
                obs_forces_x.append((point_cloud[idx][0]-self.x)/distance_readings[idx]**3)
                obs_forces_y.append((-point_cloud[idx][1]+self.y)/distance_readings[idx]**3)
        obs_force_x = (0 if len(obs_forces_x)<1 else sum(obs_forces_x)/len(obs_forces_x))
        obs_force_y = (0 if len(obs_forces_y)<1 else sum(obs_forces_y)/len(obs_forces_y))
        goal_force_x = (self.goal[0]-self.x)/goal_dist**3
        goal_force_y = (-self.goal[1]+self.y)/goal_dist**3
        force_x = -10*obs_force_x+2.5*goal_force_x
        force_y = -10*obs_force_y+2.5*goal_force_y
        if len(point_cloud)>0:
            observations = np.concatenate((distance_readings,
                                           [self.x,self.y,goal_dist,force_x,force_y]))
        return observations
            
class Graphics:
    def __init__(self,dimensions,robot_img_path,star_img_path,map_matrix,drawing,goal):
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
        self.goal = goal
        if drawing:
            self.map = pygame.display.set_mode((self.width,self.height))
            self.surf = pygame.surfarray.make_surface(map_matrix)

            self.map.blit(self.surf,(0,0))
            self.star_img = pygame.image.load(star_img_path).convert()
            self.star_img = pygame.transform.scale(self.star_img, (75, 75))
            self.star_rect = self.star_img.get_rect(center = self.goal)
            #load imgs
            self.robot = pygame.image.load(robot_img_path).convert_alpha()
            self.robot =  pygame.transform.scale(self.robot, (60, 60))
        
    def draw_robot(self,x,y,heading):
       
        rotated = pygame.transform.rotozoom(self.robot,m.degrees(heading),1)
        rect = rotated.get_rect(center = (x,y))
        self.map.blit(rotated,rect)
    def draw_map(self):
        self.map.blit(self.surf,(0,0))
        self.map.blit(self.star_img,self.star_rect)
                
    def draw_sensor_data(self,point_cloud):
        for point in point_cloud:
            pygame.draw.circle(self.map,self.red,point,3,0)

    def draw_lidar_rays(self,ray_points):
        for point in ray_points:
            self.map.set_at(point,(0,208,255))


class LaserScan:
    def __init__(self,sensor_range,angles_sep,map_matrix):
        self.sensor_range = sensor_range
        self.map_width,self.map_height = map_matrix.shape[0:2]
        self.map_matrix = map_matrix
        self.angles_sep = angles_sep
        
    def sense_obstacles(self,x,y,heading):
        obstacles = []; ray_points = []
        x1,y1 = x,y
        start_angle = heading - self.sensor_range[1]
        finish_angle = heading + self.sensor_range[1]
        num_angles = round((finish_angle-start_angle)/self.angles_sep)
        
        for angle in np.linspace(start_angle,finish_angle,num_angles):
            x2 = x1 +self.sensor_range[0]*m.cos(angle)
            y2 = y1 - self.sensor_range[0]*m.sin(angle)
            for i in range(0,100):
                u = i/100
                x = int(x2*u+x1*(1-u))
                y = int(y2*u+y1*(1-u))
                if 0<x<self.map_width and 0<y<self.map_height:
                    color = self.map_matrix[x,y,:]
                    ray_points.append([x,y])
                    if (color[0],color[1],color[2]) == (0,0,0):
                        obstacles.append([x,y])
                        break
                    elif i == 99:
                        obstacles.append([x,y])
                    
                elif x>self.map_width or y>self.map_height:
                    obstacles.append([x,y])
        return obstacles, ray_points
    

        
            
        