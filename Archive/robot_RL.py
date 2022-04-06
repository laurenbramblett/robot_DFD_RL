# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 07:30:39 2022

@author: qbr5kx
"""
import numpy as np
import math as m
import gym
from gym import error, spaces
import pygame
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import random
from collections import deque

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
        
        self.lin_v = 0.01*self.m2p
        self.ang_v = 0
        self.maxspeed = 0.02*self.m2p
        self.minspeed = 0.01*self.m2p
        
        self.min_obs_dist = 100
        self.count_down = 5
        
        self.goal = goal
    
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
    
    def get_reward(self,point_cloud):
        #Reward
        # Set the reward.
        # Car crashed when any reading == 1
        done = False
        closest_obs,distance_readings = self.check_obstacles(point_cloud)
        goal_dist = distance([self.x,self.y],self.goal)
        if closest_obs[1]<=60:
            reward = 0
            reward -= goal_dist
            done = True
        elif distance([self.x,self.y],self.goal)<60:
            reward = 50000
            done = True
        else:
            #close_obs = [i for i in distance_readings if i <= 200]
            reward = -1 #- int(np.sum(close_obs) / 100)
            #reward -= distance([self.x,self.y],self.goal)/20
        obs_norm = np.interp(distance_readings, [0, 250], [-1, 1])
        vel = np.interp(self.lin_v,[self.minspeed, self.maxspeed],[-1,1])
        ang = np.interp(self.heading,[-m.pi,m.pi],[-1,1])
        goal_angle = m.atan2(self.goal[1]-self.y, self.goal[0]-self.x)
        goal_dist = np.interp(goal_dist,[0,1200],[-1,1])
       
        states = np.append(obs_norm,[vel,ang,goal_angle,goal_dist])
        
            
        return states, reward, done
    
    def move(self,action,dt,point_cloud):
        self.ang_v = self.minspeed*action
        
    # def kinematics(self,dt):
        self.x += (self.lin_v)*m.cos(self.heading)*dt
        self.y-= (self.lin_v)*m.sin(self.heading)*dt
        self.heading += (self.ang_v)/self.w*dt
        
        if self.heading>m.pi:
            self.heading-=2*m.pi
        elif self.heading<-m.pi:
            self.heading += 2*m.pi
            
        self.lin_v = max(min(self.maxspeed,self.lin_v),self.minspeed)
        # self.vl = max(min(self.maxspeed,self.vl),self.minspeed)
        states, reward, done = self.get_reward(point_cloud)
        
        return states,reward,done
        
class Graphics:
    def __init__(self,dimensions, robot_img_path,map_img_path,goal_img_path,goal):
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
        self.map = pygame.display.set_mode((self.width,self.height))
        
        
        #load imgs
        self.robot = pygame.image.load(robot_img_path).convert_alpha()
        self.map_img = pygame.image.load(map_img_path).convert()
        self.star_img = pygame.image.load(goal_img_path).convert()
        self.star_img = pygame.transform.scale(self.star_img, (75, 75))
        self.star_rect = self.star_img.get_rect(center = goal)
        self.map.blit(self.map_img,(0,0))
        self.map.blit(self.star_img,self.star_rect)
        
        

        
    def draw_robot(self,x,y,heading):
        rotated = pygame.transform.rotozoom(self.robot,m.degrees(heading),1)
        rect = rotated.get_rect(center = (x,y))
        self.map.blit(rotated,rect)
        
    def draw_sensor_data(self,point_cloud):
        for point in point_cloud:
            pygame.draw.circle(self.map,self.red,point,3,0)
        


class LaserScan:
    def __init__(self,sensor_range,angle_space,map):
        self.sensor_range = sensor_range
        self.map_width,self.map_height = pygame.display.get_surface().get_size()
        self.map = map
        self.angle_space = angle_space
        
    def sense_obstacles(self,x,y,heading):
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
                    color = self.map.get_at((x,y))
                    self.map.set_at((x,y),(0,208,255))
                    if (color[0],color[1],color[2]) == (0,0,0):
                        obstacles.append([x,y])
                        break
                    elif i == 99:
                        obstacles.append([x,y])
                    
                elif x>self.map_width or self.map_height:
                    obstacles.append([x,y])
                    
                
        return obstacles

class robotEnv:
    def __init__(self,env_config = {}):
        self.dimensions = env_config['dimensions']
        self.robot_img_path = env_config['robot_img_path']
        self.map_img_path = env_config['map_img_path']
        self.goal_img_path = env_config['goal_img_path']
        self.startpos = env_config['start']
        self.width = env_config['width']
        self.goal = env_config['goal']
        self.angle_space = env_config['angle_space']
        self.sensor_range = env_config['sensor_range']
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(44,), dtype=np.float32
        ) #Change this if changing lidar model or state space
        self.robot = Robot(self.startpos, self.width, self.goal)
        self.graphics = Graphics(self.dimensions,self.robot_img_path,
                                 self.map_img_path,self.goal_img_path,self.goal)
        self.map = self.graphics.map
        self.laser = LaserScan(self.sensor_range,self.angle_space,self.map)
        self.last_time = pygame.time.get_ticks()
        self.running = True
        self.point_cloud = self.laser.sense_obstacles(self.robot.x,self.robot.y,self.robot.heading)
        
    def reset(self):
        self.robot = Robot(self.startpos, self.width, self.goal)
        #closest_obs,distance_readings = self.robot.check_obstacles(self.point_cloud)
        state, _, _ = self.robot.get_reward(self.point_cloud)
        return state
        
        
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        self.graphics.map.blit(self.graphics.map_img,(0,0))
        self.graphics.map.blit(self.graphics.star_img,self.graphics.star_rect)
        self.graphics.draw_robot(self.robot.x,self.robot.y,self.robot.heading)
        
        self.point_cloud = self.laser.sense_obstacles(self.robot.x,self.robot.y,self.robot.heading)
        self.graphics.draw_sensor_data(self.point_cloud)   
        #pygame.draw.circle(self.graphics.map,self.graphics.green,(250,250),10)
        pygame.display.update()
        
    
    def step(self, action=-1):
        #dt = (pygame.time.get_ticks()-self.last_time)/1000
        dt = 0.1
        self.last_time = pygame.time.get_ticks()
        self.robot.move(action,dt,self.point_cloud)
        state, reward, done = self.robot.get_reward(self.point_cloud)
        return state,reward,done
    
class nn_deep_q:
    def __init__(self,env,nn_configs):
        self.env = env
        self.nactions = env.action_space.n
        self.nstates = env.observation_space.shape[0]
        self.num_nodes = nn_configs['num_nodes']
        self.num_steps = nn_configs['max_steps']
        self.num_eps = nn_configs['n_episodes']
        self.sol_reward = nn_configs['sol_reward'] 
        self.batch = nn_configs['batch'];
        self.gamma = nn_configs['gamma']
        self.epsilon = nn_configs['epsilon']
        self.alpha = nn_configs['alpha0']
        self.nnet = self.init_nn()
        self.reward_mem = []
    
    def init_nn(self): #Define Keras Model
        nn = Sequential()
        nn.add(Dense(self.num_nodes*2, input_dim=self.nstates, activation='relu'))
        nn.add(Dense(self.num_nodes, activation='relu'))
        nn.add(Dense(self.nactions, activation='linear'))
        nn.compile(loss = 'mean_squared_error', optimizer = optimizers.Adam(lr = self.alpha)) #,metrics = ['accuracy'])
        return nn          
    
    def train(self):
        reward_mem = []; memory = deque(); 
        for i in range(self.num_eps):
            episode_reward = 0
            state1 = self.env.reset()
            state1 = np.reshape(state1,(1,self.nstates))
            if i>200:
                self.epsilon = (self.epsilon*0.99 if self.epsilon>0.01 else 0.01)
                    
            print("{:0.2f}% done".format(i/self.num_eps*100))
            for step in range(self.num_steps):
#                if i % 50 == 0:
                self.env.render()
                action1 = self.choose_action(state1)
                state2,reward,done = self.env.step(action1)
                state2 = np.reshape(state2,(1,self.nstates))
                memory.append((state1, action1, reward, state2, done))
#                if len(memory)>self.batch*10:
#                    memory.popleft()
                # NEED TO FIX MODEL FIT
                #Add rewards and reset
                episode_reward+=reward; state1 = state2
                if done:
                    break
                self.update_nn_weights(memory, reward_mem)   
            reward_mem.append(episode_reward)
            print(episode_reward);
            last_rewards = np.mean(reward_mem[-100:]) #Last 100 based on OpenAI benchmarks
            if last_rewards > self.sol_reward-100: 
                print("Training Complete")
                break
        return reward_mem
    
    def choose_action(self,state):
        action = 0; rand_num = np.random.uniform()
        if rand_num < self.epsilon:
            action = np.random.randint(0,self.nactions)
            action -= 1
        else:
            pred = np.squeeze(self.nnet.predict(state))
            action = np.argmax(pred)
            action -= 1
        return action;
    
    def update_nn_weights(self,memory,reward):        
        if len(memory)<self.batch: #Don't update weights (not enough in the list)
            return
        elif np.mean(reward[-10:])>self.sol_reward:  #Don't update weights
            return
        else:
            mem_sample = random.sample(memory,self.batch)
            states, actions, rewards, states2, done_array = self.pull_sample_attr(mem_sample)
            done_array = done_array*1
            targets = rewards + self.gamma*(np.max(self.nnet.predict_on_batch(states2),axis = 1))*(1-done_array)
            target_list = self.nnet.predict_on_batch(states)
            batch_list = np.arange(self.batch) 
            target_list[batch_list, actions] = targets
            self.nnet.fit(states, target_list, epochs=1, verbose=0)
        
    def pull_sample_attr(self,sample):
        states = np.array([sample[i][0] for i in range(len(sample))]); 
        states2 = np.array([sample[i][3] for i in range(len(sample))])
        states = np.squeeze(states); states2 = np.squeeze(states2)
        actions = np.array([sample[i][1] for i in range(len(sample))]); 
        rewards = np.array([sample[i][2] for i in range(len(sample))])
        done_arr = np.array([sample[i][4] for i in range(len(sample))])
        return states, actions, rewards, states2, done_arr
            
    def test(self,num_tests, num_steps):
        reward_mem = []
        for i in range(num_tests):
            state1 = self.env.reset()
            state1 = np.reshape(state1,(1,self.nstates))
            episode_reward = 0
            for step in range(num_steps):
                self.env.render()
                action1 = np.argmax(np.squeeze(self.nnet.predict(state1)))
                state2,reward,done = self.env.step(action1)
                state2 = np.reshape(state2,(1,self.nstates))
                state1 = state2; episode_reward+=reward
                if done:
                    break
            reward_mem.append(episode_reward)
        return reward_mem
        
            
        