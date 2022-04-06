# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:59:07 2022

@author: qbr5kx
"""
import math as m
from robotClass_collisionAvoidance import robotEnv
#from ray.tune.registry import register_env
from ray.rllib.agents import ppo
# from ray.tune.logger import pretty_print
# from ray import tune
import ray
# import cv2
# import numpy
# from pygame_recorder import ScreenRecorder
f = 'C:\\Users\\qbr5kx\\OneDrive - University of Virginia\\Desktop\\UVA\\PhD Scratch\\RL_SMD\\BARN_dataset\\grid_files\\grid_0.npy'
ray.shutdown()
map_dims = (1175,1175)
start = (75,500)
goal = (1100,500)
#sensor
sensor_range = 250,m.radians(90/2)
angle_space = 40
# map_matrix = cv2.imread('pictures/obstacleMap.png')
# map_matrix = numpy.transpose(map_matrix, (1,0,2)) 

env_configs = {
    'dimensions': map_dims,
    'drawing': False,
    'robot_img_path': 'pictures/robot_img.png',
    'map_path': f,
    'sensor_range': sensor_range,
    'goal': goal,
    'start': start,
    'angle_space': angle_space,
    'width': 0.01*3779.52,
    'goal_img_path': 'pictures/starPNG.png'}

ray.init()
env = robotEnv(env_configs)

config={
    "env": robotEnv,
    "env_config": env_configs
    }
checkpoint_path = "C:\\Users\\qbr5kx\\ray_results\\PPOTrain1\\PPO_robotEnv_cfb28_00000_0_2022-03-14_11-45-54\\checkpoint_000003\\checkpoint-3"

agent = ray.tune.run(ppo.PPOTrainer, restore=checkpoint_path)
agent.restore(checkpoint_path)

episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward