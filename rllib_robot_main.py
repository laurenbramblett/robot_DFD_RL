# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 12:52:55 2022

@author: qbr5kx
"""
import os
os.chdir("C:\\Users\\qbr5kx\\OneDrive - University of Virginia\\Desktop\\UVA\\PhD Scratch\RL_SMD\\DFD_Robot")
import math as m
from draw_background import draw_background
f = 'grid_files/grid_0.npy'
map_matrix = draw_background(f)
from robotClass_collisionAvoidance import robotEnv
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
import ray


ray.shutdown()

map_dims = (1175,1175)
start = (75,500)
goal = (1100,500)
#sensor
sensor_range = 250,m.radians(180/2)
angle_space = m.pi/8


env_configs = {
    'dimensions': map_dims,
    'drawing': False,
    'robot_img_path': 'pictures/robot_img.png',
    'map_matrix': map_matrix,
    'sensor_range': sensor_range,
    'goal': goal,
    'start': start,
    'angle_space': angle_space,
    'width': 0.01*3779.52,
    'goal_img_path': 'pictures/starPNG.png',
    'obs_space_len': 10}

ray.init(local_mode=True)
# trainer = sac.SACTrainer(env=robotEnv, config={
#     "env_config": env_configs,  # config to pass to env class
# })
# n = 0
# while True:
#     result = trainer.train()
#     print(pretty_print(result))

config = ppo.DEFAULT_CONFIG.copy()
env = robotEnv(env_configs)
stop = {"timesteps_total": 1_000}
config = {"env": robotEnv,"env_config": env_configs}
config["log_level"] = "WARN"
ck_path, analysis = env.train(config,stop)


env_configs['drawing'] = True
agent = env.load(ck_path,config, robotEnv)
env = robotEnv(env_configs)
env.test(env,agent)
# tune.run(
#     "PPO", # reinforced learning agent
#     name = "PPOTrain1",
#     # to resume training from a checkpoint, set the path accordingly:
#     # resume = True, # you can resume from checkpoint
#     # restore = r'.\ray_results\Example\SAC_RocketMeister10_ea992_00000_0_2020-11-11_22-07-33\checkpoint_3000\checkpoint-3000',
#     checkpoint_freq = 500,
#     checkpoint_at_end = True,
#     #local_dir = r'./ray_results/',
#     config={
#         "env": robotEnv,
#         "env_config": env_configs
#         },
# ,
#     )
