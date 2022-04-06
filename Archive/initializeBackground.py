# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:14:41 2022

@author: qbr5kx
"""

from ray import tune
import pygame
# tune.run('PPO', config={"env": "CartPole-v0"})
import cv2
import numpy
pygame.quit()
pygame.init()
pic = cv2.imread('pictures/obstacleMap.png')
display = pygame.display.set_mode((1200, 600))
pic = numpy.transpose(pic, (1,0,2))          
surf = pygame.surfarray.make_surface(pic)

display.blit(surf, (0, 0))
pygame.display.update()


