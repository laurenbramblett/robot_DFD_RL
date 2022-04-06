# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:35:37 2022

@author: qbr5kx
"""
from matplotlib import pyplot,colors
import cv2
import numpy as np
def draw_background(map_path,map_dims):
    data = np.load(map_path)
    cushion = np.transpose(np.tile(np.concatenate(([1],np.repeat(0,data.shape[0]-2),[1])),(5,1)))
    data2 = np.column_stack((np.ones((data.shape[0],1)),cushion,data,cushion))
    pyplot.figure(figsize=(30,40))
    colormap = colors.ListedColormap(["white","black"])
    pyplot.imshow(data2,cmap = colormap)
    pyplot.yticks([])
    pyplot.xticks([])
    pyplot.rcParams['axes.linewidth'] = 20
    pyplot.savefig('pictures/matrixFile.png',bbox_inches='tight',pad_inches = 0)
    pyplot.close()
    map_matrix = cv2.imread('pictures/matrixFile.png')
    map_matrix = np.transpose(map_matrix, (1,0,2)) 
    map_matrix = cv2.resize(map_matrix,(map_dims))
    return map_matrix
