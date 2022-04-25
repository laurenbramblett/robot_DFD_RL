# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:09:30 2022

@author: qbr5kx
"""
import numpy as np
import pandas as pd
filepath = 'characteristics_by_world.csv'
metrics_total = []
for grid_num in range(0,300):
    print(grid_num)
    f = 'metrics_files/metrics_%d.npy' % grid_num
    mets = np.load(f)
    mets = np.append(mets,grid_num)
    metrics_total.append(mets)
 
col_names = ["Dist_to_Closest_Obs","Avg_Visibility","Dispersion",
             "Characteristic_Dimension","Tortuosity","World_ID"]
df = pd.DataFrame(metrics_total, columns = col_names)  
df.to_csv(filepath,index=False)
    #Distance to Closest Obstacle, Average Visibility, Dispersion,
    #Characteristic Dimension, and Tortuosity
    
    