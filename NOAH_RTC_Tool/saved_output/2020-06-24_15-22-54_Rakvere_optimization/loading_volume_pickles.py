# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:24:26 2020

@author: magnu
"""


import pickle
import os 
import pandas as pd
import matplotlib.pyplot as plt

num = 6


volumes = np.zeros(num)
for i in range(1,num+1):
    file = 'CSO_vol_{}.pickle'.format(i)
    pickle_in = open(file,"rb")
    volumes[i-1] = pickle.load(pickle_in)
    
print(volumes)

plt.plot(volumes,'+')