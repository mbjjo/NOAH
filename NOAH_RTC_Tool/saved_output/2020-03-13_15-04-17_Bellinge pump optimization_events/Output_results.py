# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:55:42 2020

@author: Magnus Johansen

Checking the output from pump optimization in Bellinge case model. 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime,timedelta
import os 
import pyswmm
from pyswmm import Simulation, Nodes, Links, SystemStats, Subcatchments
import swmmtoolbox.swmmtoolbox as swmmtoolbox
os.getcwd()
os.listdir()
 
outfile = 'BellingeSWMM_MU_v017a_pumptest.out'

sensor = 'G80F13P'
pump = 'G80F13Pp1'
rg = 'rg5425'

CSO_id1 = 'G80F13P'
CSO_id2 = 'G80F240'
CSO_id3 = 'G80F11B'

# depth
# CSO_id1 = swmmtoolbox.extract(outfile,['node',CSO_id1,'Depth_above_invert'])
# CSO_id2 = swmmtoolbox.extract(outfile,['node',CSO_id2,'Depth_above_invert'])
# CSO_id3 = swmmtoolbox.extract(outfile,['node',CSO_id3,'Depth_above_invert'])

pump_rate = swmmtoolbox.extract(outfile,['link',pump,'Flow_rate'])

# Ponding
CSO_id1 = swmmtoolbox.extract(outfile,['node',CSO_id1,'Volume_stored_ponded'])
CSO_id2 = swmmtoolbox.extract(outfile,['node',CSO_id2,'Volume_stored_ponded'])
CSO_id3 = swmmtoolbox.extract(outfile,['node',CSO_id3,'Volume_stored_ponded'])

swmmtoolbox.catalog(outfile, 'node')

plt.figure(1)
pump_rate.plot(color = 'tab:red')
plt.show()
plt.figure(2)
CSO_id1.plot()
plt.legend(['cso1'])
CSO_id2.plot()
plt.legend(['cso2'])
CSO_id3.plot()
plt.legend(['cso3'])
plt.show()

"""
There is no flooding from these nodes but ponding is allowed instead.
How are CSO's computed in the models?
"""