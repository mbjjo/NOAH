# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 08:27:56 2021

@author: Magnus Johansen

Computes the number of events in the existing Astlingen model. 
"""
import pandas as pd
import numpy as np
# from scipy import optimize,integrate
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
# import threading
from datetime import datetime,timedelta
# import configparser
# from configparser import ConfigParser
import os 
# from threading import Thread
import swmmio
import shutil
# import the necessary modelus from pyswmm
import pyswmm
from pyswmm import Simulation, Nodes, Links, SystemStats, Subcatchments
import swmmtoolbox.swmmtoolbox as swmmtoolbox
import pdb
import CSO_statistics 


# Specify path where the file is stored
model_path = r'C:\Users\magnu\OneDrive\DTU\NOAH_project_local\real_models'


CSO_ids = ['T1','T2','T3','T4','T5','T6','CSO7','CSO8','CSO9','CSO10']
model_outfile = model_path + '\Astlingen0920_maximal_basecase_DWFDiurnal.out'
# Creates a small copy of the input from config file with only the required inputs

class inp_copy:
    def __init__(self):
        self.CSO_type = 'Flooding above ground from node'
        rpt_step_tmp = '00:05:00'
        rpt_step = datetime.strptime(rpt_step_tmp, '%H:%M:%S').time()
        self.report_times_steps = rpt_step.hour*60 + rpt_step.minute + rpt_step.second/60
inp = inp_copy()
# inp.CSO_type

# Statistics is computed 
df_base = CSO_statistics.Compute_CSO_statistics(inp,CSO_ids, model_outfile,
                                                allow_CSO_different_locations = False)

# df.head()