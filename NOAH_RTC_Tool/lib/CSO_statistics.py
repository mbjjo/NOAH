# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:19:48 2020

@author: magnu
"""
#%%

# Required external imports 
import pandas as pd
import numpy as np
from scipy import optimize,integrate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import threading
from datetime import datetime,timedelta
import configparser
from configparser import ConfigParser
import os 
from threading import Thread
import swmmio
import shutil
# import the necessary modelus from pyswmm
import pyswmm
from pyswmm import Simulation, Nodes, Links, SystemStats, Subcatchments
import swmmtoolbox.swmmtoolbox as swmmtoolbox
import pdb
import Parameters

if __name__ == "__main__":
    os.chdir(r'C:\Users\magnu\OneDrive\DTU\NOAH_project_local\github\NOAH\NOAH_RTC_Tool\lib')
    inp = Parameters.Read_Config_Parameters('Astlingen.ini')
    os.chdir(r'C:\Users\magnu\OneDrive\DTU\NOAH_project_local\github\NOAH\NOAH_RTC_Tool\saved_output\2020-03-13_15-04-17_Bellinge pump optimization_events')
    os.chdir(r'C:\Users\magnu\OneDrive\DTU\NOAH_project_local\2020-01-08_22-21-46')
    
# CSO_ids = ['22065','14','5619']
    CSO_ids = ['T4','T3','T5']
    model_outfile = 'Astlingen.out'

def Compute_CSO_statistics(inp,CSO_ids, model_outfile,allow_CSO_different_locations = True):
    """
    Takes an .out file from SWMM and computes statistics for the specified nodes. 
    This can be to compute CSO volumes or count events. 

    Parameters
    ----------
    inp : TYPE
        Input file with all parameters from the NOAH RTC Tool.
    CSO_ids : TYPE
        List of node IDs that are to be computed.
    model_outfile : TYPE
        The .out file from the SWMM simulation.
    allow_CSO_different_locations: Binary (True/False)
        Defines whether the CSO's are computed fro each node or if they are combined only the different times count as different CSO's
        This is used for one recipient only and disregards the spatial distribution of the nodes. 
        True if CSO's from each node is computed individually
        False if the CSO's are combined into one timeseries that is used for the events. 
        
    Returns 
    -------
    DataFrame with statistics for each of the nodes
    """    
    # create the dataframe with the 
    max_event_length = 1 # day
    # Time between events is assumed to be 12 hours before they are counted as seperate
    CSO_event_seperation = 6 # hours
    
    
    
    if inp.CSO_type == 'Outflow from CSO structure':
        CSO_type = 'Total_inflow'
    elif inp.CSO_type == 'Flooding above ground from node':
        CSO_type = 'Flow_lost_flooding'
    
    df = pd.concat(((swmmtoolbox.extract(model_outfile,['node',CSO_ids[i],CSO_type]))for i in range(len(CSO_ids))),axis = 1)
    # df is CMS therefore this are converted to m3/timestep. 
    df = df*60*inp.report_times_steps
    # pdb.set_trace()
    if allow_CSO_different_locations == False:
        df = pd.DataFrame(df.max(axis=1))
        CSO_ids = ['Sum']
    df.columns = CSO_ids
    # Set all timesteps with flooding to 1
    CSO_YesNo_df = df.mask(df>0,other = 1)
    # Time between events in timesteps
    time_between_events= CSO_event_seperation*60/inp.report_times_steps 
    # time_between_events = timedelta(hours = CSO_event_seperation)
    # Get df with 1 at every event start 
    CSO_start_df = CSO_YesNo_df.where(np.logical_and(CSO_YesNo_df>0,CSO_YesNo_df.rolling(int(time_between_events),min_periods=0).sum()==1),other=0)
    # Get df with 1 at every event stop
    CSO_stop_df = CSO_YesNo_df.where(np.logical_and(CSO_YesNo_df>0,CSO_YesNo_df.rolling(int(time_between_events),min_periods=0).sum().shift(-(int(time_between_events)-1)).fillna(0)==1),other=0)
    
    empty_dict = {'Event_ID':[],
                  'Node':[],
                  'Start_time':[],
                  'End_time':[],
                  'Duration':[],
                  'Volume':[]
                  }
    # pdb.set_trace()
    (start_index, node_ID) = np.where(CSO_start_df==1)
    start_time = CSO_start_df.iloc[start_index].index
    # Each start time corresponds to one event. 
    no_events = len(start_time)
    # computes statistics for each event
    df_stats = pd.DataFrame(empty_dict)
    for i in range(no_events):
        tmp_dict = empty_dict.copy()
        tmp_dict['Event_ID'] = i+1
        tmp_dict['Node'] = CSO_ids[node_ID[i]]
        tmp_dict['Start_time'] = start_time[i]
        stop_index = np.where(np.logical_and(CSO_stop_df[CSO_ids[node_ID[i]]].index >= start_time[i],CSO_stop_df[CSO_ids[node_ID[i]]] == 1))
        # pdb.set_trace()
        tmp_dict['End_time'] = CSO_stop_df.index[stop_index[0]][0]
        tmp_dict['Duration'] = tmp_dict['End_time'] - tmp_dict['Start_time']
        tmp_dict['Volume'] = df[CSO_ids[node_ID[i]]].loc[tmp_dict['Start_time']:tmp_dict['End_time']].sum()
        df_stats = df_stats.append(tmp_dict, ignore_index=True)

    # Test if any of the events are longer than the maximum allowed before they are counted as more events
    try:
        df_stats['Duration'] > timedelta(hours = max_event_length)
    except:
        print('Some events last longer than the maxium allowed duration.')

    
    # All events are ranked after volume. Biggest event has rank = 1
    df_stats['Rank'] = df_stats['Volume'].rank(ascending=False)
    
    # Events for each node are ranked after volume. Biggest event has rank = 1
    df_stats['Node Rank'] = df_stats.groupby('Node')['Volume'].rank(ascending=False)
    
    # Event_ID is used as index
    df_stats.set_index(df_stats['Event_ID'],inplace= True)
    return df_stats
    

#%%
"""
Examples on how the above function can be used 
"""
if __name__ == "__main__":
    
    df = Compute_CSO_statistics(inp,CSO_ids,model_outfile)
    df.head()
    
    # Count evetns in total
    df['Event_ID'].count()
    
    # Count evetns per node
    df.groupby('Node')['Event_ID'].count()
    
    # plot events according to size 
    plt.plot(df['Rank'],df['Volume'],'*')
    plt.bar(df['Rank'],df['Volume'])
    
    # plot events according to duration
    plt.bar(df['Rank'],df['Duration'])
    
    # See total volume per node
    df.groupby('Node')['Volume'].sum()
    # and plot it
    total_volume = df.groupby('Node')['Volume'].sum()
    plt.bar(total_volume.index,total_volume)
    
    # Count relative contribution for each event (in %)
    df['Volume'] / df['Volume'].sum() * 100
    
    # Count relative contribution for each event per node (in %)
    def count_percentage(df):
        return df['Volume'] / df['Volume'].sum() * 100
    df.groupby('Node').apply(count_percentage)
    
    
    def duration_between_events(df_stats):
        return df_stats['Start_time'].shift(-1) - df_stats['End_time']
    duration_between_events(df)
    df.groupby('Node').apply(duration_between_events)
    
    
    def duration_of_events(df_stats):
        return df_stats['Start_time'].shift(-1) - df_stats['End_time']
    duration_between_events(df)
    df.groupby('Node').apply(duration_between_events)
    
    
    CSO_event_seperation = 12
    
    
    # time_to_next = df_stats.groupby('Node').apply(duration_between_events)
    # np.where(time_to_next < time_between_events)
    
    
    
    # # df_stats = pd.concat((df_stats,pd.DataFrame(tmp_dict)),axis=0)
    # np.where(np.logical_and(CSO_stop_df[CSO_ids[node_ID[i]]].index > start_time[i],CSO_stop_df[CSO_ids[node_ID[i]]] == 1))[0]
    # np.where(CSO_stop_df[CSO_ids[node_ID[i]]].index > start_time[i])
    # np.where(CSO_stop_df[CSO_ids[node_ID[i]]] == 1)
    
    # df[CSO_ids[node_ID[i]]].loc[tmp_dict['Start_time']:tmp_dict['End_time']].sum()
    
    


#%%
# =============================================================================
# The code is tested on a case from Rakvere to see if the same resutls are obtained 
# 02-11-20
# =============================================================================
if __name__ == "__main__":
    os.chdir(r'C:\Users\magnu\OneDrive\DTU\NOAH_project_local\github\NOAH\NOAH_RTC_Tool\lib')
    inp = Parameters.Read_Config_Parameters('Rakvere.ini')
    inp
    os.chdir(r'C:\Users\magnu\OneDrive\DTU\NOAH_project_local\real_models\Rakvere\0529_Rakvere_6months_NoRTC')
    
    CSO_ids = ['5902','19260','9859','39','5832']
    model_outfile = 'Rakvere_v3_cal.out'
    
    df = Compute_CSO_statistics(inp,CSO_ids,model_outfile)

#%%

if __name__ == "__main__":
    
    test_df = pd.DataFrame(np.zeros((10,3)))
    test_df.head()
    test_df.iloc[2,1] = 1
    test_df.head()
    
    test_df.where(test_df>0)
    test_df.where(test_df.rolling(3,min_periods=0).sum().shift(-(3-1)).fillna(0)==1)
    
    
    test_df.where(np.logical_and(CSO_YesNo_df>0,CSO_YesNo_df.rolling(int(time_between_events),min_periods=0).sum().shift(-(int(time_between_events)-1)).fillna(0)==1),other=0)
        
    

#%%




# Compute CSO volume
def count_CSO_volume(self,inp,CSO_ids, model_outfile): 
    if inp.CSO_type == 'Outflow from CSO structure':
        CSO_type = 'Total_inflow'
    elif inp.CSO_type == 'Flooding above ground from node':
        CSO_type = 'Flow_lost_flooding'
    df = pd.concat(((swmmtoolbox.extract(model_outfile,['node',CSO_ids[i],CSO_type]))for i in range(len(CSO_ids))),axis = 1)
    df = df*inp.report_times_steps*60 
    # CSO_volume_events = df.sum().sum()
    CSO_volume_total = df.sum().sum()
    
    
    return CSO_volume_total

#===================================================================     
    
# Compute CSO frequency 
def count_CSO_events(self,inp,CSO_ids, model_outfile):
    # create the dataframe with the 
    max_event_length = 1 # day
    CSO_event_seperation = 12 # hours
    
    if inp.CSO_type == 'Outflow from CSO structure':
        CSO_type = 'Total_inflow'
    elif inp.CSO_type == 'Flooding above ground from node':
        CSO_type = 'Flow_lost_flooding'

    df = pd.concat(((swmmtoolbox.extract(model_outfile,['node',CSO_ids[i],CSO_type]))for i in range(len(CSO_ids))),axis = 1)
    # df is CMS therefore this are converted to m3/timestep. 
    df = df*60*inp.report_times_steps
    
    # Set all timesteps with flooding to 1
    CSO_YesNo_df = df.mask(df>0,other = 1)
    # Time between events is assumed to be 12 hours before they are counted as seperate
    time_between_events= CSO_event_seperation*60/inp.report_times_steps
    
    # Get df with 1 at every event start 
    CSO_start_df = CSO_YesNo_df.where(np.logical_and(CSO_YesNo_df>0,CSO_YesNo_df.rolling(int(time_between_events),min_periods=0).sum()==1),other=0)
    # Get df with 1 at every event stop
    CSO_stop_df = CSO_YesNo_df.where(np.logical_and(CSO_YesNo_df>0,CSO_YesNo_df.rolling(int(time_between_events),min_periods=0).sum().shift(-(int(time_between_events)-1)).fillna(0)==1),other=0)
    
    # Counter for each CSO structure
    CSO_event_counter = dict(zip(CSO_ids, np.zeros(len(CSO_ids))))
    for count,col in enumerate(CSO_start_df.columns):
        max_dur = timedelta(max_event_length).days
        start_index = np.where(CSO_start_df[col]==1)[0]
        stop_index = np.where(CSO_stop_df[col]==1)[0]
        # The end is added to the stop index if overflow continues until the end of simulation
        if len(start_index) != len(stop_index):
            stop_index = np.append(stop_index,len(CSO_start_df)-1)
        start_time = CSO_start_df.iloc[start_index].index
        stop_time = CSO_stop_df.iloc[stop_index].index
        duration = stop_time - start_time
        # If events are longer than maximum duration then the overflow is counted once for each day it lasts. 
        for i, dur in enumerate(duration):
            if dur.days > max_dur:
                CSO_event_counter[CSO_ids[count]] += dur.days
            else:
                CSO_event_counter[CSO_ids[count]] += 1
    # CSO_event_counter contain information for each CSO structure
    # total_CSO_events contain the total number of CSO's
    total_CSO_events = sum(CSO_event_counter.values())
    return total_CSO_events





#%%
def basins_cross_correlation(outfile,basins,depths):
    """
    8/2-21
    Computes the correlation between depths in the different basins. 
    Can be used to check if filling degrees are similar or if some basins vary in filling. 
    
    Parameters
    ----------
    outfile : TYPE
        DESCRIPTION.
    basins : TYPE
        DESCRIPTION.
    depths : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
# outfile = r'C:\Users\magnu\OneDrive\DTU\NOAH_project_local\github\NOAH\NOAH_RTC_Tool\output\2021-02-08_11-39-14\Astlingen_EFD.out'
# basins = ('T1','T2','T3','T4','T5','T6')
# depths = [5,5,5,5,5,5]

    frames = [ swmmtoolbox.extract(outfile,['node',basin,'Depth_above_invert'])/depths[i] for i,basin in enumerate(basins)]
    filling_degrees = pd.concat(frames,axis=1)
    filling_degrees.columns = basins

    # Compute correlations
    Corr_matrix = filling_degrees.corr()
    print('Correlation matrix is:\n',Corr_matrix)

    # Make a simple plot
    ticks = range(len(basins))
    labels = basins
    
    plt.figure()
    plt.matshow(Corr_matrix)
    plt.xticks(ticks, labels)
    # plt.yticks(ticks, labels)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    # plt.title('Correlation Matrix', fontsize=10);    
    plt.show()
#%%
# filling_degrees.plot()



# # output_storage.iloc[1:3000,:].plot()
    
    # # system = swmmtoolbox.extract(outfile,
    # #                      ['system','','Flow_leaving_outfalls'])
    
    # # print(system.sum())
    
    # # actuators = swmmtoolbox.extract(outfile,['link','V3','Flow_rate'])
    # # actuators.plot()
    
    
    # # =============================================================================
    # # # for comparison this should give around 27 to be similar to Astlingen 
    # # # when compared in the period 1/1/2000 to 03/10/2000
    # # =============================================================================
    # nodes_flooding = swmmtoolbox.extract(outfile,['node','','Flow_lost_flooding'])
    # # print(nodes_flooding.sum().sum())
    # nodes_flooding.sum()
    # return nodes_flooding.sum().sum()
    