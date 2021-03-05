# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:39:36 2020

@author: Magnus Johansen 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import pickle
import configparser
import time
import os, shutil, sys
from threading import Thread
from datetime import datetime,timedelta
import swmmtoolbox.swmmtoolbox as swmmtoolbox
import pyswmm
from pyswmm import Simulation,Nodes,Links,SystemStats, raingages
import pdb
import Parameters
import CSO_statistics


def equal_filling_degree(config_file):
    """
    3/2-21
    
    This is the main function for the Equal Filling Degree (EFD). 
    It takes input from the config file (either from GUI or the text file) 
    and calls several functions to compute the EFD. 

    Parameters
    ----------
    config_file : .ini file
        Contains all variables and parameters that are used in the EFD computation. The process is the same as the original simple RTC and the calibration procedure. 

    Returns
    -------
    None.

    """
    # All variables are stored as inp._VarName_ and can be used in functions. 
    inp = Parameters.Read_Config_Parameters(config_file)

    filename = '_EFD'
    # Redefining timeseries if these are not in the right directory. 
    # The Hiddenprints are only used to avoid printing the warning from the exception since this is not relevant for the end user. 
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
    
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout
    try:
        with HiddenPrints():
            #Changing the path of the Time series to run models.
            from swmmio.utils import modify_model
            from swmmio.utils.modify_model import replace_inp_section
            from swmmio.utils.dataframes import create_dataframeINP
            # Extracting a section (Timeseries) to a pandas DataFrame 
                
            path = inp.model_dir
            inp_file = inp.model_name
            baseline = path + '/' + inp_file + '.inp'
        
          
            # Create a dataframe of the model's time series 
            Timeseries = create_dataframeINP(baseline, '[TIMESERIES]')
            New_Timeseries = Timeseries.copy()
            # Modify the path containing the timeseries 
            for i in range(len(Timeseries)):
                string = Timeseries.iloc[i][0]
                
                if '"' in string:   # Check if string is an external file if not nothing is changed. 
                                    # This might be slow of the timeseries are long
                    Rainfile_old = string.split('"')[-2]
                    if '\\' in Rainfile_old: # If the rain file is in an absolute path this is changed (Also if is in another directory than the model)
                        Rainfile_old = Rainfile_old.split('/')[-1]
                    Rain_name = string.split('"')[-3]
                    Rainfile_new = path + '/' + Rainfile_old
                    New_Timeseries.iloc[i][0] = Rain_name + '"' + Rainfile_new + '"'
                    print('Rainfile_new: '+ Rainfile_new)
                    print('Rainfile_old: '+ Rainfile_old)
                    print('Rain_name:' + Rain_name)
                else: 
                    New_Timeseries.iloc[i][0] = np.nan
            New_Timeseries.dropna(inplace = True)
                
            # Create a temporary file with the adjusted path
            new_file = inp_file + '_tmp.inp'
            shutil.copyfile(baseline, path + '/' + new_file)
         
            # print('path:' + path ) 
            #Overwrite the TIMESERIES section of the new model with the adjusted data
            with pd.option_context('display.max_colwidth', 400): # set the maximum length of string to prit the full string into .inp
                replace_inp_section(path + '/' + new_file, '[TIMESERIES]', New_Timeseries)
            
            model_inp = inp.model_dir + '/' + inp.model_name + '_tmp.inp'
    except KeyError:
        model_inp = inp.model_dir + '/' + inp.model_name + '.inp'
    
    # DataFrame that is needed for the EFD is created 
    # input_df should be created 
        
    basins = [x.strip(' ') for x in inp.EFD_basins.split(',')]
    actuators = [x.strip(' ') for x in inp.EFD_actuators.split(',')]
    depths = [float(x.strip(' ')) for x in inp.basin_depths.split(',')]
    depths_df = pd.DataFrame(depths,index = basins) # written into a dataframe
    CSO_ids = [x.strip(' ') for x in inp.EFD_CSO_id.split(',')]

    # print(depths_df)
    no_layers = inp.no_layers
    
    # creating directory for the two output files
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.mkdir('../output/'+timestamp)   
    model_rptfile = '../output/' + timestamp + '/' + inp.model_name + filename + '.rpt'
    model_outfile = '../output/' + timestamp + '/' + inp.model_name + filename + '.out'
    
    # The basin structure is defined 
    if 'Astlingen' in inp.model_name:
        basin_structure = create_basin_structure(basins)
    else:
        print('Basin structure is not implemented for this model. Compute that manually in the function create_basin_structure() in EqualFillingDegree.py')
        
    if inp.EFD_setup_type == 'default': # if default setups are tried out
        no_layers = 3
        
        # The dataframe with the EFD setup (i.e. orifice settings and thresholds between zones) that should be computed to SWMM is created.
        EFD_input_df = create_EFD_input_df(basins,depths_df,no_layers)
        
        # A SWMM file is written with the selected EFD algorithm 
        EFD_to_SWMM(EFD_input_df,inp,timestamp,filename,basin_structure,basins,actuators,depths_df,no_layers)
 
        # Compute the SWMM simulation 
        run_SWMM(model_inp,model_rptfile,model_outfile)
    
        CSO_stats_df = CSO_statistics.Compute_CSO_statistics(inp,CSO_ids, model_outfile)
        print(CSO_stats_df.head())
        print(CSO_stats_df['Volume'].sum())
    
    elif inp.EFD_setup_type == 'optimized':
        print('optimizing...')
        
        start_value = [4,1]

        result = optimize.minimize(fun = efd_optimizer_wrapper,
                          args = (inp,timestamp,filename,
                          basin_structure,basins,actuators,depths_df,no_layers,CSO_ids,
                          model_inp,model_rptfile,model_outfile),
                          x0 = start_value, method='Nelder-Mead',
                          options = {'disp':True})

        print('result', result)        
        
    
def efd_optimizer_wrapper(metaparameters,inp,timestamp,filename,
                          basin_structure,basins,actuators,depths_df,no_layers,CSO_ids,
                          model_inp,model_rptfile,model_outfile):
    c1 = metaparameters[0]
    c2 = metaparameters[1]

    EFD_input_df  = custom_EFD(c1,c2,basins,depths_df,no_layers)
        
    EFD_to_SWMM(EFD_input_df,inp,timestamp,filename,basin_structure,basins,actuators,depths_df,no_layers)
    
    run_SWMM(model_inp,model_rptfile,model_outfile)
   
    CSO_stats_df = CSO_statistics.Compute_CSO_statistics(inp,CSO_ids, model_outfile)
    # print(CSO_stats_df.head())
    total_volume = CSO_stats_df['Volume'].sum()
    
    return total_volume
  
    
    
def run_SWMM(model_inp,model_rptfile,model_outfile):
    # pass
    with Simulation(model_inp, model_rptfile,model_outfile) as sim: 
        for step in sim:
            pass
    print('Ran SWMM file')
    
    
    
def create_basin_structure(basins):
    """
    8/2 - 21
    
    Returns
    -------
    basin_structure : dataframe
        Table with the structure of the basins. 
        Rows are the donwstream basins and the number indicate how many "steps" (i.e. basins) upstream another basin is. 
        Columns are the upstream basins and the number indocate how many "steps" downstream another basin is. 

        Ideally this is created dynamically with the structure analysis tools from the calibration procedure. This is not implemented. 
    """
    # Structure of the basins are to be computed beforehand using the system structure analysis tools from calibration
    
    # The table is created manually for the Astlingen model 
    basin_structure = pd.DataFrame(np.zeros((6,6)),index = basins,columns = basins)
    basin_structure.loc['T1','T2':'T5'] = 1
    basin_structure.loc['T1','T6'] = 2
    basin_structure.loc['T3','T6'] = 1

    return basin_structure
    
def create_EFD_input_df(basins,depths_df,no_layers):
    """
    8/2 - 21

    Creates a DataFrame that defines the EFD thresholds (i.e. the water levels that change the orifice settings) 
    
    Parameters
    ----------
    basins : list
        DESCRIPTION.
    depths_df : dataframe
        DESCRIPTION.
    no_layers : int
        DESCRIPTION.

    Returns
    -------
    EFD_input_df : dataframe
        DESCRIPTION

    """
    
    # index = basin and "type" (i.e. ds_basin, ups_basin, orifice setting)
    # columns = layer no (=no_layers)
    basins_names = [basins[int(i)] for i in (np.floor(np.arange(0,len(basins)*3)/3))]
    index = pd.MultiIndex.from_tuples(zip(basins_names,('ds_level', 'ups_level','orifice_setting')*len(basins)))
    EFD_input_df = pd.DataFrame((np.zeros((len(basins)*3,no_layers))),index = index)
    
# Possible Default setups:
    
    # 1
    # Zones are evenly distributed in the entire basin
    # for i,basin in enumerate(basins):
    #     depth = depths_df.loc[basin][0]
    #     EFD_input_df.loc[basin,'ds_level'] = np.linspace(0,depth,no_layers) 
    #     EFD_input_df.loc[basin,'ups_level'] = np.linspace(0,depth,no_layers) 
    #     EFD_input_df.loc[basin,'orifice_setting'] = np.linspace(1,0,no_layers) 
    
    # 2
    # Evenly distributed zones in the top 1/3 of all the basins.
    # for i,basin in enumerate(basins):
    #     depth = depths_df.loc[basin][0]
    #     EFD_input_df.loc[basin,'ds_level'] = np.linspace(depth - depth/3,depth,no_layers) 
    #     EFD_input_df.loc[basin,'ups_level'] = np.linspace(depth - depth/3,depth,no_layers) 
    #     EFD_input_df.loc[basin,'orifice_setting'] = np.linspace(1,0,no_layers) 
    # EFD_input_df.head()
    
    # 3     
    # Zones split at 84%, 90% and 100%.
    # Orifice settings at 0%, 30% and 100% 
    for i,basin in enumerate(basins):
        EFD_input_df.loc[basin,'ds_level'] = np.array([0.84,0.9,1])*depths_df.loc[basin][0]
        EFD_input_df.loc[basin,'ups_level'] = np.array([0.84,0.9,1])*depths_df.loc[basin][0]
        EFD_input_df.loc[basin,'orifice_setting'] = np.array([1,0.3,0])
    
    return EFD_input_df



# =============================================================================
# Functions used for the optimization
# =============================================================================
  
def parameterization_2D(c1,c2,ups_v,ds_v):
    """
    12/2-2021
    
    Computes a continuous function in the (x,y,z)-plane based on 2 input parameters (c1, and c2). 
    Based on this the orifice setting can be determined from the filling degrees of the 2 relevant basins. 
    Dimensions are:
        x: filling degree of upstream basin
        y: filling degree of downstream basin
        z: orifice setting between the two basins (0=closed, 1=open)
        
    vectorized using: 
        ups_v,ds_v = np.meshgrid(x,y)
    
    Parameters
    ----------
    c1 : float
        Parameter 1. 
        Determines the slope of the curve
    c2 : float
        Parameter 2.
        Determines the shift in the horizontal direction. 
    ups_v : array 
        The filling degree in the upstream basin. Vectorized format computed from np.meshgrid
    ds_v : array
        The filling degree in the downstream basin. Vectorized format computed from np.meshgrid 
        
    Returns
    -------
    z : float
        z-values of the control function. Same dimensions as the input
    """
    
    # vectorized parameterizaton. Based  on inputs from np.meshgrid
    z = 1/(1+np.exp(-c1*(ups_v-ds_v)-c2))
    return z

def contour_plot(Z,discrete_grid = False, discrete_x= None ,discrete_y = None):
    """
    Make a contour plot of the setting. This is only used for documentartion 
    """
    # compute z for the contour plot
    delta = 0.025
    x = np.arange(-1.0, 1.0, delta)
    y = np.arange(-1.0, 1.0, delta)
    X, Y = np.meshgrid(x, y)
    # Z1 = np.exp(-X**2 - Y**2)
    # Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    # Z = (Z1 - Z2) * 2
    Z = Z
    
    # fig, ax = plt.subplots()
    # CS = ax.contour(X, Y, Z)
    # ax.clabel(CS, inline=1, fontsize=10)
    # ax.set_title('Simplest default with labels')
    
    import matplotlib.cm as cm
    fig, ax = plt.subplots()
    im = ax.imshow(Z, interpolation='bilinear', origin='lower',
                    cmap=cm.gray, extent=(0, 1, 0, 1))
    levels = np.arange(-1.2, 1.6, 0.2)
    CS = ax.contour(Z, levels, origin='lower', cmap='flag',
                    linewidths=2, extent=(0, 1, 0, 1))
    
    # Thicken the zero contour.
    # zc = CS.collections[6]
    # plt.setp(zc, linewidth=4)
    
    # ax.clabel(CS, levels[1::2],  # label every second level
    #           inline=1, fmt='%1.1f', fontsize=14)
    
    # # make a colorbar for the contour lines
    # CB = fig.colorbar(CS, shrink=0.8, extend='both')
    
    ax.set_title('Orifice setting [0,1]')
    ax.set_xlabel('Filling degree in upsteram basin ')
    ax.set_ylabel('Filling degree in downstream basin ')
    # We can still add a colorbar for the image, too.
    CBI = fig.colorbar(im, orientation='horizontal', shrink=0.8)
    
    # This makes the original colorbar look a bit out of place,
    # so let's improve its position.
    
    # l, b, w, h = ax.get_position().bounds
    # ll, bb, ww, hh = CB.ax.get_position().bounds
    # CB.ax.set_position([ll, b + 0.1*h, ww, h*0.8])
    
    if discrete_grid == True:
        for i in range(len(discrete_x)):
            # horizontal lines
            plt.plot((discrete_y[i],discrete_y[i]),(0,1),color = 'g') 
            plt.plot((0,1),(discrete_x[i],discrete_x[i]),color = 'g') 
            plt.legend(['Discrete\nzones'],loc = 'lower right')
    
    
    plt.tight_layout()
    plt.show()


"""
18/11/2020

Writes an approximated EFD to a SWMM model with basins compared only to the one just downstream of the basin. 
Gives flexibility to optimize each pair of basins independnet of the others and allows more layes in the basins to be defined.

Assumes that all tanks are connected to one orifice 
and that they are pairs listed in same order
"""

# The purpose of the function os to write the optimal settings for x into SWMM.
# inp is the input file that is used
def EFD_to_SWMM(x,inp,timestamp,filename,basin_structure,basins,actuators,depths_df,no_layers):    
    """
    Input: dataframe with rows = no_basins * 3 and columns = no_layers 
    the Control section is written based on this dataframe. The end goal is to adjust the dataframe and thus optimize the RTC.
    
    Also runs the SWMM files and reads the output. Should not be included in the final function
    """
    
    no_layers = no_layers + 1 
            
                
    count = 0
    for actuator_num, ups_basin in enumerate(basins): # The upstream basin
        for ds_basin in basins: # The downstream basin
            if basin_structure.loc[ds_basin,ups_basin] == 1 :
                actuator = actuators[actuator_num]
                # pdb.set_trace()
                # Activation depths or "zones". Can be optimized 
                # ups and ds levels are defined. A value that is above depth is added to make rules if flooded. 
                ups_levels_tmp = np.asarray(x.loc[ups_basin,'ups_level'])
                ups_levels = np.concatenate((np.array([0]),ups_levels_tmp,np.array(depths_df.loc[ups_basin]+1))) # adding 0 at the bottom and 1 meter to the max depth to ensure that this is never exceeded. 
                
                ds_levels_tmp = np.asarray(x.loc[ds_basin,'ds_level'])
                ds_levels = np.concatenate((np.array([0]),ds_levels_tmp,np.array(depths_df.loc[ups_basin]+1))) # adding 0 at the bottom and 1 meter to the max depth to ensure that this is never exceeded. 

                # Actuator settings 
                # Can be changed and optimized. 
                orifice_settings_tmp = x.loc[ups_basin,'orifice_setting']
                orifice_settings = pd.concat([orifice_settings_tmp,pd.DataFrame([0])[0]], ignore_index=True) # Add 1 to the first orficie settings 
                # print(orifice_settings)
                # pdb.set_trace()


                # Suffix for basins in rule names 
                levels_notation = ['level_{}'.format(i) for i in range(0,no_layers)]
                
                from swmmio.utils.modify_model import replace_inp_section
                # pdb.set_trace()
                n = 7 # number of lines required for each rule 
                rules = no_layers**2 # rules per set of basins. 
                
                # counter to count the rule number per basin pair
                rules_count  = 0
                Temp_controls_section = pd.DataFrame({'[CONTROLS]':np.zeros(n)})
                # Loops to write the rules in SWMM format. 
                for i in range(0,no_layers): # ups basin
                    for j in range(0,no_layers): # ds basin
                        # Rules are defined
                        Temp_controls_section['[CONTROLS]'][0] = 'RULE {}_{}_AND_{}_{}'.format(ups_basin,levels_notation[i],ds_basin,levels_notation[j])
                        Temp_controls_section['[CONTROLS]'][1] = 'IF NODE {} DEPTH > {}'.format(ups_basin,ups_levels[i])
                        Temp_controls_section['[CONTROLS]'][2] = 'AND NODE {} DEPTH < {}'.format(ups_basin,ups_levels[i+1])
                        Temp_controls_section['[CONTROLS]'][3] = 'AND NODE {} DEPTH > {}'.format(ds_basin,ds_levels[j])
                        Temp_controls_section['[CONTROLS]'][4] = 'AND NODE {} DEPTH < {}'.format(ds_basin,ds_levels[j+1])
                        # Orifice setting is based on the difference in level in upsream and downstream. 
                        # If i-j > 0: Water level ups > water level ds => open orifice
                        # If i-j = 0: Water level ups = water level ds => ? 
                        # If i-j < 0: Water level ds > water level ds => close orifice
                        # pdb.set_trace()
                        if i<j: 
                            orifice_setting = orifice_settings[j-i]
                        elif i==j:
                            orifice_setting = orifice_settings[0]
                        elif i>j:
                            orifice_setting = orifice_settings[0]
                            # print(i,j,orifice_setting)
                        Temp_controls_section['[CONTROLS]'][5] = 'THEN ORIFICE {} SETTING = {}'.format(actuator,orifice_setting)
                        # Temp_controls_section['[CONTROLS]'][6] = 'ELSE ORIFICE {} SETTING = 1.0'.format(actuator)
                        Temp_controls_section['[CONTROLS]'][6] = ''
                    
                        # Rule is appended to the ruleset for that basin pair. 
                        if rules_count == 0:
                            Rules_controls_section = Temp_controls_section.copy()
                        else:
                            Rules_controls_section = Rules_controls_section.append(Temp_controls_section,ignore_index=True)
                        rules_count += 1
            
                # The section for each pair of basins is added to the full control setion. 
                if count == 0:
                    Full_controls_section = Rules_controls_section.copy()
                else:
                    Full_controls_section = Full_controls_section.append(Rules_controls_section,ignore_index=True)
                
                count += 1
                
    # Control section is replaced in the .inp file
    old_file = inp.model_dir +'/'+ inp.model_name + '.inp'
    new_file = inp.model_dir +'/'+ inp.model_name + filename + '.inp'
    shutil.copyfile(old_file,new_file)
    # Overwrite the CONTROLS section of the new model with the adjusted data
    with pd.option_context('display.max_colwidth', 400): # set the maximum length of string to prit the full string into .inp
        replace_inp_section(new_file, '[CONTROLS]', Full_controls_section)
    print('EFD algorithm written to {}{} with settings\n{}'.format(inp.model_name,filename,x))
    

#%% Manual / analytic solution to the parameterization
def custom_EFD(c1,c2,basins,depths_df,no_layers):
    """
    

    Parameters
    ----------
    c1 : TYPE
        DESCRIPTION.
    c2 : TYPE
        DESCRIPTION.
    basins : TYPE
        DESCRIPTION.
    depths_df : TYPE
        DESCRIPTION.
    no_layers : TYPE
        DESCRIPTION.

    Returns
    -------
    EFD_input_df : TYPE
        DESCRIPTION.

    """
    
    # The continous parameterized version is created
    x = np.arange(0,1.0,1/100)
    ups_v,ds_v = np.meshgrid(x,x)
    parameterization = parameterization_2D(c1,c2, ups_v,ds_v)
    # the space is plotted.     
    # contour_plot(parameterization)

    # Orifice settings are set to 0.99 and 0.01 instead of 1 and 0 for numerical stability. 
    # These are changed to allow the orifice to be fully open and fully closed later. 
    orifice_settings = np.linspace(0.99,0.01,no_layers)    
    
    # Two equations with two unknowns: 
        # eq1: z = 1/(1+np.exp(-c1*(ups_v-ds_v)-c2)) # z is the orifice setting, c1 and c2 are given from the optimization. 
    
        # eq2: ups_v = 1 - ds_v # This is because the parameterization is summetric and all iso-lines are at a 45 degree angle. 
        
        # The two unknows are ups_v and ds_v that is calculated as:
    
        # ds = (c1+c2+np.log(-(z-1)/z))/(2*c1)
        # ups = (-c1+c2+np.log(-(z-1)/z))/(2*c1)
        
    
    # finding upstream and downstream index of this orifice setting
    # ups_idx = np.where(ups_v[0,:] == np.abs(round(ups,2)))[0][0]
    # ds_idx = np.where(ds_v[:,0] == np.abs(round(ds,2)))[0][0]
    
    # print(parameterization[ds_idx,ups_idx])
     
    # upper and lower limits are set to unsure that the parameterization covers all orifice settings. 
    # If only some are covered it is not correct to assume that the orifice should close completely 
    upper_limit = 0.9
    lower_limit = 0.1
    if np.logical_and(np.max(parameterization) > upper_limit,np.min(parameterization) < lower_limit):
        print('true')
    
        ds_index_tmp = np.zeros(no_layers)
        ups_index_tmp = np.zeros(no_layers)
        # finds the analytical solution for each orifice setting
        for i in range(no_layers):
            z = orifice_settings[i]
            ds = (c1+c2+np.log(-(z-1)/z))/(2*c1)
            ups = (-c1+c2+np.log(-(z-1)/z))/(2*c1)
            # print(ds)
            # maximum opening is 1 (0.99 is due to pythonic indexing) 
            if ds > 1.0:
                ds = 0.99
            if ups > 1.0:
                ups = 0.99
            # finding upstream and downstream index of this orifice setting
            # ds_idx = np.where(ds_v[:,0] == np.abs(round(ds,2)))[0]
            # print(ds)
            ds_idx = int(np.abs(ds*100))
            # ups_idx = np.where(ups_v[0,:] == np.abs(round(ups,2)))[0][0]
            ups_idx = int(np.abs(ups*100))
            # print('ds_idx: ',ds_idx,'ups_idx: ',ups_idx)
            # print('z: ',z) 
            # print('param: ',parameterization[ds_idx,ups_idx])
            ds_index_tmp[i] = ds_idx
            ups_index_tmp[i] = ups_idx
            
            # Converted into integers 
            ds_index_int = [int(ds_index_tmp[i]) for i in range(no_layers)] # convert to integer
            ups_index_int = [int(ups_index_tmp[i]) for i in range(no_layers)] # convert to integer
            
            # ds_index and ups_index is now used to compute the borders between the zones that can be written to SWMM.
            
            # 1 is added as the maximum orifice setting 
            ds_values_tmp = np.append(ds_v[ds_index_int,0],1) # This marks the center of each "zone"
            ups_values_tmp = np.append(1,ups_v[0,ups_index_int]) # This marks the center of each "zone"
            
            # Borders between the zones are defined. 
            ds_values = [round((ds_values_tmp[i+1]-ds_values_tmp[i])/2 + ds_values_tmp[i],2) for i in range(no_layers)] # This marks the borders between the zones 
            ups_values = [round((ups_values_tmp[i]-ups_values_tmp[i+1])/2 + ups_values_tmp[i+1],2) for i in range(no_layers)] # This marks the borders between the zones 
            
        # Fully open and closed is set (were 0.01 and 0.99 for numerical reasons)
        orifice_settings[0] = 1
        orifice_settings[-1] = 0

        # The orders are added to the parameterized figure
        # contour_plot(parameterization,discrete_grid = True, discrete_x = ds_values, discrete_y = ups_values)


        # The EFD dataframe is written with the borders determined earlier. 
        basins_names = [basins[int(i)] for i in (np.floor(np.arange(0,len(basins)*3)/3))]
        index = pd.MultiIndex.from_tuples(zip(basins_names,('ds_level', 'ups_level','orifice_setting')*len(basins)))
        EFD_input_df = pd.DataFrame((np.zeros((len(basins)*3,no_layers))),index = index)
        # pdb.set_trace()
        for i,basin in enumerate(basins):
            EFD_input_df.loc[basin,'ds_level'] = np.array(ds_values)*depths_df.loc[basin][0]
            EFD_input_df.loc[basin,'ups_level'] = np.array(ups_values)*depths_df.loc[basin][0]
            EFD_input_df.loc[basin,'orifice_setting'] = np.array(orifice_settings)
            
            # print(EFD_input_df)
    else: 
        # Do something if the parameterization is not good enough (i.e. not covering enough orifice settings)
        print('false')
        # EFD_input_df = None
    

    return EFD_input_df

    # filling_degrees_ups = ups_values
    # filling_degrees_ds = ds_values
    
    
    # vec = np.arange((no_layers)*2)
    # vec = vec/vec.max()
    # orifice_settings_disc = np.array([vec[i:i+no_layers] for i in range(no_layers)])
    
    
    # def discretize_parameters(filling_degrees_ups,filling_degrees_ds,orifice_settings,no_layers):
    # """
    # 12/2 -2021   
    # Computes a discrete version of the (x,y,z)-plane of the orifice settings. 
    # Based on this the orifice setting can be determined from the filling degrees of the 2 relevant basins. 
    # Dimensions are:
    #     x: filling degree of upstream basin
    #     y: filling degree of downstream basin
    #     z: orifice setting between the two basins (0=closed, 1=open)
    
    # Parameters
    # ----------
    # filling_degrees_ups : array
    #     The filling degrees in the upstream basin that defines the treshold between different orifice settings. (lines on horizontal axis)
    # filling_degrees_s : array
    #     The filling degrees in the downstream basin that defines the treshold between different orifice settings. (lines on vertical axis)
    # orifice_settings : array
    #     The orifice settings that is determined by filling. Determines how much water can be led out by the orifice (lines on z-axis) 
    #     Dimensions of (no_layers X no_layers) 
    # no_layers : int
    #     Number of layers or "zones" in each basin. Many layers give a more accurate RTC setup but increases the number of rules that are to be written in SWMM.
    
    # Returns
    # -------
    # y : array
    #     The filling degrees in discretized format that can be compared with the continous (parameterized) function. 
    # """
        
    # M = (np.array(filling_degrees_ups)*100).astype(np.int32)
    # N = (np.array(filling_degrees_ds)*100).astype(np.int32)
    # # Creating the submatrix that can be merged 
    # def submatrix(m,n,val):
    #     mat = np.repeat(val,m*n).reshape((m,n))
    #     return mat 
    # # pdb.set_trace()
    # # Submatrices are created 
    # # pdb.set_trace()
    # submats = np.zeros((no_layers,no_layers), dtype=object)
    # sub_tmp = np.zeros((no_layers,no_layers), dtype=object)
    # for m in range(no_layers):
    #     for n in range(no_layers):
    #         # print(m,n)
    #         # pdb.set_trace()
    #         # submats[no_layers-(m),n] = submatrix(M[m-1]-M[m],N[n]-N[n-1],orifice_settings[m,n])
    #         if m ==no_layers-1:
    #             if n == 0:
    #                 submats[-m,n] = submatrix(M[m],N[n]-N[n],orifice_settings_disc[m,n])
    #             else:
    #                 submats[-m,n] = submatrix(M[m],N[n]-N[n-1],orifice_settings_disc[m,n])
    #         else:
    #             if n == 0:
    #                 submats[-m,n] = submatrix(M[m]-M[m+1],N[n]-N[n],orifice_settings_disc[m,n])
    #             else:
    #                 submats[-m,n] = submatrix(M[m]-M[m+1],N[n]-N[n-1],orifice_settings_disc[m,n])
    #         print(m,n,orifice_settings_disc[m,n])
    #         sub_tmp[-m,n] = (np.mean(submats[-m,n]))
    # # merging the submatrices into one big matrix with size (steps X steps) 
    # # pdb.set_trace()
    # submats_merged = np.concatenate([np.concatenate(submats[i,],axis = 1) for i in range(no_layers)],axis=0)
    # # return submats_merged 
    # submats_merged 
    # contour_plot(submats_merged) 
    # no_layers
    
    # discrete = discretize_parameters(ups_values,ds_values,orifice_settings_disc,no_layers)


if __name__ == "__main__":
    equal_filling_degree('Astlingen.ini')
