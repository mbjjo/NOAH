# -*- coding: utf-8 -*-
"""    


This is the main function for the Parameterized Control Function. 
It takes input from the config file (either from the GUI or a text file) 
and finds the near-optimal set of if-else-rules and saves these to a SWMM file. 
"""
# imports
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
import SWMMControlFunction



def main_script(config_file):       
    # All input data is found in the conifguration file
    
    # All variables are stored as inp._VarName_ and can be used in functions. 
    inp = Parameters.Read_Config_Parameters(config_file)
    
    # Rewriting file if timeseries are not in the right folder
    model_inp = SWMMControlFunction.redefine_time_series(inp) 
    print(model_inp)
    # Reads input from the .inp file.  
    basins = [x.strip(' ') for x in inp.Control_basins.split(',')]
    actuators = [x.strip(' ') for x in inp.Control_actuators.split(',')]
    depths = [float(x.strip(' ')) for x in inp.basin_depths.split(',')]
    depths_df = pd.DataFrame(depths,index = basins) # written into a dataframe
    CSO_ids = [x.strip(' ') for x in inp.Control_CSO_id.split(',')]
    no_layers = inp.no_layers
    
    # creating directory for the two output files. Only the final files are saved. 
    # These are stored in /NOAH_RTC_Tool/output/_Timestamp_
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.mkdir('../output/'+timestamp)   
    filename = '_Control'
    model_rptfile = '../output/' + timestamp + '/' + inp.model_name + filename + '.rpt'
    model_outfile = '../output/' + timestamp + '/' + inp.model_name + filename + '.out'
    # new file is the file that the control is written to and that SWMM should run
    new_file= str(model_inp.split('.')[0] + filename + '.' + model_inp.split('.')[1])
    
    
    # The basin structure is defined 
    if 'Astlingen' in inp.model_name:
        basin_structure = SWMMControlFunction.create_basin_structure(basins)
    else:
        print('Basin structure is not implemented for this model. Compute that manually in the function create_basin_structure() in SWMMControlFunction.py')
        
    # Creating pairs of basins from the basins_structure dataframe
    basin_pairs = []
    for ds_basin in basins:
        for ups_basin in basins:
            if basin_structure[ups_basin][ds_basin] ==1:
                basin_pairs.append([ups_basin,ds_basin]) 
    
    
    if inp.Control_setup_type == 'default': # if default setups are tried out
        no_layers = 3
        
        # The dataframe with the Control setup (i.e. orifice settings and thresholds between zones) that should be computed to SWMM is created.
        Input_df = SWMMControlFunction.create_default_input_df(basins,depths_df,no_layers,inp.Default_setup_selection)
        
        # A SWMM file is written with the selected rules
        SWMMControlFunction.Control_to_SWMM(Input_df,inp,timestamp,model_inp,new_file,basin_structure,basins,actuators,depths_df,no_layers)
     
        # Compute the SWMM simulation 
        SWMMControlFunction.run_SWMM(new_file,model_rptfile,model_outfile)
    
        CSO_stats_df = CSO_statistics.Compute_CSO_statistics(inp,CSO_ids, model_outfile)
        print(CSO_stats_df.head())
        print(CSO_stats_df['Volume'].sum())
    
    elif inp.Control_setup_type == 'optimized':
        
        print('optimizing...')
        
        # =============================================================================
        # Optimization routine
        # The optimization routine can be computed here or called in a seperate function
        # =============================================================================
        
        
        # Running optimizer for all metaparameters at once
        # start_value = [1,1,1,1,1,5,5,5,5,5]
        # result = optimize.minimize(fun = optimizer_wrapper_all,
        #                   args = (inp,timestamp,new_file,
        #                   basin_structure,basin_pairs,basins,actuators,depths_df,no_layers,CSO_ids,
        #                   model_inp,model_rptfile,model_outfile),
        #                   x0 = start_value, method='Nelder-Mead',
        #                   options = {'disp':True})
        
        # Running optimizer for only one set of meta parameters that are identical for all basins (i.e. 1 alpha and 1 beta)
        start_value = [0,5] # [alpha,beta] 
        result_one_set = optimize.minimize(fun = optimizer_wrapper_one_set,
                          args = (inp,timestamp,new_file,
                          basin_structure,basin_pairs,basins,actuators,depths_df,no_layers,CSO_ids,
                          model_inp,model_rptfile,model_outfile),
                          x0 = start_value, method='Nelder-Mead',
                          options = {'disp':True})
        print('Result of the one-set optimizer: ',result_one_set.x)
        
        # Running optimizer for only alpha
        start_value = np.repeat(result_one_set.x[0],len(basin_pairs))
        betas = np.repeat(result_one_set.x[1],len(basin_pairs))
        
        result_alpha = optimize.minimize(fun = optimizer_wrapper_alpha,
                          args = (betas,inp,timestamp,new_file,
                          basin_structure,basin_pairs,basins,actuators,depths_df,no_layers,CSO_ids,
                          model_inp,model_rptfile,model_outfile),
                          x0 = start_value, method='Nelder-Mead',
                          options = {'disp':True})
        
        print('Results of the alpha optimizer: ',result_alpha.x)
        
        # Running optimizer for only Beta
        start_value = np.repeat(result_one_set.x[1],len(basin_pairs))
        alphas = result_alpha.x
      
        result_beta = optimize.minimize(fun = optimizer_wrapper_beta,
                          args = (alphas, inp,timestamp,new_file,
                          basin_structure,basin_pairs,basins,actuators,depths_df,no_layers,CSO_ids,
                          model_inp,model_rptfile,model_outfile),
                          x0 = start_value, method='Nelder-Mead',
                          options = {'disp':True})
        
        
        print('Results of the beta optimizer: ',result_beta.x)
    
    
    
    
def optimizer_wrapper_all(metaparameters,inp,timestamp,new_file,
                          basin_structure,basin_pairs,basins,actuators,depths_df,no_layers,CSO_ids,
                          model_inp,model_rptfile,model_outfile):
    alphas = metaparameters[:5]
    betas = metaparameters[5:]
    
    print(alphas,betas)

    # Compute discretized if-else rules from meta parameters. 
    Input_df  = SWMMControlFunction.custom_parameterization(alphas,betas,basin_structure,basin_pairs,basins,depths_df,no_layers)
        
    # Write controls to the SWMM file 
    SWMMControlFunction.Control_to_SWMM(Input_df,inp,timestamp,model_inp,new_file,basin_structure,basins,actuators,depths_df,no_layers)
    
    # Run SWMM file
    SWMMControlFunction.run_SWMM(new_file,model_rptfile,model_outfile)
   
    # objective function is computed
    CSO_stats_df = CSO_statistics.Compute_CSO_statistics(inp,CSO_ids, model_outfile)
    # print(CSO_stats_df.head())
    total_volume = CSO_stats_df['Volume'].sum()
    print('total volumne:', total_volume)

    # Writing to Excel
    # global row_no
    # for i in range(5):
    #     my_sheet.write(row_no,i,c1s[i])
    #     my_sheet.write(row_no,i+5,c2s[i])
    # my_sheet.write(row_no,10,total_volume)
    # row_no += 1 
    
    
    # my_xls.save("Convergence.xls")
    return total_volume


def optimizer_wrapper_one_set(metaparameters,inp,timestamp,new_file,
                          basin_structure,basin_pairs,basins,actuators,depths_df,no_layers,CSO_ids,
                          model_inp,model_rptfile,model_outfile):
    # using the same parameters for all basins:
    alphas = np.repeat(metaparameters[0],len(basin_pairs)) 
    betas =np.repeat(metaparameters[1],len(basin_pairs))

    print(alphas,betas)

    # Compute discretized if-else rules from meta parameters. 
    Input_df  = SWMMControlFunction.custom_parameterization(alphas,betas,basin_structure,basin_pairs,basins,depths_df,no_layers)
        
    # Write controls to the SWMM file 
    SWMMControlFunction.Control_to_SWMM(Input_df,inp,timestamp,model_inp,new_file,basin_structure,basins,actuators,depths_df,no_layers)
    
    # Run SWMM file
    SWMMControlFunction.run_SWMM(new_file,model_rptfile,model_outfile)
   
    # objective function is computed
    CSO_stats_df = CSO_statistics.Compute_CSO_statistics(inp,CSO_ids, model_outfile)
    # print(CSO_stats_df.head())
    total_volume = CSO_stats_df['Volume'].sum()
    print('total volumne:', total_volume)

    # Writing to Excel
    # global row_no
    # for i in range(5):
    #     my_sheet.write(row_no,i,c1s[i])
    #     my_sheet.write(row_no,i+5,c2s[i])
    # my_sheet.write(row_no,10,total_volume)
    # row_no += 1 
    
    
    # my_xls.save("Convergence.xls")
    return total_volume


def optimizer_wrapper_alpha(alphas, betas, inp,timestamp,new_file,
                          basin_structure,basin_pairs,basins,actuators,depths_df,no_layers,CSO_ids,
                          model_inp,model_rptfile,model_outfile):
    """
    Only optimizing alphas
    """
    # Compute discretized if-else rules from meta parameters. 
    Input_df  = SWMMControlFunction.custom_parameterization(alphas,betas,basin_structure,basin_pairs,basins,depths_df,no_layers)
        
    # Write controls to the SWMM file 
    SWMMControlFunction.Control_to_SWMM(Input_df,inp,timestamp,model_inp,new_file,basin_structure,basins,actuators,depths_df,no_layers)
    
    # Run SWMM file
    SWMMControlFunction.run_SWMM(new_file,model_rptfile,model_outfile)
   
    # objective function is computed
    CSO_stats_df = CSO_statistics.Compute_CSO_statistics(inp,CSO_ids, model_outfile)
    # print(CSO_stats_df.head())
    total_volume = CSO_stats_df['Volume'].sum()
    print('total volumne:', total_volume)

    # Writing to Excel
    # global row_no
    # for i in range(5):
    #     my_sheet.write(row_no,i,c1s[i])
    #     my_sheet.write(row_no,i+5,c2s[i])
    # my_sheet.write(row_no,10,total_volume)
    # row_no += 1 
    
    
    # my_xls.save("Convergence.xls")
    return total_volume

def optimizer_wrapper_beta(betas, alphas, inp,timestamp,new_file,
                          basin_structure,basin_pairs,basins,actuators,depths_df,no_layers,CSO_ids,
                          model_inp,model_rptfile,model_outfile):
    """
    Only optimizing betas
    """
    # Compute discretized if-else rules from meta parameters. 
    Input_df  = SWMMControlFunction.custom_parameterization(alphas,betas,basin_structure,basin_pairs,basins,depths_df,no_layers)
        
    # Write controls to the SWMM file 
    SWMMControlFunction.Control_to_SWMM(Input_df,inp,timestamp,model_inp,new_file,basin_structure,basins,actuators,depths_df,no_layers)
    
    # Run SWMM file
    SWMMControlFunction.run_SWMM(new_file,model_rptfile,model_outfile)
   
    # objective function is computed
    CSO_stats_df = CSO_statistics.Compute_CSO_statistics(inp,CSO_ids, model_outfile)
    # print(CSO_stats_df.head())
    total_volume = CSO_stats_df['Volume'].sum()
    print('total volumne:', total_volume)

    # Writing to Excel
    # global row_no
    # for i in range(5):
    #     my_sheet.write(row_no,i,c1s[i])
    #     my_sheet.write(row_no,i+5,c2s[i])
    # my_sheet.write(row_no,10,total_volume)
    # row_no += 1 
    
    
    # my_xls.save("Convergence.xls")
    return total_volume



# Running function only if script is run. 
if __name__ == "__main__":
    config_file = 'Astlingen_SWMM.ini'
    main_script(config_file )
