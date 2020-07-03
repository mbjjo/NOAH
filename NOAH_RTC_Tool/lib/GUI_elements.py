# -*- coding: utf-8 -*-
"""
Copyright 2019 Magnus Johansen
Copyright 2019 Technical University of Denmark

This file is part of NOAH RTC Tool.

NOAH RTC Tool is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

NOAH RTC Tool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with NOAH RTC Tool. If not, see <http://www.gnu.org/licenses/>.
"""


# Required imports created for the GUI 
import pyswmm_Simulation
import noah_calibration_tools
from noah_calibration_tools import swmm_simulate, interpolate_swmm_times_to_obs_times, change_model_property, simulate_objective, create_new_model, create_runobjective_delete_model, generate_run_lhs, run_simplex_routine
from model_structure_analysis import swmm_model_inventory, backwards_network_trace
from User_defined_objective_function import User_defined_objective_function

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

import tkinter as tk
from tkinter import *
from tkinter import Tk, ttk, filedialog, scrolledtext,messagebox
from tkinter import messagebox as msg
from tkintertable.Tables import TableCanvas
from tkintertable.TableModels import TableModel


# # =============================================================================
# Write input from GUI to configuration file        
def write_config(self):
    
    config = configparser.ConfigParser()
    config['DEFAULT'] = {}
    
    config['Settings'] = {
                          'System_Units': swmmio.swmmio.inp(self.param.model_dir.get() + '/' + self.param.model_name.get() + '.inp').options.Value.FLOW_UNITS, 
                          'Reporting_timesteps': swmmio.swmmio.inp(self.param.model_dir.get() + '/' + self.param.model_name.get() + '.inp').options.Value.REPORT_STEP,
                          'Time_seperating_CSO_events':self.CSO_event_seperation.get(),
                          'Max_CSO_duration':self.CSO_event_duration.get(),
                          'CSO_type':self.CSO_type.get()
                          }
    
    config['Model'] = {'Modelname':self.param.model_name.get(),
                       'Modeldirectory':self.param.model_dir.get(),                       
                      'Rain series':''
                      }
    
    config['RuleBasedControl'] = {
                                  'actuator_type':self.param.actuator_type.get(),
                                  'sensor1_id':self.sensor1id.get(),
                                  'sensor1_critical_depth':self.sensor1setting.get(),
                                  'actuator1_id':self.actuator1id.get(),
                                  'actuator1_target_setting_True':self.actuator1setting_True.get(),
                                  'actuator1_target_setting_False':self.actuator1setting_False.get(),
                                  'sensor1_critical_depth_dryflow':self.sensor1setting_dry.get(),
                                  'actuator1_target_setting_True_dryflow':self.actuator1setting_True_dry.get(),
                                  'actuator1_target_setting_False_dryflow':self.actuator1setting_False_dry.get(),
                                   'raingage1':self.raingage1.get(),
                                   'rainfall_threshold_value':self.rainfall_threshold.get(),
                                   'rainfall_threshold_duration':self.rainfall_time.get()
                                  }
    
    config['Optimization'] = {'UseOptimization':self.param.UseOptimization.get(),
                              'optimization_method':self.opt_method.get(),
                              'CSO_objective':self.param.CSO_objective.get(),
                              'CSO_id1':self.CSO_id1.get(),
                              'CSO_id2':self.CSO_id2.get(),
                              'CSO_id3':self.CSO_id3.get(),
                              'Custom_CSO_ids':self.Custom_CSO_ids.get(),
                              'Optimized_parameter':self.optimized_parameter.get(),
                              'expected_min_Xvalue':self.expected_min_Xvalue.get(),
                              'expected_max_Xvalue':self.expected_max_Xvalue.get(),
                              'max_initial_iterations':self.max_initial_iterations.get(),
                              'max_iterations_per_minimization':self.max_iterations_per_minimization.get()                              
                              }    
    
    config['Calibration'] = {'Calibrate_perc_imp':self.param.Calib_perc_imp.get(),
                             'Percent_imp_min':self.percent_imp_min.get(),
                             'Percent_imp_max':self.percent_imp_max.get(),
                             'Calibrate_width':self.param.Calib_width.get(),
                             'Width_min':self.Width_min.get(),
                             'Width_max':self.Width_max.get(),
                             'Calibrate_initial_loss':self.param.Calib_Dstore.get(),
                             'Initial_loss_min':self.Dstore_min.get(),
                             'Initial_loss_max':self.Dstore_max.get(),
                             'Calibrate_roughness_pipe':self.param.Calib_n_pipe.get(),
                             'Roughness_pipe_min':self.n_pipe_min.get(),
                             'Roughness_pipe_max':self.n_pipe_max.get(),
                             'Observations':self.param.obs_data.get(),
                             'Observations_directory':self.param.obs_data_path.get(),
                             'Calibration_sensor':self.sensor_calib.get(),
                             'Objective_function':self.Cal_section.get(),
                             'Calibration_start':self.Start_calib_time.get(),
                             'Calibration_end':self.End_calib_time.get(),
                             'Use_hotstart':self.param.use_hotstart.get(),
                             'Hotstart_period':self.hotstart_period_h.get(),
                             'Calibratied_area':self.param.Calib_area.get(),
                             'lhs_simulations':self.max_initial_iterations_calib.get(),
                             'Simplex_simulations':self.max_optimization_iterations_calib.get(),
                             'Optimization_method':self.optimization_method_calib.get(),
                             'Output_time_step':self.output_time_step.get(),
                             'Save_file_as':self.save_calib_file.get()
                             }
    
    
    
    # save the file
    config_name = self.param.model_name.get()
    config_path = '../config/saved_configs/'
    with open(config_path + config_name + '.ini','w') as configfile: 
        config.write(configfile)
    msg.showinfo('','Saved to configuration file')
#===================================================================    
        
class parameters(object):
    def __init__(self):
        
        self.model_name = StringVar()
        self.model_dir = StringVar()
        self.Overwrite_config = BooleanVar()
        
        # RTC settings
        self.actuator_type = StringVar()
        
        # Optimization
        self.UseOptimization = BooleanVar()
        self.CSO_objective = StringVar()
        
        self.use_hotstart_sim = BooleanVar()

        # Results
        self.Benchmark_model = IntVar()
        self.results_vol = IntVar()
        self.results_CSO_vol_id = StringVar()
        self.results_freq = IntVar()
        self.results = {}       
        self.optimal_setting = {}
        
        # Calibration
        self.obs_data = StringVar()
        self.obs_data_path = StringVar()
        
        self.use_hotstart = BooleanVar()
        self.Calib_perc_imp = BooleanVar()
        self.Calib_width = BooleanVar()
        self.Calib_Dstore = BooleanVar()
        self.Calib_n_pipe = BooleanVar()
        
        self.Calib_area = StringVar()

# =============================================================================
class Read_Config_Parameters:
    def __init__(self,config_file):
        config = configparser.ConfigParser()
        config.read('../config/saved_configs/'+config_file)
        # try except are only applied on float() lines. This ensures that these can be left blank. 
        # If they are needed in the simulation an error will occur at that point

        self.system_units = config['Settings']['System_Units']
        rpt_step_tmp = config['Settings']['Reporting_timesteps']      
        rpt_step = datetime.strptime(rpt_step_tmp, '%H:%M:%S').time()
        self.report_times_steps = rpt_step.hour*60 + rpt_step.minute + rpt_step.second/60
        try:
            self.CSO_event_seperation = float(config['Settings']['Time_seperating_CSO_events'])
        except ValueError:
            # print('\nWarning: Some fields are left blank or not specified correctly\n')
            pass
        try:
            self.CSO_event_duration = float(config['Settings']['Max_CSO_duration'])
        except ValueError:
            pass
        self.CSO_type = config['Settings']['CSO_type']
        self.model_name = config['Model']['modelname']
        self.model_dir = config['Model']['modeldirectory']

        # Rule Based Control                    
        RBC = config['RuleBasedControl']
        self.actuator_type = RBC['actuator_type']
        self.sensor1_id = RBC['sensor1_id']    
        self.actuator1_id = RBC['actuator1_id']
        try:
            self.sensor1_critical_depth = float(RBC['sensor1_critical_depth']) # It could make sense to leave this blank
            self.actuator1_target_setting_True = float(RBC['actuator1_target_setting_True'])
            self.actuator1_target_setting_False = float(RBC['actuator1_target_setting_False'])
        except ValueError:
            # print('Parameters for the rule based control are not specified correctly or left blank.')    
            pass
        try:
            self.sensor1_critical_depth_dry = float(RBC['sensor1_critical_depth_dryflow'])
            self.actuator1_target_setting_True_dry = float(RBC['actuator1_target_setting_true_dryflow'])
            self.actuator1_target_setting_False_dry = float(RBC['actuator1_target_setting_false_dryflow'])
        except ValueError:
            # print('Parameters for the rule based control are not specified correctly.')    
            pass
        self.RG1 = RBC['raingage1']
        try:
            self.rainfall_threshold_value = float(RBC['rainfall_threshold_value'])
            self.rainfall_threshold_time = float(RBC['rainfall_threshold_duration'])
        except ValueError:
            pass
        
        # RTC Optimization
        Optimization = config['Optimization']
        self.useoptimization = bool(Optimization['useoptimization'])
        self.optimization_method = Optimization['optimization_method']
        self.CSO_objective = Optimization['CSO_objective']
        self.CSO_id1 = Optimization['CSO_id1']
        self.CSO_id2 = Optimization['CSO_id2']
        self.CSO_id3 = Optimization['CSO_id3']
        self.Custom_CSO_ids = Optimization['Custom_CSO_ids']
        self.optimized_parameter = Optimization['optimized_parameter']
        try:
            self.min_expected_Xvalue = float(Optimization['expected_min_xvalue'])
            self.max_expected_Xvalue = float(Optimization['expected_max_xvalue'])
            self.max_initial_iterations = int(Optimization['max_initial_iterations'])
            self.maxiterations = int(Optimization['max_iterations_per_minimization'])
        except ValueError:
            # print('Optimization parameters are not specified correctly or left blank.')
            pass
        
        # Calibration
        Calibration = config['Calibration']
        
        try:
            self.calibrate_perc_imp = eval(Calibration['calibrate_perc_imp'])
            self.percent_imp_min = float(Calibration['percent_imp_min'])
            self.percent_imp_max = float(Calibration['percent_imp_max'])
        except ValueError:
            pass
        try:    
            self.calibrate_width = eval(Calibration['calibrate_width'])
            self.width_min = float(Calibration['width_min'])
            self.width_max = float(Calibration['width_max'])
        except ValueError:
            pass
        try:
            self.Calibrate_initial_loss = eval(Calibration['Calibrate_initial_loss'])
            self.Initial_loss_min = float(Calibration['Initial_loss_min'])
            self.Initial_loss_max = float(Calibration['Initial_loss_max'])
        except ValueError:
            pass
        try:
            self.Calibrate_roughness_pipe = eval(Calibration['Calibrate_roughness_pipe'])
            self.Roughness_pipe_min = float(Calibration['Roughness_pipe_min'])
            self.Roughness_pipe_max = float(Calibration['Roughness_pipe_max'])
        except ValueError:
            pass
        try: 
            self.Use_hotstart = eval(Calibration['Use_hotstart'])
            self.hotstart_period_h = float(Calibration['Hotstart_period'])
        except ValueError:
            pass
        try:
            self.lhs_simulations = int(Calibration['lhs_simulations'])
        except ValueError:
            pass
        try:
            self.Simplex_simulations = int(Calibration['Simplex_simulations'])
        except ValueError:
            pass
        try:
            self.Output_time_step = int(Calibration['Output_time_step'])
        except ValueError:
            pass
        # inpput as strings: 
        self.Observations  = Calibration['Observations']
        self.Observateions_dir = Calibration['Observations_directory']
        self.Calibration_sensor = Calibration['Calibration_sensor']
        self.Objective_function = Calibration['Objective_function']
        self.simulationStartTime = Calibration['Calibration_start']
        self.simulationEndTime = Calibration['Calibration_end']
        self.Calibratied_area = Calibration['Calibratied_area']
        self.Optimization_method = Calibration['Optimization_method']
        self.Save_file_as = Calibration['Save_file_as']

# Not important:
# def Calibrate(self):
    # config_file = self.param.model_name.get() + '.ini'
    # These two lines are required in order to read the configuration file



def Calibrate_with_config(self):
    if self.param.Overwrite_config.get() == True:
        write_config(self) # Writes configuration file 
    
    msg.showinfo('','Running calibration \nSee run status in console window')
    config_file = self.param.model_name.get() + '.ini'
    Calibrate(config_file)

def Calibrate(config_file):
    # All variables are stored as inp._VarName_ and can be used in functions. 
    inp = Read_Config_Parameters(config_file)
    
    try:
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
        
        
    print(model_inp)
    # get parameters to be calibrated: 
    model_param = [inp.calibrate_perc_imp,inp.calibrate_width,inp.Calibrate_initial_loss,inp.Calibrate_roughness_pipe]
    all_parameter_ranges = [[inp.percent_imp_min,inp.percent_imp_max],
                           [inp.width_min,inp.width_max],
                           [inp.Initial_loss_min,inp.Initial_loss_max],
                           [inp.Roughness_pipe_min,inp.Roughness_pipe_max]]
    all_parameter_sections = ["[SUBCATCHMENTS]", "[SUBCATCHMENTS]", "[SUBAREAS]", "[CONDUITS]"] # the section name for each of the parameters (as specified in the .inp file)
    all_parameter_names = ['PercImperv', 'Width', 'S-Imperv', 'ManningN'] # the name of each parameter as defined in the .inp file
    
    num_model_parameters = sum(model_param) #number of model parameters
    parameter_sections = [x for i,x in enumerate(all_parameter_sections) if model_param[i]]
    parameter_names = [x for i,x in enumerate(all_parameter_names) if model_param[i]]
    parameter_ranges = [x for i,x in enumerate(all_parameter_ranges) if model_param[i]]
    
    selected_nodes = [inp.Calibration_sensor]
    selected_links = list()
    selected_subcatchments = list()
    
    # select objective function from input
    if inp.Objective_function == 'RMSE':
        objective_func = RMSE_objective
    elif inp.Objective_function == 'NSE':
        objective_func = NSE_neg_objective
    elif inp.Objective_function == 'MAE':
        objective_func = MAE_objective
    elif inp.Objective_function == 'ARPE':
        objective_func = abs_rel_peak_objective
    elif inp.Objective_function == 'User defined function':
        objective_func = User_defined_objective_function
    
    # Load observations
    data_file_path = inp.Observateions_dir + '/' + inp.Observations + '.csv'
    observations_loaded = pd.read_csv(data_file_path, sep=',')
    observations_loaded['time'] = pd.to_datetime(observations_loaded['time'])
    
    # Create temp folder and temp file name for new SWMM models that are run and deleted again
    
         
    # create folder with the time stamp for the first simulation. All results are stored in this.  
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.mkdir('../output/'+timestamp)   
   
    temp_folder_name = "temp_calibration_folder"
    temp_folder_path = os.path.join('..','output', timestamp, temp_folder_name)
    
    ##################### Calculations ######################
    
    
    ## Get information about model (all nodes, links, subcatchments, rain gauges)
    model_nodes, model_links, model_subs, model_rgs = swmm_model_inventory(model_inp)
    if inp.Calibratied_area == 'all':
        nodes_to_modify, links_to_modify, subs_to_modify = (model_nodes, model_links, model_subs)
    elif inp.Calibratied_area == 'upstream':
        nodes_to_modify, links_to_modify, subs_to_modify = backwards_network_trace(model_nodes, model_links, model_subs, inp.Calibration_sensor)
    
    ### Sample parameters with LHS
    if inp.Optimization_method == 'lhs' or inp.Optimization_method == 'Combined':
        lhs_evaluated_parameters, lhs_objective_values = generate_run_lhs(inp.lhs_simulations, parameter_ranges,
                                                                  model_inp, temp_folder_path,
                                                                  parameter_sections, parameter_names, num_model_parameters,
                                                                  nodes_to_modify, links_to_modify, subs_to_modify,
                                                                  inp.simulationStartTime, inp.simulationEndTime,
                                                                  selected_nodes, selected_links, selected_subcatchments, 
                                                                  inp.Output_time_step, inp.Use_hotstart, inp.hotstart_period_h,
                                                                  observations_loaded,
                                                                  objective_func,
                                                                  only_best = False)

        best_lhs_parameter_set = lhs_evaluated_parameters[np.argmin(lhs_objective_values),:]
        best_lhs_objective_value = np.min(lhs_objective_values)
        
    elif inp.Optimization_method == 'Simplex':
        lhs_evaluated_parameters = []
        lhs_objective_values = []
        best_lhs_parameter_set = list(np.ones(num_model_parameters))
        best_lhs_objective_value = []
    
    if inp.Optimization_method == 'Simplex' or inp.Optimization_method == 'Combined':
        ### Run Scipy's simplex routine
        # Use LHS results - best parameter set - as input to Scipy

        evaluated_simplex_parameter_sets, evaluated_simplex_function_values = run_simplex_routine(best_lhs_parameter_set, 
                                                                      model_inp, temp_folder_path,
                                                                      parameter_sections, parameter_names, parameter_ranges, num_model_parameters,
                                                                      nodes_to_modify, links_to_modify, subs_to_modify,
                                                                      inp.simulationStartTime, inp.simulationEndTime,
                                                                      selected_nodes, selected_links, selected_subcatchments,
                                                                      inp.Output_time_step, inp.Use_hotstart, inp.hotstart_period_h,
                                                                      observations_loaded,
                                                                      objective_func,
                                                                      max_iterations = inp.Simplex_simulations,
                                                                      only_best = False)

        
        best_simplex_parameter_set = evaluated_simplex_parameter_sets[np.argmin(evaluated_simplex_function_values),:]
        best_simplex_objective_value = np.min(evaluated_simplex_function_values)
    
    elif inp.Optimization_method == 'lhs':
        evaluated_simplex_parameter_sets = []
        evaluated_simplex_function_values = []
        best_simplex_parameter_set = []
        best_simplex_objective_value = []
        
        
    all_parameter_names_plot = ['%Imprv', 'Width', 'Initial loss', 'Roughness (pipes)'] # the name of each parameter as defined in the .inp file
    parameter_names_plot = [x for i,x in enumerate(all_parameter_names_plot) if model_param[i]]
  
    create_calibrate_plot(inp.Optimization_method,num_model_parameters,parameter_names_plot,timestamp,
                   lhs_evaluated_parameters,lhs_objective_values,
                   best_lhs_parameter_set,best_lhs_objective_value,
                   evaluated_simplex_parameter_sets,evaluated_simplex_function_values,
                   best_simplex_parameter_set,best_simplex_objective_value)


    # Save calibrated file 
    if inp.Optimization_method == 'lhs':
        multiplying_values = best_lhs_parameter_set
    else: 
        multiplying_values = best_simplex_parameter_set
        
    new_file_path = inp.model_dir + '/' + inp.Save_file_as + '.inp'
    create_new_model(multiplying_values, 
                      model_inp, new_file_path,
                      parameter_sections, parameter_names, num_model_parameters,
                      
                      subs_to_modify, links_to_modify, nodes_to_modify)
    
# A simple RMSE objective function
def RMSE_objective(obs, mod):
    error = mod - obs
    RMSE = np.sqrt(np.mean(np.square(error)))
    return(RMSE)

# The negative Nash-Sutcliffe Efficiency (has to be negative for calibration, as the objective function is minimized)
def NSE_neg_objective(obs, mod):
    error = mod - obs
    benchmark = obs - np.mean(obs)
    NSE_neg = - ( 1 - np.sum(np.square(error)) / np.mean(np.square(benchmark)) )
    return(NSE_neg)

# Mean Absolute Error objective function
def MAE_objective(obs, mod):
    error = mod - obs
    MAE = np.mean(np.abs(error))
    return(MAE)

# Absolute Relative Peak Error
def abs_rel_peak_objective(obs, mod):
    abs_rel_peak_error = abs((max(mod) - max(obs))/max(obs))
    return(abs_rel_peak_error)





def sensor_validation_with_config(self):
    if self.param.Overwrite_config.get() == True:
        write_config(self) # Writes configuration file 
    
    config_file = self.param.model_name.get() + '.ini'
    sensor_validation(config_file)



def sensor_validation(config_file):
    # Loading parameters
    inp = Read_Config_Parameters(config_file)
    
    model_inp = inp.model_dir + '/' + inp.model_name + '.inp'
    selected_nodes = [inp.Calibration_sensor]
    selected_links = list()
    selected_subcatchments = list()
    
    # select objective function from input
    if inp.Objective_function == 'RMSE':
        objective_func = RMSE_objective
    elif inp.Objective_function == 'NSE':
        objective_func = NSE_neg_objective
    elif inp.Objective_function == 'MAE':
        objective_func = MAE_objective
    elif inp.Objective_function == 'ARPE':
        objective_func = abs_rel_peak_objective
    elif inp.Objective_function == 'User defined function':
        objective_func = User_defined_objective_function
     
    # Load observations
    data_file_path = inp.Observateions_dir + '/' + inp.Observations + '.csv'
    observations_loaded = pd.read_csv(data_file_path, sep=',')
    observations_loaded['time'] = pd.to_datetime(observations_loaded['time'])
    # Running simulation with objective function
    single_sim_calib = simulate_objective(model_inp, 
                       inp.simulationStartTime, inp.simulationEndTime, 
                       selected_nodes, selected_links, selected_subcatchments, 
                       inp.Output_time_step, inp.Use_hotstart, inp.hotstart_period_h,
                       observations_loaded,
                       objective_func)
    
    
    
    print('''\nCalculated model-data fit
objective value is:
{:.2f} with the objective function: {}
'''.format(single_sim_calib,inp.Objective_function))
        
        
# =============================================================================
# Tooltip
class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tip_window = None

    def show_tip(self, tip_text):
        "Display text in a tooltip window"
        if self.tip_window or not tip_text:
            return
        x, y, _cx, cy = self.widget.bbox("insert")      # get size of widget
        x = x + self.widget.winfo_rootx() + 25          # calculate to display tooltip 
        y = y + cy + self.widget.winfo_rooty() + 25     # below and to the right
        self.tip_window = tw = Toplevel(self.widget)    # create new tooltip window
        tw.wm_overrideredirect(True)                    # remove all Window Manager (wm) decorations
#         tw.wm_overrideredirect(False)                 # uncomment to see the effect
        tw.wm_geometry("+%d+%d" % (x, y))               # create window size

        label = tk.Label(tw, text=tip_text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

# Tooltip function            
def create_ToolTip(widget, text):
    toolTip = ToolTip(widget)       # create instance of class
    def enter(event):
        toolTip.show_tip(text)
    def leave(event):
        toolTip.hide_tip()
    widget.bind('<Enter>', enter)   # bind mouse events
    widget.bind('<Leave>', leave)
    
# =============================================================================
# Small functions
# =============================================================================

# Define actuator type and whether dry flow is active
def orifice_or_pump(self):
    if self.param.actuator_type.get() == 'orifice':
        state = 'disabled'
    if self.param.actuator_type.get() == 'weir':
        state = 'disabled'
    elif self.param.actuator_type.get() == 'pump':
        state = 'normal'        
    self.sensor1id_dry['state'] = state
    self.sensor1setting_dry['state'] = state
    self.actuator1id_dry['state'] = state
    self.actuator1setting_True_dry['state'] = state
    self.actuator1setting_False_dry['state'] = state
    
# Determine whether several sensor locations can be chosen 
def enable_sensor_location(self):
    if self.optimized_parameter.get() == 'Sensor location' or self.optimized_parameter.get() == 'Sensor location and activation depth':
        self.sensor_loc1['state'] = 'normal'
        self.sensor_loc2['state'] = 'normal'
        self.sensor_loc3['state'] = 'normal'
    elif self.optimized_parameter.get() == 'Activation depth':
        self.sensor_loc1['state'] = 'disabled'
        self.sensor_loc2['state'] = 'disabled'
        self.sensor_loc3['state'] = 'disabled'
    
    
def enable_calib_method(self):
    if self.optimization_method_calib.get() == 'lhs':
        self.max_initial_iterations_calib['state'] = 'normal'
        self.max_optimization_iterations_calib['state'] = 'disabled'
    elif self.optimization_method_calib.get() == 'Simplex':
        self.max_initial_iterations_calib['state'] = 'disabled'        
        self.max_optimization_iterations_calib['state'] = 'normal'
    elif self.optimization_method_calib.get() == 'Combined':
        self.max_initial_iterations_calib['state'] = 'normal'
        self.max_optimization_iterations_calib['state'] = 'normal'
        
def enable_RTC_optimization(self):
    if self.param.UseOptimization.get() == True:
        state = 'normal'
        self.opt_method['state'] = 'readonly'
        self.optimized_parameter['state'] = 'readonly'
    
    elif self.param.UseOptimization.get() == False:
        state = 'disabled'
        self.opt_method['state'] = state
        self.optimized_parameter['state'] = state
    self.expected_min_Xvalue['state'] = state
    self.expected_max_Xvalue['state'] = state
    self.max_initial_iterations['state'] = state
    self.max_iterations_per_minimization['state'] = state
    
    
        
        
# Update the text of one entry based on another.
def update(entry_in, entry_out):
    entry_out.delete(0, END)
    entry_out.insert(0,entry_in.get())

def enableEntry(entry):
    entry.configure(state='normal')
    entry.update()

def disableEntry():
    entry.configure(state = 'disabled')
    entry.update()
 
def update_Hotstart(self,checkbutton_param,entry):
    if checkbutton_param.get() == True:
        entry['state'] = 'normal'
    elif checkbutton_param.get() == False:
        entry['state'] = 'disabled'
        
    
 
    
def update_min_max_calib(self,input_param,min_entry,max_entry,min_val,max_val):
    if input_param.get() == True:
        min_entry.configure(state= 'normal')
        max_entry.configure(state= 'normal')
        # if min_entry.get() == '':
        #     min_entry.insert(END,min_val)
        # if max_entry.get() == '':
        #     max_entry.insert(END,max_val)
    else:
        min_entry.configure(state= 'disabled')
        max_entry.configure(state= 'disabled')
        
        
def check_custom_ids(self):
    if self.Custom_CSO_ids.get()=='':
        state = 'normal'
    else:
        state = 'disabled'
    self.CSO_id1['state'] = state
    self.CSO_id2['state'] = state
    self.CSO_id3['state'] = state

# =============================================================================
# functions for the GUI buttons
# =============================================================================

#===================================================================     
# Define run function
def run(self):
    
    if self.param.Overwrite_config.get() == True:
        write_config(self) # Writes configuration file 
    
    msg.showinfo('','Running simulation \nSee run status in console window')
    config_file = self.param.model_name.get() + '.ini'
    # generate_SWMM_file(self) # Does not work 
    pyswmm_Simulation.Optimizer(config_file)

# =============================================================================
def OpenFile(self):
    try:
        path = filedialog.askopenfilename(filetypes =(("SWMM model", "*.inp"),("All Files","*.*")),
                       title = "Choose a file.")
        modelname = path.split('/')[-1].split('.')[-2]
        self.param.model_name.set(modelname)
        # Define path of the model     
        directory = os.path.split(path)[0]
        self.param.model_dir.set(directory)
        if self.save_calib_file.get() == '':
            self.save_calib_file.insert(END,modelname + '_Calibrated')
    except IndexError: 
        print('No model selected.')

# =============================================================================
def select_obs(self):
    try:
        path = filedialog.askopenfilename(filetypes =(("csv", "*.csv"),("All Files","*.*")),
                           title = "Choose a file.")
        obs_data = path.split('/')[-1].split('.')[-2]
        self.param.obs_data.set(obs_data)
        # Define path of the model     
        directory = os.path.split(path)[0]
        self.param.obs_data_path.set(directory)
    except IndexError:
        print('No observation file selected') 
# =============================================================================

def generate_SWMM_file(self):
    msg.showerror('','Not implemented')
#     name = filedialog.asksaveasfilename(initialdir = self.param.model_dir.get() ,filetypes =(("SWMM model", "*.inp"),("All Files","*.*")),
#                             title = "Save file as", defaultextension='.inp')
    
#     # Read the .inp file from the model specified. 
#     read_file = self.param.model_dir.get()+'/'+self.param.model_name.get()+'.inp'
#     with open(read_file) as f:
#         with open(name, "w") as f1:
#             for line in f:
#                 f1.write(line)

# # Write Control rules 
#     write_file = open(name,'a')
#     write_file.write('\n[CONTROLS]\nRULE RBC_1\n\n')
#     write_file.write('IF NODE ' + self.sensor1id.get() + ' DEPTH > ' + self.sensor1setting.get() + '\n')
#     write_file.write('THEN ORIFICE ' + self.actuator1id.get() + ' SETTING = '+ self.actuator1setting_True.get() +  '\n')
#     write_file.write('ELSE ORIFICE ' + self.actuator1id.get() + ' SETTING = '+self.actuator1setting_False.get()+'\n')
    
#     write_file.close()
#     msg.showinfo('','Saved to SWMM file')

# =============================================================================
# User message to be printet in console:
def user_msg(self):
    msg = """
Clompleted simulation {} of {}.
Time of simulation is {}.
Expected time of completion is {}.
"""
    
    sim_num = self.sim_num
    total_sim = self.max_initial_iterations + self.maxiterations
    sim_time = self.sim_end_time - self.sim_start_time
    Complete_time = datetime.now() + sim_time*int((total_sim-sim_num))
    print(msg.format(sim_num,total_sim,sim_time,Complete_time.strftime("%H:%M")))

# =============================================================================
# Show plots 
def Results_plot(location,timestamp_folder):
    import pickle
    pickle_in = open("../output/"+ timestamp_folder + "/First_step_simulations.pickle","rb")
    df = pickle.load(pickle_in)
        # plt.xticks(fontsize = size-2)
        # plt.yticks(fontsize = size-2)
        # ax.legend() 
    size = 14 # fontsize
    figure = plt.Figure(figsize=(7,5), dpi=80)
    ax = figure.add_subplot(111)
    ax.scatter(df['starting points'],df['objective values'], color = 'black',marker = '+',s = 50)
    scatter = FigureCanvasTkAgg(figure,location) 
    scatter.get_tk_widget().grid(row = 0)
    ax.set_xlabel('starting points',fontsize = size)
    ax.set_ylabel('Objective value',fontsize = size)
    ax.set_title('Results of first step of the optimization',fontsize = size)
    figure.savefig("../output/" + timestamp_folder + '/Plot of first step of the optimization.png')

# Shows the results of the optimization in a plot
def First_step_optimization_plot(timestamp):
    popup_plot = tk.Tk()
    popup_plot.wm_title('Results')   
    tframe = Frame(popup_plot)
    tframe.grid(row=0,column = 0,sticky ='nsew')
    mframe = Frame(popup_plot)
    mframe.grid(row=1,column = 0,sticky ='nsew')
    bframe = Frame(popup_plot)
    bframe.grid(row = 2,sticky = 'nsew')

    with open('../output/' + timestamp+ '/optimized_results.txt','r') as file:
        resultfile = file.read()  
    res_text = scrolledtext.ScrolledText(tframe, height = 15, width = 70, wrap = "word")
    res_text .grid(row = 0, column = 0)
    res_text.insert(INSERT,resultfile)
    Results_plot(mframe,timestamp)
    B1 = ttk.Button(bframe, text="Quit", command = popup_plot.destroy)
    B1.grid(row = 0,column = 1, sticky = E)
    
    
def create_calibrate_plot(calibrate_method,num_model_parameters,parameter_names,timestamp,
                   lhs_evaluated_parameters,lhs_objective_values,
                   best_lhs_parameter_set,best_lhs_objective_value,
                   evaluated_simplex_parameter_sets,evaluated_simplex_function_values,
                   best_simplex_parameter_set,best_simplex_objective_value):
    # Creates plots of the optimization of the calibration. 
    # These are saved as .png files in the output folder. 
    
    if calibrate_method == 'lhs':
        if num_model_parameters > 1:
            # Plot LHS results
            fig_lhs, ax = plt.subplots(1,num_model_parameters)
            for i in range(num_model_parameters):
                ax[i].plot(lhs_evaluated_parameters[:,i], lhs_objective_values, 'g.',label = 'evaluated lhs simulations')
                ax[i].plot(best_lhs_parameter_set[i], best_lhs_objective_value, 'r.',label = 'best lhs simulation')
                ax[i].set_title(parameter_names[i])
                ax[i].set_xlabel("Parameter Value")
            ax[0].legend(bbox_to_anchor=(num_model_parameters/2.,1.05),ncol = 2,loc = 'lower center')
        else:
            plt.figure()
            plt.plot(lhs_evaluated_parameters[:], lhs_objective_values, 'g.',label = 'evaluated lhs simulations')
            plt.plot(best_lhs_parameter_set, best_lhs_objective_value,'r.',label = 'best lhs simulation')
            plt.title(parameter_names[0])
            plt.xlabel("Parameter Value")
            plt.legend(bbox_to_anchor=(num_model_parameters/2.,1.05),ncol = 2,loc = 'lower center')
        
        plt.savefig('../output/' + timestamp +'/lhs_figure.png',bbox_inches='tight')
  
    elif calibrate_method == 'Combined':
        if num_model_parameters > 1:
            fig_both, ax = plt.subplots(1,num_model_parameters)
            for i in range(num_model_parameters):
                ax[i].plot(lhs_evaluated_parameters[:,i], lhs_objective_values, 'g.',label = 'evaluated lhs simulations')
                ax[i].plot(evaluated_simplex_parameter_sets[:,i], evaluated_simplex_function_values,'.',label = 'evaluated simplex simulations')
                ax[i].plot(best_lhs_parameter_set[i], best_lhs_objective_value,'r.',label = 'best lhs simulation')
                ax[i].plot(best_simplex_parameter_set[i], best_simplex_objective_value,'y.',label = 'best simplex simulation')
                ax[i].set_title(parameter_names[i])
                ax[i].set_xlabel("Parameter Value")
                
            ax[0].legend(bbox_to_anchor=(num_model_parameters/2.,1.05),ncol = 2,loc = 'lower center')
        else: 
            plt.figure()
            plt.plot(lhs_evaluated_parameters[:], lhs_objective_values, 'g.',label = 'evaluated lhs simulations')
            plt.plot(evaluated_simplex_parameter_sets[:], evaluated_simplex_function_values,'.',label = 'evaluated simplex simulations')
            plt.plot(best_lhs_parameter_set, best_lhs_objective_value,'r.',label = 'best lhs simulation')
            plt.plot(best_simplex_parameter_set, best_simplex_objective_value,'y.',label = 'best simplex simulation')
            plt.title(parameter_names[0])
            plt.xlabel("Parameter Value")
            plt.legend(bbox_to_anchor=(num_model_parameters/2.,1.05),ncol = 2,loc = 'lower center')
        
        plt.savefig('../output/' + timestamp +'/lhs_and_simplex_figure.png',bbox_inches='tight')
    
        # plot of simplex objective function values over time (to check convergence)
        fig_simplex_iterations, ax = plt.subplots(1,1)
        ax.plot(evaluated_simplex_function_values,'.')
        ax.set_title('Check model convergence')
        ax.set_xlabel("Model iterations")
        plt.savefig('../output/' + timestamp +'/simplex_convergence.png',bbox_inches='tight')
    
    elif calibrate_method == 'Simplex':
        if num_model_parameters > 1:
            fig_simplex, ax = plt.subplots(1,num_model_parameters)
            for i in range(num_model_parameters):
                ax[i].plot(evaluated_simplex_parameter_sets[:,i], evaluated_simplex_function_values,'.',label = 'evaluated simplex simulations')
                ax[i].plot(best_simplex_parameter_set[i], best_simplex_objective_value,'y.',label = 'best simplex simulation')
                ax[i].set_title(parameter_names[i])
                ax[i].set_xlabel("Parameter Value")
                fig_simplex.legend()
            ax[0].legend(bbox_to_anchor=(num_model_parameters/2.,1.05),ncol = 2,loc = 'lower center')
            
        else: 
            plt.figure()
            plt.plot(evaluated_simplex_parameter_sets[:], evaluated_simplex_function_values,'.',label = 'evaluated simplex simulations')
            plt.plot(best_simplex_parameter_set, best_simplex_objective_value,'y.',label = 'best simplex simulation')
            plt.title(parameter_names[0])
            plt.xlabel("Parameter Value")
            plt.legend(bbox_to_anchor=(num_model_parameters/2.,1.05),ncol = 2,loc = 'lower center')
        
        plt.savefig('../output/' + timestamp +'/simplex_figure.png',bbox_inches='tight')
        # plot of simplex objective function values over time (to check convergence)
        fig_simplex_iterations, ax = plt.subplots(1,1)
        ax.plot(evaluated_simplex_function_values,'.')
        ax.set_title('Check model convergence')
        ax.set_xlabel("Model iterations")
       
        
        plt.savefig('../output/' + timestamp +'/simplex_convergence.png',bbox_inches='tight')
   
def calibrate_results(timestamp):
    from PIL import ImageTk, Image
    
    popup_plot = tk.Tk()
    popup_plot.wm_title('Results')   
    tframe = Frame(popup_plot)
    tframe.grid(row=0,column = 0,sticky ='nsew')
    mframe = Frame(popup_plot)
    mframe.grid(row=1,column = 0,sticky ='nsew')
    bframe = Frame(popup_plot)
    bframe.grid(row = 2,sticky = 'nsew')

    
    with open('../output/' + timestamp+ '/results.txt','r') as file:
        resultfile = file.read()  
    res_text = scrolledtext.ScrolledText(tframe, height = 15, width = 70, wrap = "word")
    res_text .grid(row = 0, column = 0)
    res_text.insert(INSERT,resultfile)
        
    canvas = Canvas(mframe, width = 700, height = 200)      
    canvas.grid(row = 0) 
    
    images = os.listdir('../output/' + timestamp+ '/')
    for im in images[:]:
        if not im.endswith(".png"):
            images.remove(im)
    
    # img = Image.open("File.jpg")  # PIL solution
# img = img.resize((250, 250), Image.ANTIALIAS) #The (250, 250) is (height, width)
# img = ImageTk.PhotoImage(img) # convert to PhotoImage
# image = C.create_image(1500,0, anchor = NE, image = img)
# image.pack() # canvas objects do not use pack() or grid()

    print(images)
    for i, image in enumerate(images):
        img = Image.open('../output/' + timestamp+ '/' + image) # PIL solution
        img = img.resize((2, 2), Image.ANTIALIAS) #The (250, 250) is (height, width)
        img = PhotoImage(img)      
        img = PhotoImage(file='../output/' + timestamp+ '/' + image)      
        canvas.create_image(0,i*100, anchor=NW, image=img)      
    
    
    
    
    B1 = ttk.Button(bframe, text="Exit", command = popup_plot.destroy)
    B1.grid(row = 0,column = 0, sticky = E)
    popup_plot.mainloop()
    
    
    
# if __name__ == "__main__": 
#     calibrate_results('2020-06-16_13-02-45')
# To be deleted: 
# =============================================================================
# 
# def Results_table(title, msg):
  
#     popup = tk.Tk()
#     popup.wm_title(title)   
#     tframe = Frame(popup)
#     tframe.grid(row=0,column = 0,sticky ='nsew')
#     popup.columnconfigure(0, weight=4)
#     popup.rowconfigure(0, weight=4)
#     table = TableCanvas(tframe,data = msg)
#     table.show()

#     bframe = Frame(popup)
#     bframe.grid(row = 1,sticky = 'nsew')
        
#     B1 = ttk.Button(bframe, text="Quit", command = popup.destroy)
#     B1.grid(row = 2,column = 1, sticky = E)
#     popup.mainloop()


# =============================================================================
# def Results_text(title, msg):
  
#     popup_opt = tk.Tk()
#     popup_opt.wm_title(title)   
#     tframe = Frame(popup_opt)
#     tframe.grid(row=0,column = 0,sticky ='nsew')

#     bframe = Frame(popup_opt)
#     bframe.grid(row = 1,sticky = 'nsew')
#     res_text = scrolledtext.ScrolledText(popup_opt, height = 16, width = 100, wrap = NONE)
#     label = ttk.Label(popup_opt, text=msg)
#     res_text .grid(row = 0, column = 0)
#     res_text.insert(INSERT,msg)
#     B1 = ttk.Button(bframe, text="Quit", command = popup_opt.destroy)
#     B1.grid(row = 3,column = 1, sticky = E)
# #    B2 = ttk.Button(bframe, text="Show graphs", command = lambda:optimal_solution_graph())
# #    B2.grid(row = 3,column = 0, sticky = E)
#     popup_opt.mainloop()
    
   
#===================================================================    
                        
# class StartUp_window(object):
#     def __init__(self):
#         self.StartUp_window = tk.Tk()
#         self.StartUp_window.wm_title('Load model')   
#         self.StartUp_window.iconbitmap('./GUI_files/noah_logo_OAb_icon.ico')
#         self.StartUp_window.configure(background = 'white')
#     #    self.StartUp_window.geometry('520x00')
#         self.StartUp_window.resizable(False,False)
            
#         tframe = Frame(self.StartUp_window)
#         tframe.grid(row=0,column = 0,sticky=N+S+E+W, padx=10, pady=30)
#         Label(tframe,text = 'Choose how to begin the project',bg = 'white').grid(row=0,column = 0,columnspan = 4,sticky=E+W)
#         B1 = ttk.Button(tframe, text="Load new SWMM model", command = lambda: GUI_elements.OpenFile()).grid(row=1,column=0)
#         B2 = ttk.Button(tframe, text="Load most recent config file", command = lambda: GUI_elements.OpenFile).grid(row=1,column=1)
#         B3 = ttk.Button(tframe, text="Load new config file", command = lambda: GUI_elements.OpenFile).grid(row=1,column=2)
#         B4 = ttk.Button(tframe, text="Quit", command = self.StartUp_window.destroy).grid(row = 1,column = 3, sticky = E)
#         self.StartUp_window.mainloop()
        