# -*- coding: utf-8 -*-
"""
Copyright 2019 Magnus Johansen
Copyright 2019 Technical University of Denmark

This file is part of NOAH RTC Tool.

NOAH RTC Tool is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

NOAH RTC Tool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with NOAH RTC Tool. If not, see <http://www.gnu.org/licenses/>.
"""
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

# =============================================================================
# # This module contains the prameters that are given from the GUI to the functions in the tool. 
# # This includes: Write configuration file, Read configuration file. 
# # These two functions are linked with a unit test to ensure that the same parameters are included in both of them. 
# =============================================================================



# =============================================================================
# # Write input from GUI to configuration file        
# =============================================================================

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
    
    print(swmmio.swmmio.inp(self.param.model_dir.get() + '/' + self.param.model_name.get() + '.inp').options.Value.REPORT_STEP)
    
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
                                    'rainfall_threshold_duration':self.rainfall_time.get(),
                                    'Several_basins_RTC':self.param.several_basins_RTC.get(),
                                    'Several_basins_basins':self.several_basins_basin_list.get(), 
                                    'Several_basins_actuators':self.several_basins_actuators_list.get(),
                                    'Several_basins_sensors':self.several_basins_sensors_list.get(),
                                    'Several_basins_num_simulations':self.several_basins_num_simulations.get() 
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
    
    
    config['EqualFillingDegree'] = {'EFD_basins':self.EFD_basins.get(),
                                    'EFD_actuators':self.EFD_actuators.get(),
                                    'Basin_depths':self.Basin_depths.get(),
                                    'no_layers':self.no_layers.get(),
                                    'EFD_CSO_id':self.EFD_CSO_id.get(),
                                    'EFD_setup_type':self.param.EFD_setup_type.get()
                                    }
    
    
    # save the file
    config_name = self.param.model_name.get()
    config_path = '../config/saved_configs/'
    with open(config_path + config_name + '.ini','w') as configfile: 
        config.write(configfile)
    msg.showinfo('','Saved to configuration file')


# =============================================================================
# # READ
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
        
        # several_basins
        try:
            self.several_basins_RTC = eval(RBC['Several_basins_RTC'])
            self.several_basins_basins_list = RBC['Several_basins_basins']
            self.several_basins_actuators_list = RBC['Several_basins_actuators']
            self.several_basins_sensors_list = RBC['Several_basins_sensors']
            self.several_basins_num_simulations = int(RBC['Several_basins_num_simulations'])
        except ValueError:
            pass
                                    
        
        # RTC Optimization
        Optimization = config['Optimization']
        self.useoptimization = eval(Optimization['useoptimization'])
        self.optimization_method = Optimization['optimization_method']
        self.CSO_objective = Optimization['CSO_objective']
        self.CSO_id1 = Optimization['CSO_id1']
        self.CSO_id2 = Optimization['CSO_id2']
        self.CSO_id3 = Optimization['CSO_id3']
        self.Custom_CSO_ids = Optimization['Custom_CSO_ids']
        self.optimized_parameter = Optimization['optimized_parameter']
        try:
            self.expected_min_xvalue = float(Optimization['expected_min_xvalue'])
            self.expected_max_xvalue = float(Optimization['expected_max_xvalue'])
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
        
        # EqualFillingDegree
        EqualFillingDegree = config['EqualFillingDegree']
        try:
            self.EFD_basins = EqualFillingDegree['efd_basins']
            self.EFD_actuators= EqualFillingDegree['efd_actuators']
            self.basin_depths  = EqualFillingDegree['basin_depths']
            self.no_layers = int(EqualFillingDegree['no_layers'])
            self.EFD_CSO_id = EqualFillingDegree['efd_cso_id']
        except ValueError:
            pass
        
        self.EFD_setup_type = EqualFillingDegree['EFD_setup_type']
    