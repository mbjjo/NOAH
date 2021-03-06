# -*- coding: utf-8 -*-
"""
Copyright 2019 Magnus Johansen
Copyright 2019 Technical University of Denmark

This file is part of NOAH RTC Tool.

NOAH RTC Tool is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

NOAH RTC Tool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with NOAH RTC Tool. If not, see <http://www.gnu.org/licenses/>.
"""

import Parameters

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
import GUI_elements
import pdb
# =============================================================================

class Optimizer:
    def __init__(self,config_file):
        starttime = time.time()
        self.sim_num = 0

        # =============================================================================
        # # INP: Everygthin that is read from the input file is defined as inp.
        # # SELF: Everything that is either calculated along the way or a function of the class is defined as self. 
        # =============================================================================        
        inp = Parameters.Read_Config_Parameters(config_file)

#         create folder with the time stamp for the first simulation. All results are stored in this.  
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.mkdir('../output/'+self.timestamp)   
        model_outfile = '../output/' + str(self.timestamp) + '/' + str(inp.model_name) + '.out'
        
        # The Hiddenprints are only used to avoid printing the warning from the exception since this is not relevant for the end user. 
        class HiddenPrints:
            def __enter__(self):
                self._original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
        
            def __exit__(self, exc_type, exc_val, exc_tb):
                sys.stdout.close()
                sys.stdout = self._original_stdout
        # copies the existing         
        try:
            with HiddenPrints():
                self.Redefine_Timeseries(inp)
                self.Timeseries = True
        except KeyError:
            self.Timeseries = False


    # running several basins simulation
        # pdb.set_trace()
        if inp.several_basins_RTC == True:
                                
            result = self.several_basin_optimization(inp)

        
    # Else the "simpel normal" simulation
        else:
                
            if inp.useoptimization == False:
                result = self.simulation(inp,[inp.sensor1_critical_depth]) # single simulation
                print('Simulation ran without optimization')
            elif inp.useoptimization == True:
                result = self.Two_step_optimizer(inp) # 2 step optimization  
                print(result)
                endtime = time.time()
                runtime = (endtime-starttime)/60
                self.write_optimal_result(inp,result,runtime,config_file)
                GUI_elements.First_step_optimization_plot(self.timestamp)

            
        
        print('simulation ended:')
        print(datetime.now())

        # Computing final simulation with optimal results
               
#        # saving results to variables
#        self.res.optimization_results = str(result)
#        self.res.optimal_setting.set(result.x)
#        self.res.volume_at_optimum.set(0.0)
#        self.res.CSO_events_at_optimum.set(10000.0)

    # configuration file is read and variables are defined according to that. 
                                    # def read_config(self,config_file):
                                    #     config = configparser.ConfigParser()
                                    #     config.read('../config/saved_configs/'+config_file)
                                    #     # try except are only applied on float() lines. This ensures that these can be left blank. 
                                    #     # If they are needed in the simulation an error will occur at that point
                                    #     self.system_units = config['Settings']['System_Units']
                                    #     rpt_step_tmp = config['Settings']['Reporting_timesteps']      
                                    #     rpt_step = datetime.strptime(rpt_step_tmp, '%H:%M:%S').time()
                                    #     self.report_times_steps = rpt_step.hour*60 + rpt_step.minute + rpt_step.second/60
                                    #     try:
                                    #         self.CSO_event_seperation = float(config['Settings']['Time_seperating_CSO_events'])
                                    #     except ValueError:
                                    #         # print('\nWarning: Some fields are left blank or not specified correctly\n')
                                    #         pass
                                    #     try:
                                    #         self.CSO_event_duration = float(config['Settings']['Max_CSO_duration'])
                                    #     except ValueError:
                                    #         pass
                                    #     self.CSO_type = config['Settings']['CSO_type']
                                    #     self.model_name = config['Model']['modelname']
                                    #     self.model_dir = config['Model']['modeldirectory']
                                    #     #    rain_series = config['Model']['rain_series']
                                            
                                    #     RBC = config['RuleBasedControl']
                                    #     self.actuator_type = RBC['actuator_type']
                                    #     self.sensor1_id = RBC['sensor1_id']    
                                    #     self.actuator1_id = RBC['actuator1_id']
                                    #     try:
                                    #         self.sensor1_critical_depth = float(RBC['sensor1_critical_depth']) # It could make sense to leave this blank
                                    #         self.actuator1_target_setting_True = float(RBC['actuator1_target_setting_True'])
                                    #         self.actuator1_target_setting_False = float(RBC['actuator1_target_setting_False'])
                                    #     except ValueError:
                                    #         # print('Parameters for the rule based control are not specified correctly or left blank.')    
                                    #         pass
                                    #     try:
                                    #         self.sensor1_critical_depth_dry = float(RBC['sensor1_critical_depth_dryflow'])
                                    #         self.actuator1_target_setting_True_dry = float(RBC['actuator1_target_setting_true_dryflow'])
                                    #         self.actuator1_target_setting_False_dry = float(RBC['actuator1_target_setting_false_dryflow'])
                                    #     except ValueError:
                                    #         # print('Parameters for the rule based control are not specified correctly.')    
                                    #         pass
                                    #     self.RG1 = RBC['raingage1']
                                    #     try:
                                    #         self.rainfall_threshold_value = float(RBC['rainfall_threshold_value'])
                                    #         self.rainfall_threshold_time = float(RBC['rainfall_threshold_duration'])
                                    #     except ValueError:
                                    #         pass
                                        
                                    #     Optimization = config['Optimization']
                                    #     self.useoptimization = eval(Optimization['useoptimization'])
                                    #     self.optimization_method = Optimization['optimization_method']
                                    #     self.CSO_objective = Optimization['CSO_objective']
                                    #     self.CSO_id1 = Optimization['CSO_id1']
                                    #     self.CSO_id2 = Optimization['CSO_id2']
                                    #     self.CSO_id3 = Optimization['CSO_id3']
                                    #     self.Custom_CSO_ids = Optimization['Custom_CSO_ids']
                                    #     self.optimized_parameter = Optimization['optimized_parameter']
                                    #     try:
                                    #         self.expected_min_xvalue = float(Optimization['expected_min_xvalue'])
                                    #         self.expected_max_xvalue = float(Optimization['expected_max_xvalue'])
                                    #         self.max_initial_iterations = int(Optimization['max_initial_iterations'])
                                    #         self.maxiterations = int(Optimization['max_iterations_per_minimization'])
                                    #     except ValueError:
                                    #         # print('Optimization parameters are not specified correctly or left blank.')
                                    #         pass
        
    #        self.expected_min_xvalue = float(Optimization['expected_min_xvalue'])
    #        self.expected_max_xvalue = float(Optimization['expected_max_xvalue'])
    ##        self.initial_value = float(Optimization['initial_value'])
    #        self.max_initial_iterations = int(Optimization['max_initial_iterations'])
    #        self.max_iteration_per_minimization = int(Optimization['max_iteration_per_minimization'])
# =============================================================================

    def write_simpel_control_to_SWMM(self,inp,x,filename):
        from swmmio.utils.modify_model import replace_inp_section
    # # The following line are only necessary if the existing controls are to be copied
    # # The Hiddenprints are only used to avoid printing the warning from the exception since this is not relevant for the end user. 
    # class HiddenPrints:
    #     def __enter__(self):
    #         self._original_stdout = sys.stdout
    #         sys.stdout = open(os.devnull, 'w')
    
    #     def __exit__(self, exc_type, exc_val, exc_tb):
    #         sys.stdout.close()
    #         sys.stdout = self._original_stdout
    # # copies the existing         
    # try:
    #     with HiddenPrints():
    #         controls_section = create_dataframeINP(self.model_dir +'/'+ self.model_name+'.inp','[CONTROLS]')
    #     New_controls_section = controls_section.copy() 
    # except KeyError:
        
    
        if inp.actuator_type == 'orifice':
            New_controls_section = pd.DataFrame({'[CONTROLS]':np.zeros(4)})
            New_controls_section['[CONTROLS]'][0] = 'RULE NOAH_RTC_tool'
            New_controls_section['[CONTROLS]'][1] = 'IF NODE {} DEPTH > {:}'.format(inp.sensor1_id,x)
            New_controls_section['[CONTROLS]'][2] = 'THEN ORIFICE {} SETTING = {}'.format(inp.actuator1_id ,int(inp.actuator1_target_setting_True))
            New_controls_section['[CONTROLS]'][3] = 'ELSE ORIFICE {} SETTING = {}'.format(inp.actuator1_id ,int(inp.actuator1_target_setting_False))
        elif inp.actuator_type == 'weir':
            New_controls_section = pd.DataFrame({'[CONTROLS]':np.zeros(4)})
            New_controls_section['[CONTROLS]'][0] = 'RULE NOAH_RTC_tool'
            New_controls_section['[CONTROLS]'][1] = 'IF NODE {} DEPTH > {:}'.format(inp.sensor1_id,x)
            New_controls_section['[CONTROLS]'][2] = 'THEN WEIR {} SETTING = {}'.format(inp.actuator1_id ,int(inp.actuator1_target_setting_True))
            New_controls_section['[CONTROLS]'][3] = 'ELSE WEIR {} SETTING = {}'.format(inp.actuator1_id ,int(inp.actuator1_target_setting_False))
        else:
            # print('Actuator type not defined')
            # pass
            New_controls_section = pd.DataFrame({'[CONTROLS]':np.zeros(4)})
            New_controls_section['[CONTROLS]'][0] = 'RULE NOAH_RTC_tool'
            New_controls_section['[CONTROLS]'][1] = 'IF NODE {} DEPTH > {:}'.format(inp.sensor1_id,x)
            New_controls_section['[CONTROLS]'][2] = 'THEN PUMP {} SETTING = {}'.format(inp.actuator1_id ,int(inp.actuator1_target_setting_True))
            New_controls_section['[CONTROLS]'][3] = 'ELSE PUMP {} SETTING = {}'.format(inp.actuator1_id ,int(inp.actuator1_target_setting_False))
        # Create a temporary file with the adjusted path
        
        if self.Timeseries == True:
            old_file = inp.model_dir +'/'+ inp.model_name + '_tmp' + '.inp'
            new_file = inp.model_dir +'/'+ inp.model_name + '_tmp' + filename + '.inp'
        elif self.Timeseries ==False:
            old_file = inp.model_dir +'/'+ inp.model_name + '.inp'
            new_file = inp.model_dir +'/'+ inp.model_name + filename + '.inp'
        shutil.copyfile(old_file,new_file)
    
        #Overwrite the CONTROLS section of the new model with the adjusted data
        with pd.option_context('display.max_colwidth', 400): # set the maximum length of string to prit the full string into .inp
            replace_inp_section(new_file, '[CONTROLS]', New_controls_section)


    def Redefine_Timeseries(self,inp):
        """
        Required inputs:
            model
            directory
        """
    #Changing the path of the Time series to run models.
        timeserie_time_start = datetime.now()
    
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
                # print('Rainfile_new: '+ Rainfile_new)
                # print('Rainfile_old: '+ Rainfile_old)
                # print('Rain_name:' + Rain_name)
            else: 
                New_Timeseries.iloc[i][0] = np.nan
        New_Timeseries.dropna(inplace = True)
        
        # Create a temporary file with the adjusted path
        new_file = inp_file + '_tmp.inp'
        shutil.copyfile(baseline, path + '/' + new_file)
        # print('path:' + path ) 
        # print('new_file:' + new_file) 
        # print('path:' + path ) 
        #Overwrite the TIMESERIES section of the new model with the adjusted data
        with pd.option_context('display.max_colwidth', 400): # set the maximum length of string to prit the full string into .inp
            replace_inp_section(path + '/' + new_file, '[TIMESERIES]', New_Timeseries)
        timeserie_time_stop = datetime.now()
        
                
    def Two_step_optimizer(self,inp):
        """
        Required inputs:
            model
            directory
            timestamp
            optimization parameters (x4) 
            Everything to simulation()
            
        """
        # First step init: determine starting point from sequencial simulations
        starting_points = np.arange(inp.expected_min_xvalue,inp.expected_max_xvalue,(inp.expected_max_xvalue-inp.expected_min_xvalue)/inp.max_initial_iterations)
        initial_simulation = np.zeros(len(starting_points))
        for i in range(len(starting_points)):
            initial_simulation[i] = self.simulation([starting_points[i]],inp)
            
        # Save the result to a pickle file
        xy = np.array([starting_points,initial_simulation]).T
        df = pd.DataFrame(xy, columns = ['starting points','objective values'])
        pickle_path = '../output/'+self.timestamp
        pickle_out = open(pickle_path + "/First_step_simulations.pickle","wb")
        pickle.dump(df, pickle_out)
        pickle_out.close()
        
        start_value = str(starting_points[initial_simulation.argmin()])
        
        if inp.maxiterations > 0:
            # Second step: Simple optimization with simplex
            print('begins optimization with simplex')
            print('start_value ' + str(start_value))
        
            result = optimize.minimize(fun = self.simulation, args=(inp),
                                  x0 = start_value, method='Nelder-Mead',
                                  options = {'disp':True,'maxfev':inp.maxiterations})
        else:
            result = {'Completed':'Only ran the inital simulations. No optimization was performed afterwards.',
                      'x': [start_value]}
        
        # print(result)
        
        # Run one more simulation to ensure that the saved .rpt and .out are the ones from the optimal setup. 
        # self.simulation(result['x'][0])
        # Write the final result to a SWMM file 
        self.write_simpel_control_to_SWMM(inp,result['x'][0],'_RTC')

        return result
# =============================================================================
    # optimized parameter is 'Activation depth'
    def simulation(self, x, inp):
        """
        Required inputs:
            model
            directory
            timestamp
            CSO objective
            CSO ids
            things to user mesage
        """
        
        
        self.sim_num += 1 
        self.sim_start_time = datetime.now()
        
        self.write_simpel_control_to_SWMM(inp,x[0],'_tmp')
            # Run simulation
        if self.Timeseries == True:
            with Simulation(inp.model_dir +'/'+ inp.model_name + '_tmp_tmp.inp' , '../output/'+ self.timestamp + '/' + inp.model_name + '.rpt', '../output/' + self.timestamp + '/' + inp.model_name + '.out') as sim: 
                for step in sim:
                    pass
        elif self.Timeseries == False:
            with Simulation(inp.model_dir +'/'+ inp.model_name + '_tmp.inp' , '../output/'+ self.timestamp + '/' + inp.model_name + '.rpt', '../output/' + self.timestamp + '/' + inp.model_name + '.out') as sim: 
                for step in sim:
                    pass
        
        # if self.actuator_type == 'orifice':
        #     # Run simulation
        #     with Simulation(self.model_dir +'/'+ self.model_name + '.inp' , '../output/'+ self.timestamp + '/' + self.model_name + '.rpt', '../output/' + self.timestamp + '/' + self.model_name + '.out') as sim: 
        #         for step in sim:
        #             # Compute control steps
        #             if Nodes(sim)[self.sensor1_id].depth > x:
        #                 Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_True
        #             else:
        #                 Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_False
        #             # print('Complete: {:.2f}'.format(sim.percent_complete))
        # elif self.actuator_type == 'pump':
        #     # Run simulation
        #     with Simulation(self.model_dir + '/' + self.model_name + '.inp' , '../output/' + self.timestamp + '/' + self.model_name + '.rpt', '../output/' + self.timestamp + '/' + self.model_name + '.out') as sim: 
        #         precipitation = list()
        #         for step in sim:
        #             precipitation.append(raingages.RainGages(sim)[self.RG1].total_precip)
                    
        #             # Compute control steps
        #             # Wet weather rule
        #             if np.mean(precipitation[-int(self.rainfall_threshold_time/self.report_times_steps):]) > self.rainfall_threshold_value:
        #             # if rainfall < X or upstream basin < X or something else:
        #                 if Nodes(sim)[self.sensor1_id].depth > self.sensor1_critical_depth: # Setting start level
        #                     Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_True
        #                 elif Nodes(sim)[self.sensor1_id].depth < 0.1: # Setting stop level
        #                     Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_False
            
        #             # Dry weather rule
        #             else:
        #                 if Nodes(sim)[self.sensor1_id].depth > x: # Parameter to be optimized (self.sensor1_critical_depth_dry):
        #                     Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_True_dry
        #                 elif Nodes(sim)[self.sensor1_id].depth < 0.1 : # Setting stop level
        #                     Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_False_dry
        #             # print('Complete: {:.2f}'.format(sim.percent_complete))
                
        # Output file is defined
        model_outfile = '../output/' + self.timestamp + '/' + str(inp.model_name) + '.out'
        if inp.Custom_CSO_ids == '':
            CSO_ids = [i for i in [inp.CSO_id1,inp.CSO_id2,inp.CSO_id3] if i] 
        else:
            CSO_ids = [x.strip(' ') for x in inp.Custom_CSO_ids.split(',')]

        if inp.CSO_objective == 'volume':
            objective_value = self.count_CSO_volume(inp,CSO_ids,model_outfile)
            
            # Saving the other function as pickle:
            df = self.count_CSO_events(inp,CSO_ids,model_outfile)
            pickle_path = '../output/'+self.timestamp
            pickle_name = '/CSO_events_{}.pickle'.format(self.sim_num)
            pickle_out = open(pickle_path + pickle_name,"wb")
            pickle.dump(df, pickle_out)
            pickle_out.close()
        
            
            
        elif inp.CSO_objective =='frequency':
            objective_value = self.count_CSO_events(inp,CSO_ids,model_outfile)
            
            # Saving the other function as pickle:
            df = self.count_CSO_volume(inp,CSO_ids,model_outfile)
            pickle_path = '../output/'+self.timestamp
            pickle_name = '/CSO_vol_{}.pickle'.format(self.sim_num)
            pickle_out = open(pickle_path + pickle_name,"wb")
            pickle.dump(df, pickle_out)
            pickle_out.close()
        
        self.sim_end_time = datetime.now()
        
        # # Output file is defined
        # model_outfile = '../output/' + self.timestamp + '/' + str(self.model_name) + '.out'
        # if self.CSO_objective == 'volume':
        #     if self.Custom_CSO_ids == '':
        #         if self.CSO_id2 == '': # Assume that CSO_id3 is also '' (empty) 
        #             objective_value = self.count_CSO_volume([self.CSO_id1],model_outfile)
        #         elif self.CSO_id3 == '':
        #             objective_value = self.count_CSO_volume([self.CSO_id1,self.CSO_id2],model_outfile)
        #         else:
        #             objective_value = self.count_CSO_volume([self.CSO_id1,self.CSO_id2,self.CSO_id3],model_outfile)
        #     else:
        #         objective_value = self.count_CSO_volume([x.strip(' ') for x in self.Custom_CSO_ids.split(',')]
        #                                                 ,model_outfile)
        # elif self.CSO_objective =='frequency':
        #     if self.Custom_CSO_ids == '':
        #         if self.CSO_id2 == '': # Assume that CSO_id3 is also '' (empty) 
        #             objective_value = self.count_CSO_events([self.CSO_id1],model_outfile)
        #         elif self.CSO_id3 == '':
        #             objective_value = self.count_CSO_events([self.CSO_id1,self.CSO_id2],model_outfile)
        #         else:
        #             objective_value = self.count_CSO_events([self.CSO_id1,self.CSO_id2,self.CSO_id3],model_outfile)
        #     else:
        #         objective_value = self.count_CSO_events([x.strip(' ') for x in self.Custom_CSO_ids.split(',')]
        #                                                 ,model_outfile)
        # self.sim_end_time = datetime.now()
        
        if inp.useoptimization == True:
            print('Objective value: {:.2f}'.format(objective_value))
            GUI_elements.user_msg(self,inp)
        return objective_value

# =============================================================================
# # several_basin_optimization
# =============================================================================
    # def several_basin_optimization(self)
    def several_basin_optimization(self,inp):
        """
        Optimizes the activation depths of actuators in several basins in a system. 
        Uses the algorithm in FUNC() to optimize x.
        See function: write_several_basins_control_to_SWMM() to see how RTC is computed.
        """        
        # pdb.set_trace()
        
        basins = [x.strip(' ') for x in inp.several_basins_basins_list.split(',')]
        actuators = [x.strip(' ') for x in inp.several_basins_actuators_list.split(',')]
        sensors = [x.strip(' ') for x in inp.several_basins_sensors_list.split(',')]
        
        for sim_num in range(inp.several_basins_num_simulations):
            # optimization algorithm: 
            # redefine x at every step based on a dataframe with CSO statistics. 
            # x = optimization_algorithm(df_CSO_stats)
            x = np.ones(4)
            # create model with the new x
            self.write_several_basins_control_to_SWMM(inp,basins,actuators,sensors,x)

            # run model
            
            if self.Timeseries == True:
                model_suffix = '_tmp_Controls.inp'
            elif self.Timeseries ==False:
                model_suffix = '_Controls.inp'
            
            with Simulation(inp.model_dir +'/'+ inp.model_name + model_suffix , '../output/'+ self.timestamp + '/' + inp.model_name + '.rpt', '../output/' + self.timestamp + '/' + inp.model_name + '.out') as sim: 
                for step in sim:
                    pass
            output_file = '../output/' + self.timestamp + '/' + inp.model_name + '.out'
            print('simulation works')
            """
            Missing steps:
                calculate input df
                Global variables for convergence
                return value
                input from GUI and parameters
                optimization algorithm
            """
            # calculate objective function (i.e. df with CSO statistics)
            # df_CSO_stats = CSO_statistics_df(output_file)
            
            # Save some parameters to a global variable to ensure transparenc yand convergence. 
        print('return is missing')
        return 'works'
    
    
    def write_several_basins_control_to_SWMM(self,inp,basins,actuators,sensors,x):
        """ 
        Assumes that all basins are connected to one orifice that is located just downstream of the basin
        and that they are pairs listed in same order 
        """
        
        # It is required that x1 > x2 
        # try:
        #     x1 > x2 == True
        # except Exception:
        #     message = "x1 must be bigger than x2"
        #     raise
        #     print(message)
        
        from swmmio.utils.modify_model import replace_inp_section
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        
        # Creating an identical set of rules for each of the basins. 
        for i,basin in enumerate(basins):
    
            n = 10
            New_controls_section = pd.DataFrame({'[CONTROLS]':np.zeros(10 + n*len(sensors))})
            
            # Rulescomparing with the sensor
            for j,sensor in enumerate(sensors,0):
                # Rule to close actuator based on high level in sensor 
                New_controls_section['[CONTROLS]'][0+j*n] = 'RULE {}_{}_open'.format(basin,sensor)
                New_controls_section['[CONTROLS]'][1+j*n] = 'IF NODE {} DEPTH > {}'.format(sensor,x1)
                New_controls_section['[CONTROLS]'][2+j*n] = 'THEN ORIFICE {} SETTING = 0'.format(actuators[i])
                New_controls_section['[CONTROLS]'][3+j*n] = 'PRIORITY 2' # higher number => higher priority (e.g. PRIORITY 5 outranks PRIORITY 1)
                New_controls_section['[CONTROLS]'][4+j*n] = ''
                
                # Rule to open actuator based on low level in sensor 
                New_controls_section['[CONTROLS]'][5+j*n] = 'RULE {}_{}_close'.format(basin,sensor)
                New_controls_section['[CONTROLS]'][6+j*n] = 'IF NODE {} DEPTH < {}'.format(sensor,x2)
                New_controls_section['[CONTROLS]'][7+j*n] = 'THEN ORIFICE {} SETTING = 1'.format(actuators[i])
                New_controls_section['[CONTROLS]'][8+j*n] = 'PRIORITY 2' # higher number => higher priority (e.g. PRIORITY 5 outranks PRIORITY 1)
                New_controls_section['[CONTROLS]'][9+j*n] = ''
            
            # Rule to open actuator based on high level in basin
            New_controls_section['[CONTROLS]'][10+j*n] = 'RULE {}_open'.format(basin)
            New_controls_section['[CONTROLS]'][11+j*n] = 'IF NODE {} DEPTH > {}'.format(basin,x3)
            New_controls_section['[CONTROLS]'][12+j*n] = 'THEN ORIFICE {} SETTING = 1'.format(actuators[i])
            New_controls_section['[CONTROLS]'][13+j*n] = 'PRIORITY 3' # higher number => higher priority (e.g. PRIORITY 5 outranks PRIORITY 1)
            New_controls_section['[CONTROLS]'][14+j*n] = ''
            
            # Rule to open actuator based on lower level in basin to ensure that the basin empties during dry weather
            # The basin only opens half (or other fraction)
            # This one might not be necessary if rule #2 ensures that the actuator opens when downstream water is low. 
            New_controls_section['[CONTROLS]'][15+j*n] = 'RULE {}_open_0.5'.format(basin)
            New_controls_section['[CONTROLS]'][16+j*n] = 'IF NODE {} DEPTH > {}'.format(basin,x4)
            New_controls_section['[CONTROLS]'][17+j*n] = 'THEN ORIFICE {} SETTING = 0.5'.format(actuators[i])
            New_controls_section['[CONTROLS]'][18+j*n] = 'PRIORITY 1' # higher number => higher priority (e.g. PRIORITY 5 outranks PRIORITY 1)
            New_controls_section['[CONTROLS]'][19+j*n] = ''
            
            if i == 0:
                Full_controls_section = New_controls_section.copy()
            else:
                Full_controls_section = Full_controls_section.append(New_controls_section,ignore_index=True)
            
            pdb.set_trace()
            
            if self.Timeseries == True:
                old_file = inp.model_dir +'/'+ inp.model_name + '_tmp.inp'
                new_file = inp.model_dir +'/'+ inp.model_name +'_tmp_Controls' + '.inp'
            elif self.Timeseries ==False:
                old_file = inp.model_dir +'/'+ inp.model_name + '.inp'
                new_file = inp.model_dir +'/'+ inp.model_name + '_Controls' + '.inp'
            shutil.copyfile(old_file,new_file)
        
            #Overwrite the CONTROLS section of the new model with the adjusted data
            with pd.option_context('display.max_colwidth', 400): # set the maximum length of string to prit the full string into .inp
                replace_inp_section(new_file, '[CONTROLS]', Full_controls_section)
    
            
        
        
# =============================================================================
# # Objective functions    
# =============================================================================

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

# =============================================================================
# Writing the optimal results to a text file 
        
    def write_optimal_result(self,inp,result,runtime, config_file):
        # Calculate optimial results
        opt_setting = float(result['x'][0])
        print(result)
        opt_value = (result['fun'])
        print(opt_value)
        print(opt_setting)
        # opt_value = 1
        model_outfile = '../output/' + self.timestamp + '/' + str(inp.model_name) + '.out'
        if inp.Custom_CSO_ids == '':
            if inp.CSO_id2 == '': # Assume that CSO_id3 is also '' (empty) 
                opt_vol = self.count_CSO_volume(inp,[inp.CSO_id1],model_outfile)
                opt_CSO_events = self.count_CSO_events(inp,[inp.CSO_id1],model_outfile)
            elif inp.CSO_id3 == '':
                opt_vol = self.count_CSO_volume(inp,[inp.CSO_id1,inp.CSO_id2],model_outfile)
                opt_CSO_events = self.count_CSO_events(inp,[inp.CSO_id1,inp.CSO_id2],model_outfile)
            else:
                opt_vol = self.count_CSO_volume(inp,[inp.CSO_id1,inp.CSO_id2,inp.CSO_id3],model_outfile)
                opt_CSO_events = self.count_CSO_events(inp,[inp.CSO_id1,inp.CSO_id2,inp.CSO_id3],model_outfile)
        else:
                opt_vol = self.count_CSO_volume(inp,[x.strip(' ') for x in inp.Custom_CSO_ids.split(',')],model_outfile)
                opt_CSO_events = self.count_CSO_events([x.strip(' ') for x in inp.Custom_CSO_ids.split(',')],model_outfile)

        with open('../output/' + self.timestamp + '/optimized_results.txt','w') as file:
            file.write("""This is the file with the optimization results from NOAH RTC Tool optimization. \n
The model used is {}\n
Total time of computation is {:.1f} minutes\n
""".format(inp.model_name,runtime))
            
            file.write("""The best setting of the RTC setup is {:.2f}.\n\n""".format(opt_setting))
                       # , which will result in an objective value of {.:2f}.""".format(opt_setting,opt_value))
# This will result in a CSO volume from the specified CSO structures of {:.0f} m3 and {:.1f} CSO events \n\n
            
            file.write('Optimization output: \n' + str(result))
            
            file.write('\n\nThe used config file is \n\n')
            
            config = configparser.ConfigParser()
            config.read('../config/saved_configs/'+config_file)
            config.write(file)

            config_file = config.read('../../config/saved_configs/'+config_file)
    
    








