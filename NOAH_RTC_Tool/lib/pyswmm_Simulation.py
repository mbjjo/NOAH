# -*- coding: utf-8 -*-
"""
Copyright 2019 Magnus Johansen
Copyright 2019 Technical University of Denmark

This file is part of NOAH RTC Tool.

NOAH RTC Tool is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

NOAH RTC Tool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with NOAH RTC Tool. If not, see <http://www.gnu.org/licenses/>.
"""
# This is the single simulation

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import pickle
import configparser
import time
import os, shutil
from threading import Thread
from datetime import datetime
import swmmtoolbox.swmmtoolbox as swmmtoolbox
import pyswmm
from pyswmm import Simulation,Nodes,Links,SystemStats, raingages
import GUI_elements
#================================================    

#================================================    



# The config file has been validated before this step. 
def single_simulation(config_file):
#    starttime = time.time()
# configuration file is read and variables are defined according o that. 
    config = configparser.ConfigParser()
    config.read('../config/saved_configs/'+config_file)
    
    model_name = config['Model']['modelname']
    model_dir = config['Model']['modeldirectory']
    rpt_step_tmp = config['Settings']['Reporting_timesteps']      
    rpt_step = datetime.strptime(rpt_step_tmp, '%H:%M:%S').time()
    report_times_steps = rpt_step.hour*60 + rpt_step.minute + rpt_step.second/60

    # output files saved:
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.mkdir('../output/'+timestamp)   
    
    RBC = config['RuleBasedControl']
    actuator_type = RBC['actuator_type']
    if actuator_type == 'orifice':
        sensor1_id = RBC['sensor1_id']    
        actuator1_id = RBC['actuator1_id']
        sensor1_critical_depth = float(RBC['sensor1_critical_depth'])
        actuator1_target_setting_True = float(RBC['actuator1_target_setting_True'])
        actuator1_target_setting_False = float(RBC['actuator1_target_setting_False'])
        
        # Simulation is running
        with Simulation(model_dir + '/' + model_name + '.inp' , '../output/' + timestamp + '/' + model_name + '.rpt', '../output/' + timestamp + '/' + model_name + '.out') as sim: 
            for step in sim:
                # Compute control steps
                if Nodes(sim)[sensor1_id].depth > sensor1_critical_depth:
                    Links(sim)[actuator1_id].target_setting = actuator1_target_setting_True
                else:
                    Links(sim)[actuator1_id].target_setting = actuator1_target_setting_False
                print('Complete: {:.2f}'.format(sim.percent_complete))
    elif actuator_type == 'pump':

        # Parameters: 
        sensor1_id = RBC['sensor1_id']    
        actuator1_id = RBC['actuator1_id']
        sensor1_critical_depth = float(RBC['sensor1_critical_depth'])
        actuator1_target_setting_True = float(RBC['actuator1_target_setting_True'])
        actuator1_target_setting_False = float(RBC['actuator1_target_setting_False'])
        
        sensor1_critical_depth_dry = float(RBC['sensor1_critical_depth_dryflow'])
        actuator1_target_setting_True_dry = float(RBC['actuator1_target_setting_true_dryflow'])
        actuator1_target_setting_False_dry = float(RBC['actuator1_target_setting_false_dryflow'])
        
        RG1 = RBC['raingage1']
        rainfall_threshold_value = float(RBC['rainfall_threshold_value'])
        rainfall_threshold_time = float(RBC['rainfall_threshold_duration'])
    
        precipitation = list()
        
        # Simulation is running
        with Simulation(model_dir + '/' + model_name + '.inp' , '../output/' + timestamp + '/' + model_name + '.rpt', '../output/' + timestamp + '/' + model_name + '.out') as sim: 
            # help(Pump1)
            for step in sim:
                precipitation.append(raingages.RainGages(sim)[RG1].total_precip)
                # Compute control steps
                # Wet weather rule
                if np.mean(precipitation[-int(rainfall_threshold_time/report_times_steps):]) > rainfall_threshold_value:
                    if Nodes(sim)[sensor1_id].depth > sensor1_critical_depth: # Setting start level
                        Links(sim)[actuator1_id].target_setting = actuator1_target_setting_True
                    elif Nodes(sim)[sensor1_id].depth < 0.1: # Setting stop level
                        Links(sim)[actuator1_id].target_setting = actuator1_target_setting_False
        
                # Dry weather rule
                else:
                    if Nodes(sim)[sensor1_id].depth > sensor1_critical_depth_dry: # Setting start level
                        Links(sim)[actuator1_id].target_setting = actuator1_target_setting_True_dry
                    elif Nodes(sim)[sensor1_id].depth < 0.1 : # Setting stop level
                        Links(sim)[actuator1_id].target_setting = actuator1_target_setting_False_dry
                print('Complete: {:.2f}'.format(sim.percent_complete))
                
    print('Simulation ran without optimization')
                

# =============================================================================
#config_file = 'example.ini'

class Optimizer:
    def __init__(self,config_file):
        starttime = time.time()
        self.read_config(config_file)
        
#         create folder with the time stamp for the first simulation. All results are stored in this.  
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.mkdir('../output/'+self.timestamp)   
        model_outfile = '../output/' + str(self.timestamp) + '/' + str(self.model_name) + '.out'
        
        # self.Redefine_Timeseries()
        
        
#        result = self.optimized_simulation(self.initial_value) # one simulation 
#        result = self.global_optimizer() # several simulations 
        result = self.Two_step_optimizer() # 2 step optimization  
        
        print(result)
#        print(type(result))
        endtime = time.time()
        runtime = (endtime-starttime)/60
        self.write_optimal_result(result,runtime,config_file)
        GUI_elements.First_step_optimization_plot(self.timestamp)
        
        print(datetime.now())

        # Computing final simulation with optimal results
               
#        # saving results to variables
#        self.res.optimization_results = str(result)
#        self.res.optimal_setting.set(result.x)
#        self.res.volume_at_optimum.set(0.0)
#        self.res.CSO_events_at_optimum.set(10000.0)

    # configuration file is read and variables are defined according to that. 
    def read_config(self,config_file):
        config = configparser.ConfigParser()
        config.read('../config/saved_configs/'+config_file)
        
        self.system_units = config['Settings']['System_Units']
        rpt_step_tmp = config['Settings']['Reporting_timesteps']      
        rpt_step = datetime.strptime(rpt_step_tmp, '%H:%M:%S').time()
        self.report_times_steps = rpt_step.hour*60 + rpt_step.minute + rpt_step.second/60
        self.model_name = config['Model']['modelname']
        self.model_dir = config['Model']['modeldirectory']
        
    #    rain_series = config['Model']['rain_series']
        
        RBC = config['RuleBasedControl']
        self.actuator_type = RBC['actuator_type']
        self.sensor1_id = RBC['sensor1_id']    
        self.actuator1_id = RBC['actuator1_id']
        self.sensor1_critical_depth = float(RBC['sensor1_critical_depth'])
        self.actuator1_target_setting_True = float(RBC['actuator1_target_setting_True'])
        self.actuator1_target_setting_False = float(RBC['actuator1_target_setting_False'])
        self.sensor1_critical_depth_dry = float(RBC['sensor1_critical_depth_dryflow'])
        self.actuator1_target_setting_True_dry = float(RBC['actuator1_target_setting_true_dryflow'])
        self.actuator1_target_setting_False_dry = float(RBC['actuator1_target_setting_false_dryflow'])
        self.RG1 = RBC['raingage1']
        self.rainfall_threshold_value = float(RBC['rainfall_threshold_value'])
        self.rainfall_threshold_time = float(RBC['rainfall_threshold_duration'])

        Optimization = config['Optimization']
        self.useoptimization = int(Optimization['useoptimization'])
        self.optimization_method = Optimization['optimization_method']
        self.CSO_objective = Optimization['CSO_objective']
        self.CSO_id1 = Optimization['CSO_id1']
        self.CSO_id2 = Optimization['CSO_id2']
        self.CSO_id3 = Optimization['CSO_id3']
        self.optimized_parameter = Optimization['optimized_parameter']
        
        self.min_expected_Xvalue = int(Optimization['expected_min_xvalue'])
        self.max_expected_Xvalue = int(Optimization['expected_max_xvalue'])
        self.max_initial_iterations = int(Optimization['max_iterations_bashop'])
        self.maxiterations = int(Optimization['max_iterations_per_minimization'])
#        self.min_expected_Xvalue = float(Optimization['min_expected_Xvalue'])
#        self.max_expected_Xvalue = float(Optimization['max_expected_Xvalue'])
##        self.initial_value = float(Optimization['initial_value'])
#        self.max_iterations_bashop = int(Optimization['max_iterations_bashop'])
#        self.max_iteration_per_minimization = int(Optimization['max_iteration_per_minimization'])
    
# Creating a 
    def create_tmp_file(self):
        
        
        
        self.Redefine_Timeseries()

        
    
    def Redefine_Timeseries(self):
    #Changing the path of the Time series to run models.
        from swmmio.utils import modify_model
        from swmmio.utils.modify_model import replace_inp_section
        from swmmio.utils.dataframes import create_dataframeINP

        # Extracting a section (Timeseries) to a pandas DataFrame 
        path = self.model_dir
        inp_file = self.model_name
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
        # Create a temporary file with the adjusted path
        new_file = inp_file + '_tmp_.inp'
        shutil.copyfile(baseline, path + '/' + new_file)
        print('path:' + path ) 
        print('new_file:' + new_file) 
        # print('path:' + path ) 
        #Overwrite the TIMESERIES section of the new model with the adjusted data
        with pd.option_context('display.max_colwidth', 400): # set the maximum length of string to prit the full string into .inp
            replace_inp_section(path + '/' + new_file, '[TIMESERIES]', New_Timeseries)
        
        
    def Two_step_optimizer(self):
       
        # First step init: determine starting point from sequencial simulations
        starting_points = np.arange(self.min_expected_Xvalue,self.max_expected_Xvalue,(self.max_expected_Xvalue-self.min_expected_Xvalue)/self.max_initial_iterations)
        initial_simulation = np.zeros(len(starting_points))
        for i in range(len(starting_points)):
            initial_simulation[i] = self.optimized_simulation(starting_points[i])
        
        # Save the result to a pickle file
        xy = np.array([starting_points,initial_simulation]).T

        df = pd.DataFrame(xy, columns = ['starting points','objective values'])
        
        pickle_path = '../output/'+self.timestamp
        pickle_out = open(pickle_path + "/First_step_simulations.pickle","wb")
        pickle.dump(df, pickle_out)
        pickle_out.close()
        
        
        start_value = starting_points[initial_simulation.argmin()]
        
        
        if self.maxiterations > 0:
            # Second step: Simple optimization with simplex
            print('begins optimization with simplex')
            print('start_value ' + str(start_value))
            result = optimize.minimize(fun = self.optimized_simulation,
                                  x0 = start_value, method='Nelder-Mead',
                                  options = {'disp':True,'maxiter':self.maxiterations})
        else:
            result = {'Only ran the inital simulations. No optimization was performed afterwards.':[],
                      'x': [start_value]}
        print(result)
        # Run one more simulation to ensure that the saved .rpt and .out are the ones from the optimal setup. 
        self.optimized_simulation(result['x'][0])

        return result


#     def global_optimizer(self): # Not used

# #        # Define the parameter that is to be optimized
# #        if self.optimized_parameter == 'Critical depth':
# #            self.opt_param = self.sensor1_critical_depth
# #        elif self.optimized_parameter == 'Sensor location':
# #            raise('optimization of sensor location is not implemented yet')
# #        elif self.optimized_parameter == 'Actuator target setting':
# #            self.opt_param = self.actuator1_target_setting

#     # Actual optimization command (with minimization i.e. only one starting point)
# #        result = optimize.minimize(fun = self.optimized_simulation,
# #                          x0 = self.initial_value, method='Nelder-Mead',
# #                          options = {'disp':True,'maxiter':self.max_iterations})
    
#     # Actual optimization command (with basinhopping i.e. several starting points)
        
# #    parameters: 
# #        x0 = 1
# #        max_iter = 3
        
#         result = optimize.basinhopping(func = self.optimized_simulation,
#                               x0 = 1,niter = max_iter, T = 20, stepsize = 1, 
#                               minimizer_kwargs = {'method':'Nelder-Mead'},
#                           disp = True, niter_success = 5)
        
#         return result

    # optimized parameter is 'Critical depth'
    def optimized_simulation(self, x):
        
        if self.actuator_type == 'orifice':
            # Run simulation
            with Simulation(self.model_dir +'/'+ self.model_name + '.inp' , '../output/'+ self.timestamp + '/' + self.model_name + '.rpt', '../output/' + self.timestamp + '/' + self.model_name + '.out') as sim: 
                for step in sim:
                    # Compute control steps
                    if Nodes(sim)[self.sensor1_id].depth > x:
                        Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_True
                    else:
                        Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_False
        elif self.actuator_type == 'pump':
            # Run simulation
            with Simulation(self.model_dir + '/' + self.model_name + '.inp' , '../output/' + self.timestamp + '/' + self.model_name + '.rpt', '../output/' + self.timestamp + '/' + self.model_name + '.out') as sim: 
                precipitation = list()
                for step in sim:
                    precipitation.append(raingages.RainGages(sim)[self.RG1].total_precip)
                    
                    # Compute control steps
                    # Wet weather rule
                    if np.mean(precipitation[-int(self.rainfall_threshold_time/self.report_times_steps):]) > self.rainfall_threshold_value:
                    # if rainfall < X or upstream basin < X or something else:
                        if Nodes(sim)[self.sensor1_id].depth > self.sensor1_critical_depth: # Setting start level
                            Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_True
                        elif Nodes(sim)[self.sensor1_id].depth < 0.1: # Setting stop level
                            Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_False
            
                    # Dry weather rule
                    else:
                        if Nodes(sim)[self.sensor1_id].depth > x: # Parameter to be optimized (self.sensor1_critical_depth_dry):
                            Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_True_dry
                        elif Nodes(sim)[self.sensor1_id].depth < 0.1 : # Setting stop level
                            Links(sim)[self.actuator1_id].target_setting = self.actuator1_target_setting_False_dry
                    # print('Complete: {:.2f}'.format(sim.percent_complete))
                
        # Output file is defined
        model_outfile = '../output/' + self.timestamp + '/' + str(self.model_name) + '.out'
        if self.CSO_objective == 'volume':
            if self.CSO_id2 == '': # Assume that CSO_id3 is also '' (empty) 
                objective_value = self.count_CSO_volume([self.CSO_id1],model_outfile)
                print('cso1')
            elif self.CSO_id3 == '':
                objective_value = self.count_CSO_volume([self.CSO_id1,self.CSO_id2],model_outfile)
                print('cso 1 + 2')
            else:
                objective_value = self.count_CSO_volume([self.CSO_id1,self.CSO_id2,self.CSO_id3],model_outfile)
                print('all cso')
            print(objective_value)
            
        elif self.CSO_objective =='frequency':
            if self.CSO_id2 == '': # Assume that CSO_id3 is also '' (empty) 
                objective_value = self.count_CSO_events([self.CSO_id1],model_outfile)
                print('cso1')
            elif self.CSO_id3 == '':
                objective_value = self.count_CSO_events([self.CSO_id1,self.CSO_id2],model_outfile)
                print('cso 1 + 2')
            else:
                objective_value = self.count_CSO_events([self.CSO_id1,self.CSO_id2,self.CSO_id3],model_outfile)
                print('all cso')
            print(objective_value)
            
        return objective_value
                    
#===================================================================     

# Objective functions    

    # Compute CSO volume
    
    
    def count_CSO_volume(self,CSO_ids, model_outfile): 
    # Note that the VSO's are now computed as regular nodes where flooding simulates overflow. 
    # If CSO's are outlets the 'Flow_lost_flooding' should be changed to 'Total_inflow'
        df = pd.concat(((swmmtoolbox.extract(model_outfile,['node',CSO_ids[i],'Flow_lost_flooding']))for i in range(len(CSO_ids))),axis = 1)
        df = df*self.report_times_steps*60 
        CSO_volume = df.sum().sum()
        return CSO_volume 

        #===================================================================     
    
    # Compute CSO frequency 
       
    def count_CSO_events(self,CSO_ids, model_outfile):
    # Note that the VSO's are now computed as regular nodes where flooding simulates overflow. 
    # If CSO's are outlets the 'Flow_lost_flooding' should be changed to 'Total_inflow'
    
        # input:
        # CSO_ids in a list of ids on CSO to be monitored
        
    
        def fill_zero(df,dt,max_length,val):
            # dt is the time between two CSO events before they are counted as seperate events 
            # max_lengt is the maximum time allowed for one continous CSO event before it is counted as several events. 
            import numpy as np
        
            for i in range(df.shape[1]): # .shape[1] = the number of columns
                # np.equal(df.iloc[:, i].values, 0) = gives True if two consecutive number are the same 
                #  np.concatenate(...) = join sequence and adds 0 at the start and end of the array 
                iszero = np.concatenate(([0], np.equal(df.iloc[:, i].values, 0).view(np.int8), [0]))
                
                
                absdiff = np.abs(np.diff(iszero))
                # find where the zeros are located in the dataframe
                zerorange = np.where(absdiff == 1)[0].reshape(-1, 2) # Contains "dry periods" 
                
                 
                # for loop - writes 1 if the number of consecutive zeros are less than 120 
                # timesteps (dt). 1 dt is 0.5 seconds 
                for j in range(len(zerorange)):
                    if zerorange[j][1] - zerorange[j][0] < dt: # If dry periods are less than dt these are counted. 
                        df.iloc[zerorange[j][0]:zerorange[j][1], i] = val
                # Max_lengt is the maximum duration of an overflow. before it is counted several times. 
                for j in range(0,len(zerorange)-1): 
                    if zerorange[j+1][0]- zerorange[j][1] > max_length:
                        df.iloc[zerorange[j][0]:zerorange[j][1], i] = val* (zerorange[j+1][0]- zerorange[j][1])/max_length
                if df.shape[0] - zerorange[-1][1] > max_length:# last row
                    df.iloc[zerorange[-1][1]:, i] = val* (df.shape[0] - zerorange[-1][1])/max_length
        # If all CSO's are to be counted as one id the following line should be changed with one merged time series. 
        df = pd.concat(((swmmtoolbox.extract(model_outfile,['node',CSO_ids[i],'Flow_lost_flooding']))for i in range(len(CSO_ids))),axis = 1)
        # changing the dataframe to zeroes and ones 
        onezero = np.sign(df)
        
        # calling fill_zero function. 1440 is the amount of minutes (24hr) chosen to be the limit between CSO events
        # Time steps i simualtion are 5 minutes.
        # if there is less than this time between overflow events these are considered as the same overflow event.
        
        fill_zero(onezero,dt = 1440/self.report_times_steps,max_length = 1440/self.report_times_steps,val = 1)
      
        group_ids = onezero != onezero.shift()
        CSO_events = onezero[group_ids].sum().sum()
        
        # number of CSO is then max(group_ids)
        # We add all events together. 
#        CSO_events = sum([max(group_ids['node_' + str(CSO_ids[i]) + '_Flow_lost_flooding'])for i in range(len(CSO_ids))])
        
        return CSO_events
     
    #===================================================================     

# Writing the optimal results to a text file 
        
    def write_optimal_result(self,result,runtime, config_file):
        # Calculate optimial results
        opt_setting = result['x'][0]

        model_outfile = '../output/' + self.timestamp + '/' + str(self.model_name) + '.out'
        if self.CSO_id2 == '': # Assume that CSO_id3 is also '' (empty) 
            opt_vol = self.count_CSO_volume([self.CSO_id1],model_outfile)
            opt_CSO_events = self.count_CSO_events([self.CSO_id1],model_outfile)
        elif self.CSO_id3 == '':
            opt_vol = self.count_CSO_volume([self.CSO_id1,self.CSO_id2],model_outfile)
            opt_CSO_events = self.count_CSO_events([self.CSO_id1,self.CSO_id2],model_outfile)
        else:
            opt_vol = self.count_CSO_volume([self.CSO_id1,self.CSO_id2,self.CSO_id3],model_outfile)
            opt_CSO_events = self.count_CSO_events([self.CSO_id1,self.CSO_id2,self.CSO_id3],model_outfile)
        
        with open('../output/' + self.timestamp + '/optimized_results.txt','w') as file:
            file.write("""This is the file with the optimization results from NOAH RTC Tool optimization. \n
The model used is {}\n
Total time of computation is {} minutes\n
""".format(self.model_name,runtime))
            
            file.write("""The best setting of the RTC setup is {:.2f}.
This will result in a CSO volume from the specified CSO structures of {:.0f} m3 and {:.1f} CSO events \n\n
""".format(opt_setting,opt_vol,opt_CSO_events))
            
            file.write('Optimization output: \n' + str(result))
            
            file.write('\n\nThe used config file is \n\n')
            
            config = configparser.ConfigParser()
            config.read('../config/saved_configs/'+config_file)
            config.write(file)

#            config_file = config.read('../../config/saved_configs/'+config_file)
            
            
        
#    with Simulation('../../model/'+ model_name + '.inp' , '../../output/' + time + '/' + model_name + '.rpt', '../../output/' + time + '/' + model_name + '.out') as sim: 

#================================================     
