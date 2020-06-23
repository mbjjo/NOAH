# -*- coding: utf-8 -*-
"""
Copyright 2019 Magnus Johansen
Copyright 2019 Technical University of Denmark

This file is part of NOAH RTC Tool.

NOAH RTC Tool is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

NOAH RTC Tool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with NOAH RTC Tool. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import os
# ME = os.path.basename(sys.argv[0])
# DIR = os.path.dirname(sys.argv[0])
# LIB_DIR = os.sep.join([DIR,'..','interface'])
# ABS_LIB_DIR = os.path.abspath(LIB_DIR)
# sys.path.append(ABS_LIB_DIR)
 
# LIB_DIR = os.sep.join([DIR,'..','simulation'])
# ABS_LIB_DIR = os.path.abspath(LIB_DIR)
# sys.path.append(ABS_LIB_DIR)

# Required imports created for the GUI 
import GUI_elements
import pyswmm_Simulation

# Required external imports 
import pandas as pd
import numpy as np
from scipy import optimize,integrate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from datetime import datetime,timedelta
import configparser
import threading

# import the necessary modelus from pyswmm
import pyswmm
from pyswmm import Simulation, Nodes, Links, SystemStats, Subcatchments

import tkinter as tk
from tkinter import *
from tkinter import Tk, ttk, filedialog, scrolledtext,messagebox
from tkinter import messagebox as msg
from tkintertable.Tables import TableCanvas
from tkintertable.TableModels import TableModel


class pyswmm_GUI:

    def __init__(self,*args,**kwargs):
        
        # Define window properties
        self.window = Tk()
        self.window.iconbitmap('./GUI_files/noah_logo_OAb_icon.ico')
        self.window.title('NOAH RTC Tool') 
        # window.configure(background = 'white')
#        self.window.geometry('600x300')
        self.window.resizable(False,False)
    
        # parameters that are used before the config file is written are defined here
        self.param = GUI_elements.parameters()

        # Widgets are created
        self.create_widgets()
        
    def create_widgets(self):
# =============================================================================
# Define top frame
        # Model selsction
        self.label1 = Label(self.window, text = "Model Name")
        self.label1.grid(row=0, column = 1)
    
        self.model_button = Button(self.window, text ='Choose model file',width = 15, command = lambda: GUI_elements.OpenFile(self))
        self.model_button.grid(row = 0, column = 0,sticky = E)
        GUI_elements.create_ToolTip(self.model_button,"Choose the SWMM model that shoul be used for the simulations")
                
        self.model_label = Label(self.window, width = 30, bg = 'white', textvariable = self.param.model_name)
        self.model_label.grid(row=0, column = 2)
        
        # Creates a logo in the corner 
        self.Interreg_logo = PhotoImage(file = './GUI_files/Interreg_logo_05.gif')
        Label(self.window,image = self.Interreg_logo).grid(row = 0, column = 3,sticky = E)

# =============================================================================
# Define middle frame (tabs) instances of classes
        
        self.tabControl = ttk.Notebook(self.window)
        self.tabControl.grid(row = 2, columnspan = 5)

        self.Simulation_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.Simulation_tab,text = 'RTC setup')
        
        self.Rain_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.Rain_tab,text = 'rain series')
        
        self.Control_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.Control_tab,text = 'Control objective')
        
        self.Optimize_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.Optimize_tab,text = 'RTC Optimization')
        
        self.Calibration_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.Calibration_tab,text = 'Model Calibration')
        
        self.Result_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.Result_tab,text = 'Results')
        
# =============================================================================
# Define bottom frame  
        # Define executeable buttons 
        self.run_button = Button(self.Simulation_tab, text ='Run RTC ', width = 15, command = lambda:GUI_elements.run(self))
        self.run_button.grid(row = 6, column = 0,sticky = W,pady = 5, padx = 5)
        GUI_elements.create_ToolTip(self.run_button,'Runs the simulation with the configuration file.')
        
        self.write_config = Button(self.Simulation_tab, text ='Save config', width = 15, command = lambda:GUI_elements.write_config(self))
        self.write_config.grid(row = 6, column = 1,sticky = W,pady = 5, padx = 5)
        GUI_elements.create_ToolTip(self.write_config,"Save the configuration file without running.")
        
        self.overwrite_button_sim = Checkbutton(self.Simulation_tab, text = "Overwrite existing configuation file", variable = self.param.Overwrite_config)
        self.overwrite_button_sim.deselect()    
        self.overwrite_button_sim.grid(row = 5, column = 0,sticky = W, columnspan = 3)
        GUI_elements.create_ToolTip(self.overwrite_button_sim,"Check if the parameters specified in the GUI should be written to configuration file. Else the existing file is used.")

        # self.SWMM_results = Button(self.window, text ='Write to SWMM file', width = 15, command = lambda: GUI_elements.generate_SWMM_file(self))
        # self.SWMM_results.grid(row = 5, column = 2,sticky = W,pady = 5, padx = 5)
        # GUI_elements.create_ToolTip(self.SWMM_results,"Write the current RTC setup (specified in the simulation tab) to a SWMM file.")
        
        Button(self.Simulation_tab, text ='Exit',width = 15, command = lambda:self.window.destroy()).grid(row = 6, column = 2,sticky =E, padx = 5)
        
# =============================================================================
    # Create content for tabs
# =============================================================================
# Creatng the content for the simulation tab 

   
    # RBC frame
        self.RBC_frame = ttk.LabelFrame(self.Simulation_tab, text = 'Control rules setup')
        self.RBC_frame.grid(row = 1, column = 0,columnspan = 2, pady =5, padx = 5,sticky = NSEW)
        GUI_elements.create_ToolTip(self.RBC_frame,'Define the rules that are applied as Rule Based Control in SWMM.')
        
         # Radiobuttons for actuator type 
        Label(self.RBC_frame, text = "Select the type of the actuator:").grid(row=1,column = 0, sticky = 'W',columnspan = 2)
        self.orifice_active = Radiobutton(self.RBC_frame,text = "Orifice",var = self.param.actuator_type,value = 'orifice', command = lambda:GUI_elements.orifice_or_pump(self))
        self.orifice_active.grid(row = 2,column = 0,sticky = W,columnspan = 3,pady = 5)
        self.orifice_active.select()
        self.weir_active = Radiobutton(self.RBC_frame,text = "Weir",var = self.param.actuator_type,value = 'weir', command = lambda:GUI_elements.orifice_or_pump(self))
        self.weir_active.grid(row=2,column = 1,sticky = W,columnspan = 2,pady = 5)
        self.weir_active.deselect()
        self.pump_active = Radiobutton(self.RBC_frame,text = "Pump",var = self.param.actuator_type,value = 'pump', command = lambda:GUI_elements.orifice_or_pump(self))
        self.pump_active.grid(row=2,column = 2,sticky = W,columnspan = 2,pady = 5)
        self.pump_active.deselect()
        
        # Wet weather rule
        Label(self.RBC_frame, text = "Rule applied during rainfall").grid(row=9,column = 0,columnspan = 3,sticky = 'W')
        
        # Sensor1
        Label(self.RBC_frame, text = "IF depth in (Sensor)").grid(row=10,column = 0, sticky = 'W')
        self.sensor1id = Entry(self.RBC_frame,width = 15)
        self.sensor1id.bind("<FocusOut>", lambda x: GUI_elements.update(self.sensor1id, self.sensor1id_dry))
        self.sensor1id.grid(row=10, column=1, sticky = 'W')
        GUI_elements.create_ToolTip(self.sensor1id,'Type in the node id of the sensor')
        
        self.sign = Label(self.RBC_frame, text = ">")
        self.sign.grid(row=10,column = 3)
                
        self.sensor1setting = Entry(self.RBC_frame,width = 10)
        self.sensor1setting.grid(row=10, column=4, sticky = 'W')
        GUI_elements.create_ToolTip(self.sensor1setting,'Type in the depth that activates the actuator [m]')
        
        # Actuator1
        Label(self.RBC_frame, text = "THEN Actuator").grid(row=11,column = 0, sticky = 'W')
        self.actuator1id = Entry(self.RBC_frame,width = 15)
        self.actuator1id.bind("<FocusOut>", lambda x: GUI_elements.update(self.actuator1id, self.actuator1id_dry))
        self.actuator1id.grid(row=11, column=1, sticky = 'W')
        GUI_elements.create_ToolTip(self.actuator1id,'Type in the id of the actuator')
    
        Label(self.RBC_frame, text = "Setting").grid(row=11,column = 3, sticky = 'W')
        self.actuator1setting_True = Entry(self.RBC_frame,width = 10)
        self.actuator1setting_True.grid(row=11, column=4, sticky = 'W')
        # self.actuator1setting.insert(0,'1')
        GUI_elements.create_ToolTip(self.actuator1setting_True,'Type in the setting of the actuator when critical depth is exceeded')
        
        Label(self.RBC_frame, text = "ELSE Actuator").grid(row=12,column = 0, sticky = 'W')
        Label(self.RBC_frame, text = "Setting").grid(row=12,column = 3, sticky = 'W')
        self.actuator1setting_False = Entry(self.RBC_frame,width = 10)
        self.actuator1setting_False.grid(row=12, column=4, sticky = 'W')
        # self.actuator1setting_False.insert(0,'0')
        GUI_elements.create_ToolTip(self.actuator1setting_False,'Type in the setting of the actuator when critical depth is NOT exceeded')

        self.Dry_flow_frame = ttk.LabelFrame(self.Control_tab, text = '')
        self.Dry_flow_frame.grid(row = 1, column = 0,rowspan = 1, pady =5, padx = 5,sticky = NSEW)
        


    # Dry flow or rainfall threshold
        Label(self.Dry_flow_frame, text = 'Rainfall or Dryflow').grid(row = 3,column = 0,sticky = W,columnspan = 2,pady = 5)
        Label(self.Dry_flow_frame, text = 'If rainfall exceeds').grid(row = 4,column = 0,sticky = W,columnspan = 1)
        self.rainfall_threshold = Entry(self.Dry_flow_frame,width = 10)
        self.rainfall_threshold.grid(row= 4, column = 1, sticky = W)
        GUI_elements.create_ToolTip(self.rainfall_threshold,'The average precipitation [mm/hr] over a certain duration that activates the rainfall rule')
        Label(self.Dry_flow_frame, text = 'During more than').grid(row = 5,column = 0,sticky = W,columnspan = 1)
        self.rainfall_time= Entry(self.Dry_flow_frame,width = 10)
        self.rainfall_time.grid(row= 5, column = 1, sticky = W)
        GUI_elements.create_ToolTip(self.rainfall_time,'The duration of the rainfall before the Raifall rule applies. [minutes]')
        Label(self.Dry_flow_frame, text = 'Raingage id').grid(row = 6,column = 0,sticky = W,columnspan = 1)
        self.raingage1 = Entry(self.Dry_flow_frame,width = 10)
        self.raingage1.grid(row = 6, column = 1,sticky = W)
        GUI_elements.create_ToolTip(self.raingage1,'ID of the raingage in the system')
        
    # Dry weather rule
        Label(self.Dry_flow_frame, text = "Rule applied during dry weather flow").grid(row=14,column = 0,columnspan = 3)

        Label(self.Dry_flow_frame, text = "IF depth in (Sensor)").grid(row=15,column = 0, sticky = 'W')
        self.sensor1id_dry = Entry(self.Dry_flow_frame,width = 15)
        self.sensor1id_dry.bind("<FocusOut>", lambda x: GUI_elements.update(self.sensor1id_dry, self.sensor1id))
        # self.sensor1id_dry = Label(self.Dry_flow_frame,width = 15)
        self.sensor1id_dry.grid(row=15, column=1, sticky = 'W')
        GUI_elements.create_ToolTip(self.sensor1id_dry,'Type in the node id of the sensor')
                
        self.sign = Label(self.Dry_flow_frame, text = ">")
        self.sign.grid(row=15,column = 3)
                
        self.sensor1setting_dry = Entry(self.Dry_flow_frame,width = 10)
        self.sensor1setting_dry.grid(row=15, column=4, sticky = 'W')
        GUI_elements.create_ToolTip(self.sensor1setting_dry,'Type in the depth that activates the actuator [m]')
        
        # Actuator1
        Label(self.Dry_flow_frame, text = "THEN Actuator").grid(row=16,column = 0, sticky = 'W')
        self.actuator1id_dry = Entry(self.Dry_flow_frame,width = 15)
        self.actuator1id_dry.bind("<FocusOut>", lambda x: GUI_elements.update(self.actuator1id_dry, self.actuator1id))
        self.actuator1id_dry.grid(row=16, column=1, sticky = 'W')
        GUI_elements.create_ToolTip(self.actuator1id_dry,'Type in the id of the actuator')
    
        Label(self.Dry_flow_frame, text = "Setting").grid(row=16,column = 3, sticky = 'W')
        self.actuator1setting_True_dry = Entry(self.Dry_flow_frame,width = 10)
        self.actuator1setting_True_dry.grid(row=16, column=4, sticky = 'W')
        # self.actuator1setting.insert(0,'1')
        GUI_elements.create_ToolTip(self.actuator1setting_True_dry,'Type in the setting of the actuator when critical depth is exceeded')
        
        Label(self.Dry_flow_frame, text = "ELSE Actuator").grid(row=17,column = 0, sticky = 'W')
        Label(self.Dry_flow_frame, text = "Setting").grid(row=17,column = 3, sticky = 'W')
        self.actuator1setting_False_dry = Entry(self.Dry_flow_frame,width = 10)
        self.actuator1setting_False_dry.grid(row=17, column=4, sticky = 'W')
        # self.actuator1setting_False.insert(0,'0')
        GUI_elements.create_ToolTip(self.actuator1setting_False_dry,'Type in the setting of the actuator when critical depth is NOT exceeded')
        
        
        # # Simulation period frame
        # self.RTC_period_frame = ttk.LabelFrame(self.Simulation_tab, text = 'Simulation period')
        # self.RTC_period_frame.grid(row =2, column = 2,rowspan = 1, pady =5, padx = 5,sticky = NSEW)
        # # Simulation period
        # Label(self.RTC_period_frame, text = "Start time").grid(row= 1,column =0, sticky = E)
        # Label(self.RTC_period_frame, text = "End time").grid(row= 2,column =0,sticky = E)
        # self.Start_sim_time = Entry(self.RTC_period_frame,width = 15)
        # self.Start_sim_time.grid(row=1, column=1,sticky = W)
        # GUI_elements.create_ToolTip(self.Start_sim_time,'Specify start time of the simulation in the format "yyyy-mm-dd HH:MM".')
        # self.End_sim_time= Entry(self.RTC_period_frame,width = 15)
        # self.End_sim_time.grid(row=2, column=1,sticky = W)
        # GUI_elements.create_ToolTip(self.End_sim_time,'Specify end time of the simulation in the format "yyyy-mm-dd HH:MM".')
        
        # self.UseHotstart_sim = Checkbutton(self.RTC_period_frame, text = "Use hotstart", variable = self.param.use_hotstart_sim,command = lambda:GUI_elements.update_Hotstart(self,self.param.use_hotstart_sim,self.hotstart_period_h_sim))
        # # self.UseHotstart.deselect()
        # self.UseHotstart_sim.grid(row = 3, column = 0,columnspan = 2,sticky = W)
        # GUI_elements.create_ToolTip(self.UseHotstart_sim,"Choose whether a Hotstart period is used.")

        # Label(self.RTC_period_frame, text = "Hotstart period").grid(row=4,column = 0,sticky='E')
        # self.hotstart_period_h_sim = Entry(self.RTC_period_frame,width = 5)
        # self.hotstart_period_h_sim.grid(row=4, column=1,sticky = W)
        # self.hotstart_period_h_sim.insert(END,5)
        # self.hotstart_period_h_sim.configure(state = 'disabled')
        # GUI_elements.create_ToolTip(self.hotstart_period_h_sim,"Specify the length of the hotstart period [hours].")
        
# =============================================================================

# Creatng the content for the rain tab 
        
        Label(self.Rain_tab, text = "Contains a tool for designing rain events that can be imported to the SWMM model", bg = 'white').grid(row=0,column = 0, columnspan = 3, pady  =8)
                
        self.hotstart_button = Button(self.Rain_tab, text ='Choose hotstart file',width = 15, command = lambda:msg.showerror('','Hotstart is not implemented yet'))
        self.hotstart_button.grid(row = 2, column = 1,sticky = 'W')
        GUI_elements.create_ToolTip(self.hotstart_button,"Choose the SWMM model that shoul be used for the simulations")
        
        # Define frame 
        self.Rain_event_frame = ttk.LabelFrame(self.Rain_tab, text = 'Definition of rain event')
        self.Rain_event_frame.grid(row = 2, column = 0,rowspan = 1, pady =5, padx = 15,sticky = NSEW)
        
        Label(self.Rain_event_frame, text = "Rain event minimum intensity").grid(row=0,column = 0, sticky = 'W')
        self.rain_event_intensity = Entry(self.Rain_event_frame,width = 10)
        self.rain_event_intensity.grid(row=0, column=1, sticky = 'W')
        # self.rain_event_intensity.insert(0,'1')
        GUI_elements.create_ToolTip(self.rain_event_intensity ,'Type in the minimum intensity of rain that is counted as an event [Unit]')
        
        Label(self.Rain_event_frame, text = "Rain event minimum duration").grid(row=1,column = 0, sticky = 'W')
        self.rain_event_duration = Entry(self.Rain_event_frame,width = 10)
        self.rain_event_duration.grid(row=1, column=1, sticky = 'W')
        # self.rain_event_intensity.insert(0,'1')
        GUI_elements.create_ToolTip(self.rain_event_duration,'Type in the minimum duration of continous rain that is counted as an event [Unit]')
        
        Label(self.Rain_event_frame, text = "More options...").grid(row=2,column = 0, sticky = 'W')
        # self.rain_event_duration = Entry(self.Rain_event_frame,width = 10)
        # self.rain_event_duration.grid(row=1, column=1, sticky = 'W')
        # # self.rain_event_intensity.insert(0,'1')
        # GUI_elements.create_ToolTip(self.rain_event_duration,'Type in the minimum duration of continous rain that is counted as an event [Unit]')
        
# =============================================================================
# Creatng the content for the objective tab 

        Label(self.Control_tab,text = 'Choose the objective of the optimization.').grid(row = 0,columnspan = 3)
        
        # Objective function frame
        self.objective_frame= ttk.LabelFrame(self.Simulation_tab,width=385, height=460, text = 'Control objective')
        self.objective_frame.grid(row =1, column = 2,rowspan = 1, pady =5, padx = 5,sticky = NSEW)
        
        # Define radiobuttons for CSO objective
        self.CSO_volume = Radiobutton(self.objective_frame,text = "Reduce CSO volume",var = self.param.CSO_objective,value = 'volume')
        self.CSO_volume.grid(row = 1,column = 0,columnspan = 2,sticky = W)
        self.CSO_volume.select()
        self.CSO_freq = Radiobutton(self.objective_frame,text = "Reduce CSO frequency",var = self.param.CSO_objective,value = 'frequency')
        self.CSO_freq.grid(row=2,column = 0,columnspan = 2,sticky = W)
        # self.CSO_freq.select()

        Label(self.objective_frame,text = "Reduce CSO from").grid(row=3,column = 0, columnspan = 1)
        self.CSO_id1 = Entry(self.objective_frame,width = 15)
        self.CSO_id1.grid(row = 3,column = 1)
        self.CSO_id2 = Entry(self.objective_frame,width = 15)
        self.CSO_id2.grid(row = 4,column = 1)
        self.CSO_id3 = Entry(self.objective_frame,width = 15)
        self.CSO_id3.grid(row = 5,column = 1)
        GUI_elements.create_ToolTip(self.CSO_id1,"Write the node ID of the CSO struture")
        GUI_elements.create_ToolTip(self.CSO_id2,"Write the node ID of the CSO struture")
        GUI_elements.create_ToolTip(self.CSO_id3,"Write the node ID of the CSO struture")
        
        Label(self.objective_frame,text = "OR provide list:").grid(row=6,column = 0, columnspan = 1)
        self.Custom_CSO_ids= Entry(self.objective_frame,width = 30)
        self.Custom_CSO_ids.grid(row = 7,column = 0, columnspan = 3)
        GUI_elements.create_ToolTip(self.Custom_CSO_ids,"Provide a list of CSO's. seperate with , \nIf applied the 3 above are ignored")
        self.Custom_CSO_ids.bind("<FocusOut>", lambda x: GUI_elements.check_custom_ids(self))
        
        # CSO Settings frame
        self.CSO_settings_frame = ttk.LabelFrame(self.Simulation_tab, text = 'CSO settings')
        self.CSO_settings_frame.grid(row = 4, column = 0,rowspan = 1, columnspan = 2, pady =5, padx = 5,sticky= NSEW)
        
        Label(self.CSO_settings_frame, text = "Time seperation CSO events").grid(row=0,column = 0,sticky = E)
        self.CSO_event_seperation = Entry(self.CSO_settings_frame,width = 5)
        self.CSO_event_seperation.grid(row = 0, column = 1,sticky = W)
        self.CSO_event_seperation.insert(END, '12')
        GUI_elements.create_ToolTip(self.CSO_event_seperation,"Define the minimum time (hours) seperating two events.")
        Label(self.CSO_settings_frame, text = "Maximum duration of CSO event").grid(row=1,column = 0,sticky='E')
        self.CSO_event_duration = Entry(self.CSO_settings_frame,width = 5)
        self.CSO_event_duration.grid(row = 1, column = 1,sticky = W)
        self.CSO_event_duration.insert(END, '24')
        GUI_elements.create_ToolTip(self.CSO_event_duration,"Define the time (hours) before an event is counted as more events.")
        Label(self.CSO_settings_frame, text = "Define CSO as").grid(row=2,column = 0,sticky='E')
        self.CSO_type = ttk.Combobox(self.CSO_settings_frame,width = 20,values = ('Outflow from CSO structure','Flodding above ground'),state = 'readonly')
        self.CSO_type.current(0)
        self.CSO_type .grid(row=2, column=1)      

# =============================================================================
# Creatng the content for the optimization tab 
         # optimization frame
        self.optimization_frame = ttk.LabelFrame(self.Simulation_tab, text = 'Optimization parameters')
        self.optimization_frame.grid(row =2, column = 0,columnspan = 2, pady =5, padx = 5,sticky = NSEW)
        # Label(self.Optimize_tab, text = "Choose optimization parameters").grid(row=2,column = 0, columnspan = 3, pady  =8)
         
        self.optimize_check = Checkbutton(self.optimization_frame, text = "Use optimization", variable = self.param.UseOptimization, command = lambda: GUI_elements.enable_RTC_optimization(self))
        self.optimize_check.deselect()
        self.optimize_check.grid(row = 0, column = 1,sticky = W)
        GUI_elements.create_ToolTip(self.optimize_check,"Choose whether optimization should be used. If not only a single simulation is run.")
        
        Label(self.optimization_frame, text = "Optimization method").grid(row=3,column = 0,sticky='E')
        self.opt_method = ttk.Combobox(self.optimization_frame,width = 20,values = ('Two-step-optimization'),state = 'disabled')
        self.opt_method.grid(row=3, column=1)
        
        Label(self.optimization_frame, text = "Parameter to be optimized").grid(row=4,column = 0,sticky='E')
        self.optimized_parameter= ttk.Combobox(self.optimization_frame,width = 20,values = ('Activation depth','Sensor location', 'Sensor location and activation depth'),state = 'disabled')
        self.optimized_parameter.bind("<<ComboboxSelected>>",lambda x: GUI_elements.enable_sensor_location(self))
        self.optimized_parameter.grid(row=4, column=1)
        
        Label(self.optimization_frame, text = "Minimum value of parameter range").grid(row=5,column = 0,sticky='E')
        self.expected_min_Xvalue = Entry(self.optimization_frame,width = 5,state='disabled')
        self.expected_min_Xvalue.grid(row=5, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.expected_min_Xvalue,"Specify the minimum value of the parameter range. The optimizer will not find a value below this point. ")
        
        Label(self.optimization_frame, text = "Maximum value of parameter range").grid(row=6,column = 0,sticky='E')
        self.expected_max_Xvalue = Entry(self.optimization_frame,width = 5,state='disabled')
        self.expected_max_Xvalue.grid(row=6, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.expected_max_Xvalue,"Specify the maximum value of the parameter range. The optimizer will not find a value above this point. ")
        
        Label(self.optimization_frame, text = "Max initial simulations").grid(row=7,column = 0,sticky='E')
        self.max_initial_iterations = Entry(self.optimization_frame,width = 5,state='disabled')
        self.max_initial_iterations.grid(row=7, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.max_initial_iterations,"Specify the number of simulations that are done in the initial screening of the optimization.")
        
        Label(self.optimization_frame, text = "Max simplex simulations").grid(row=8,column = 0,sticky='E')
        self.max_iterations_per_minimization = Entry(self.optimization_frame,width = 5,state='disabled')
        self.max_iterations_per_minimization.grid(row=8, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.max_iterations_per_minimization,"Specify the maximum allowed number of simulations for the second part of the optimization.")
        
         # Sensor location frame
        self.sensor_loc_frame = ttk.LabelFrame(self.Optimize_tab, text = 'Possible sensor locations')
        self.sensor_loc_frame.grid(row =1, column = 1,rowspan = 20, pady =5, padx = 5,sticky = NSEW)
        
        Label(self.sensor_loc_frame, text = "Select possible sensor locations").grid(row=0,column = 0,sticky = W)
        self.sensor_loc1 = Entry(self.sensor_loc_frame,width = 15)
        self.sensor_loc1.grid(row = 1, column = 0,sticky = W)
        self.sensor_loc2 = Entry(self.sensor_loc_frame,width = 15)
        self.sensor_loc2.grid(row = 2, column = 0,sticky = W)
        self.sensor_loc3 = Entry(self.sensor_loc_frame,width = 15)
        self.sensor_loc3.grid(row = 3, column = 0,sticky = W)
        GUI_elements.create_ToolTip(self.sensor_loc_frame,"Define the ID of nodes with possibility of a sensor.")
        # self.sensor_loc1.insert(END,self.sensor1id.get())
             
# =============================================================================
# Content for the calibraion tab

        # Clibration parameter frame
        self.calib_param_frame = ttk.LabelFrame(self.Calibration_tab, text = 'Calibration parameter')
        self.calib_param_frame.grid(row =1, column = 0,columnspan = 3 ,rowspan = 1, pady =5, padx = 5,sticky = NSEW)
        GUI_elements.create_ToolTip(self.calib_param_frame,"Check the parameters that are to be calibrated and specify minimum and maximum allowed values.")
        
        Label(self.calib_param_frame, text = "Parameter").grid(row=1,column = 0)
        Label(self.calib_param_frame, text = "Minimum value").grid(row=1,column = 1)
        Label(self.calib_param_frame, text = "maximum value").grid(row=1,column = 2)
        
        self.percent_imp = Checkbutton(self.calib_param_frame, text = "% Imperv", variable = self.param.Calib_perc_imp,
                                       command = lambda: GUI_elements.update_min_max_calib(self,self.param.Calib_perc_imp,self.percent_imp_min,self.percent_imp_max,0.5,1.5))
        self.percent_imp.deselect()
        self.percent_imp.grid(row = 2, column = 0,sticky = W)
        self.percent_imp_min = Entry(self.calib_param_frame,width = 5)#,state = 'disabled') 
        self.percent_imp_min.grid(row=2,column=1)
        self.percent_imp_max = Entry(self.calib_param_frame,width = 5)#,state = 'disabled') 
        self.percent_imp_max.grid(row=2,column=2)
        self.percent_imp_min.insert(END,0.5)
        self.percent_imp_min.configure(state= 'disabled')
        self.percent_imp_max.insert(END,1.5)
        self.percent_imp_max.configure(state= 'disabled')
    
        self.Width = Checkbutton(self.calib_param_frame, text = "Width", variable = self.param.Calib_width,
                                 command = lambda: GUI_elements.update_min_max_calib(self,self.param.Calib_width,self.Width_min,self.Width_max,0.2,5.0))
        self.Width.deselect()
        self.Width.grid(row = 3, column = 0,sticky = W)
        self.Width_min = Entry(self.calib_param_frame,width = 5)#,state = 'disabled') 
        self.Width_min.grid(row=3,column=1)
        self.Width_max = Entry(self.calib_param_frame,width = 5)#,state = 'disabled') 
        self.Width_max.grid(row=3,column=2)
        self.Width_min.insert(END,0.2) 
        self.Width_min.configure(state = 'disabled')
        self.Width_max.insert(END,5.0)
        self.Width_max.configure(state = 'disabled')
        
        self.Dstore = Checkbutton(self.calib_param_frame, text = "Initial loss", variable = self.param.Calib_Dstore,
                                  command = lambda: GUI_elements.update_min_max_calib(self,self.param.Calib_Dstore,self.Dstore_min,self.Dstore_max,0.33,3.0))
        self.Dstore.deselect()
        self.Dstore.grid(row = 4, column = 0,sticky = W)
        self.Dstore_min = Entry(self.calib_param_frame,width = 5)#,state = 'disabled') 
        self.Dstore_min.grid(row=4,column=1)
        self.Dstore_max = Entry(self.calib_param_frame,width = 5)#,state = 'disabled') 
        self.Dstore_max.grid(row=4,column=2)
        self.Dstore_min.insert(END,0.33)
        self.Dstore_min.configure(state= 'disabled')
        self.Dstore_max.insert(END,3.0)
        self.Dstore_max.configure(state= 'disabled')
        self.n_pipe = Checkbutton(self.calib_param_frame, text = "Roughness (pipes)", variable = self.param.Calib_n_pipe,
                                  command = lambda: GUI_elements.update_min_max_calib(self,self.param.Calib_n_pipe,self.n_pipe_min,self.n_pipe_max,0.7,1.3))
        self.n_pipe.deselect()
        self.n_pipe.grid(row = 5, column = 0,sticky = W)
        self.n_pipe_min = Entry(self.calib_param_frame,width = 5)#,state = 'disabled') 
        self.n_pipe_min.grid(row=5,column=1)
        self.n_pipe_max = Entry(self.calib_param_frame,width = 5)#,state = 'disabled') 
        self.n_pipe_max.grid(row=5,column=2)
        self.n_pipe_min.insert(END,0.7)
        self.n_pipe_min.configure(state= 'disabled')
        self.n_pipe_max.insert(END,1.3)
        self.n_pipe_max.configure(state= 'disabled')
        
        #Observations area frame
        self.calib_obs_frame = ttk.LabelFrame(self.Calibration_tab, text = 'Observations')
        self.calib_obs_frame .grid(row =2, column = 0,columnspan =3, pady =5, padx = 5,sticky = NSEW)

        self.obs_button = Button(self.calib_obs_frame, text ='Select observations',width = 15, command = lambda: GUI_elements.select_obs(self))
        self.obs_button.grid(row = 2, column = 0,sticky = E)
        GUI_elements.create_ToolTip(self.obs_button,"Choose the file that contains the observed data.")
        self.obs_label = Label(self.calib_obs_frame, width = 20, bg = 'white', textvariable = self.param.obs_data)
        self.obs_label.grid(row=2, column = 1,sticky = E)
        Label(self.calib_obs_frame, text = "Sensor location").grid(row= 3,column =0,columnspan = 1,sticky = E)
        self.sensor_calib = Entry(self.calib_obs_frame,width = 10)
        self.sensor_calib.grid(row=3, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.sensor_calib,"Specify the ID of the sensor.")

        # Objection function
        Label(self.calib_obs_frame, text = "Objective function").grid(row=4,column = 0,sticky = E)
        self.Cal_section = ttk.Combobox(self.calib_obs_frame,width = 20,values = ('NSE','RMSE','MAE','Abs peak','Corr peak'),state = 'readonly')
        self.Cal_section.grid(row=4, column=1)
        GUI_elements.create_ToolTip(self.Cal_section,"Choose objective function.")
        
        # Clibration period frame
        self.calib_period_frame = ttk.LabelFrame(self.Calibration_tab, text = 'Calibration period')
        self.calib_period_frame.grid(row =1, column = 3,rowspan = 1, pady =5, padx = 5,sticky = NSEW)
        # Calibration period
        # Label(self.calib_period_frame, text = "Calibration period").grid(row= 0,column =0,columnspan = 2)
        Label(self.calib_period_frame, text = "Start time").grid(row= 1,column =0, sticky = E)
        Label(self.calib_period_frame, text = "End time").grid(row= 2,column =0,sticky = E)
        self.Start_calib_time = Entry(self.calib_period_frame,width = 15)
        self.Start_calib_time.grid(row=1, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.Start_calib_time,'Specify start time of the calibrated period in the format "yyyy-mm-dd HH:MM".')
        self.End_calib_time= Entry(self.calib_period_frame,width = 15)
        self.End_calib_time.grid(row=2, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.End_calib_time,'Specify end time of the calibrated period in the format "yyyy-mm-dd HH:MM".')
        
        self.UseHotstart = Checkbutton(self.calib_period_frame, text = "Use hotstart", variable = self.param.use_hotstart,command = lambda:GUI_elements.update_Hotstart(self,self.param.use_hotstart,self.hotstart_period_h))
        # self.UseHotstart.deselect()
        self.UseHotstart.grid(row = 3, column = 0,columnspan = 2,sticky = W)
        GUI_elements.create_ToolTip(self.UseHotstart ,"Choose whether a Hotstart period is used.")

        Label(self.calib_period_frame, text = "Hotstart period").grid(row=4,column = 0,sticky='E')
        self.hotstart_period_h = Entry(self.calib_period_frame,width = 5)
        self.hotstart_period_h.grid(row=4, column=1,sticky = W)
        self.hotstart_period_h.insert(END,5)
        self.hotstart_period_h.configure(state = 'disabled')
        GUI_elements.create_ToolTip(self.hotstart_period_h,"Specify the length of the hotstart period [hours].")
        #Calibration area frame
        self.calib_area_frame = ttk.LabelFrame(self.Calibration_tab, text = 'Calibration area')
        self.calib_area_frame.grid(row =2, column = 3,columnspan =1, pady =5, padx = 5,sticky = NSEW)
        GUI_elements.create_ToolTip(self.calib_area_frame,"Choose the area in the model that is to be calibrated.")
        
        # Calibration area. 
        # Label(self.calib_area_frame, text = "Calibration area").grid(row= 1,column =2)
        self.Cal_all = Radiobutton(self.calib_area_frame,text = "All",var = self.param.Calib_area,value = 'all')
        self.Cal_all.grid(row = 2,column = 2,sticky = W)
        self.Cal_all.select()
        self.Cal_upstream = Radiobutton(self.calib_area_frame,text = "Upstream",var = self.param.Calib_area,value = 'upstream')
        self.Cal_upstream.grid(row = 3,column = 2,sticky = W)
        self.Cal_upstream.deselect()
        self.Cal_custom = Radiobutton(self.calib_area_frame,text = "Custom",var = self.param.Calib_area,value = 'custom',state = 'disabled')
        self.Cal_custom.grid(row = 5,column = 2,sticky = W)
        self.Cal_custom.deselect()
        self.Cal_custom_entry = Entry(self.calib_area_frame,width = 10,state='disabled')
        self.Cal_custom_entry.grid(row=5, column=3,sticky = W)
        GUI_elements.create_ToolTip(self.Cal_custom_entry,"List the tag on the elements where calibration should be done. The tag is specified in SWMM")

        #Settings frame
        self.settings_calib= ttk.LabelFrame(self.Calibration_tab, text = 'Settings')
        self.settings_calib.grid(row =3, column = 0,columnspan =3, pady =5, padx = 5,sticky = NSEW)

        Label(self.settings_calib, text = "Number of lhs simulations").grid(row=1,column = 0,sticky='E')
        self.max_initial_iterations_calib = Entry(self.settings_calib,width = 5)
        self.max_initial_iterations_calib.grid(row=1, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.max_initial_iterations_calib,"Specify the number of simulations that are done in the latin hypercube sample of the optimization.")
        
        Label(self.settings_calib, text = "Optimization method").grid(row=0,column = 0,sticky='E')
        self.optimization_method_calib= ttk.Combobox(self.settings_calib,width = 20,values = ('lhs','Simplex','Combined'),state = 'readonly')
        self.optimization_method_calib.bind("<<ComboboxSelected>>",lambda x: GUI_elements.enable_calib_method(self))
        self.optimization_method_calib.grid(row=0, column=1)
        GUI_elements.create_ToolTip(self.optimization_method_calib,"Select optimization method.")
        
        Label(self.settings_calib, text = "Max simplex simulations").grid(row=2,column = 0,sticky='E')
        self.max_optimization_iterations_calib= Entry(self.settings_calib,width = 5)
        self.max_optimization_iterations_calib.grid(row=2, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.max_optimization_iterations_calib,"Specify the maximum allowed number of simulations for the simplex optimization.")
        
        Label(self.settings_calib, text = "Output time steps").grid(row=3,column = 0,sticky='E')
        self.output_time_step = Entry(self.settings_calib,width = 5)
        self.output_time_step.grid(row=3, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.output_time_step,"Specify the time step length in the output file [seconds].")
        
        # Save calibrated model as:
        Label(self.Calibration_tab, text = "Save calibrated file as:").grid(row= 5,column =0,columnspan = 1)
        self.save_calib_file = Entry(self.Calibration_tab,width = 40)
        self.save_calib_file.grid(row=5, column=1,sticky = W,columnspan = 3)
        GUI_elements.create_ToolTip(self.save_calib_file,"Select the name of the calibrated file.")
        
        
        self.overwrite_button = Checkbutton(self.Calibration_tab, text = "Overwrite existing configuation file", variable = self.param.Overwrite_config)
        self.overwrite_button.deselect()    
        self.overwrite_button.grid(row = 6, column = 0,sticky = W, columnspan = 3)
        GUI_elements.create_ToolTip(self.overwrite_button,"Check if the parameters specified in the GUI should be written to configuration file. Else the existing file is used.")

        
        # Calibration button
        self.Auto_calibration = Button(self.Calibration_tab, text ='Run calibration', width = 15, command = lambda:GUI_elements.Calibrate_with_config(self))
        self.Auto_calibration.grid(row = 7, column = 0,sticky = W, padx = 5,columnspan = 1)
        
        # self.write_config = Button(self.Calibration_tab, text ='Save config', width = 15, command = lambda:GUI_elements.write_config(self))
        # self.write_config.grid(row = 6, column = 0,sticky = W,pady = 5, padx = 5)
        # GUI_elements.create_ToolTip(self.write_config,"Save the configuration file without running.")
        
        Button(self.Calibration_tab, text ='Exit',width = 15, command = lambda:self.window.destroy()).grid(row = 7, column = 2,sticky = W, padx = 5)

# =============================================================================
# Content for the results tab

        # Parameters for comparison: 
        self.Result_parameter_frame = ttk.LabelFrame(self.Result_tab, text = 'Parameters for comparison')
        self.Result_parameter_frame.grid(row = 1, column = 0,rowspan = 5, columnspan  =5, pady =5, padx = 15,sticky = NSEW)
        
        self.result_CSO_vol = Checkbutton(self.Result_parameter_frame, text = "Compare CSO volumne", variable = self.param.results_vol)
        self.result_CSO_vol.grid(row=0, column=0, sticky = 'W')
        self.result_CSO_vol.deselect()
        
        self.result_CSO_vol_id_obj_fun = Radiobutton(self.Result_parameter_frame, text = "Same nodes as in objective function", var = self.param.results_CSO_vol_id, value = 'objective_function')
        self.result_CSO_vol_id_obj_fun.grid(row = 0, column = 1, sticky = 'W')
        self.result_CSO_vol_id_obj_fun.select()
        
        self.result_CSO_vol_id_custom = Radiobutton(self.Result_parameter_frame, text = "Custom nodes", var = self.param.results_CSO_vol_id, value = 'new_nodes')
        self.result_CSO_vol_id_custom.grid(row = 0, column = 2, sticky = 'W')
        self.result_CSO_vol_id_custom.deselect()
        
        self.result_CSO_freq = Checkbutton(self.Result_parameter_frame, text = "Compare number of CSO events", variable = self.param.results_freq)
        self.result_CSO_freq.grid(row=1, column=0, sticky = 'W')
        self.result_CSO_freq.deselect()
        
        Label(self.Result_parameter_frame, text = "More options...").grid(row=4,column = 0, sticky = 'W')
        
        # Benchmark for comparison:
        self.Benchmark_model = Radiobutton(self.Result_tab, text = "Compute new benchmark model", var = self.param.Benchmark_model, value = 'new')
        self.Benchmark_model.select()
        self.Benchmark_model.grid(row = 9, column = 0,sticky = W)
        GUI_elements.create_ToolTip(self.Benchmark_model,"Select if a benchmark simulation is needed for comparison of the results with and without RTC.")
        
        self.Benchmark_model_existing = Radiobutton(self.Result_tab, text = "Compute new benchmark model", var = self.param.Benchmark_model, value = 'existing')
        self.Benchmark_model_existing.deselect()
        self.Benchmark_model_existing.grid(row = 9, column = 1,sticky = W)
        GUI_elements.create_ToolTip(self.Benchmark_model_existing,"Select if a benchmark simulation already exists for comparison of the results with and without RTC.")
        
        self.Benchmark_model_file = Button(self.Result_tab, text ='Select output file for comparison', width = 30, command = lambda:msg.showerror('','Not implemented'))
        self.Benchmark_model_file.grid(row = 10, column = 1 ,sticky = W, padx = 5,columnspan = 2)
        GUI_elements.create_ToolTip(self.Benchmark_model_file,"Choose an output file from an already run SWMM model for comparison.")
        
# =============================================================================
# Run the program
if __name__ == "__main__":
    GUI = pyswmm_GUI()
    GUI.window.mainloop()