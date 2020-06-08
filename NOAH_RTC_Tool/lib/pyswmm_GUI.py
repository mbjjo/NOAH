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
                
        self.model_label = Label(self.window, width = 15, bg = 'white', textvariable = self.param.model_name)
        self.model_label.grid(row=0, column = 2)
        
        self.overwrite_button = Checkbutton(self.window, text = "Overwrite existing configuation file", variable = self.param.Overwrite_config)
        self.overwrite_button.deselect()    
        self.overwrite_button.grid(row = 4, column = 0,sticky = W, columnspan = 2)
        GUI_elements.create_ToolTip(self.overwrite_button,"Check if the parameters specified in the GUI should be written to configuration file. Else the existing file is used.")

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
        self.run_button = Button(self.window, text ='Run', width = 15, command = lambda:GUI_elements.run(self))
        self.run_button.grid(row = 5, column = 0,sticky = W,pady = 5, padx = 5)
        GUI_elements.create_ToolTip(self.run_button,'Runs the simulation with the configuration file.')
        
        self.write_config = Button(self.window, text ='Save config', width = 15, command = lambda:GUI_elements.write_config(self))
        self.write_config.grid(row = 5, column = 1,sticky = W,pady = 5, padx = 5)
        GUI_elements.create_ToolTip(self.write_config,"Save the configuration file without running.")
        
        self.SWMM_results = Button(self.window, text ='Write to SWMM file', width = 15, command = lambda: GUI_elements.generate_SWMM_file(self))
        self.SWMM_results.grid(row = 5, column = 2,sticky = W,pady = 5, padx = 5)
        GUI_elements.create_ToolTip(self.SWMM_results,"Write the current RTC setup (specified in the simulation tab) to a SWMM file.")
        
        Button(self.window, text ='Exit',width = 15, command = lambda:self.window.destroy()).grid(row = 5, column = 3,sticky = W, padx = 5)

# =============================================================================
    # Create content for tabs
# =============================================================================
# Creatng the content for the simulation tab 

    # Radiobuttons for actuator type 
        Label(self.Simulation_tab, text = "Select the type of the actuator:").grid(row=1,column = 0, sticky = 'W',columnspan = 2)
        self.orifice_active = Radiobutton(self.Simulation_tab,text = "Orifice",var = self.param.actuator_type,value = 'orifice', command = lambda:GUI_elements.orifice_or_pump(self))
        self.orifice_active.grid(row = 2,column = 0,sticky = W,columnspan = 2,pady = 5)
        self.orifice_active.deselect()
        self.pump_active = Radiobutton(self.Simulation_tab,text = "Pump",var = self.param.actuator_type,value = 'pump', command = lambda:GUI_elements.orifice_or_pump(self))
        self.pump_active.grid(row=2,column = 1,sticky = W,columnspan = 2,pady = 5)
        self.pump_active.select()
        
    # Dry flow or rainfall threshold
        Label(self.Simulation_tab, text = 'Rainfall or Dryflow').grid(row = 3,column = 0,sticky = W,columnspan = 2,pady = 5)
        Label(self.Simulation_tab, text = 'If rainfall exceeds').grid(row = 4,column = 0,sticky = W,columnspan = 1)
        self.rainfall_threshold = Entry(self.Simulation_tab,width = 10)
        self.rainfall_threshold.grid(row= 4, column = 1, sticky = W)
        GUI_elements.create_ToolTip(self.rainfall_threshold,'The average precipitation [mm/hr] over a certain duration that activates the rainfall rule')
        Label(self.Simulation_tab, text = 'During more than').grid(row = 5,column = 0,sticky = W,columnspan = 1)
        self.rainfall_time= Entry(self.Simulation_tab,width = 10)
        self.rainfall_time.grid(row= 5, column = 1, sticky = W)
        GUI_elements.create_ToolTip(self.rainfall_time,'The duration of the rainfall before the Raifall rule applies. [minutes]')
        Label(self.Simulation_tab, text = 'Raingage id').grid(row = 6,column = 0,sticky = W,columnspan = 1)
        self.raingage1 = Entry(self.Simulation_tab,width = 15)
        self.raingage1.grid(row = 6, column = 1,sticky = W)
        GUI_elements.create_ToolTip(self.raingage1,'ID of the raingage in the system')
    # RBC frame
        self.RBC_frame = ttk.LabelFrame(self.Simulation_tab, text = 'Control rules setup')
        self.RBC_frame.grid(row = 1, column = 4,rowspan = 20, pady =5, padx = 15)
        # Wet weather rule
        # self.Rainfall_frame = 
        Label(self.RBC_frame, text = "Rule applied during rainfall").grid(row=9,column = 0,columnspan = 3)
        
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
        
    # Dry weather rule
        Label(self.RBC_frame, text = "Rule applied during dry weather flow").grid(row=14,column = 0,columnspan = 3)

        Label(self.RBC_frame, text = "IF depth in (Sensor)").grid(row=15,column = 0, sticky = 'W')
        self.sensor1id_dry = Entry(self.RBC_frame,width = 15)
        self.sensor1id_dry.bind("<FocusOut>", lambda x: GUI_elements.update(self.sensor1id_dry, self.sensor1id))
        # self.sensor1id_dry = Label(self.RBC_frame,width = 15)
        self.sensor1id_dry.grid(row=15, column=1, sticky = 'W')
        GUI_elements.create_ToolTip(self.sensor1id_dry,'Type in the node id of the sensor')
                
        self.sign = Label(self.RBC_frame, text = ">")
        self.sign.grid(row=15,column = 3)
                
        self.sensor1setting_dry = Entry(self.RBC_frame,width = 10)
        self.sensor1setting_dry.grid(row=15, column=4, sticky = 'W')
        GUI_elements.create_ToolTip(self.sensor1setting_dry,'Type in the depth that activates the actuator [m]')
        
        # Actuator1
        Label(self.RBC_frame, text = "THEN Actuator").grid(row=16,column = 0, sticky = 'W')
        self.actuator1id_dry = Entry(self.RBC_frame,width = 15)
        self.actuator1id_dry.bind("<FocusOut>", lambda x: GUI_elements.update(self.actuator1id_dry, self.actuator1id))
        self.actuator1id_dry.grid(row=16, column=1, sticky = 'W')
        GUI_elements.create_ToolTip(self.actuator1id_dry,'Type in the id of the actuator')
    
        Label(self.RBC_frame, text = "Setting").grid(row=16,column = 3, sticky = 'W')
        self.actuator1setting_True_dry = Entry(self.RBC_frame,width = 10)
        self.actuator1setting_True_dry.grid(row=16, column=4, sticky = 'W')
        # self.actuator1setting.insert(0,'1')
        GUI_elements.create_ToolTip(self.actuator1setting_True_dry,'Type in the setting of the actuator when critical depth is exceeded')
        
        Label(self.RBC_frame, text = "ELSE Actuator").grid(row=17,column = 0, sticky = 'W')
        Label(self.RBC_frame, text = "Setting").grid(row=17,column = 3, sticky = 'W')
        self.actuator1setting_False_dry = Entry(self.RBC_frame,width = 10)
        self.actuator1setting_False_dry.grid(row=17, column=4, sticky = 'W')
        # self.actuator1setting_False.insert(0,'0')
        GUI_elements.create_ToolTip(self.actuator1setting_False_dry,'Type in the setting of the actuator when critical depth is NOT exceeded')
        
# =============================================================================

# Creatng the content for the rain tab 
        
        Label(self.Rain_tab, text = "Contains a tool for designing rain events that can be imported to the SWMM model", bg = 'white').grid(row=0,column = 0, columnspan = 3, pady  =8)
                
        self.hotstart_button = Button(self.Rain_tab, text ='Choose hotstart file',width = 15, command = lambda:msg.showerror('','Hotstart is not implemented yet'))
        self.hotstart_button.grid(row = 2, column = 1,sticky = 'W')
        GUI_elements.create_ToolTip(self.hotstart_button,"Choose the SWMM model that shoul be used for the simulations")
        
        # Define frame 
        self.Rain_event_frame = ttk.LabelFrame(self.Rain_tab, text = 'Definition of rain event')
        self.Rain_event_frame.grid(row = 2, column = 0,rowspan = 20, pady =5, padx = 15,sticky = 'W')
        
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
        
        
        # Define radiobuttons for CSO objective
        self.CSO_volume = Radiobutton(self.Control_tab,text = "Reduce CSO volume",var = self.param.CSO_objective,value = 'volume')
        self.CSO_volume.grid(row = 1,column = 0,sticky = W)
        self.CSO_volume.select()
        self.CSO_freq = Radiobutton(self.Control_tab,text = "Reduce CSO frequency",var = self.param.CSO_objective,value = 'frequency')
        self.CSO_freq.grid(row=1,column = 1,sticky = W,pady=10)
        # self.CSO_freq.select()
        
        Label(self.Control_tab,text = "Reduce CSO from").grid(row=3,column = 0, columnspan = 1)
        self.CSO_id1 = Entry(self.Control_tab,width = 15)
        self.CSO_id1.grid(row = 3,column = 1)
        self.CSO_id2 = Entry(self.Control_tab,width = 15)
        self.CSO_id2.grid(row = 4,column = 1)
        self.CSO_id3 = Entry(self.Control_tab,width = 15)
        self.CSO_id3.grid(row = 5,column = 1)
        GUI_elements.create_ToolTip(self.CSO_id1,"Write the node ID of the CSO struture")
        GUI_elements.create_ToolTip(self.CSO_id2,"Write the node ID of the CSO struture")
        GUI_elements.create_ToolTip(self.CSO_id3,"Write the node ID of the CSO struture")
        
        # CSO Settings frame
        self.Settings_frame = ttk.LabelFrame(self.Control_tab, text = 'Settings for CSO')
        self.Settings_frame.grid(row = 10, column = 0,rowspan = 20, pady =5, padx = 15)
        
        Label(self.Settings_frame, text = "Time seperation CSO events").grid(row=0,column = 0,sticky = W)
        self.CSO_event_seperation = Entry(self.Settings_frame,width = 5)
        self.CSO_event_seperation.grid(row = 0, column = 1,sticky = W)
        self.CSO_event_seperation.insert(END, '12')
        GUI_elements.create_ToolTip(self.CSO_event_seperation,"Define the minimum time (hours) seperating two events.")
        Label(self.Settings_frame, text = "Maximum duration of CSO event").grid(row=1,column = 0)
        self.CSO_event_duration = Entry(self.Settings_frame,width = 5)
        self.CSO_event_duration.grid(row = 1, column = 1,sticky = W)
        self.CSO_event_duration.insert(END, '24')
        GUI_elements.create_ToolTip(self.CSO_event_duration,"Define the time (hours) before an event is counted as more events.")

# =============================================================================
# Creatng the content for the optimization tab 
        optimize_check = Checkbutton(self.Optimize_tab, text = "Use optimization", variable = self.param.UseOptimization)
        optimize_check.deselect()
        optimize_check.grid(row = 0, column = 0,sticky = W)
        GUI_elements.create_ToolTip(optimize_check,"Choose whether optimization should be used. If not only a single simulation is run.")
        
         # Sensor location frame
        self.optimization_frame = ttk.LabelFrame(self.Optimize_tab, text = 'Optimization parameters')
        self.optimization_frame.grid(row =1, column = 0,rowspan = 20, pady =5, padx = 5)
        # Label(self.Optimize_tab, text = "Choose optimization parameters").grid(row=2,column = 0, columnspan = 3, pady  =8)
         
        Label(self.optimization_frame, text = "Optimization method").grid(row=3,column = 0,sticky='E')
        self.opt_method = ttk.Combobox(self.optimization_frame,width = 20,values = ('Two-step-optimization'),state = 'readonly')
        self.opt_method.grid(row=3, column=1)
        
        Label(self.optimization_frame, text = "Parameter to be optimized").grid(row=4,column = 0,sticky='E')
        self.optimized_parameter= ttk.Combobox(self.optimization_frame,width = 20,values = ('Activation depth','Sensor location', 'Sensor location and activation depth'),state = 'readonly')
        self.optimized_parameter.bind("<<ComboboxSelected>>",lambda x: GUI_elements.enable_sensor_location(self))
        self.optimized_parameter.grid(row=4, column=1)
        
        Label(self.optimization_frame, text = "Minimum value of parameter range").grid(row=5,column = 0,sticky='E')
        self.expected_min_Xvalue = Entry(self.optimization_frame,width = 5)
        self.expected_min_Xvalue.grid(row=5, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.expected_min_Xvalue,"Specify the minimum value of the parameter range. The optimizer will not find a value below this point. ")
        
        Label(self.optimization_frame, text = "Maximum value of parameter range").grid(row=6,column = 0,sticky='E')
        self.expected_max_Xvalue = Entry(self.optimization_frame,width = 5)
        self.expected_max_Xvalue.grid(row=6, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.expected_max_Xvalue,"Specify the maximum value of the parameter range. The optimizer will not find a value above this point. ")
        
        Label(self.optimization_frame, text = "Number of initial simulations").grid(row=7,column = 0,sticky='E')
        self.max_iterations_bashop = Entry(self.optimization_frame,width = 5)
        self.max_iterations_bashop.grid(row=7, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.max_iterations_bashop,"Specify the number of simulations that are done in the initial screening of the optimization.")
        
        Label(self.optimization_frame, text = "Maximum number of simulations for optimization").grid(row=8,column = 0,sticky='E')
        self.max_iterations_per_minimization = Entry(self.optimization_frame,width = 5)
        self.max_iterations_per_minimization.grid(row=8, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.max_iterations_per_minimization,"Specify the maximum allowed number of simulations for the second part of the optimization.")
        
         # Sensor location frame
        self.sensor_loc_frame = ttk.LabelFrame(self.Optimize_tab, text = 'Possible sensor locations')
        self.sensor_loc_frame.grid(row =1, column = 2,rowspan = 20, pady =5, padx = 5)
        
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
# NOT ADDED TO PARAM!
        
        Label(self.Calibration_tab, text = "Section").grid(row=1,column = 0)
        self.Cal_section = ttk.Combobox(self.Calibration_tab,width = 20,values = ('Junctions','Conduits','Xsections')  )
        self.Cal_section.grid(row=2, column=0)
        Label(self.Calibration_tab, text = "Parameter").grid(row=1,column = 1)
        self.Cal_param = ttk.Combobox(self.Calibration_tab,width = 20,values = ('Elevation', 'MaxDepth', 'InitDepth')  )
        self.Cal_param.grid(row=2, column=1)
        
        # Define radiobuttons for calibration target. 
        # NOT ADDED TO PARAM!
        Label(self.Calibration_tab, text = "Calibration target").grid(row= 1,column =2)
        self.Cal_all = Radiobutton(self.Calibration_tab,text = "All",value = 'all')
        self.Cal_all.grid(row = 2,column = 2,sticky = W)
        self.Cal_all.select()
        self.Cal_upstream = Radiobutton(self.Calibration_tab,text = "Upstream",value = 'upstream')
        self.Cal_upstream.grid(row = 3,column = 2,sticky = W)
        self.Cal_upstream.deselect()
        self.Cal_downstream = Radiobutton(self.Calibration_tab,text = "Downstream",value = 'downstream')
        self.Cal_downstream.grid(row = 4,column = 2,sticky = W)
        self.Cal_downstream.deselect()
        
        self.Cal_custom_entry = Entry(self.Calibration_tab,width = 10,state='disabled')
        self.Cal_custom_entry.grid(row=5, column=3,sticky = W)
        GUI_elements.create_ToolTip(self.Cal_custom_entry,"List the elements where calibration should be done. Seperate by ,")
        
        self.Cal_custom = Radiobutton(self.Calibration_tab,text = "Custom",value = 'custom', command = GUI_elements.enableEntry(self.Cal_custom_entry))
        self.Cal_custom.grid(row = 5,column = 2,sticky = W)
        self.Cal_custom.deselect()
        
        Label(self.Calibration_tab, text = "Calibrate by").grid(row= 6,column =0)
        self.Cal_multiply = Radiobutton(self.Calibration_tab,text = "multiply value",value = 'multiply')
        self.Cal_multiply.grid(row = 7,column = 0,sticky = W)
        self.Cal_multiply.select()
        self.Cal_add= Radiobutton(self.Calibration_tab,text = "add value",value = 'add')
        self.Cal_add.grid(row = 8,column = 0,sticky = W)
        self.Cal_add.deselect()
        self.Cal_replace = Radiobutton(self.Calibration_tab,text = "replace value",value = 'replace')
        self.Cal_replace.grid(row = 9,column = 0,sticky = W)
        self.Cal_replace.deselect()
        self.Cal_value = Entry(self.Calibration_tab,width = 20)
        self.Cal_value.grid(row=8, column=1,sticky = W)
        GUI_elements.create_ToolTip(self.Cal_value,"Specify the value that needs to be calibrated with.")
        
        self.Auto_calibration = Button(self.Calibration_tab, text ='Automatic calibration', width = 20, command = lambda:msg.showerror('','Automatic calibration is not possible'))
        self.Auto_calibration.grid(row = 8, column = 2,sticky = W, padx = 5,columnspan = 2)
        # GUI_elements.create_ToolTip(self.Auto_calibration ,"Write the current RTC setup (specified in the simulation tab) to a SWMM file.")
     
# =============================================================================
# Content for the results tab

        # Parameters for comparison: 
        self.Result_parameter_frame = ttk.LabelFrame(self.Result_tab, text = 'Parameters for comparison')
        self.Result_parameter_frame.grid(row = 1, column = 0,rowspan = 5, columnspan  =5, pady =5, padx = 15)
        
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