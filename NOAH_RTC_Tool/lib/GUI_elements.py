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
# import the necessary modelus from pyswmm
import pyswmm
from pyswmm import Simulation, Nodes, Links, SystemStats, Subcatchments

import tkinter as tk
from tkinter import *
from tkinter import Tk, ttk, filedialog, scrolledtext,messagebox
from tkinter import messagebox as msg
from tkintertable.Tables import TableCanvas
from tkintertable.TableModels import TableModel



#===================================================================     
# Define run function
def run(self):
    
    if self.param.Overwrite_config.get() == 1:
        write_config(self) # Writes configuration file 
    
    msg.showinfo('','Running simulation \nSee run status in console window')
    config_file = self.param.model_name.get() + '.ini'
    if self.param.UseOptimization.get() == 0:
        pyswmm_Simulation.single_simulation(config_file)
        msg.showinfo('','Ran one simulation')
        results_text = ''
        
    elif self.param.UseOptimization.get() == 1:
        pyswmm_Simulation.Optimizer(config_file)
        msg.showinfo('','Found optimal RTC setup\nSee file optimized_results.txt in the \output folder with the latest time for the results.')
        results_text = 'See file optimized_results.txt in the \output folder with the latest time for the results.'
        # generate_SWMM_file(self) # Does not work 

# # =============================================================================
# Write input from GUI to configuration file        
def write_config(self):
    
    config = configparser.ConfigParser()
    config['DEFAULT'] = {}
    
    config['Settings'] = {
                          'System_Units': swmmio.swmmio.inp(self.param.model_dir.get() + '/' + self.param.model_name.get() + '.inp').options.Value.FLOW_UNITS, 
                          'Reporting_timesteps': swmmio.swmmio.inp(self.param.model_dir.get() + '/' + self.param.model_name.get() + '.inp').options.Value.REPORT_STEP,
                          'Time_seperating_CSO_events':self.CSO_event_seperation.get(),
                          'Max_CSO_duration':self.CSO_event_duration.get()
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
                              'Optimized_parameter':self.optimized_parameter.get(),
                              'expected_min_Xvalue':self.expected_min_Xvalue.get(),
                              'expected_max_Xvalue':self.expected_max_Xvalue.get(),
                              'max_iterations_bashop':self.max_iterations_bashop.get(),
                              'max_iterations_per_minimization':self.max_iterations_per_minimization.get()                              
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
        self.Overwrite_config = IntVar()
        
        # RTC settings
        self.actuator_type = StringVar()
        
        # Optimization
        self.UseOptimization = IntVar()
        self.CSO_objective = StringVar()
       
        # Results
        self.Benchmark_model = IntVar()
        self.results_vol = IntVar()
        self.results_CSO_vol_id = StringVar()
        self.results_freq = IntVar()
        self.results = {}       
        self.optimal_setting = {}
        
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
 
# =============================================================================
# functions for the GUI buttons
# =============================================================================

def OpenFile(self):
    path = filedialog.askopenfilename(filetypes =(("SWMM model", "*.inp"),("All Files","*.*")),
                       title = "Choose a file.")
    modelname = path.split('/')[-1].split('.')[-2]
    self.param.model_name.set(modelname)
    # Define path of the model     
    directory = os.path.split(path)[0]
    self.param.model_dir.set(directory)

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
def First_step_optimization_plot(timestamp_folder):
    # timestamp_folder = '2020-02-05_10-38-09'
    popup_plot = tk.Tk()
    popup_plot.wm_title('Results')   
    tframe = Frame(popup_plot)
    tframe.grid(row=0,column = 0,sticky ='nsew')
    mframe = Frame(popup_plot)
    mframe.grid(row=1,column = 0,sticky ='nsew')
    bframe = Frame(popup_plot)
    bframe.grid(row = 2,sticky = 'nsew')

    with open('../output/' + timestamp_folder + '/optimized_results.txt','r') as file:
        resultfile = file.read()  
    res_text = scrolledtext.ScrolledText(tframe, height = 15, width = 70, wrap = "word")
    res_text .grid(row = 0, column = 0)
    res_text.insert(INSERT,resultfile)
    Results_plot(mframe,timestamp_folder)
    B1 = ttk.Button(bframe, text="Quit", command = popup_plot.destroy)
    B1.grid(row = 0,column = 1, sticky = E)
    
    
    
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
        