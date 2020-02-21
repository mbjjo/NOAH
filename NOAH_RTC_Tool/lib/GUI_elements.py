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
from datetime import datetime,timedelta
import configparser
from configparser import ConfigParser
import os 
from threading import Thread
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
    
    config_file = self.param.model_name.get() + '.ini'
    
    self.param.status.set('Running simulation')
    if self.param.UseOptimization.get() == 0:
        pyswmm_Simulation.single_simulation(config_file)
        msg.showinfo('','Ran one simulation')
        results_text = ''
        
    elif self.param.UseOptimization.get() == 1:
        self.param.status.set('Running optimization')
        pyswmm_Simulation.Optimizer(config_file)
        msg.showinfo('','Found optimal RTC setup\nSee file optimized_results.txt in the \output folder with the latest time for the results.')
        results_text = 'See file optimized_results.txt in the \output folder with the latest time for the results.'
#        GUI_elements.Results_text('Run complete',results_text)
#        GUI_elements.Results_plot()
#        self.progress_bar.start()
        # generate_SWMM_file(self) # Works but not used 
    self.param.status.set('Run Complete')
    
#        print('Run complete')
    #     exec(open('SimpelSimulation.py').read())
#    except:
#        msg.showerror('Error','Not sufficient input to run the model')
  
    

# #===================================================================     

#     # Define run function
# def run(self):
    
#     if self.param.Overwrite_config.get() == 1:
#         GUI_elements.write_config(self) # Writes configuration file 
    
#     config_file = self.param.model_name.get() + '.ini'
    
#     if self.param.UseOptimization.get() == 0:
#         self.create_Thread(pyswmm_Simulation.single_simulation(config_file))
#         self.param.status.set('Running simulation')
#         # pyswmm_Simulation.single_simulation(config_file)
#         msg.showinfo('','Ran one simulation')
#         results_text = ''
        
#     elif self.param.UseOptimization.get() == 1:
        
#         self.param.status.set('Running optimization')
#         pyswmm_Simulation.Optimizer(config_file)
#         msg.showinfo('','Found optimal RTC setup\nSee file optimized_results.txt in the \output folder with the latest time for the results.')
#         # results_text = 'See file optimized_results.txt in the \output folder with the latest time for the results.'
# #        GUI_elements.Results_text('Run complete',results_text)
# #        GUI_elements.Results_plot()
# #        self.progress_bar.start()
#     self.param.status.set('Run Complete')
    
# #        print('Run complete')
#     #     exec(open('SimpelSimulation.py').read())
# #    except:
# #        msg.showerror('Error','Not sufficient input to run the model')
  
# # =============================================================================

#     
# =============================================================================
#================================================         
        
def write_config(self):
    
    config = configparser.ConfigParser()
    config['DEFAULT'] = {}
    config['Settings'] = {'System_Units': 'SI', 
                          'ReportingTimesteps_min': '5',
                          'Random_setting': '0'}
    config['Model'] = {'Modelname':self.param.model_name.get(),
                      'Rain series':''
                      }
    
    
    config['RuleBasedControl'] = {'sensor1_id':self.sensor1id.get(),
                                  'sensor1_critical_depth':self.sensor1setting.get(),
                                  'actuator1_id':self.actuator1id.get(),
                                  'actuator1_target_setting_True':self.actuator1setting_True.get(),
                                  'actuator1_target_setting_False':self.actuator1setting_False.get(),
                                  'actuator_type':self.actuator_type.get()
                                  }
    config['Optimization'] = {'UseOptimization':self.param.UseOptimization.get(),
                              'optimization_method':self.opt_method.get(),
                              'CSO_objective':self.param.CSO_objective.get(),
                              'CSO_id1':self.CSO_id1.get(),
                              'CSO_id2':self.CSO_id2.get(),
                              'CSO_id3':self.CSO_id3.get(),
                              'Optimized_parameter':self.param.optimized_parameter.get(),

                              'expected_min_Xvalue':self.expected_min_Xvalue.get(),
                              'expected_max_Xvalue':self.expected_max_Xvalue.get(),
#                              'Initial_value':self.initial_value.get(),
#                              'stepsize':
                              'max_iterations_bashop':self.max_iterations_bashop.get(),
                              'max_iterations_per_minimization':self.max_iterations_per_minimization.get()
                              
                              }

#                              'reduce_flood':self.param.red_flood.get(),
#                              'reduce_CSO':self.param.red_CSO.get(),
#                              'Stable_Water_Level':self.param.StableWaterLevel.get()
#                              }
    
    
    
    # save the file
    config_name = self.param.model_name.get()
    
    config_path = '../../config/saved_configs/'
    with open(config_path + config_name + '.ini','w') as configfile: 
#              '../../config/saved_configs/example.ini','w') as configfile:
        config.write(configfile)
    msg.showinfo('','Saved to configuration file')
#===================================================================    
        
class parameters(object):
    def __init__(self):
        
        # Define model
        self.model_name = StringVar()
        self.Overwrite_config = IntVar()
        self.status = StringVar()
        self.status.set('Ready')
        self.sim_time = StringVar()
        self.sim_time.set('LOOONG time')
        # Define Control settings in the model 
        self.Control_tag = StringVar()
        self.control_weight1 = IntVar()
        self.control_weight1.set(0)
        self.control_weight2 = IntVar()
        self.control_weight2.set(0)
        self.control_weight3 = IntVar()
        self.control_weight3.set(0)
    
        self.Control_type = IntVar()
        # Optimization tab
        self.red_flood = IntVar()
        self.red_CSO = IntVar()
        self.StableWaterLevel = IntVar()
        
        
        self.sensors = ''
        self.target_settings = 0
        self.actuators = ''
        self.settings = 0
        
        self.UseOptimization = IntVar()
        self.optimization_method = StringVar()
        self.CSO_objective = StringVar()
        self.CSO_id1 = StringVar()
        self.CSO_id2 = StringVar()
        self.CSO_id3 = StringVar()
        self.optimized_parameter = StringVar()
        self.min_expected_Xvalue = IntVar()
        self.max_expected_Xvalue = IntVar()
        self.max_iterations_bashop = IntVar()
        self.max_iterations_per_minimization = IntVar()
        
        
        
        # # Results
        self.results = {}       
        
        self.optimal_setting = {}
        
    
#================================================    
        
class Results(object):        
    def __init__(self):
        self.optimization_results = StringVar()
        self.optimal_setting = DoubleVar()
        self.volume_at_optimum = DoubleVar()
        self.CSO_events_at_optimum = DoubleVar()
        
#        
    
#================================================    

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
            
#===================================================================          

def create_ToolTip(widget, text):
    toolTip = ToolTip(widget)       # create instance of class
    def enter(event):
        toolTip.show_tip(text)
    def leave(event):
        toolTip.hide_tip()
    widget.bind('<Enter>', enter)   # bind mouse events
    widget.bind('<Leave>', leave)
    

#===================================================================     
def enableEntry(entry):
    entry.configure(state='normal')
    entry.update()

def disableEntry():
    entry.configure(state = 'disabled')
    entry.update()
 
#===================================================================     
#===================================================================     
      
    
# functions for the GUI buttons

def OpenFile(self):
    init_dir = "../../model"
    name = filedialog.askopenfilename(initialdir=init_dir,
                       filetypes =(("SWMM model", "*.inp"),("All Files","*.*")),
                       title = "Choose a file.")
    name = name.split('/')[-1].split('.')[-2]
    self.param.model_name.set(name)
    #Using try in case user types in unknown file or closes without choosing a file.
    #     try:
    #         name.split('.')[-1] == 'inp'
    # #         with open(name,'r') as UseFile:
    # #             print(UseFile.read())
    #     except:
    #         model_label.configure(text = 'not a SWMM model')

#===================================================================     

def generate_SWMM_file(self):
    init_dir = "../../model"
    name = filedialog.asksaveasfilename(initialdir=init_dir,
                            filetypes =(("SWMM model", "*.inp"),("All Files","*.*")),
                            title = "Save file as", defaultextension='.inp')
    # name = init_dir + "/" + self.param.model_name.get()+'_RTC.inp'
# Read the .inp file from the model specified. 
    os.getcwd()
    read_file = '../../model/'+self.param.model_name.get()+'.inp'
    with open(read_file) as f:
        with open(name, "w") as f1:
            for line in f:
                f1.write(line)

# Write Control rules 
    write_file = open(name,'a')
    write_file.write('\n[CONTROLS]\nRULE RBC_1\n\n')
    write_file.write('IF NODE ' + self.sensor1id.get() + ' DEPTH > ' + self.sensor1setting.get() + '\n')
    write_file.write('THEN ORIFICE ' + self.actuator1id.get() + ' SETTING = '+ self.actuator1setting_True.get() +  '\n')
    write_file.write('ELSE ORIFICE ' + self.actuator1id.get() + ' SETTING = '+self.actuator1setting_False.get()+'\n')
    
    write_file.close()
    msg.showinfo('','Saved to SWMM file')
#================================================    

#def config_validation():
        
# from https://gist.github.com/tsuriga/5bc5bbfaf21c6a51cda7
# First one is a example of the config file
            #[DEFAULT]
            #; Operation mode
            #; This is a global value for all sections
            #mode = master
            #
            #[server]
            #
            #; Connection lifetime
            #timeout = 3600
            #
            #; Garbage collection mode
            #; Accepted values: none, aggressive, smart, auto
            #gc_mode = smart
            #
            #; Notice there is no mode set under this section - it will be read from defaults
            #
            #
            #[client]
            #
            #; Fallback procedure for clients
            #; Accepted values: none, polling, auto
            #; Invalid value as an example here
            #fallback = socket
            #
            #; Overriding global value here
            #mode = slave


class MyException(Exception):
    pass


#class MyConfig(ConfigParser):
#    def __init__(self, config_file):
#        super(MyConfig, self).__init__()
#
#        self.read(config_file)
#        self.validate_config()

class MyConfig(ConfigParser):
    def __init__(self, config_file):
        super(MyConfig, self).__init__()

        self.read(config_file)
        self.validate_config()


    def validate_config(self):
        config = configparser.ConfigParser()
        config['DEFAULT'] = {}
        config['Settings'] = {'Setting1': 'True',
                              'Setting2': '2.2',
                              'Setting3': '0'}
        config['Model'] = {'Modelname':str,
                              'Rain series':''
                              }
    
        config['RuleBasedControl'] = {'sensor1_id':str,
                              'sensor1_critical_depth':str,
                              'actuator1_id':str,
                              'actuator1_target_setting':str
                              }
        config['Optimization'] = {'UseOptimization':str,
                              'reduce_flood':str,
                              'reduce_CSO':str,
                              'Stable_Water_Level':str
                              }
        
        required_values = config
#        required_values = {
#            'Settings': {
#                'setting1': 'True', b 
#                'setting2': float,
#                'setting3':int
#                },
##                'gc_mode': ('none', 'aggressive', 'smart', 'auto'),
##                'mode': ('master')
##            },
#            'Model': {'modelname': str,
##                ('none', 'polling', 'auto'),
##                'mode': ('master', 'slave')
#                },
#            'RuleBasedControl':{
#                'sensor1_id':str,
#                'sensor1_critical_depth': str,
#                'actuator1_id' : str,
#                'actuator1_target_setting': str
#                },
#            'optimization':{
#                'param' : str,
#                }
#            
#            }   
        """
        Notice the different mode validations for global mode setting: we can
        enforce different value sets for different sections
        """
        config = configparser.ConfigParser()
        config.read('example.ini')





      
#
        for section in required_values.sections():
            if section not in config.sections():
                raise MyException(
                    'Missing section %s in the config file' % section)
            for (key,value) in required_values.items(section):
                        
#            for (key, value) in keys.items(section):
#            for key, values in section.items(section):                
                if key not in config.items(section): #or config.items(section)[key] == '':
                    raise MyException((
                        'Missing value for %s under section %s in ' +
                        'the config file') % (key, section))

#                if value:
#                    if self[section][key] not in values:
#                        raise MyException((
#                            'Invalid value for %s under section %s in ' +
#                            'the config file') % (key, section))

#cfg = {}
#try:
#    # The example config file has an invalid value so cfg will stay empty first
#    cfg = MyConfig('example.ini')
#except MyException as e:
#    # Initially you'll see this due to the invalid value
#    print(e)
#else:
#    # Once you fix the config file you'll see this
#    print(cfg['client']['fallback'])
#                        
        
        
        
#===================================================================    
        
                        
class StartUp_window(object):
    def __init__(self):
        self.StartUp_window = tk.Tk()
        self.StartUp_window.wm_title('Load model')   
        self.StartUp_window.iconbitmap('./GUI_files/noah_logo_OAb_icon.ico')
        self.StartUp_window.configure(background = 'white')
    #    self.StartUp_window.geometry('520x00')
        self.StartUp_window.resizable(False,False)
            
        tframe = Frame(self.StartUp_window)
        tframe.grid(row=0,column = 0,sticky=N+S+E+W, padx=10, pady=30)
        Label(tframe,text = 'Choose how to begin the project',bg = 'white').grid(row=0,column = 0,columnspan = 4,sticky=E+W)
        B1 = ttk.Button(tframe, text="Load new SWMM model", command = lambda: GUI_elements.OpenFile()).grid(row=1,column=0)
        B2 = ttk.Button(tframe, text="Load most recent config file", command = lambda: GUI_elements.OpenFile).grid(row=1,column=1)
        B3 = ttk.Button(tframe, text="Load new config file", command = lambda: GUI_elements.OpenFile).grid(row=1,column=2)
        B4 = ttk.Button(tframe, text="Quit", command = self.StartUp_window.destroy).grid(row = 1,column = 3, sticky = E)
        self.StartUp_window.mainloop()
#===================================================================    

def Results_table(title, msg):
  
    popup = tk.Tk()
    popup.wm_title(title)   
    tframe = Frame(popup)
    tframe.grid(row=0,column = 0,sticky ='nsew')
    popup.columnconfigure(0, weight=4)
    popup.rowconfigure(0, weight=4)
    table = TableCanvas(tframe,data = msg)
    table.show()

    bframe = Frame(popup)
    bframe.grid(row = 1,sticky = 'nsew')
        
    B1 = ttk.Button(bframe, text="Quit", command = popup.destroy)
    B1.grid(row = 2,column = 1, sticky = E)
    popup.mainloop()


#================================================     

# Show plots 
def Results_plot(location,timestamp_folder):
    import pickle
    pickle_in = open("../../output/"+ timestamp_folder + "/First_step_simulations.pickle","rb")
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
    figure.savefig("../../output/" + timestamp_folder + '/Plot of first step of the optimization.png')

# Shows the results of the optimization
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

    with open('../../output/' + timestamp_folder + '/optimized_results.txt','r') as file:
        resultfile = file.read()  
    res_text = scrolledtext.ScrolledText(tframe, height = 15, width = 70, wrap = "word")
    res_text .grid(row = 0, column = 0)
    res_text.insert(INSERT,resultfile)
    Results_plot(mframe,timestamp_folder)
    B1 = ttk.Button(bframe, text="Quit", command = popup_plot.destroy)
    B1.grid(row = 0,column = 1, sticky = E)

#===================================================================     

def Results_text(title, msg):
  
    popup_opt = tk.Tk()
    popup_opt.wm_title(title)   
    tframe = Frame(popup_opt)
    tframe.grid(row=0,column = 0,sticky ='nsew')

    bframe = Frame(popup_opt)
    bframe.grid(row = 1,sticky = 'nsew')
    res_text = scrolledtext.ScrolledText(popup_opt, height = 16, width = 100, wrap = NONE)
    label = ttk.Label(popup_opt, text=msg)
    res_text .grid(row = 0, column = 0)
    res_text.insert(INSERT,msg)
    B1 = ttk.Button(bframe, text="Quit", command = popup_opt.destroy)
    B1.grid(row = 3,column = 1, sticky = E)
#    B2 = ttk.Button(bframe, text="Show graphs", command = lambda:optimal_solution_graph())
#    B2.grid(row = 3,column = 0, sticky = E)
    popup_opt.mainloop()
    
   
