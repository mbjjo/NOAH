
# function that allows you to simulate a period of time with SWMM, 
# and define the output that you want from the simulation.
def swmm_simulate(inpFile, simulationStartTime, simulationEndTime, 
                  selected_nodes = list(), selected_links = list(), selected_subcatchments = list(), 
                  output_time_step = 300, add_and_remove_hotstart_period = False, hotstart_period_h = 5):

    import numpy as np
    import pyswmm
    import datetime
    from dateutil.relativedelta import relativedelta
 
    
    # import pdb
    # pdb.set_trace()
    # create time objects in Python
    simulationStartTime = datetime.datetime.strptime(simulationStartTime,"%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)
    simulationStartTime -= datetime.timedelta(seconds=output_time_step) # subtract one time step from the start time - otherwise the first time is a bit late
    simulationEndTime  =  datetime.datetime.strptime(simulationEndTime,"%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)

    ## Allocate space in arrays for storing the results later 
    # we allocate a 10\% more of space just in case after the simulation it is freed...
    expected_timeseries_length = int(np.ceil(1.1*(simulationEndTime - simulationStartTime).total_seconds()/output_time_step))
    # "ynode" is a dict with time series for the specified points of interest
    output_dict = {}
    output_dict["time"] = [datetime.datetime.now() for i in range(expected_timeseries_length)]
    for node in selected_nodes:
        output_dict[node] = np.zeros(expected_timeseries_length)
    for link in selected_links:
        output_dict[link] = np.zeros(expected_timeseries_length)
    for subcatchment in selected_subcatchments:
        output_dict[subcatchment] = np.zeros(expected_timeseries_length)
    
    
    # define which time steps to tell the user how far the simulation has come
    message_at_step = (np.linspace(0.25,0.75,num=3) * ((expected_timeseries_length/1.1) + hotstart_period_h*60/(output_time_step/60))).round()
    
    
    with pyswmm.Simulation(inpFile) as sim:
        # preparing the settings for the simulation like start and end time
        sim.start_time = simulationStartTime
        sim.end_time = simulationEndTime
        sim.step_advance(output_time_step)
        
        # if it has been selected that a "transient period" should be added in front the of the event that is simulated, then we add it here
        if add_and_remove_hotstart_period: 
            sim.start_time = simulationStartTime - relativedelta(hours=hotstart_period_h)
            #Afterwards do not register the hotstart_period_h first hours!!!
    
        if len(selected_nodes)>0:
            l_node=[]
            allNodes = pyswmm.Nodes(sim)
            for node in selected_nodes:    
                l_node.append(allNodes[node])
    
        if len(selected_links)>0:
            allLinks = pyswmm.Links(sim)
            l_link=[]
            for link in selected_links:
                l_link.append(allLinks[link])
    
        if len(selected_subcatchments)>0:
            allSubcatchments = pyswmm.Subcatchments(sim)
            l_subcatchment=[]
            for subcatchment in selected_subcatchments:
                l_subcatchment.append(allSubcatchments[subcatchment])
        
        print("Simulation has started...")
        # run the model by moving one time step forward in time
        #nstep_hotstart = 0 # counter for number of time steps in the hotstart
        nstep = 0 # counter for the number of time steps in the model simulation
        message_step = 0
        for step in sim:
            tt = sim.current_time.replace(tzinfo=datetime.timezone.utc)
            #nstep_hotstart += 1
            message_step += 1
            if (tt-simulationStartTime).total_seconds() >= 0:   
                output_dict["time"][nstep] = tt.replace(tzinfo=datetime.timezone.utc) # save the time stamps for each time step
                
                for index, node in enumerate(selected_nodes):
                    output_dict[node][nstep] = l_node[index].depth # save the values for all the variables of interest
    
                for index, link in enumerate(selected_links):
                     output_dict[link][nstep] = l_link[index].flow
                
                for index, subcatchment in enumerate(selected_subcatchments):
                     output_dict[subcatchment][nstep]  = l_subcatchment[index].runoff
                
                nstep += 1 # increase the counter for the number of time steps
                
            if np.isin(message_step, message_at_step):
                print("Percent completed: " + str(round(sim.percent_complete*100)))
    
    # cut off the space that was intentionally allocated too much
    output_dict["time"] = output_dict["time"][:nstep] # there are only "nstep" number of time steps
    for i in output_dict:
        output_dict[i] = output_dict[i][:nstep]
    
    print("Simulation finished !!!!")
    return(output_dict)



## Function that takes a simulated time series with uneven time steps (like the ones you get from pyswmm)
## and a vector of time steps that the output is desired at (the time where you have observations), and then
## returns a data frame where the original simulation results have been interpolated in time to the desired output time steps.
def interpolate_swmm_times_to_obs_times(simulation_times, simulation_values, observation_times, observation_values):
    import pandas as pd
    
    # for the interpolation pandas needs a data frame with time stamps as "index" and values as a regular column
    new_df = pd.DataFrame({"sim_values": simulation_values})
    new_df.index = pd.to_datetime(simulation_times)
    
    # then pandas needs another data frame, which has index values that are the observation time stamps
    # new_idx = observations_loaded['time'][obs_start_idx:obs_end_idx].reset_index(drop=True)
    new_idx = observation_times
    new_idx = new_idx.dt.tz_localize(tz='UTC')
    new_idx.index = new_idx
    
    # create a new data frame that contains both simulation and observation indices (with NAs for the observation time steps)
    res = new_df.reindex(new_df.index.union(new_idx))
    # in the previous line, pandas converted the index to an "object". We need it back as a "DateTimeIndex"
    res.index = pd.to_datetime(res.index,utc=True)
    # Interpolate (linearly) all of the NAs based on the time stamps indeices
    res = res.interpolate(method = 'values')
    # Only keep the time stamps and values that are 
    res = res.reindex(new_idx)
    res['obs_values'] = observation_values.tolist()
    
    return(res)












### Function that can change a property or parameter in a SWMM model,
# and either creates a new .inp file or overwrites the original one.
    
## replacement_values, multiplying_value, and add_value has to be specified as a single float within a list

## Section options in swmmio and their parameters:j

# "[OPTIONS]":          ['Value']
# "[RAINGAGES]":        ['RainType', 'TimeIntrv', 'SnowCatch', 'DataSourceType','DataSourceName', ';']
# "[SUBCATCHMENTS]":    ['Raingage', 'Outlet', 'Area', 'PercImperv', 'Width', 'PercSlope', 'CurbLength', 'SnowPack', ';']
# "[SUBAREAS]":         ['N-Imperv', 'N-Perv', 'S-Imperv', 'S-Perv', 'PctZero', 'RouteTo', 'PctRouted', ';']
# "[INFILTRATION]":     Depends on method (if you change infiltration method in SWMM gui, you will get new parameters to change here for each subcatchment)
# "[JUNCTIONS]":        ['InvertElev', 'MaxDepth', 'InitDepth', 'SurchargeDepth', 'PondedArea', ';']
# "[OUTFALLS]":         ['InvertElev', 'OutfallType', 'StageOrTimeseries', 'TideGate', ';']
# "[STORAGE]":          ['InvertElev', 'MaxD', 'InitDepth', 'StorageCurve', 'Coefficient', 'Exponent', 'Constant', 'PondedArea', 'EvapFrac', 'SuctionHead', 'Conductivity', 'InitialDeficit', ';']
# "[CONDUITS]":         ['InletNode', 'OutletNode', 'Length', 'ManningN', 'InletOffset', 'OutletOffset', 'InitFlow', 'MaxFlow', ';']
# "[PUMPS]":            ['InletNode', 'OutletNode', 'PumpCurve', 'InitStatus', 'Depth', 'ShutoffDepth', ';']
# "[ORIFICES]":         ['InletNode', 'OutletNode', 'OrificeType', 'CrestHeight', 'DischCoeff', 'FlapGate', 'OpenCloseTime', ';']
# "[WEIRS]":            ['InletNode', 'OutletNode', 'WeirType', 'CrestHeight', 'DischCoeff', 'FlapGate', 'EndCon', 'EndCoeff', 'Surcharge', 'RoadWidth', 'RoadSurf', ';']
# "[OUTLETS]":          Not possible
# "[XSECTIONS]":        ['Shape', 'Geom1', 'Geom2', 'Geom3', 'Geom4', 'Barrels', ';']
# "[LOSSES]":           ['Inlet', 'Outlet', 'Average', 'FlapGate', ';']
# "[CONTROLS]":         [CONTROLS]
# "[DWF]":              ['Parameter', 'AverageValue', 'TimePatterns', ';']
# "[CURVES]":           Difficult
# "[PATTERNS]":         Difficult
# "[REPORTING]":        Difficult
# "[MAP]":              Mistake in the code
# "[COORDINATES]":      ['X', 'Y', ';']
# "[VERTICES]":         ['X', 'Y', ';']
# "[Polygons]":         ['X', 'Y', ';']
# "[SYMBOLS]":          Difficult
# "[LABELS]":           Difficult
    
def change_model_property(inp_path_original, section, ID_list, parameter, replacement_values = list(), multiplying_value = list(), add_value = list(), new_file_path = list()):
    import shutil
    from swmmio import create_dataframeINP
    from swmmio.utils.modify_model import replace_inp_section
    
    section_df = create_dataframeINP(inp_path_original, section)
    if len(replacement_values) > 0:
        section_df.loc[ID_list,parameter] = replacement_values[0]
    elif len(multiplying_value)  > 0:
        section_df.loc[ID_list,parameter] = section_df.loc[ID_list,parameter] * multiplying_value[0]
    else:
        section_df.loc[ID_list,parameter] = section_df.loc[ID_list,parameter] + add_value[0]
        
    # inp_path, can it not be assigned?
    
    if new_file_path:
        shutil.copyfile(inp_path_original, new_file_path) # here make a copy of the original .inp file, then overwrite its content later with "replace_inp_section()"
        inp_path_new = new_file_path
    else:
        inp_path_new = inp_path_original
    
    replace_inp_section(inp_path_new, section, section_df)






# A function that take a model, and simulation settings, observations, and an objective function.
# It returns the value of the objective function.
def simulate_objective(inp_file, simulationStartTime, simulationEndTime, 
                       selected_nodes, selected_links, selected_subcatchments, 
                       output_time_step, add_and_remove_hotstart_period, hotstart_period_h,
                       observations_df,
                       objective_function):
    ### Function that runs a swmm model, loads observations, aligns sim+obs, returns objective function
    import numpy as np
    import pandas as pd
    
    # simulate desired time period
    swmm_simulation_results = swmm_simulate(inp_file, simulationStartTime, simulationEndTime,
                                            selected_nodes, selected_links, selected_subcatchments,
                                            output_time_step, add_and_remove_hotstart_period, hotstart_period_h)
    
    simulated_variables = list(swmm_simulation_results.keys())[1] # get the name of the simulated variable (only works for one simulated variable, not multiple ones)
    
    # locate observations for simulation period
    obs_start_idx = np.where(observations_df['time'] == pd.to_datetime(simulationStartTime))
    obs_start_idx = int(obs_start_idx[0])
    obs_end_idx = np.where(observations_df['time'] == pd.to_datetime(simulationEndTime))
    obs_end_idx = int(obs_end_idx[0]) - 1
    
    # align model and observation time steps
    swmm_results_interpolated = interpolate_swmm_times_to_obs_times(swmm_simulation_results['time'], 
                                                                    swmm_simulation_results[simulated_variables], 
                                                                    observations_df.loc[obs_start_idx:obs_end_idx,'time'],
                                                                    observations_df.loc[obs_start_idx:obs_end_idx,'value_no_errors'])
    
    # obs = observations_df.loc[obs_start_idx:obs_end_idx,'value_no_errors'].reset_index(drop=True)
    # mod = swmm_results_interpolated['values'].reset_index(drop=True)
    
    nan_bool = np.isnan(swmm_results_interpolated['sim_values']) | np.isnan(swmm_results_interpolated['obs_values'])
    
    obs = swmm_results_interpolated['obs_values'][np.invert(nan_bool)]
    mod = swmm_results_interpolated['sim_values'][np.invert(nan_bool)]
    
    # calculate objective function
    objective_output = objective_function(obs, mod)
    
    return(objective_output)




# Define all potential sections, where something could be calibrated... needs to be defined for if-else part that determines object_ids
# subs are ids: "[SUBCATCHMENTS]", "[SUBAREAS]", "[INFILTRATION]"
# conduits/links are ids: "[CONDUITS]"
# nodes/junctions are ids: "[DWF]"

# Function that can change multiple parameters in a SWMM model, and save it as a new model. Builds on the "change_model_property"-function.
def create_new_model(multiplying_values, 
                     original_model_inp_file, new_file_path,
                     parameter_sections, parameter_names, num_model_parameters,
                     subs_to_modify, links_to_modify, nodes_to_modify):
    
#    breakpoint()
    
    # Create a model with the new parameter set
    for param_num in range(num_model_parameters):
        
        if parameter_sections[param_num] in ["[SUBCATCHMENTS]", "[SUBAREAS]", "[INFILTRATION]"]:
            object_ids = subs_to_modify["id"]
        elif parameter_sections[param_num] == "[CONDUITS]":
            conduit_idx = links_to_modify["type"] == "conduit"
            object_ids = links_to_modify["id"][conduit_idx]
        elif parameter_sections[param_num] == "[DWF]":
            junction_idx = nodes_to_modify["type"] == "junction"
            object_ids = nodes_to_modify["id"][junction_idx]
            
#        breakpoint()
        
        
        # if it is the first parameter, we create a new .inp-file, otherwise we overwrite the new file
        if param_num == 0:
            change_model_property(original_model_inp_file, 
                                  parameter_sections[param_num], 
                                  object_ids, 
                                  parameter_names[param_num], 
                                  multiplying_value = [multiplying_values[param_num]],
                                  new_file_path = new_file_path)
 #           breakpoint()
        else:
            change_model_property(new_file_path, 
                                  parameter_sections[param_num], 
                                  object_ids, 
                                  parameter_names[param_num], 
                                  multiplying_value = [multiplying_values[param_num]])   
    


# A function that takes a desired parameter specification, creates a new model based on those, and executes the "simulate_objective"-function.
def create_runobjective_delete_model(multiplying_values, 
                                     original_model_inp_file, new_file_path,
                                     parameter_sections, parameter_names, num_model_parameters,
                                     nodes_to_modify, links_to_modify, subs_to_modify,
                                     simulationStartTime, simulationEndTime, 
                                     selected_nodes, selected_links, selected_subcatchments, 
                                     output_time_step, add_and_remove_hotstart_period, hotstart_period_h,
                                     observations_loaded,
                                     objective_function):
    import os
    
    
    create_new_model(multiplying_values, 
                     original_model_inp_file, new_file_path,
                     parameter_sections, parameter_names, num_model_parameters,
                     subs_to_modify, links_to_modify, nodes_to_modify)
    
    # Run new model and get objective
    my_objective = simulate_objective(new_file_path, simulationStartTime, simulationEndTime, 
                                      selected_nodes, selected_links, selected_subcatchments, 
                                      output_time_step, add_and_remove_hotstart_period, hotstart_period_h,
                                      observations_loaded,
                                      objective_function)
    
    # Delete the .inp file that this simulation was based on
    os.remove(new_file_path)
    
    # Return the value of the objective function
    return(my_objective)




def generate_run_lhs(num_lhs_samples, parameter_ranges,
                     original_model_inp, temp_folder_path,
                     parameter_sections, parameter_names, num_model_parameters,
                     nodes_to_modify, links_to_modify, subs_to_modify,
                     simulationStartTime, simulationEndTime, 
                     selected_nodes, selected_links, selected_subcatchments, 
                     output_time_step, add_and_remove_hotstart_period, hotstart_period_h,
                     observations_df,
                     objective_function,
                     only_best = False):
    
    import os, shutil
    import numpy as np
    import pyDOE2_lhs # used for Latin Hypercube Sampling

    os.mkdir(temp_folder_path)

    new_file_name = "new_temp_model.inp"
    new_file_path = os.path.join(temp_folder_path, new_file_name)


    params_latin_pydoe = pyDOE2_lhs.lhs(num_model_parameters, samples=num_lhs_samples, criterion="maximin", iterations=50)
    
    for par_id in range(num_model_parameters):
        for sample_id in range(num_lhs_samples):
            params_latin_pydoe[sample_id, par_id] = parameter_ranges[par_id][0] + params_latin_pydoe[sample_id, par_id] * np.diff(parameter_ranges[par_id])
    
    # Include a model run with the default parameters
    params_latin_pydoe = np.concatenate((np.ones((1, num_model_parameters), dtype=float), params_latin_pydoe), axis = 0)
    
    
        
    ### Loop over each parameter set from LHS
    
    lhs_objective_values = np.zeros(num_lhs_samples+1,dtype=float)
    
    for lhs_model_num in range(num_lhs_samples+1):
        print("Running LHS parameter set " + str(lhs_model_num+1) + " out of " + str(num_lhs_samples+1))
        
        lhs_objective_values[lhs_model_num] = create_runobjective_delete_model(params_latin_pydoe[lhs_model_num,:],
                                                                               original_model_inp, new_file_path,
                                                                               parameter_sections, parameter_names, num_model_parameters,
                                                                               nodes_to_modify, links_to_modify, subs_to_modify,
                                                                               simulationStartTime, simulationEndTime, 
                                                                               selected_nodes, selected_links, selected_subcatchments, 
                                                                               output_time_step, add_and_remove_hotstart_period, hotstart_period_h,
                                                                               observations_df,
                                                                               objective_function)
    
    # Delete temp folder
    shutil.rmtree(temp_folder_path)
    
    best_lhs_parameter_set = params_latin_pydoe[np.argmin(lhs_objective_values),:]
    best_lhs_objective_value = np.min(lhs_objective_values)

    if only_best == True:
        return(best_lhs_parameter_set, best_lhs_objective_value)
    else:
        return(params_latin_pydoe, lhs_objective_values)
    
    
    
def run_simplex_routine(start_param_values, 
                        original_model_inp, temp_folder_path,
                        parameter_sections, parameter_names, parameter_ranges, num_model_parameters,
                        nodes_to_modify, links_to_modify, subs_to_modify,
                        simulationStartTime, simulationEndTime,
                        selected_nodes, selected_links, selected_subcatchments,
                        output_time_step, add_and_remove_hotstart_period, hotstart_period_h,
                        observations_df,
                        objective_function,
                        max_iterations = None,
                        only_best = False):
    
    import os, shutil
    import numpy as np
    from scipy.optimize import minimize
    
    os.mkdir(temp_folder_path)

    new_file_name = "new_temp_model.inp"
    new_file_path = os.path.join(temp_folder_path, new_file_name)
    
    global simplex_params, simplex_objectives, simplex_iteration # unfornately, we have to make these variables global in order to obtain them - otherwise Scipy won't let you get them
    
    # Define varialbes to store all simplex trials    
    simplex_params = []
    simplex_objectives = []
    simplex_iteration = 0
    
    # specify arguments that are formatted for scipy's "minimize" function
    args_for_scipy = (original_model_inp, new_file_path,
     parameter_sections, parameter_names, parameter_ranges, num_model_parameters,
     nodes_to_modify, links_to_modify, subs_to_modify,
     simulationStartTime, simulationEndTime,
     selected_nodes, selected_links, selected_subcatchments,
     output_time_step, add_and_remove_hotstart_period, hotstart_period_h,
     observations_df,
     objective_function)
    
    # Define whether the simplex algorithm should have a maximum number of iterations
    if max_iterations == None:
        sim_options = {'disp': True}   #{'maxiter': 10, 'disp': True},
    else:
        sim_options = {'maxfev': max_iterations, 'disp': True}
    
    # Run the simplex algorithm
    res = minimize(simplex_objective, start_param_values, args = args_for_scipy, method='nelder-mead', callback = simplex_callback,
                   options = sim_options)
    
    # Delete temp folder
    shutil.rmtree(temp_folder_path)
    #os.rmdir(temp_folder_path)
    
    # gather results of which parameter sets that were evaluated, and which objective function values were obtained
    simplex_objectives = np.asarray(simplex_objectives)
    evaluated_parameter_sets = np.vstack(simplex_params[0:len(simplex_objectives)])
    
    # remove parameter sets that were outside the user-specified range
    appropriate_parameter_sets = simplex_objectives != 10e9
    evaluated_function_values = simplex_objectives[appropriate_parameter_sets]
    evaluated_parameter_sets = evaluated_parameter_sets[appropriate_parameter_sets, :]
    
    # get the best performing parameter set
    best_simplex_parameter_set = evaluated_parameter_sets[np.argmin(evaluated_function_values),:]
    best_simplex_objective_value = np.min(evaluated_function_values)

    if only_best == True:
        return(best_simplex_parameter_set, best_simplex_objective_value)
    else:
        return(evaluated_parameter_sets, evaluated_function_values)
    
    
# callback is a function that does *something* (here, printing the iteration of the optimization) between each simplex trial
def simplex_callback(x):
    global simplex_iteration
    simplex_iteration += 1
    #simplex_iteration = simplex_iteration + 1
    print(simplex_iteration)
    
    
def simplex_objective(params, *args):
    #global simplex_params, simplex_objectives # unfornately, we have to make these variables global in order to obtain them - otherwise Scipy won't let you get them
    
    original_model_inp, new_file_path = args[0], args[1]
    parameter_sections, parameter_names, parameter_ranges, num_model_parameters = args[2], args[3], args[4], args[5]
    nodes_to_modify, links_to_modify, subs_to_modify = args[6], args[7], args[8]
    simulationStartTime, simulationEndTime = args[9], args[10]
    selected_nodes, selected_links, selected_subcatchments = args[11], args[12], args[13]
    output_time_step, add_and_remove_hotstart_period, hotstart_period_h = args[14], args[15], args[16]
    observations_df = args[17]
    objective_function = args[18]
    
    simplex_params.append(params) # save the parameter set that was tested by the lhs algorithm
    
    param_bounds = tuple([tuple(parameter_ranges[i]) for i in range(len(parameter_ranges))])
    
    # make a check to see that the parameter values that the algorithm wants to test, is within the user-specified parameter boundaries
    param_bound_check = [False]*len(params)
    for i in range(len(params)):
        param_bound_check[i] = params[i] < param_bounds[i][0] or params[i] > param_bounds[i][1]
    
    if sum(param_bound_check) == 0:
        my_objective = create_runobjective_delete_model(params,
                                                        original_model_inp, new_file_path,
                                                        parameter_sections, parameter_names, num_model_parameters,
                                                        nodes_to_modify, links_to_modify, subs_to_modify,
                                                        simulationStartTime, simulationEndTime,
                                                        selected_nodes, selected_links, selected_subcatchments,
                                                        output_time_step, add_and_remove_hotstart_period, hotstart_period_h,
                                                        observations_df,
                                                        objective_function)
    else:
        my_objective = 10e9 # if selected parameters are outside the user-specified range, return a very large objective function value


    simplex_objectives.append(my_objective) # save the objective function value that the optimization calculated
    
    return(my_objective)
    
