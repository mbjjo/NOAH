# -*- coding: utf-8 -*-
"""
Copyright 2019 Magnus Johansen
Copyright 2019 Technical University of Denmark

This file is part of NOAH RTC Tool.

NOAH RTC Tool is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

NOAH RTC Tool is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with NOAH RTC Tool. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np

def User_defined_objective_function(obs,mod):
    # This function can be edited and used as an objective function during the calibration. 
    # Takes input observed data (obs) and modeled data (mod)
    # Returns a single value
    # An example is provided below:
    abs_rel_peak_error = abs((max(mod) - max(obs))/max(obs))
    correlation_coef = np.corrcoef(mod,obs)[0][1]
    mixed_objective = abs_rel_peak_error - correlation_coef
    return(mixed_objective)
