# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:03:24 2020

@author: Magnus Johansen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import swmmtoolbox.swmmtoolbox as swmmtoolbox



class Optimizer:
    def __init__(self):
        self.report_times_steps = 1
        self.model_outfile = 'BellingeSWMM_MU_v017a_pumptest.out'
        # sensor = 'G80F13P'
        # pump = 'G80F13Pp1'
        # rg = 'rg5425'
        self.CSO_id1 = 'G80F13P'
        self.CSO_id2 = 'G80F240'
        self.CSO_id3 = 'G80F11B'

        # self.objective_value = self.count_CSO_volume([self.CSO_id1,self.CSO_id2,self.CSO_id3],self.model_outfile)
        self.objective_value = self.count_CSO_events([self.CSO_id1,self.CSO_id2,self.CSO_id3],self.model_outfile)

        print(self.objective_value)
            
            
model_outfile = 'BellingeSWMM_MU_v017a_pumptest.out'
CSO_id1 = 'G80F13P'
CSO_id2 = 'G80F240'
CSO_id3 = 'G80F11B'
CSO_ids = [CSO_id1,CSO_id2,CSO_id3]
            
model_outfile = 'Astlingen.out'
CSO_id1 = 'T3'
CSO_id2 = 'T6'
CSO_id3 = 'T1'
CSO_ids = [CSO_id1,CSO_id2,CSO_id3]
    def count_CSO_events(self,CSO_ids, model_outfile):
        # create the dataframe with the 
            
        df = pd.concat(((swmmtoolbox.extract(model_outfile,['node',CSO_ids[i],'Flow_lost_flooding']))for i in range(len(CSO_ids))),axis = 1)
        # df is CMS therefore this are converted to m3/timestep. 
        df = df*60*5#*self.report_times_steps
        
        # Set all timesteps with flooding to 1
        CSO_YesNo_df = df.mask(df>0,other = 1)
        # Time between events is assumed to be 12 hours before they are counted as seperate
        time_between_events= 12*60/5 #/self.report_times_steps
        
        # This is ca. time. How to correct for this exact number?!
    
        CSO_start_df = CSO_YesNo_df.where(np.logical_and(CSO_YesNo_df>0,CSO_YesNo_df.rolling(int(time_between_events)).sum()==1),other=0)
        
        # Cummulative sum to count the CSO's 
        CSO_events_counter = CSO_start_df.cumsum(axis = 0)
        def my_fun():
            CSO_events = CSO_events_counter.iloc[-1]
            return CSO_events 
        import timeit
        timeit.timeit(my_fun,number=10000) # in seconds
        
        total_CSO_events = CSO_events.sum()
        CSO_events
        
        
# index = np.where(df['node_T6_Flow_lost_flooding'] > 0)[0]
# index
# times = df.index[index]

# for i in range(len(times)-1):
#     timediff = times[i+1]-times[i]    
#     if timediff > times[2]-times[1]:
#         print('Event started at ' + str(times[i+1]))
#         print('and lasted ' + str(timediff))




#!!! OBS NOT SET MAX LENGT YET
        
        return total_CSO_events
    
    
    
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
          
        group_ids = onezero != onezero.shift() # Mistake! 
        CSO_events = onezero[group_ids].sum().sum()
        
        # number of CSO is then max(group_ids)
        # We add all events together. 
        #        CSO_events = sum([max(group_ids['node_' + str(CSO_ids[i]) + '_Flow_lost_flooding'])for i in range(len(CSO_ids))])
        
        return CSO_events
        
test = Optimizer()


"""
Checking the result:    
"""
df = pd.concat(((swmmtoolbox.extract(model_outfile,['node',CSO_ids[i],'Flow_lost_flooding']))for i in range(len(CSO_ids))),axis = 1)
df.head()
df.plot()

