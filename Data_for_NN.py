#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pickle
import os
import random
import time
#import necessary notebooks
#import import_ipynb
from Jobs_and_Machines import *
from States_and_Policies import *
from Global_Variables import *


# In[6]:


#take information of state to create normalized data for Neural Network
def state_input(state):
    
    #normalized information about the jobs
    jobs_data = np.asarray([[(proc_time/max_runtime) for i, proc_time in enumerate(job.processing_time)
                             if state.machines_on_duty[i] == 1]
                            + [max(job.deadline-state.time,0)/max_deadline,
                               job.weight/max_weight]
                            for job in list_jobs if state.jobs_remaining[job.index] == 1])
    
    #normalized information about the machines
    machines_data = np.asarray([[state.machine_runtimes[machine.index]/max_runtime,
                                 max(machine.deadline-state.time,0)/max_deadline,
                                 machine.weight/max_weight] 
                                for machine in list_machines if state.machines_on_duty[machine.index] == 1])
                             
    #sort them and save permutation
    machines_perm = machines_data[:,0].argsort() #sort machines by remaining runtime
    orig_order = np.arange(len(machines_perm)) #just an array of the form [0,1,...,m_state-1]
    jobs_data[:,orig_order] = jobs_data[:,machines_perm] #reorder processing time of jobs by new order of machines
    jobs_perm = jobs_data[:,0].argsort() #order of jobs by processing time for current free machine
    jobs_data = jobs_data[jobs_perm] #sort jobs by this order
    machines_data = machines_data[machines_perm] #sort machines by their remaining runtime
    
    #merge
    state.input = [jobs_data, machines_data]
    state.permutation = [jobs_perm,machines_perm]


# In[7]:


def state_target(state):
    
    #number of remaining jobs
    n_state = sum(state.jobs_remaining)
    #get Qvalues of all feasible actions
    target = np.array([qvalue for qvalue in state.Qvalues if qvalue != None])
    #sort by permutation
    target[np.arange(n_state)] = target[state.permutation[0]]
    state.target = [target, 
                    np.eye(target.shape[0])[np.argmin(target)], #one_hot_vector of optimal action
                   np.min(target)/target] #normalize by scaling through minimum value and then taking inverse value


# In[9]:


#we set a maximum to how many data points we want to have for each (n,m)-situation with Job i being the optimal action
def create_data(all_states, data_points_max, save=False):
    
    #measure start time
    st = time.time()
    
    #the minimum amount of jobs and machines a state has to have for us to be interesteing enough to save its data
    n_min = 3
    m_min = 2
    
    #will be a tuple consisting of inputs list and targets list
    data_dictionary = dict(((n_state,m_state),([],[])) 
                           for n_state in range(n_min,n+1) for m_state in range(m_min,m+1))
    #counter of how many data points there are already for each job i to be the optimal action (+option of machine shut down)
    data_points_counter = dict(((n_state,m_state,i),0) 
                               for n_state in range(n_min,n+1) for m_state in range(m_min,m+1) for i in range(n_state+1))
    
    #permutations that will be added in data later on
    #we add the max_runtime+1 as last entry, so that "n" (=turning off machine) is always the last entry of permutation
    permutations = [np.argsort([job.processing_time[i] for job in list_jobs]+[max_runtime+1]) for i in range(m)]
    
    #create data for states
    for state in all_states:
        n_state = sum(state.jobs_remaining)
        m_state = sum(state.machines_on_duty)
        if n_state >= n_min and m_state >= m_min:
            #find out which of the n_state jobs + machine shut down is best action
            #rev_perm_target = np.array([state.Qvalues[i] for i in permutations[state.machine] if state.Qvalues[i] != None][::-1])
            rev_perm_target = np.array([qvalue for qvalue in np.array(state.Qvalues)[permutations[state.machine]][::-1] if qvalue != None])
            opt_action = len(rev_perm_target) - np.argmin(rev_perm_target) - 1 #reversed for emphasis on higher indices equality cases
            #opt_action = minimum(state)[1][0] #which job action corresponds to minimum cost
            if len(data_dictionary[(n_state,m_state,opt_action)][1]) < data_points_max:
            #if data_points_counter[(n_state,m_state,opt_action)] < data_points_max/len(rev_perm_target):
                state_input(state)
                data_dictionary[(n_state,m_state)][0].append(state.input)
                state_target(state)
                data_dictionary[(n_state,m_state)][1].append(state.target[2])
                data_points_counter[(n_state,m_state,opt_action)] += 1
                
    #measure end time
    et = time.time()
    
    #tell how much the entire process took
    print(round(et-st,2), "seconds to compute", sum(len(data_dictionary[key][0]) for key in data_dictionary), "data points.")
    
    if save:
        with open('data.pickle', 'wb') as f:
            pickle.dump(data_dictionary, f, pickle.HIGHEST_PROTOCOL)
            
    return data_dictionary


# In[ ]:


def store_data(all_states, data_points_max, MVS, JS):
    
    data = create_data(all_states, data_points_max)
    
    MVS_str = "0"*(2-len(str(MVS))) + str(MVS)
    JS_str = "0"*(4-len(str(JS))) + str(JS)
    path = f'MaxValuesSets/MaxValues_{MVS_str}/Data_{MVS_str}/data_{MVS_str}_{JS_str}.pickle'
    with open(path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


# In[2]:


"""def create_MLP_data(all_states, n_max, m_max):
    training_dictionary = dict(((n+1,m+1),([],[])) for n in range(n_max) for m in range(m_max))
    for state in all_states:
        n = sum(state.jobs_remaining)
        m = sum(state.machines_on_duty)
        if (n == 1 and m > 1) or n > 2:
            training_dictionary[(n,m)][0].append(np.concatenate((state.input[0].flatten(),state.input[1].flatten())))
            training_dictionary[(n,m)][1].append(state.target[2]) #for regression
            #training_dictionary[(n,m)][1].append(state.target[1]) #for classification
            #training_dictionary[(n,m)][1].append(Softmax(state.target[2])) #for mix

    for tupl in training_dictionary:
        (x, y) = training_dictionary[tupl]
        #print(len([np.concatenate(x)]), len([np.concatenate(y)]))
        if x:
            training_dictionary[tupl] = ([np.stack(x)],[np.stack(y)])
        
    return training_dictionary""";


# In[ ]:


"""def store_MLP_data(all_states, n_max, m_max , MVS, JS):
    
    MLP_data = create_MLP_data(all_states, n_max, m_max)
    
    MVS_str = "0"*(2-len(str(MVS))) + str(MVS)
    JS_str = "0"*(4-len(str(JS))) + str(JS)
    path = f'MaxValuesSets/MaxValues_{MVS_str}/MLP_Data_{MVS_str}/MLP_data_{MVS_str}_{JS_str}.pickle'
    with open(path, 'wb') as f:
        pickle.dump(MLP_data, f, pickle.HIGHEST_PROTOCOL)""";

