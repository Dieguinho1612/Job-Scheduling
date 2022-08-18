#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
#import necessary notebooks
#import import_ipynb
from Jobs_and_Machines import *
from States_and_Policies import *


# In[2]:


def create_MLP_data(all_states, n_max, m_max):
    training_dictionary = dict(((n+1,m+1),([],[])) for n in range(n_max) for m in range(m_max))
    for state in all_states:
        n = sum(state.jobs_remaining)
        m = sum(state.machines_on_duty)
        if n > 0:
            training_dictionary[(n,m)][0].append(np.concatenate((state.input[0].flatten(),state.input[1].flatten())))
            training_dictionary[(n,m)][1].append(state.target[2]) #for regression
            #training_dictionary[(n,m)][1].append(state.target[1]) #for classification
            #training_dictionary[(n,m)][1].append(Softmax(state.target[2])) #for mix

    for tupl in training_dictionary:
        (x, y) = training_dictionary[tupl]
        #print(len([np.concatenate(x)]), len([np.concatenate(y)]))
        if x:
            training_dictionary[tupl] = ([np.stack(x)],[np.stack(y)])
        
    return training_dictionary


# In[ ]:


def store_MLP_data(all_states, n_max, m_max , MVS, JS):
    
    MLP_data = create_MLP_data(all_states, n_max, m_max)
    
    MVS_str = "0"*(2-len(str(MVS))) + str(MVS)
    JS_str = "0"*(4-len(str(JS))) + str(JS)
    path = f'MaxValuesSets/MaxValues_{MVS_str}/JobScheduling_{JS_str}/MLP_data_{MVS_str}_{JS_str}.pickle'
    with open(path, 'wb') as f:
        pickle.dump(all_states, f, pickle.HIGHEST_PROTOCOL)

