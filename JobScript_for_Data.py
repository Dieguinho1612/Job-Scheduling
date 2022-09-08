#!/usr/bin/env python
# coding: utf-8

# ## Import Modules

# In[1]:


import sys
import os
import pickle
import random
#import import_ipynb
from Jobs_and_Machines import *


# ## Set JS Environment

# In[2]:


DS = 1
DS_str = "0"*(2-len(str(DS))) + str(DS)


# In[3]:


if not os.path.exists('Data'):
    os.mkdir('Data')
    os.mkdir(f'Data/DataSet_{DS_str}')
    sys.exit()
elif not os.path.exists(f'Data/DataSet_{DS_str}'):
    os.mkdir(f'Data/DataSet_{DS_str}')
    sys.exit()


# In[4]:


n = 6
m = 3


# In[ ]:


max_weight = 10
max_deadline = 30
max_runtime = random.randint(round(max_deadline/3),round(max_deadline*1.5))
max_init_runtime = random.randint(round(max_runtime/3), max_runtime)


# In[5]:


#input = sys.argv[1]
input = int(input) - 1


# ## Pass Parameters as Global Variables to imported Notebooks

# In[6]:


import Global_Variables

#create dictionary of all constant variables of this JS
list_var = ['n', 'm', 'max_init_runtime', 'max_runtime', 'max_deadline', 'max_weight']
dict_var = dict((var,eval(var)) for var in list_var)
#pass them as global variables, so that imported notebooks can access them
Global_Variables.set_var_to_global(dict_var)


# ## Create Random List of Jobs and Machines

# In[7]:


from Random_Generator import *


# In[8]:


#create txt file of parameters and required folders if they dont exist already
#generate_max_values_sets(MVS)


# In[9]:


list_jobs, list_machines = generate_random_JS(DS, input, prnt=False)


# In[10]:


"""zeros = "0"*(4-len(str(input)))
random_lists_path = f'MaxValuesSets/MaxValues_{MVS_str}/Jobs_and_Machines_{MVS_str}/random_lists_{MVS_str}_{zeros}{input}.pickle'
with open(random_lists_path, 'rb') as f:
        list_jobs, list_machines = pickle.load(f)""";


# In[11]:


#pass list of jobs and machines to global variables
Global_Variables.set_var_to_global({'list_jobs':list_jobs,
                                   'list_machines':list_machines})


# ## Compute All States

# In[12]:


from States_and_Policies import *


# In[13]:


all_states = compute_all_states(list_jobs, list_machines, MVS=DS, JS=input)


# ## Save Data

# In[14]:


from Data_for_NN import *


# In[15]:


data_points_max = 100


# In[16]:


random.shuffle(all_states)


# In[17]:


store_data(all_states, data_points_max, DS, input)


# ## Back-Up-Code

# In[18]:


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


# In[19]:


"""training_data = create_MLP_data(all_states, n, m)""";


# In[20]:


"""store_MLP_data(all_states, n, m, MVS=1, JS=input)""";


# Ändere Ordner Struktur zu:<br>
# - Ordner MaxValuesSets
#     - File MaxValues 01
#     - Ordner MaxValues 01
#         - Ordner JS 0001
#             - File RandomLists 0001
#             - File States 0001
#             - File NN-Data 0001
#             
# Diese Struktur muss auch geändert werden bei allen anderen Notebooks!  

# Siehe, was passiert, wenn du weitere Jobs oder Maschinen auf NN anwendest (also mehr, als das, worauf es trainiert wurde).
# Vergleiche die Kosten mit den Kosten verschiedener Approx-Algorithmen.
# Für jedes (n,m)-Tupel 10.000 JS mit jedem Algo. und mit NN durchlaufen lassen und average Werte vergleichen.
# Auch für (n,m)-Werte innerhalb der Trainingskala die Ergebnisse so mit Algorithmen vergleichen.
