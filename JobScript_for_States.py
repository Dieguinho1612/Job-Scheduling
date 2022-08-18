#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pickle
#import import_ipynb
from Jobs_and_Machines import *
from States_and_Policies import *
from Random_Generator import *
from Data_for_NN import *


# In[2]:


MVS = 1
MVS_str = "0"*(2-len(str(MVS))) + str(MVS)


# In[3]:


[n,m,max_init_runtime,max_runtime,max_deadline,max_weight] = read_max_parameters(MVS)


# In[4]:


input = sys.argv[1]


# In[5]:


generate_random_JS(MVS, input, prnt=False)


# In[6]:


zeros = "0"*(4-len(str(input)))
random_lists_path = f'MaxValuesSets/MaxValues_{MVS_str}/Jobs_and_Machines_{MVS_str}/random_lists_{MVS_str}_{zeros}{input}.pickle'
with open(random_lists_path, 'rb') as f:
        list_jobs, list_machines = pickle.load(f)


# In[7]:


all_states = compute_all_states(list_jobs, list_machines, max_runtime, max_deadline, max_weight, MVS=MVS, JS=input)


# In[ ]:


store_MLP_data(all_states, n, m, MVS=1, JS=input)


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
