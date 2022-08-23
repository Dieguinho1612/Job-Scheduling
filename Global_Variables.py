#!/usr/bin/env python
# coding: utf-8

# This is a dummy notebook to save global variables that remain constant throughout a Job Scheduling problems.
# In specific, these are:
#    - list_jobs
#    - list_machines
#    - n
#    - m
#    - max_init_runtime
#    - max_runtime
#    - max_deadline
#    - max_weight

# In[1]:


#import import_ipynb
from Jobs_and_Machines import *


# In[5]:


def set_var_to_global(dict_of_var):
    
    global_variables = globals()
    global_variables.update(dict_of_var)

