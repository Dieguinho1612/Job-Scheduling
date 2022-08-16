#!/usr/bin/env python
# coding: utf-8

# # Deterministic Job Scheduling
# The problem of Deterministic Job Scheduling is a well known mathematical problem of the field of Discrete Optimization.<br>
# We consider the special case of having unrelated parallel Machines.<br>
# While there already exist many approximation algorithms, we want to encounter this problem using a Deep Reinforcement Learning approach to see if we can further improve the results.<br>
# The idea is to obtain a Neural Network that provides the user with a reasonable Policy of action for any given set of Jobs and Machines.<br>
# The Neural Network is defined by the amount <em>m</em> of Machines, but can handle any amount of Jobs up to a maximum amount of <em>n</em> Jobs.
# <em>n</em> and <em>m</em> depend on how it was trained.<br>
# For the training we will deviate a bit from the classical way Deep Reinforcement Learning is handled by priorly computing the Optimal Policy to a certain Job Scheduling Problem and train the Neural Network on these known Q-Values, but still use the underlying decision of training structure of Reinforcement Learning, therefore turning it into some form of supervised learning. 

# ## Description of Notebook
# This is the main notebook for the problem of Deterministic Job Scheduling.<br>
# Here we will merge all the classes, functions and methods defined in the other notebooks.<br>
# 
# The user starts be defining the Machines he wants to use together with their properties.<br>
# Afterwards the Jobs that need to be processed get specified.<br>
# 
# From here on, by executing the corresponding cells, all States get computed together with their Q-Values as well as the Optimal Policy.<br>
# 
# These then are used to compute the Neural Network together with the Inputs and Targets for training, validation and testing.<br>
# 
# If the user wishes to create further list of Jobs, Machines and States with their Optimal Policy, this can be done by adding <br><em>"name=..."</em><br>
# into the input of the specific function.<br>
# Alternatively, the user can use the "Side"-notebook for this.

# ## Code

# In[1]:


import random
#import import_ipynb


# ### Jobs and Machines

# In[2]:


from Jobs_and_Machines import *


# #### Max Values
# Give the maximum values for (initial) processing time, deadline and weight.

# In[3]:


max_init_runtime = 5 #only important for generating machines randomly
max_runtime = 20
max_deadline = 30
max_weight = 20

print(f"The maximal Initial Runtime is {max_init_runtime}.")
print(f"The maximal Runtime is {max_runtime}.")
print(f"The maximal Deadline is {max_deadline}.")
print(f"The maximal Weight is {max_weight}.")


# #### Machines
# Create the set of Machines.<br>
# They will be sorted by their weight in descending order and then be indexed und numbered. In case of equal weights, the sorting will rely on their deadlines.<br>
# At least one Machine has to start with an initial runtime of zero. 

# In[4]:


#machines(initial runtime, deadline, weight)
list_machines = [machines(4,10,5), machines(2,9,1), machines(0,12,3)]

#by adding "name=..." in the input, the list can be saved under a different name (default name is list_machines)
list_machines = prep_machines(list_machines, save=True, prnt=True)


# #### Jobs
# Follow by creating the set of Jobs.<br>
# These Jobs will then be sorted by their weights in descending order and indexed and numbered as well. Again, if weights are equal for two machines, they get ordered by their deadline.<br>
# When giving the processing times of every job on each machine it has to be considered that the order of the machines may have change due to their weights from the order in which they were originally given.

# In[5]:


#jobs(list of processing times, deadline, weight)
list_jobs = [jobs([4,7,5], 12, 1), jobs([1,12,14], 8, 2), jobs([8,8,8], 15, 3),
             jobs([9,4,2], 7, 4), jobs([2,2,6], 12, 16), jobs([5,7,2], 9, 12),
             jobs([10,12,20],21, 3), jobs([15,10,8],6, 3), jobs([6,4,12],26,4)]

#by adding "name=..." in the input, the list can be saved under a different name (default name is list_jobs)
list_jobs = prep_jobs(list_jobs, save=True, prnt=True)


# #### Dimensions

# In[6]:


#define dimensions
n = len(list_jobs)
m = len(list_machines)

print(n,"Jobs and",m,"Machines")


# ### States and Policies

# In[7]:


from States_and_Policies import *


# #### States
# Compute all States together with their Q-Values corresponding to every possible action.

# In[8]:


#by adding "name=..." in the input, the list can be saved under a different name (default name is all_states)
all_states = compute_all_states(list_jobs, list_machines, max_runtime, max_deadline, max_weight, save=True)


# #### Policies
# We compute the Optimal Policy as well as a random one, to have some base of comparision.

# In[9]:


#compute a random policy
random.seed(3)
rand_policy = random_policy(all_states)


# In[10]:


#compute the optimal policy
    #by adding "name=..." in the input, the policy can be saved under a different name (default name is opt_policy)
opt_policy = optimal_policy(all_states)


# ### Visualization
# We wish to visualize the Optimal Policy.<br>

# In[11]:


#visualize random example
from Visualizations import *


# This can be done by:
#    - Having it explained in words
#    - Plotting it as a graph
#    - Plotting it as a gantt chart

# In[12]:


#by words
explain_policy(opt_policy)


# In[13]:


#as a graph
draw_policy_graph(opt_policy)


# In[14]:


#as a gantt chart
draw_gantt_chart(opt_policy)


# We also might want to compare it to a random Policy.

# In[15]:


#plot random policy as a gantt chart
draw_gantt_chart(rand_policy)


# ### Input and Targets for Neural Network
# We  can now use our obtained data to train, validate and test our Neural Network.
# 
# The obtained States will be used to create the Inputs for our Neural Network, while their Q-Values will be the Target values that shall be learned.

# ### Commentaries for future ideas:
# 
# - One could also limit deadlines of jobs to be greater than their processing times.
# - Different max values for deadline and weight could be given for jobs vs machines.
# - use Boosting
# - Embedding Layer für One-Hot-vektoren
